"""
Record one rollout: headless sim -> drive a scenario -> rosbag2 + events JSON.

This is Phase A, and it never touches a VLM. The architectural rule for this
machine is that the simulator and the model never run at the same time; keeping
them in separate commands is what enforces it.

A scenario is a JSON file so that p1..p4 can declare rollouts without importing
harness code:

    {
      "id": "stuck_wall",
      "label": "drive straight into the wall until wheels stall",
      "expect": "stuck",
      "nav": false,
      "spawn": {"x": -2.0, "y": -0.5, "yaw": 0.0},
      "mode": "script",
      "script": [{"linear": 0.2, "angular": 0.0, "seconds": 25,
                  "label": "drive forward into the wall"}]
    }

Usage:
    python record.py scenarios/stuck_wall.json --out-dir ../p1-failure-vlm/data
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCH_FILE = HERE / "launch" / "rollout.launch.py"
DRIVE_SCRIPT = HERE / "drive.py"

# What lands in the bag. /rgb_camera is why any of this exists; the rest is what
# makes an event timeline reconstructable afterwards without a rerun.
DEFAULT_TOPICS = [
    "/rgb_camera", "/odom", "/gt_pose", "/cmd_vel", "/scan", "/tf", "/tf_static", "/clock",
]


class Proc:
    """A child process group we can reliably kill, sim included."""

    def __init__(self, name: str, cmd: list[str], log_path: Path, env: dict):
        self.name = name
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "w")
        self.p = subprocess.Popen(
            cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,  # own process group, so children die with it
        )

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self, grace_s: float = 10.0):
        if not self.alive():
            self.log.close()
            return
        try:
            os.killpg(os.getpgid(self.p.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        deadline = time.time() + grace_s
        while self.alive() and time.time() < deadline:
            time.sleep(0.2)
        if self.alive():
            print(f"[record] {self.name} ignored SIGINT — sending SIGKILL")
            try:
                os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.p.wait(timeout=5)
        self.log.close()


def ros_env() -> dict:
    env = os.environ.copy()
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    # Keep rollouts on this machine only; a stray ROS peer on the LAN would
    # otherwise inject topics into the bag.
    env.setdefault("ROS_LOCALHOST_ONLY", "1")
    return env


def wait_for_topics(wanted: list[str], timeout_s: float, env: dict) -> list[str]:
    """Poll `ros2 topic list` until every wanted topic exists (or we give up)."""
    deadline = time.time() + timeout_s
    seen: set[str] = set()
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["ros2", "topic", "list"], capture_output=True, text=True,
                timeout=20, env=env,
            ).stdout
        except subprocess.TimeoutExpired:
            continue
        seen = set(out.split())
        missing = [t for t in wanted if t not in seen]
        if not missing:
            return []
        time.sleep(2.0)
    return [t for t in wanted if t not in seen]


def wait_for_messages(topic: str, timeout_s: float, env: dict) -> bool:
    """
    A topic existing is not the same as it publishing.

    The camera in particular advertises early and then takes many seconds to
    produce its first frame, because the first offscreen render has to build the
    whole scene in software. Starting the bag before that yields an empty bag.
    """
    try:
        subprocess.run(
            ["ros2", "topic", "echo", "--once", topic],
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        return True
    except subprocess.TimeoutExpired:
        return False


def record(args) -> int:
    scenario = json.loads(Path(args.scenario).read_text())
    run_id = args.run_id or scenario.get("id") or Path(args.scenario).stem

    out_dir = Path(args.out_dir).resolve() / run_id
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} already exists — pass --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    bag_dir = out_dir / "bag"
    logs = out_dir / "logs"

    env = ros_env()
    spawn = scenario.get("spawn", {})
    use_nav = "true" if scenario.get("nav", False) else "false"

    launch_cmd = [
        "ros2", "launch", str(LAUNCH_FILE),
        f"nav:={use_nav}",
        f"x_pose:={spawn.get('x', -2.0)}",
        f"y_pose:={spawn.get('y', -0.5)}",
        f"yaw:={spawn.get('yaw', 0.0)}",
    ]
    if scenario.get("world"):
        launch_cmd.append(f"world:={scenario['world']}")

    print(f"[record] run     : {run_id}")
    print(f"[record] scenario: {scenario.get('label', '')}")
    print(f"[record] expect  : {scenario.get('expect', '(unlabelled)')}")
    print(f"[record] nav2    : {use_nav}")
    print(f"[record] out     : {out_dir}")

    sim = Proc("sim", launch_cmd, logs / "sim.log", env)
    bag = None
    drive = None
    status = "unknown"

    try:
        print("[record] waiting for topics...")
        needed = ["/odom", "/rgb_camera"]
        missing = wait_for_topics(needed, args.startup_timeout, env)
        if missing:
            status = "sim_startup_failed"
            raise SystemExit(
                f"topics never appeared: {missing}\n"
                f"  check {logs / 'sim.log'}"
            )
        print("[record] topics up; waiting for the first camera frame...")
        if not wait_for_messages("/rgb_camera", args.first_frame_timeout, env):
            status = "no_camera_frames"
            raise SystemExit(
                f"/rgb_camera advertised but published nothing within "
                f"{args.first_frame_timeout}s — check {logs / 'sim.log'}"
            )
        print("[record] camera live")

        topics = scenario.get("topics", DEFAULT_TOPICS)
        bag = Proc(
            "bag",
            ["ros2", "bag", "record", "-o", str(bag_dir), *topics],
            logs / "bag.log",
            env,
        )
        time.sleep(3.0)  # let the recorder subscribe before the robot moves

        drive_cmd = [
            sys.executable, str(DRIVE_SCRIPT),
            "--events-out", str(out_dir / "events.json"),
            "--mode", scenario.get("mode", "script"),
            "--label", scenario.get("label", run_id),
            "--expect", scenario.get("expect", ""),
        ]
        if scenario.get("mode") == "goal":
            g = scenario["goal"]
            drive_cmd += ["--goal", f"{g['x']},{g['y']},{g.get('yaw_deg', 0)}"]
            drive_cmd += ["--timeout", str(scenario.get("timeout_s", 180))]
        else:
            drive_cmd += ["--script", json.dumps(scenario["script"])]

        print("[record] driving...")
        drive = Proc("drive", drive_cmd, logs / "drive.log", env)
        cap = scenario.get("duration_cap_s", args.duration_cap)
        deadline = time.time() + cap
        while drive.alive() and time.time() < deadline:
            time.sleep(1.0)
        if drive.alive():
            print(f"[record] duration cap of {cap}s hit — stopping the driver")
            status = "duration_capped"
        else:
            status = "completed" if drive.p.returncode == 0 else "drive_failed"

    except SystemExit as e:
        print(f"[record] {e}", file=sys.stderr)
    except KeyboardInterrupt:
        status = "interrupted"
        print("\n[record] interrupted — shutting down cleanly")
    finally:
        for proc in (drive, bag, sim):
            if proc is not None:
                proc.stop()
        # Gazebo occasionally survives its launcher; make sure nothing is left
        # holding a GB of RAM after the run.
        subprocess.run(["pkill", "-f", "gz sim"], capture_output=True)

    meta = {
        "run_id": run_id,
        "status": status,
        "scenario": scenario,
        "bag": str(bag_dir),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[record] status : {status}")
    print(f"[record] bag    : {bag_dir}")
    print(f"[record] events : {out_dir / 'events.json'}")
    print(f"[record] logs   : {logs}")
    if status == "completed":
        print(
            f"\nnext:\n  python extract_keyframes.py {bag_dir} "
            f"--out {out_dir} --run-id {run_id}"
        )
    return 0 if status in ("completed", "duration_capped") else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("scenario", help="scenario JSON file")
    p.add_argument("--out-dir", required=True, help="parent dir; a run_id subdir is created")
    p.add_argument("--run-id", default=None)
    p.add_argument("--force", action="store_true", help="overwrite an existing run dir")
    p.add_argument("--startup-timeout", type=float, default=180.0,
                   help="seconds to wait for sim topics (software rendering is slow to boot)")
    p.add_argument("--first-frame-timeout", type=float, default=120.0)
    p.add_argument("--duration-cap", type=float, default=300.0,
                   help="hard stop for the driver, seconds")
    sys.exit(record(p.parse_args()))


if __name__ == "__main__":
    main()
