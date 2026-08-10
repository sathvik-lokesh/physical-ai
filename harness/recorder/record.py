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
    # Unbuffered, so a child's log survives being killed. A capped or hung run
    # is exactly when its last output matters most, and Python's default block
    # buffering throws it away on SIGKILL -- which turned one hang into a long
    # guessing game about a process that had in fact printed nothing.
    env.setdefault("PYTHONUNBUFFERED", "1")
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


class _Terminated(Exception):
    """SIGTERM, raised so the cleanup in `finally` actually runs."""


def _on_sigterm(_signum, _frame):
    raise _Terminated()


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
    # Capitalised on purpose. These are forwarded to nav2_bringup, whose own
    # launch file evaluates them with PythonExpression -- eval("false and True")
    # raises NameError, and it surfaces as an unexplained sim startup timeout.
    # "True"/"False" satisfy both Python's eval and launch's IfCondition.
    use_nav = "True" if scenario.get("nav", False) else "False"
    use_slam = "True" if scenario.get("slam", False) else "False"
    # nav + slam together is legitimate: nav2_bringup then runs slam_toolbox in
    # place of AMCL, which is how you navigate a room you are still mapping.
    # What must never happen is a standalone slam_toolbox alongside AMCL, and
    # the launch file prevents that rather than this check.

    launch_cmd = [
        "ros2", "launch", str(LAUNCH_FILE),
        f"nav:={use_nav}",
        f"slam:={use_slam}",
        f"x_pose:={spawn.get('x', -2.0)}",
        f"y_pose:={spawn.get('y', -0.5)}",
        f"yaw:={spawn.get('yaw', 0.0)}",
    ]
    if scenario.get("world"):
        launch_cmd.append(f"world:={scenario['world']}")
    if scenario.get("map"):
        launch_cmd.append(f"map:={scenario['map']}")
    if scenario.get("params_file"):
        launch_cmd.append(f"params_file:={scenario['params_file']}")

    print(f"[record] run     : {run_id}")
    print(f"[record] scenario: {scenario.get('label', '')}")
    print(f"[record] expect  : {scenario.get('expect', '(unlabelled)')}")
    print(f"[record] nav2    : {use_nav}")
    print(f"[record] slam    : {use_slam}")
    print(f"[record] out     : {out_dir}")

    # Without this, an external `kill` skips the finally-block below and leaves
    # the whole child tree running. A Nav2 bringup is fifteen processes, and
    # they kept this machine at load 20 until they were killed by hand.
    signal.signal(signal.SIGTERM, _on_sigterm)

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

        topics = list(scenario.get("topics", DEFAULT_TOPICS))
        if use_slam == "True":
            # The occupancy grid is the whole point of a mapping run, and it is
            # latched, so it has to be in the bag or the map is lost with the sim.
            topics += ["/map", "/slam_toolbox/graph_visualization"]
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
        if scenario.get("mode") in ("goal", "goto"):
            g = scenario["goal"]
            # "--goal=..." not "--goal", "...": argparse reads a value with a
            # leading minus as another option flag, and map-frame coordinates
            # are negative half the time because the frame is centred on wherever
            # the robot started. The equals form is unambiguous.
            drive_cmd += [f"--goal={g['x']},{g['y']},{g.get('yaw_deg', 0)}"]
            drive_cmd += ["--timeout", str(scenario.get("timeout_s", 180))]
            # AMCL will not localise until it is told roughly where it is, and
            # Nav2 never becomes active until it does. The robot is spawned at a
            # known pose, so hand that over rather than waiting forever.
            ip = scenario.get("initial_pose")
            if ip:
                drive_cmd += [
                    f"--initial-pose={ip['x']},{ip['y']},{ip.get('yaw_deg', 0)}"]
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
    except _Terminated:
        status = "terminated"
        print("\n[record] SIGTERM — shutting down cleanly")
    finally:
        for proc in (drive, bag, sim):
            if proc is not None:
                proc.stop()
        # Gazebo occasionally survives its launcher; make sure nothing is left
        # holding a GB of RAM after the run.
        subprocess.run(["pkill", "-f", "gz sim"], capture_output=True)
        # Nav2 nodes are launched by `ros2 launch` and do not always die with
        # it. Left behind they hold several GB and peg every core.
        subprocess.run(["pkill", "-f", "/opt/ros/jazzy/lib/nav2_"],
                       capture_output=True)

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
