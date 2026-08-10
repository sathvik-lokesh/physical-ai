"""
Drives one rollout and writes the ground-truth event timeline beside the bag.

Two drive modes, because Phase 1 needs both:

  goal    — hand a pose to Nav2 and let the full stack run. Used for nominal
            runs and for failures Nav2 itself has to notice (unreachable goal).
  script  — open-loop (linear, angular, seconds) triples published straight to
            /cmd_vel with Nav2 out of the picture. This is how you *reliably*
            cause a failure: driving into a wall is a one-line script and a
            planner fight to reproduce.

The events JSON is the ground truth the VLM is later scored against, so it is
derived from what the robot *did*, never from what the scenario intended:
"commanded forward while the true pose moved <2 cm in 6 s" is an observation,
and the label `stuck` follows from it. Labelling from intent would score the VLM
against what we meant to happen rather than what happened.

Crucially, "what the robot did" comes from the simulator's ground-truth pose
(`/gt_pose`), **not** from `/odom`. Wheel odometry integrates wheel rotation, so
a robot pinned against a wall with its wheels spinning reports travel it never
made — measured at 5.66 m of phantom movement on the `stuck_wall` scenario.
Odometry is still recorded, as `odom_path_m`, because the gap between it and the
true path is itself a result: it is precisely the situation where a telemetry-only
monitor is fooled and a camera might not be.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_msgs.msg import TFMessage

# Motion below this over the stall window counts as "not moving"
STALL_DIST_M = 0.02
STALL_WINDOW_S = 6.0

# The robot's model name in Gazebo, as spawned by spawn_tb3.launch.py
ROBOT_MODEL = "turtlebot3_waffle"

# How often the last known pose is sampled into the stall window
POSE_SAMPLE_HZ = 5.0

# Safety valve for sim-time waits: if the simulation is far slower than this
# fraction of real time (or /clock stalls entirely), give up rather than hang.
MIN_RTF = 0.05


class RolloutDriver(Node):
    def __init__(self, out_path: Path, stall_window_s: float = STALL_WINDOW_S):
        # Everything this node times is measured on the *simulation* clock.
        #
        # Wall-clock timing makes scenarios irreproducible: this machine renders
        # in software, so the real-time factor swings with load. Two recordings
        # of the same "drive forward for 40 s" script produced 7.8 m and 0.4 m of
        # travel purely because one run got a worse RTF slice. On the sim clock a
        # 40 s step is 40 simulated seconds on any machine, every time.
        super().__init__(
            "rollout_driver",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.out_path = out_path
        self.stall_window_s = stall_window_s

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Odometry, "/odom", self._on_odom, sensor_qos)
        self.create_subscription(TFMessage, "/gt_pose", self._on_gt_pose, sensor_qos)

        # Ground truth: the simulator's true pose. Everything that decides a
        # label is computed from this.
        self.pose = None          # (x, y, yaw)
        self.path_m = 0.0
        # What an onboard monitor would see: wheel-integrated odometry, which
        # counts wheel slip as travel. Recorded for comparison, never for labels.
        self.odom_pose = None
        self.odom_path_m = 0.0

        self.t_start = None   # set once the sim clock is alive
        self.events = []
        self.recent = []          # (t, x, y) ring for stall detection
        self.commanding = False   # are we currently asking the robot to move?
        self.stalled_since = None

        # Sample the pose on a clock rather than on message arrival.
        #
        # The PosePublisher plugin only emits when the pose *changes* (its
        # static mode crashes gz-sim 8, see waffle_rgbd.sdf.xacro), so a robot
        # that is truly stuck stops publishing entirely. Sampling on arrival
        # would therefore go quiet in exactly the situation it must detect,
        # while sampling the last known pose on a timer records the flat line
        # that proves the robot is not moving.
        self.create_timer(1.0 / POSE_SAMPLE_HZ, self._sample_pose)

    # --- state tracking -------------------------------------------------

    def _on_odom(self, msg: Odometry):
        """Wheel odometry — the deceptive signal, tracked but not authoritative."""
        p = msg.pose.pose.position
        if self.odom_pose is not None:
            self.odom_path_m += math.hypot(p.x - self.odom_pose[0], p.y - self.odom_pose[1])
        self.odom_pose = (p.x, p.y)

    def _on_gt_pose(self, msg: TFMessage):
        for tr in msg.transforms:
            if tr.child_frame_id != ROBOT_MODEL:
                continue
            p = tr.transform.translation
            q = tr.transform.rotation
            yaw = math.atan2(
                2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
            )
            if self.pose is not None:
                self.path_m += math.hypot(p.x - self.pose[0], p.y - self.pose[1])
            self.pose = (p.x, p.y, yaw)
            return

    def _sample_pose(self):
        if self.pose is None:
            return
        now = self.t()
        self.recent.append((now, self.pose[0], self.pose[1]))
        while self.recent and now - self.recent[0][0] > self.stall_window_s:
            self.recent.pop(0)
        self._check_stall(now)

    def _check_stall(self, now: float):
        """A stall is only meaningful while we are actually commanding motion."""
        if not self.commanding or len(self.recent) < 2:
            return
        span = now - self.recent[0][0]
        if span < self.stall_window_s:
            return
        moved = math.hypot(
            self.recent[-1][1] - self.recent[0][1],
            self.recent[-1][2] - self.recent[0][2],
        )
        if moved < STALL_DIST_M:
            if self.stalled_since is None:
                self.stalled_since = now
                self.log(
                    "stall_detected",
                    f"commanded motion but moved {moved*100:.1f} cm in {span:.1f} s",
                    moved_cm=round(moved * 100, 1),
                )
        elif self.stalled_since is not None:
            self.log("stall_cleared", f"moving again after {now - self.stalled_since:.1f} s")
            self.stalled_since = None

    def sim_now(self) -> float:
        """Seconds on the simulation clock (follows /clock via use_sim_time)."""
        return self.get_clock().now().nanoseconds / 1e9

    def t(self) -> float:
        """Seconds of simulated time since the rollout began."""
        if self.t_start is None:
            return 0.0
        return self.sim_now() - self.t_start

    def wait_sim(self, seconds: float, publish: "Twist | None" = None) -> None:
        """
        Spin until `seconds` of *simulated* time have passed.

        Bounded by a wall-clock deadline so a stalled /clock cannot hang the run.
        """
        end = self.sim_now() + seconds
        wall_deadline = time.time() + seconds / MIN_RTF + 30.0
        while self.sim_now() < end and rclpy.ok():
            if publish is not None:
                self.cmd_pub.publish(publish)
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() > wall_deadline:
                self.log(
                    "sim_too_slow",
                    f"gave up waiting for {seconds}s of sim time "
                    f"(RTF below {MIN_RTF})",
                )
                return

    def log(self, kind: str, detail: str = "", **extra):
        event = {
            "t_s": round(self.t(), 2),
            "kind": kind,
            "detail": detail,
            "x": round(self.pose[0], 3) if self.pose else None,
            "y": round(self.pose[1], 3) if self.pose else None,
            "path_m": round(self.path_m, 3),
            "odom_path_m": round(self.odom_path_m, 3),
            **extra,
        }
        self.events.append(event)
        self.get_logger().info(f"[{event['t_s']:7.2f}s] {kind}: {detail}")

    def spin_for(self, seconds: float):
        """Process callbacks for a duration of simulated time."""
        self.wait_sim(seconds)

    def wait_for_pose(self, timeout_s: float = 60.0) -> bool:
        """
        Block until both the ground-truth pose and the sim clock are alive.

        The clock matters as much as the pose: every duration in this node is
        measured on it, so starting before /clock ticks would time the first
        step against zero.
        """
        end = time.time() + timeout_s   # wall-clock, since the sim clock may not exist yet
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.pose is not None and self.sim_now() > 0.0:
                self.t_start = self.sim_now()
                return True
        return False

    # --- drive modes ----------------------------------------------------

    def run_script(self, steps: list[dict]):
        """steps: [{"linear": 0.2, "angular": 0.0, "seconds": 8, "label": "..."}]"""
        for i, step in enumerate(steps):
            lin = float(step.get("linear", 0.0))
            ang = float(step.get("angular", 0.0))
            secs = float(step.get("seconds", 1.0))
            self.commanding = abs(lin) > 1e-3 or abs(ang) > 1e-3
            self.stalled_since = None
            self.recent.clear()
            self.log(
                "step_start",
                step.get("label", f"step {i}: lin={lin} ang={ang} for {secs}s"),
                step=i, linear=lin, angular=ang, seconds=secs,
            )

            msg = Twist()
            msg.linear.x = lin
            msg.angular.z = ang
            self.wait_sim(secs, publish=msg)

            self.log("step_end", f"step {i} complete")

        self.commanding = False
        self.cmd_pub.publish(Twist())  # always leave the robot stopped
        self.spin_for(1.0)

    def run_goto(self, gx: float, gy: float, gyaw: float, timeout_s: float,
                 pos_tol: float = 0.15, yaw_tol: float = 0.15):
        """
        Drive to a pose in the map frame with a plain turn-then-go controller.

        This exists because Nav2's full stack does not fit this machine. Fifteen
        non-composed nodes alongside a software-rendered Gazebo drove the load
        average to 21 on four cores, and the driver was starved to 2.8% CPU
        before anything moved. slam_toolbox on its own is one node, so the pose
        estimate is still a real localiser's rather than the simulator's, and
        the room between the patrol ring and the objects is open floor, which is
        what makes a straight-line controller adequate here. It would not be
        adequate in a cluttered room, and that is the honest limit of this demo.
        """
        from tf2_ros import Buffer, TransformListener, LookupException, \
            ConnectivityException, ExtrapolationException

        buf = Buffer()
        TransformListener(buf, self)
        self.commanding = True
        self.log("goto_start", f"heading for map ({gx:.2f}, {gy:.2f}, "
                               f"{math.degrees(gyaw):.0f} deg)")

        deadline = self.sim_now() + timeout_s
        wall_deadline = time.time() + timeout_s / MIN_RTF + 60.0
        phase, reached = "drive", False

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.sim_now() > deadline or time.time() > wall_deadline:
                self.log("goto_timeout", f"no arrival within {timeout_s}s of sim time")
                break
            try:
                tf = buf.lookup_transform("map", "base_footprint", rclpy.time.Time())
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue                       # slam_toolbox has not published yet

            px = tf.transform.translation.x
            py = tf.transform.translation.y
            q = tf.transform.rotation
            pyaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                              1 - 2 * (q.y * q.y + q.z * q.z))

            dx, dy = gx - px, gy - py
            dist = math.hypot(dx, dy)
            msg = Twist()

            if phase == "drive" and dist > pos_tol:
                want = math.atan2(dy, dx)
                err = math.atan2(math.sin(want - pyaw), math.cos(want - pyaw))
                if abs(err) > 0.35:            # point roughly at it before moving
                    msg.angular.z = 0.4 if err > 0 else -0.4
                else:
                    msg.linear.x = min(0.15, 0.4 * dist)
                    msg.angular.z = max(-0.5, min(0.5, 1.2 * err))
            else:
                if phase == "drive":
                    self.log("goto_arrived",
                             f"within {dist:.2f} m of the goal, turning to face it")
                    phase = "turn"
                err = math.atan2(math.sin(gyaw - pyaw), math.cos(gyaw - pyaw))
                if abs(err) < yaw_tol:
                    reached = True
                    break
                msg.angular.z = 0.35 if err > 0 else -0.35

            self.cmd_pub.publish(msg)

        self.cmd_pub.publish(Twist())
        self.commanding = False
        if reached:
            self.log("goto_reached", "final heading reached; stopped")
        self.spin_for(1.0)

    def run_goal(self, x: float, y: float, yaw: float, timeout_s: float,
                 initial: "tuple[float, float, float] | None" = None):
        from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
        from geometry_msgs.msg import PoseStamped

        nav = BasicNavigator()

        # AMCL publishes no map->odom until it has a pose estimate, and
        # waitUntilNav2Active() blocks on exactly that, so without this the run
        # hangs until its timeout with nothing in the log to say why. The robot
        # is spawned at a pose we chose, so there is no reason to make the
        # localiser guess it.
        if initial is not None:
            ix, iy, iyaw = initial
            start = PoseStamped()
            start.header.frame_id = "map"
            start.header.stamp = nav.get_clock().now().to_msg()
            start.pose.position.x = ix
            start.pose.position.y = iy
            start.pose.orientation.z = math.sin(iyaw / 2)
            start.pose.orientation.w = math.cos(iyaw / 2)
            nav.setInitialPose(start)
            self.log("initial_pose", f"seeded AMCL at ({ix}, {iy}, "
                                     f"{math.degrees(iyaw):.0f} deg)")

        self.log("nav2_wait", "waiting for Nav2 to become active")
        nav.waitUntilNav2Active()
        self.log("nav2_active", "Nav2 active")

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = nav.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.orientation.w = math.cos(yaw / 2)

        self.commanding = True
        self.log("goal_sent", f"goal ({x}, {y}, {math.degrees(yaw):.0f} deg)",
                 goal_x=x, goal_y=y)
        nav.goToPose(goal)

        deadline = self.sim_now() + timeout_s
        wall_deadline = time.time() + timeout_s / MIN_RTF + 30.0
        while not nav.isTaskComplete() and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.sim_now() > deadline or time.time() > wall_deadline:
                nav.cancelTask()
                self.log("goal_timeout", f"no result within {timeout_s}s")
                break

        result = nav.getResult()
        mapping = {
            TaskResult.SUCCEEDED: ("goal_succeeded", "Nav2 reported success"),
            TaskResult.CANCELED: ("goal_canceled", "Nav2 task canceled"),
            TaskResult.FAILED: ("goal_failed", "Nav2 reported failure"),
        }
        kind, detail = mapping.get(result, ("goal_unknown", f"result={result}"))
        self.log(kind, detail)
        self.commanding = False
        nav.destroy_node()

    # --- output ---------------------------------------------------------

    def save(self, extra: dict):
        payload = {
            **extra,
            "duration_s": round(self.t(), 2),
            "path_m": round(self.path_m, 3),
            "odom_path_m": round(self.odom_path_m, 3),
            "odom_overreport_m": round(self.odom_path_m - self.path_m, 3),
            "final_pose": (
                {"x": round(self.pose[0], 3), "y": round(self.pose[1], 3),
                 "yaw_deg": round(math.degrees(self.pose[2]), 1)}
                if self.pose else None
            ),
            "events": self.events,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[drive] events -> {self.out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--events-out", required=True, help="where to write the events JSON")
    p.add_argument("--mode", choices=["goal", "goto", "script"], required=True)
    p.add_argument("--goal", default=None, help="x,y,yaw_deg for --mode goal")
    p.add_argument("--script", default=None,
                   help="JSON file or inline JSON list of drive steps")
    p.add_argument("--timeout", type=float, default=180.0, help="goal timeout, seconds")
    p.add_argument("--initial-pose", default=None,
                   help="x,y,yaw_deg to seed AMCL with, in the map frame")
    p.add_argument("--label", default="", help="scenario name recorded into the events file")
    p.add_argument("--expect", default="", help="expected ground-truth outcome label")
    args = p.parse_args()

    rclpy.init()
    driver = RolloutDriver(Path(args.events_out))
    try:
        if not driver.wait_for_pose():
            print("[drive] no /gt_pose within 60s — is the sim up?", file=sys.stderr)
            return 2
        driver.log("rollout_start", args.label or args.mode)

        if args.mode == "goto":
            if not args.goal:
                raise SystemExit("--mode goto requires --goal x,y,yaw_deg")
            x, y, yaw_deg = (float(v) for v in args.goal.split(","))
            driver.run_goto(x, y, math.radians(yaw_deg), args.timeout)
        elif args.mode == "goal":
            if not args.goal:
                raise SystemExit("--mode goal requires --goal x,y,yaw_deg")
            x, y, yaw_deg = (float(v) for v in args.goal.split(","))
            initial = None
            if args.initial_pose:
                ix, iy, iyaw = (float(v) for v in args.initial_pose.split(","))
                initial = (ix, iy, math.radians(iyaw))
            driver.run_goal(x, y, math.radians(yaw_deg), args.timeout, initial)
        else:
            if not args.script:
                raise SystemExit("--mode script requires --script")
            raw = args.script.strip()
            # Decide by shape, not by Path.exists(): a long inline JSON script
            # exceeds the filename length limit and makes exists() raise OSError
            # rather than return False.
            if raw.startswith(("[", "{")):
                steps = json.loads(raw)
            else:
                steps = json.loads(Path(raw).read_text())
            driver.run_script(steps)

        driver.log("rollout_end", "")
        driver.save({"scenario": args.label, "mode": args.mode, "expect": args.expect})
    finally:
        driver.cmd_pub.publish(Twist())
        driver.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
