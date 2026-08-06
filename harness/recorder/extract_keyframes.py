"""
rosbag2 -> keyframe PNGs + manifest.jsonl (Phase B input).

Sampling is by **distance travelled**, not by time. A robot that is stuck or
oscillating emits thousands of near-identical frames; sampling on a clock would
fill the manifest with them and spend hours of VLM time re-answering the same
question. Distance sampling keeps one frame per ~0.5 m of actual progress, with
a stall-timeout escape hatch so a genuinely stuck robot still contributes the
frames that prove it (which is the whole point of Phase 1).

The bag is streamed message-by-message rather than loaded — a rollout bag is
larger than the memory budget this machine can spare.

Usage:
    python extract_keyframes.py bags/run01 --out data/run01 --every-m 0.5
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py

DEFAULT_IMAGE_TOPIC = "/rgb_camera"
DEFAULT_ODOM_TOPIC = "/odom"
DEFAULT_CMD_VEL_TOPIC = "/cmd_vel"
DEFAULT_GT_POSE_TOPIC = "/gt_pose"
ROBOT_MODEL = "turtlebot3_waffle"


def open_bag(path: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, types


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def image_to_array(msg) -> np.ndarray:
    """sensor_msgs/Image -> HxWx3 uint8 RGB."""
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(msg.height, msg.step // 3, 3)[:, : msg.width, :]
        return arr[:, :, ::-1] if enc == "bgr8" else arr
    if enc in ("rgba8", "bgra8"):
        arr = buf.reshape(msg.height, msg.step // 4, 4)[:, : msg.width, :3]
        return arr[:, :, ::-1] if enc == "bgra8" else arr
    if enc == "mono8":
        arr = buf.reshape(msg.height, msg.step)[:, : msg.width]
        return np.stack([arr] * 3, axis=-1)
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


def extract(args) -> int:
    bag_path = Path(args.bag).resolve()
    if not bag_path.exists():
        raise SystemExit(f"bag not found: {bag_path}")

    out_dir = Path(args.out).resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    reader, types = open_bag(bag_path)
    for topic in (args.image_topic, args.odom_topic):
        if topic not in types:
            raise SystemExit(
                f"topic {topic!r} not in bag. Recorded topics: {sorted(types)}"
            )
    has_gt = args.gt_pose_topic in types
    if not has_gt:
        print(
            f"WARNING: {args.gt_pose_topic} not in bag — keyframes will carry no "
            f"ground-truth pose, and labelling will fall back to wheel odometry, "
            f"which counts wheel slip as travel. Re-record with a current "
            f"rollout.launch.py to get trustworthy labels.",
            file=sys.stderr,
        )

    msg_classes = {name: get_message(t) for name, t in types.items()}

    # Streaming state.
    #
    # Two position signals are tracked on purpose. `odom_*` is what the robot
    # believes from its wheels, and is what gets shown to the VLM — it is what a
    # real onboard monitor would have. `gt_*` is the simulator's true pose and is
    # what labels are derived from, because wheel odometry counts slip as travel
    # and would mislabel exactly the stuck-against-a-wall case.
    last_odom = None          # (x, y, yaw)
    last_cmd = None           # (linear_x, angular_z)
    path_length = 0.0         # metres of wheel-reported travel
    last_gt = None            # (x, y, yaw) ground truth
    gt_path_length = 0.0
    kept_at_path = None       # path_length when the last keyframe was kept
    kept_at_time = None       # bag time (s) when the last keyframe was kept
    t0 = None

    manifest = []
    n_images = 0

    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        t = stamp_ns / 1e9
        if t0 is None:
            t0 = t
        rel_t = t - t0

        if topic == args.odom_topic:
            odom = deserialize_message(data, msg_classes[topic])
            p = odom.pose.pose.position
            yaw = yaw_from_quaternion(odom.pose.pose.orientation)
            if last_odom is not None:
                path_length += math.hypot(p.x - last_odom[0], p.y - last_odom[1])
            last_odom = (p.x, p.y, yaw)
            continue

        if topic == args.gt_pose_topic:
            tf_msg = deserialize_message(data, msg_classes[topic])
            for tr in tf_msg.transforms:
                if tr.child_frame_id != ROBOT_MODEL:
                    continue
                gp = tr.transform.translation
                gq = tr.transform.rotation
                gyaw = math.atan2(
                    2 * (gq.w * gq.z + gq.x * gq.y),
                    1 - 2 * (gq.y * gq.y + gq.z * gq.z),
                )
                if last_gt is not None:
                    gt_path_length += math.hypot(gp.x - last_gt[0], gp.y - last_gt[1])
                last_gt = (gp.x, gp.y, gyaw)
                break
            continue

        if topic == args.cmd_vel_topic:
            cmd = deserialize_message(data, msg_classes[topic])
            # Nav2 on Jazzy may publish TwistStamped; handle both shapes
            twist = getattr(cmd, "twist", cmd)
            last_cmd = (twist.linear.x, twist.angular.z)
            continue

        if topic != args.image_topic:
            continue

        n_images += 1
        # Decide whether this frame earns a slot in the manifest
        progress = gt_path_length if has_gt else path_length
        if kept_at_path is None:
            keep = True
        else:
            moved = progress - kept_at_path
            stalled_for = rel_t - kept_at_time
            keep = moved >= args.every_m or stalled_for >= args.stall_s
        if not keep:
            continue

        msg = deserialize_message(data, msg_classes[topic])
        try:
            arr = image_to_array(msg)
        except ValueError as e:
            print(f"  skipping frame at t={rel_t:.1f}s: {e}", file=sys.stderr)
            continue

        idx = len(manifest)
        name = f"kf{idx:04d}.png"
        PILImage.fromarray(arr).save(frames_dir / name)

        # Hash the pixels, not the PNG, so re-encoding cannot change the answer.
        #
        # A stalled robot in a static scene renders the same image every time,
        # and — measured across four `stuck` rollouts that varied approach angle,
        # obstacle shape and lighting — so does a *different* rollout that ends
        # pressed against a flat surface: 30 of those frames were byte-identical
        # to each other. Without this field a class can look like thirty trials
        # while being one picture, and every accuracy over it is really the score
        # on a single image. Scoring uses it to weight duplicates down.
        pixel_sha1 = hashlib.sha1(arr.tobytes()).hexdigest()

        x, y, yaw = last_odom if last_odom else (None, None, None)
        gx, gy, gyaw_v = last_gt if last_gt else (None, None, None)
        manifest.append(
            {
                "id": f"{args.run_id or bag_path.name}_kf{idx:04d}",
                "image": f"frames/{name}",
                "meta": {
                    "run_id": args.run_id or bag_path.name,
                    "frame": idx,
                    "t_s": round(rel_t, 2),
                    "pixel_sha1": pixel_sha1,
                    "x": round(x, 3) if x is not None else None,
                    "y": round(y, 3) if y is not None else None,
                    "yaw_deg": round(math.degrees(yaw), 1) if yaw is not None else None,
                    "path_m": round(path_length, 3),
                    "cmd_linear_x": round(last_cmd[0], 3) if last_cmd else None,
                    "cmd_angular_z": round(last_cmd[1], 3) if last_cmd else None,
                    # Ground truth — for labelling and scoring, never for prompts
                    "gt_x": round(gx, 3) if gx is not None else None,
                    "gt_y": round(gy, 3) if gy is not None else None,
                    "gt_yaw_deg": round(math.degrees(gyaw_v), 1) if gyaw_v is not None else None,
                    "gt_path_m": round(gt_path_length, 3),
                },
            }
        )
        kept_at_path = progress
        kept_at_time = rel_t

    if not manifest:
        raise SystemExit(f"no frames extracted — bag had {n_images} images on {args.image_topic}")

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

    duration = manifest[-1]["meta"]["t_s"]
    n_distinct = len({it["meta"]["pixel_sha1"] for it in manifest})
    print(
        f"bag       : {bag_path}\n"
        f"images    : {n_images} on {args.image_topic}\n"
        f"keyframes : {len(manifest)} (every {args.every_m} m, stall timeout {args.stall_s}s)\n"
        f"distinct  : {n_distinct} of {len(manifest)} keyframes are pixel-unique\n"
        f"duration  : {duration}s\n"
        f"path      : {gt_path_length:.2f} m true"
        + (f", {path_length:.2f} m by wheel odometry "
           f"(+{path_length - gt_path_length:.2f} m of slip)" if has_gt else " (odometry only)")
        + "\n"
        f"frames    -> {frames_dir}\n"
        f"manifest  -> {manifest_path}"
    )
    est_min = len(manifest) * 80 / 60
    print(f"\nat ~80 s/frame that is ~{est_min:.0f} min of VLM time per model.")
    return 0


def main():
    p = argparse.ArgumentParser(description="rosbag2 -> keyframes + manifest")
    p.add_argument("bag", help="rosbag2 directory")
    p.add_argument("--out", required=True, help="output dir for frames/ + manifest.jsonl")
    p.add_argument("--run-id", default=None, help="id prefix (default: bag dir name)")
    p.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    p.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    p.add_argument("--cmd-vel-topic", default=DEFAULT_CMD_VEL_TOPIC)
    p.add_argument("--gt-pose-topic", default=DEFAULT_GT_POSE_TOPIC,
                   help="simulator ground-truth pose; labels are derived from this")
    p.add_argument("--every-m", type=float, default=0.5,
                   help="keep a frame each N metres of path travelled")
    p.add_argument("--stall-s", type=float, default=5.0,
                   help="keep a frame anyway after N seconds without progress "
                        "(this is what captures a stuck robot)")
    sys.exit(extract(p.parse_args()))


if __name__ == "__main__":
    main()
