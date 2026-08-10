"""
Add slam_toolbox's estimated robot pose to each keyframe in a manifest.

Without this the project is not really semantic *SLAM*. The map comes from
slam_toolbox, but `semantic_map.py` anchors labels using `gt_*`, the simulator's
true pose, which no real robot has. That was a deliberate first-pass choice —
it isolates "can the VLM's labels be localised at all" from "does the localiser
drift" — but leaving it there would be quietly claiming a result the system has
not earned.

This composes the two transforms slam_toolbox and the drivetrain publish:

    map -> odom            (slam_toolbox's drift correction)
    odom -> base_footprint (wheel odometry)

and writes the result as `slam_x`, `slam_y`, `slam_yaw_deg` in the map frame,
alongside the existing ground truth. `semantic_map.py --poses slam` then runs
the identical pipeline on estimated poses, and the gap between the two runs is
the localisation error's contribution, reported rather than assumed.

Ground truth is in the world frame and SLAM is in the map frame, so the
comparison needs the world->map transform. That is fixed by the robot's spawn
pose, which the scenario file states exactly.

    python add_slam_poses.py
"""

import argparse
import bisect
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def compose(a: tuple, b: tuple) -> tuple:
    """(x, y, yaw) of b expressed in a's parent frame."""
    ax, ay, ayaw = a
    bx, by, byaw = b
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (ax + c * bx - s * by, ay + s * bx + c * by, ayaw + byaw)


def collect(bag: Path):
    """Timestamped map->odom and odom->base_footprint series from the bag."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    cls = {n: get_message(t) for n, t in types.items()}

    m2o: list[tuple[float, tuple]] = []
    o2b: list[tuple[float, tuple]] = []
    t0 = None
    while r.has_next():
        topic, data, stamp = r.read_next()
        t = stamp / 1e9
        if t0 is None:
            t0 = t                      # same convention as extract_keyframes.py
        if topic != "/tf":
            continue
        msg = deserialize_message(data, cls[topic])
        for tr in msg.transforms:
            p = (tr.transform.translation.x, tr.transform.translation.y,
                 yaw_of(tr.transform.rotation))
            if tr.header.frame_id == "map" and tr.child_frame_id == "odom":
                m2o.append((t - t0, p))
            elif tr.header.frame_id == "odom" and tr.child_frame_id == "base_footprint":
                o2b.append((t - t0, p))
    return m2o, o2b


def latest_at(series: list[tuple[float, tuple]], keys: list[float], t: float):
    """Most recent transform at or before t — what a live consumer would have."""
    i = bisect.bisect_right(keys, t) - 1
    return series[i][1] if i >= 0 else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--bag", default=str(HERE / "data/map_room/bag"))
    p.add_argument("--manifest", default=str(HERE / "data/map_room/manifest.jsonl"))
    p.add_argument("--scenario", default=str(HERE / "scenarios/map_room.json"))
    a = p.parse_args()

    m2o, o2b = collect(Path(a.bag))
    if not m2o or not o2b:
        raise SystemExit("bag has no map->odom or odom->base_footprint — "
                         "was it recorded with slam:=true?")
    km, ko = [t for t, _ in m2o], [t for t, _ in o2b]
    print(f"map->odom {len(m2o)} transforms, odom->base_footprint {len(o2b)}")

    spawn = json.loads(Path(a.scenario).read_text())["spawn"]
    sx, sy, syaw = spawn.get("x", 0.0), spawn.get("y", 0.0), spawn.get("yaw", 0.0)
    c, s = math.cos(-syaw), math.sin(-syaw)

    def world_to_map(px, py):
        dx, dy = px - sx, py - sy
        return (c * dx - s * dy, s * dx + c * dy)

    items = [json.loads(l) for l in open(a.manifest) if l.strip()]
    errs, filled = [], 0
    for it in items:
        m = it["meta"]
        t = m.get("t_s")
        if t is None:
            continue
        A, B = latest_at(m2o, km, t), latest_at(o2b, ko, t)
        if A is None or B is None:
            continue
        x, y, yaw = compose(A, B)
        m["slam_x"], m["slam_y"] = round(x, 3), round(y, 3)
        m["slam_yaw_deg"] = round(math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))), 1)
        filled += 1
        if m.get("gt_x") is not None:
            gx, gy = world_to_map(m["gt_x"], m["gt_y"])
            e = math.hypot(x - gx, y - gy)
            m["slam_pos_error_m"] = round(e, 3)
            errs.append(e)

    with open(a.manifest, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    print(f"wrote slam poses to {filled}/{len(items)} keyframes")
    if errs:
        errs.sort()
        print(f"SLAM vs ground truth: median {errs[len(errs)//2]:.3f} m  "
              f"max {errs[-1]:.3f} m  mean {sum(errs)/len(errs):.3f} m")
        print("(this is the localisation error the semantic layer has to tolerate)")


if __name__ == "__main__":
    main()
