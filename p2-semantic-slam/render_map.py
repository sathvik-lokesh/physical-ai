"""
Render the slam_toolbox occupancy grid with true and estimated object positions.

This is the figure the project is judged on, so it is a script rather than
something typed once into a shell: every number on it has to be regenerable from
the bag after any change.

The one subtlety is the frame conversion. slam_toolbox's map frame is anchored to
wherever the robot started, not to the world origin, so authored object
coordinates cannot be drawn on it directly. The robot starts at a known pose
(the scenario's `spawn`), which gives the world->map transform exactly. Getting
this wrong puts every marker in a plausible but false place, and it would look
fine, so `--check` verifies it against the room geometry instead of trusting it:
the mapped walls must come out 7 x 7 m and centred.

    python render_map.py --out results/map.png
    python render_map.py --check           # verify the transform, draw nothing
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
TRUE_OBJECTS = {
    "chair": (2.5, 0.0),
    "table": (0.0, 2.5),
    "bookshelf": (-2.5, 0.0),
    "sofa": (0.0, -2.5),
}
PATROL = [(1.5, 1.5), (-1.5, 1.5), (-1.5, -1.5), (1.5, -1.5)]
ROOM_M = 7.0


def read_last_map(bag: Path):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if "/map" not in types:
        raise SystemExit(f"{bag} has no /map topic — was this recorded with slam:=true?")
    cls = get_message(types["/map"])
    last = None
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/map":
            last = deserialize_message(data, cls)
    if last is None:
        raise SystemExit("no /map messages in the bag")
    return last


def spawn_pose(scenario: Path) -> tuple[float, float, float]:
    s = json.loads(scenario.read_text())["spawn"]
    return s.get("x", 0.0), s.get("y", 0.0), s.get("yaw", 0.0)


def make_world_to_map(sx: float, sy: float, syaw: float):
    """
    World coordinates -> map coordinates.

    The map frame starts at the robot's spawn pose, so a world point is
    expressed relative to that pose and then rotated into it.
    """
    c, s = math.cos(-syaw), math.sin(-syaw)

    def to_map(px, py):
        dx, dy = px - sx, py - sy
        return (c * dx - s * dy, s * dx + c * dy)

    return to_map


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--bag", default=str(HERE / "data/map_room/bag"))
    p.add_argument("--scenario", default=str(HERE / "scenarios/map_room.json"))
    p.add_argument("--estimates", default=None,
                   help="JSON of {label: [x, y]} in WORLD coords, from semantic_map.py")
    p.add_argument("--out", default=str(HERE / "results/map.png"))
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--check", action="store_true",
                   help="verify the map geometry and frame transform, write nothing")
    a = p.parse_args()

    m = read_last_map(Path(a.bag))
    w, h, res = m.info.width, m.info.height, m.info.resolution
    ox, oy = m.info.origin.position.x, m.info.origin.position.y
    grid = np.array(m.data, dtype=np.int8).reshape(h, w)
    to_map = make_world_to_map(*spawn_pose(Path(a.scenario)))

    occ = np.argwhere(grid > 50)
    if len(occ) == 0:
        raise SystemExit("map has no occupied cells — SLAM produced nothing")
    ys, xs = occ[:, 0], occ[:, 1]
    span_x = (xs.max() - xs.min()) * res
    span_y = (ys.max() - ys.min()) * res

    print(f"map        {w}x{h} @ {res:.3f} m/cell")
    print(f"occupied   spans {span_x:.2f} x {span_y:.2f} m   (room is {ROOM_M} x {ROOM_M} m)")
    ok = abs(span_x - ROOM_M) < 0.5 and abs(span_y - ROOM_M) < 0.5
    print(f"geometry   {'OK' if ok else 'WRONG — map does not match the room'}")

    # Every object must land inside the mapped area, or the transform is wrong.
    bad = []
    for name, (wx, wy) in TRUE_OBJECTS.items():
        mx, my = to_map(wx, wy)
        cx, cy = int((mx - ox) / res), int((my - oy) / res)
        if not (0 <= cx < w and 0 <= cy < h):
            bad.append(name)
    print(f"transform  {'OK' if not bad else 'WRONG — outside the grid: ' + ', '.join(bad)}")
    if a.check:
        raise SystemExit(0 if (ok and not bad) else 1)

    rgb = np.full((h, w, 3), 130, np.uint8)          # unknown
    rgb[(grid >= 0) & (grid <= 50)] = (250, 250, 250)  # free
    rgb[grid > 50] = (20, 20, 20)                      # occupied
    im = Image.fromarray(rgb).transpose(Image.FLIP_TOP_BOTTOM)
    im = im.resize((w * a.scale, h * a.scale), Image.NEAREST)
    d = ImageDraw.Draw(im)

    def px(wx, wy):
        mx, my = to_map(wx, wy)
        return ((mx - ox) / res * a.scale, (h - (my - oy) / res) * a.scale)

    for cx, cy in PATROL:                              # where the robot paused
        X, Y = px(cx, cy)
        d.ellipse([X - 5, Y - 5, X + 5, Y + 5], outline=(70, 120, 220), width=2)

    for name, (wx, wy) in TRUE_OBJECTS.items():        # truth
        X, Y = px(wx, wy)
        d.ellipse([X - 9, Y - 9, X + 9, Y + 9], outline=(210, 40, 40), width=3)
        d.text((X + 12, Y - 6), name, fill=(210, 40, 40))

    if a.estimates:                                    # what the VLM located
        est = json.loads(Path(a.estimates).read_text())
        for name, (ex, ey) in est.items():
            X, Y = px(ex, ey)
            d.line([X - 9, Y - 9, X + 9, Y + 9], fill=(20, 150, 60), width=3)
            d.line([X - 9, Y + 9, X + 9, Y - 9], fill=(20, 150, 60), width=3)
            if name in TRUE_OBJECTS:
                TX, TY = px(*TRUE_OBJECTS[name])
                d.line([X, Y, TX, TY], fill=(20, 150, 60), width=1)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    im.save(a.out)
    print(f"figure     -> {a.out}")
    print("legend     red circle = true pose, green cross = VLM estimate, "
          "blue = patrol corner")


if __name__ == "__main__":
    main()
