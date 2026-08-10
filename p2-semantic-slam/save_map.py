"""
Write the mapping run's occupancy grid out as a map_server pgm + yaml pair.

The demo navigates to a pose the semantic map returned, and Nav2's planner needs
a map to plan through. Running slam_toolbox again during the demo would work in
principle, but the global costmap starts empty and the planner then has to route
through unknown space on a machine that renders in software. Planning against
the map we already built is both more robust and more honest about what the
system does: map once, then answer questions about the room.

    python save_map.py --out results/room_map
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_last_map(bag: Path):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    cls = get_message(types["/map"])
    last = None
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/map":
            last = deserialize_message(data, cls)
    if last is None:
        raise SystemExit(f"no /map in {bag}")
    return last


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--bag", default=str(HERE / "data/map_room/bag"))
    p.add_argument("--out", default=str(HERE / "results/room_map"))
    p.add_argument("--occupied-thresh", type=float, default=0.65)
    p.add_argument("--free-thresh", type=float, default=0.25)
    a = p.parse_args()

    m = read_last_map(Path(a.bag))
    w, h, res = m.info.width, m.info.height, m.info.resolution
    g = np.array(m.data, dtype=np.int8).reshape(h, w)

    # map_server's convention: 254 free, 0 occupied, 205 unknown, and the image
    # runs top-down while the grid runs bottom-up.
    img = np.full((h, w), 205, np.uint8)
    img[(g >= 0) & (g <= 25)] = 254
    img[g >= 65] = 0
    img = np.flipud(img)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pgm, yaml = out.with_suffix(".pgm"), out.with_suffix(".yaml")
    with open(pgm, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(img.tobytes())

    o = m.info.origin.position
    yaml.write_text(
        f"image: {pgm.name}\n"
        f"mode: trinary\n"
        f"resolution: {res}\n"
        f"origin: [{o.x}, {o.y}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: {a.occupied_thresh}\n"
        f"free_thresh: {a.free_thresh}\n"
    )
    occ = int((img == 0).sum())
    print(f"{w}x{h} @ {res} m/cell, origin ({o.x:.2f}, {o.y:.2f})")
    print(f"  {occ} occupied, {(img == 254).sum()} free, {(img == 205).sum()} unknown")
    print(f"  -> {pgm}\n  -> {yaml}")
    if occ == 0:
        raise SystemExit("map has no occupied cells — Nav2 would have nothing to avoid")


if __name__ == "__main__":
    main()
