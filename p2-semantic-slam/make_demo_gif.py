"""
Build the demo GIF: the robot's own camera while it drives to a queried object.

The point of the animation is that the chair starts off to one side and ends up
centred and close, having been found by asking a question in English rather than
by being given a coordinate. The caption carries the query and the error so the
picture cannot be read as better than it is.

    python make_demo_gif.py --out results/demo.gif
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent


def frames_from_bag(bag: Path, topic: str = "/rgb_camera"):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    cls = get_message(types[topic])
    t0, out = None, []
    while r.has_next():
        name, data, stamp = r.read_next()
        t = stamp / 1e9
        if t0 is None:
            t0 = t
        if name != topic:
            continue
        m = deserialize_message(data, cls)
        buf = np.frombuffer(m.data, dtype=np.uint8)
        enc = m.encoding.lower()
        if enc in ("rgb8", "bgr8"):
            a = buf.reshape(m.height, m.step // 3, 3)[:, : m.width, :]
            if enc == "bgr8":
                a = a[:, :, ::-1]
        else:
            raise SystemExit(f"unhandled encoding {m.encoding}")
        out.append((t - t0, a.copy()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--bag", default=str(HERE / "data/goto_chair/bag"))
    p.add_argument("--scenario", default=str(HERE / "scenarios/goto_query.json"))
    p.add_argument("--events", default=str(HERE / "data/goto_chair/events.json"))
    p.add_argument("--out", default=str(HERE / "results/demo.gif"))
    p.add_argument("--every", type=int, default=2, help="keep 1 frame in N")
    p.add_argument("--ms", type=int, default=180, help="frame duration")
    p.add_argument("--scale", type=int, default=2)
    a = p.parse_args()

    scen = json.loads(Path(a.scenario).read_text())
    ev = json.loads(Path(a.events).read_text())
    fp = ev["final_pose"]
    truth = (2.5, 0.0) if scen["_target"] == "chair" else None
    stop_err = math.hypot(fp["x"] - truth[0], fp["y"] - truth[1]) if truth else None

    frames = frames_from_bag(Path(a.bag))
    if not frames:
        raise SystemExit("no camera frames in the bag")
    kept = frames[:: a.every]
    print(f"{len(frames)} camera frames -> {len(kept)} in the gif")

    W, H = kept[0][1].shape[1], kept[0][1].shape[0]
    BAR = 46
    imgs = []
    for t, arr in kept:
        im = Image.new("RGB", (W, H + BAR), (18, 18, 18))
        im.paste(Image.fromarray(arr), (0, BAR))
        d = ImageDraw.Draw(im)
        d.text((6, 5), f'"{scen["_query"]}"', fill=(255, 255, 255))
        d.text((6, 19), f'{scen["_target"]} believed at '
                        f'({scen["_object_world"][0]:+.2f}, {scen["_object_world"][1]:+.2f}), '
                        f'{scen["_object_error_m"]:.2f} m off', fill=(150, 200, 150))
        d.text((6, 33), f"t={t:5.1f}s"
                        + (f"   stopped {stop_err:.2f} m from it" if stop_err else ""),
               fill=(160, 160, 160))
        if a.scale != 1:
            im = im.resize((im.width * a.scale, im.height * a.scale), Image.NEAREST)
        imgs.append(im)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:],
                 duration=a.ms, loop=0, optimize=True)
    kb = Path(a.out).stat().st_size / 1024
    print(f"-> {a.out}  ({len(imgs)} frames, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
