"""
DriveLM-nuScenes `behavior` QA -> a manifest.jsonl the existing harness scores.

The behavior answer is one sentence pair — "The ego vehicle is going straight.
The ego vehicle is driving fast." — which factorises into two independent labels
with wildly different difficulty:

    steer  going straight | slightly steering left/right | steering left/right
    speed  not moving | driving slowly | with normal speed | fast | very fast

Scoring them separately is the whole point. `going straight` is ~79% of the
training set, so a model that answers it every time posts a number that reads
like success and is worth nothing; `speed` is near-uniform, so it is where any
real discrimination has to show up. `scoring/score.py` prints the majority
baseline beside every accuracy, which is what makes that visible instead of
flattering.

Two arms, because p1 found that output *format* moved results more than the
choice of model did:

    single  one CAM_FRONT frame. Speed is not recoverable from it even in
            principle, so this arm measures what the model does when the
            evidence is absent — guess the prior, or say so.
    pair    the same frame plus the previous keyframe, with the true gap in
            seconds stated in the prompt. Now speed IS recoverable, by how far
            the scene moved. Any gain is the model using temporal evidence.

`--pair-mode composite` puts the two frames side by side in one image (works
everywhere); `--pair-mode two-images` sends them as two images in one request
(vlmrunner supports it, llava-phi3's projector may not). Running both measures
the harness's own format sensitivity rather than assuming it away.

Usage:
    python build_manifest.py --arm single --n 200
    python build_manifest.py --arm pair --n 200 --pair-mode composite
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVELM = HERE / "data" / "drivelm"
QA_FILE = DRIVELM / "v1_1_train_nus.json"

# The official file's sha256, as reported by the Hugging Face API for
# OpenDriveLab/DriveLM. A same-length, different-content copy of this file is
# being served by at least one ungated mirror, so identity is checked by hash.
# `fetch_drivelm.sh` verifies this on download; this is the second gate, here
# because a manifest built from the wrong bytes would look completely normal.
OFFICIAL_QA_SHA256 = None  # filled in by --record-hash once fetched officially

BEHAVIOR_RE = re.compile(
    r"^The ego vehicle is (.+?)\.\s*The ego vehicle is (.+?)\.?\s*$"
)

STEER_VALUES = [
    "going straight",
    "slightly steering to the left",
    "slightly steering to the right",
    "steering to the left",
    "steering to the right",
]
SPEED_VALUES = [
    "not moving",
    "driving slowly",
    "driving with normal speed",
    "driving fast",
    "driving very fast",
]

# Keyframe spacing in DriveLM varies from 0.4 s to 13 s. A model cannot convert
# displacement into speed without knowing the interval, and an interval that
# varies by 30x across items is a confound rather than a variable. The pair arm
# is restricted to this band and told the exact gap.
PAIR_DT_MIN, PAIR_DT_MAX = 1.5, 3.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frame_timestamp(image_path: str) -> float:
    """nuScenes encodes capture time in microseconds in the filename."""
    m = re.search(r"__CAM_FRONT__(\d+)\.jpg", image_path)
    if not m:
        raise ValueError(f"no CAM_FRONT timestamp in {image_path!r}")
    return int(m.group(1)) / 1e6


def parse_behavior(answer: str) -> tuple[str, str] | None:
    m = BEHAVIOR_RE.match(answer.strip())
    if not m:
        return None
    steer, speed = m.group(1), m.group(2)
    if steer not in STEER_VALUES or speed not in SPEED_VALUES:
        return None
    return steer, speed


def load_frames(qa_path: Path) -> list[dict]:
    """
    Flatten the scene/keyframe tree into a time-ordered list per scene, keeping
    only frames that carry a parseable behavior label.
    """
    with open(qa_path) as f:
        data = json.load(f)

    frames: list[dict] = []
    unparsed = 0
    for scene_id, scene in data.items():
        rows = []
        for frame_id, frame in scene["key_frames"].items():
            behavior = frame["QA"].get("behavior") or []
            if not behavior:
                continue
            labels = parse_behavior(behavior[0]["A"])
            if labels is None:
                unparsed += 1
                continue
            front = frame["image_paths"]["CAM_FRONT"]
            rows.append(
                {
                    "scene": scene_id,
                    "frame": frame_id,
                    "t": frame_timestamp(front),
                    # image_paths are written relative to a sibling nuscenes/
                    "image": front.replace("../", "", 1),
                    "steer": labels[0],
                    "speed": labels[1],
                    "question": behavior[0]["Q"],
                }
            )
        rows.sort(key=lambda r: r["t"])
        for i, row in enumerate(rows):
            row["prev"] = rows[i - 1] if i else None
            frames.append(row)

    if unparsed:
        print(f"note: {unparsed} behavior answers did not parse and were dropped")
    return frames


def eligible(frames: list[dict], arm: str) -> list[dict]:
    if arm == "single":
        return frames
    out = []
    for row in frames:
        prev = row["prev"]
        if prev is None:
            continue
        dt = row["t"] - prev["t"]
        if PAIR_DT_MIN <= dt <= PAIR_DT_MAX:
            row["dt_s"] = round(dt, 2)
            out.append(row)
    return out


def sample(frames: list[dict], n: int, seed: int) -> list[dict]:
    """
    Stratify by (speed, steer-is-straight), one frame per scene.

    One per scene because keyframes inside a nuScenes scene are seconds apart on
    the same stretch of road: they are not identical images, so p1's pixel_sha1
    de-duplication will not catch them, but they are far from independent
    trials. p1's lesson was that correlated frames inflate a score silently; the
    cheap fix here is to spend the sample budget on distinct scenes.

    Stratifying deliberately changes the class balance away from the corpus
    prior. That is safe only because score.py derives its majority baseline from
    the manifest that was actually run, so the comparison stays like-for-like.
    """
    rng = random.Random(seed)
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in frames:
        by_scene[row["scene"]].append(row)

    # One candidate per scene, chosen at random among that scene's frames.
    candidates = [rng.choice(rows) for rows in by_scene.values()]
    rng.shuffle(candidates)

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        strata[(row["speed"], row["steer"] == "going straight")].append(row)

    # Round-robin across strata so the rare turning classes survive.
    picked: list[dict] = []
    keys = sorted(strata, key=lambda k: (str(k[0]), k[1]))
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    picked.sort(key=lambda r: (r["scene"], r["t"]))
    return picked


def composite(a: Path, b: Path, out: Path) -> None:
    """Left = earlier frame, right = current, with a divider between them."""
    from PIL import Image

    left, right = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    height = min(left.height, right.height)
    scale = lambda im: im.resize(  # noqa: E731
        (int(im.width * height / im.height), height), Image.LANCZOS
    )
    left, right = scale(left), scale(right)
    gap = 8
    canvas = Image.new("RGB", (left.width + gap + right.width, height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["single", "pair"], required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--pair-mode", choices=["composite", "two-images"],
                   default="composite")
    p.add_argument("--qa", default=str(QA_FILE))
    p.add_argument("--out", default=None)
    p.add_argument("--allow-missing-images", action="store_true",
                   help="build the manifest before the 3.5 GB image zip is "
                        "fetched; the runner will refuse to run it")
    args = p.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"missing {qa_path} — run ./fetch_drivelm.sh first", file=sys.stderr)
        return 2
    if OFFICIAL_QA_SHA256 and sha256_file(qa_path) != OFFICIAL_QA_SHA256:
        print(f"{qa_path} does not match the official DriveLM sha256 — refusing "
              f"to build a manifest from it", file=sys.stderr)
        return 3

    frames = eligible(load_frames(qa_path), args.arm)
    if not frames:
        print("no eligible frames", file=sys.stderr)
        return 1
    picked = sample(frames, args.n, args.seed)

    name = args.arm if args.arm == "single" else f"{args.arm}_{args.pair_mode}"
    out_dir = Path(args.out) if args.out else HERE / "data" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    missing: list[str] = []
    rows = []
    for row in picked:
        image = DRIVELM / row["image"]
        images = [image]
        if args.arm == "pair":
            images.insert(0, DRIVELM / row["prev"]["image"])
        for path in images:
            if not path.exists():
                missing.append(str(path))

        meta = {
            "scene": row["scene"],
            "frame": row["frame"],
            "t_s": round(row["t"], 2),
            "question": row["question"],
        }
        item = {"id": f"{row['scene'][:8]}_{row['frame'][:8]}"}

        # Paths are written relative to the manifest's own directory, which is
        # what vlmrunner resolves against by default. Absolute paths would work
        # on this machine and nowhere else, and this manifest gets committed.
        def rel(path: Path) -> str:
            return str(Path(os.path.relpath(path, out_dir)))

        if args.arm == "single":
            item["image"] = rel(image)
        else:
            meta["dt_s"] = row["dt_s"]
            if args.pair_mode == "composite":
                out_img = out_dir / "pairs" / f"{row['frame'][:8]}.jpg"
                if not missing:
                    composite(images[0], images[1], out_img)
                item["image"] = rel(out_img)
            else:
                item["images"] = [rel(x) for x in images]

        # pixel_sha1 feeds score.py's duplicate-weighted accuracy. nuScenes
        # frames should all be distinct, so this is expected to be a no-op —
        # which is exactly why it is worth computing rather than assuming.
        primary = out_dir / (item.get("image") or item["images"][-1])
        if primary.exists():
            meta["pixel_sha1"] = hashlib.sha1(primary.read_bytes()).hexdigest()

        item["meta"] = meta
        item["gt"] = {"steer": row["steer"], "speed": row["speed"]}
        rows.append(item)

    if missing and not args.allow_missing_images:
        print(f"{len(missing)} image(s) not found, e.g. {missing[0]}", file=sys.stderr)
        print("run ./fetch_drivelm.sh --images", file=sys.stderr)
        return 4

    with open(manifest, "w") as f:
        for item in rows:
            f.write(json.dumps(item) + "\n")

    steer = Counter(r["gt"]["steer"] for r in rows)
    speed = Counter(r["gt"]["speed"] for r in rows)
    n = len(rows)
    print(f"wrote {manifest} — {n} items from {len({r['meta']['scene'] for r in rows})} scenes")
    if missing:
        print(f"WARNING: {len(missing)} images missing; manifest is not runnable yet")
    print(f"  steer majority baseline {100 * steer.most_common(1)[0][1] / n:5.1f}%"
          f"  ({steer.most_common(1)[0][0]})")
    print(f"  speed majority baseline {100 * speed.most_common(1)[0][1] / n:5.1f}%"
          f"  ({speed.most_common(1)[0][0]})")
    for label, counter in (("steer", steer), ("speed", speed)):
        print(f"  {label}:", ", ".join(f"{k}={v}" for k, v in counter.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
