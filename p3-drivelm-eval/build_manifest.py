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
#
# Left None until the official file has actually been fetched once. Until then
# the expected hash comes from the sidecar `<qa>.sha256` that fetch_drivelm.sh
# writes with the value the HF API reported, or from `--record-hash`, which asks
# the API directly. If neither exists this script refuses to build, because a
# gate that silently passes when it has nothing to compare against is not a gate
# — which is what this constant being permanently None used to mean.
OFFICIAL_QA_SHA256 = None  # pin here once known; see --record-hash

HF_REPO = "OpenDriveLab/DriveLM"

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


def sidecar(qa_path: Path) -> Path:
    """Where fetch_drivelm.sh records the hash the HF API reported."""
    return qa_path.with_name(qa_path.name + ".sha256")


def api_sha256(name: str) -> str:
    """Ask the Hugging Face API for the official file's sha256 (its LFS oid)."""
    import urllib.error
    import urllib.request

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; cannot ask the API for the official hash")
    req = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            tree = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"HF API returned {e.code}: accept the licence at "
            f"https://huggingface.co/datasets/{HF_REPO} and check the token"
        ) from e
    for entry in tree:
        if entry.get("path") == name:
            oid = (entry.get("lfs") or {}).get("oid", "")
            if not oid:
                raise SystemExit(f"{name}: the API reports no LFS sha256 for it")
            return oid
    raise SystemExit(f"{name}: not found in {HF_REPO}")


def expected_sha256(qa_path: Path) -> str | None:
    """The hash to check the QA file against: pinned constant, else sidecar."""
    if OFFICIAL_QA_SHA256:
        return OFFICIAL_QA_SHA256
    side = sidecar(qa_path)
    if side.exists():
        text = side.read_text().split()
        if text:
            return text[0]
    return None


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
    p.add_argument("--record-hash", action="store_true",
                   help="ask the HF API for the official sha256, check the local "
                        "QA file against it, and record it beside the file")
    p.add_argument("--allow-unverified", action="store_true",
                   help="build without a hash to check against. Only for synthetic "
                        "fixtures — never for a manifest whose numbers get published")
    args = p.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"missing {qa_path} — run ./fetch_drivelm.sh first", file=sys.stderr)
        return 2

    if args.record_hash:
        want = api_sha256(qa_path.name)
        got = sha256_file(qa_path)
        if got != want:
            print(f"{qa_path} sha256 {got}\n  does not match the official "
                  f"{want}\n  refusing to record a hash for the wrong file",
                  file=sys.stderr)
            return 3
        sidecar(qa_path).write_text(f"{want}  {qa_path.name}\n")
        print(f"recorded official sha256 -> {sidecar(qa_path)}")

    # The second gate. It only means anything if there is something to compare
    # against, so "no known hash" is a refusal rather than a pass.
    want = expected_sha256(qa_path)
    if want is None:
        if not args.allow_unverified:
            print(f"no official sha256 to check {qa_path} against.\n"
                  f"  run ./fetch_drivelm.sh (it records one), or --record-hash "
                  f"with HF_TOKEN set.\n"
                  f"  An ungated mirror serves a same-length, different-content "
                  f"copy of this file, so this check is the point.",
                  file=sys.stderr)
            return 3
        print("WARNING: building without a hash check — do not publish these numbers")
    else:
        got = sha256_file(qa_path)
        if got != want:
            print(f"{qa_path} does not match the official DriveLM sha256 — refusing "
                  f"to build a manifest from it\n  expected {want}\n  got      {got}",
                  file=sys.stderr)
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

    # Images live in a `nuscenes/` tree beside the QA file, because that is how
    # the zip unpacks and how `image_paths` are written relative to it. Derived
    # from --qa rather than from the module constant, so that pointing --qa at a
    # second copy repoints its images too instead of half-repointing the build.
    image_root = qa_path.parent

    missing: list[str] = []
    rows = []
    for row in picked:
        image = image_root / row["image"]
        images = [image]
        if args.arm == "pair":
            images.insert(0, image_root / row["prev"]["image"])
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

    # Ids are truncated tokens because a full 32-hex pair is unreadable in a
    # progress line. Truncation can collide, and vlmrunner refuses a manifest
    # with duplicate ids -- so widen here rather than emit one it will reject.
    width = 8
    while width <= 32:
        ids = [f"{r['meta']['scene'][:width]}_{r['meta']['frame'][:width]}" for r in rows]
        if len(set(ids)) == len(ids):
            break
        width += 4
    else:
        raise SystemExit("scene/frame tokens are not unique even at full length")
    if width > 8:
        print(f"note: widened item ids to {width} hex chars to keep them unique")
    for item, new_id in zip(rows, ids):
        item["id"] = new_id

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
