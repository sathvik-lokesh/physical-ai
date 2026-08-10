"""
Turn per-frame VLM labels into located objects, and measure what corroboration buys.

A label on its own is not a position. It says "there is a chair somewhere along
the direction this camera was pointing", which is a *bearing*, so a single
observation cannot place anything. Two observations of the same class from two
different positions can: the rays cross where the object is.

That geometry is also the filter. The p2 smoke test caught `llava-phi3`
hallucinating a bookshelf onto a bare wall, and a false label anchored to a real
pose becomes a landmark that is not there. But a hallucination that depends on
one particular view does not survive being looked at again from two metres to
the left: its ray crosses nothing, or crosses somewhere different every time.
Real objects produce rays that agree.

Two estimators are compared, which is the actual experiment:

  naive        every label is trusted and dropped at a fixed range along its
               bearing; the estimate is the centroid of those points.
  corroborated pairs of rays for the same class are intersected, pairs from
               near-identical viewpoints are discarded because they carry no
               parallax, and the estimate is the median intersection. A class
               needs MIN_SUPPORT surviving pairs to be reported at all.

Ground truth is exact: the objects are authored constants in the world file and
are `static`, so nothing can move them.

    python semantic_map.py                       # score both estimators
    python semantic_map.py --query "where did you last see the sofa"
"""

import argparse
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Authored object poses from harness/recorder/worlds/tb3_semantic.sdf.xacro.
TRUE_OBJECTS = {
    "chair": (2.5, 0.0),
    "table": (0.0, 2.5),
    "bookshelf": (-2.5, 0.0),
    "sofa": (0.0, -2.5),
}

ROOM_HALF = 3.5          # walls at +-3.5 m; an object cannot be outside them
MIN_BASELINE_M = 0.5     # two views closer than this give no parallax
MIN_SUPPORT = 3          # ray pairs a class needs before it is believed
MAX_RANGE_M = 6.0        # an intersection further than this is a near-parallel artefact

# The naive estimator drops each label at a fixed range along its bearing, and
# that range is a free parameter. The first version used 2.0 m, which happens to
# sit almost exactly on the true median object range for this route (1.96 m) —
# so it had been handed the answer's scale, and then "beat" triangulation with
# it. Rather than pick a flattering number, the naive estimator is now swept and
# reported at its *best* value. Losing to a baseline tuned in its own favour is
# the honest bar to clear.
NAIVE_RANGE_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# A label alone puts the object anywhere in the 60 deg field of view. Asking
# which third it sits in narrows that to ~20 deg; measured on this dataset it
# takes the median bearing error from 23.5 deg to 7.4 deg. Image left is +y in
# the robot frame, hence the positive offset.
SIDE_OFFSET_DEG = {"left": +20.0, "centre": 0.0, "right": -20.0, "none": 0.0}


def map_to_world(scenario: Path):
    """
    slam_toolbox reports in a map frame anchored at the robot's spawn pose.

    Object ground truth is in world coordinates, so SLAM poses are converted to
    world coordinates rather than the other way round — that keeps every error
    in this file in the same frame as the authored object positions.
    """
    s = json.loads(scenario.read_text())["spawn"]
    sx, sy, syaw = s.get("x", 0.0), s.get("y", 0.0), s.get("yaw", 0.0)
    c, sn = math.cos(syaw), math.sin(syaw)

    def to_world(mx, my, myaw):
        return (sx + c * mx - sn * my, sy + sn * mx + c * my, myaw + syaw)

    return to_world


def load_observations(results_path: Path, manifest_path: Path,
                      poses: str = "gt", use_side: bool = True,
                      scenario: Path | None = None) -> list[dict]:
    """
    (pose, heading, label) for every frame the model named an object in.

    `poses="slam"` uses slam_toolbox's estimate instead of the simulator's true
    pose, which is what a real robot would have. `use_side` applies the model's
    left/centre/right answer as a bearing offset when the task provides one.
    """
    meta = {}
    for line in open(manifest_path):
        if not line.strip():
            continue
        it = json.loads(line)
        meta[it["id"]] = it

    to_world = map_to_world(scenario) if (poses == "slam" and scenario) else None

    obs, sided = [], 0
    for line in open(results_path):
        if not line.strip():
            continue
        row = json.loads(line)
        parsed = row.get("parsed") or {}
        label = parsed.get("object_choice")
        if label in (None, "nothing"):
            continue
        m = meta.get(row["id"], {}).get("meta", {})

        if poses == "slam":
            if m.get("slam_x") is None or to_world is None:
                continue
            # to_world takes and returns radians for yaw.
            x, y, yaw = to_world(m["slam_x"], m["slam_y"],
                                 math.radians(m["slam_yaw_deg"]))
        else:
            if m.get("gt_x") is None or m.get("gt_yaw_deg") is None:
                continue
            x, y, yaw = m["gt_x"], m["gt_y"], math.radians(m["gt_yaw_deg"])

        side = parsed.get("side")
        if use_side and side in SIDE_OFFSET_DEG and side != "none":
            yaw += math.radians(SIDE_OFFSET_DEG[side])
            sided += 1

        obs.append({
            "id": row["id"], "label": label, "x": x, "y": y, "yaw": yaw,
            "truth": (meta[row["id"]].get("gt") or {}).get("object_choice"),
            "visibility": m.get("visibility"), "side": side,
        })
    if sided:
        print(f"  ({sided} of {len(obs)} labels carried a left/centre/right offset)")
    return obs


def ray_intersection(a: dict, b: dict) -> tuple[float, float] | None:
    """
    Where two bearing rays cross, or None if they cannot sensibly be crossed.

    Rejects four cases, each of which otherwise manufactures a confident
    landmark out of nothing: near-parallel rays, a crossing *behind* either
    camera, a crossing outside the room, and a crossing so far away that it is
    really just two almost-parallel rays meeting at infinity.
    """
    d1 = (math.cos(a["yaw"]), math.sin(a["yaw"]))
    d2 = (math.cos(b["yaw"]), math.sin(b["yaw"]))
    det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(det) < 1e-6:
        return None                                   # parallel
    rx, ry = b["x"] - a["x"], b["y"] - a["y"]
    t1 = (rx * (-d2[1]) - ry * (-d2[0])) / det
    t2 = (d1[0] * ry - d1[1] * rx) / det
    if t1 <= 0 or t2 <= 0:
        return None                                   # behind one of the cameras
    if t1 > MAX_RANGE_M or t2 > MAX_RANGE_M:
        return None                                   # effectively parallel
    px, py = a["x"] + t1 * d1[0], a["y"] + t1 * d1[1]
    if abs(px) > ROOM_HALF or abs(py) > ROOM_HALF:
        return None                                   # outside the room
    return (px, py)


def estimate_naive(obs: list[dict], range_m: float) -> dict:
    """Trust every label; drop it at a fixed range along its bearing."""
    out = {}
    for label in {o["label"] for o in obs}:
        pts = [
            (o["x"] + range_m * math.cos(o["yaw"]),
             o["y"] + range_m * math.sin(o["yaw"]))
            for o in obs if o["label"] == label
        ]
        out[label] = {
            "pos": (statistics.fmean(p[0] for p in pts),
                    statistics.fmean(p[1] for p in pts)),
            "support": len(pts),
        }
    return out


def best_naive(obs: list[dict]) -> tuple[dict, float, float]:
    """The naive estimator at whichever fixed range suits it best."""
    best, best_rng, best_err = None, None, float("inf")
    for rng in NAIVE_RANGE_SWEEP:
        est = estimate_naive(obs, rng)
        errs = [error_m(est[l]["pos"], l) for l in TRUE_OBJECTS if l in est]
        if not errs:
            continue
        med = statistics.median(errs)
        if med < best_err:
            best, best_rng, best_err = est, rng, med
    return best or {}, best_rng, best_err


def estimate_corroborated(obs: list[dict]) -> tuple[dict, dict]:
    """Intersect same-class ray pairs that have real parallax between them."""
    out, rejected = {}, {}
    for label in {o["label"] for o in obs}:
        group = [o for o in obs if o["label"] == label]
        pts = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < MIN_BASELINE_M:
                    continue                          # same viewpoint, no parallax
                p = ray_intersection(a, b)
                if p is not None:
                    pts.append(p)
        if len(pts) >= MIN_SUPPORT:
            out[label] = {
                "pos": (statistics.median(p[0] for p in pts),
                        statistics.median(p[1] for p in pts)),
                "support": len(pts),
            }
        else:
            rejected[label] = {"support": len(pts), "observations": len(group)}
    return out, rejected


def error_m(est: tuple[float, float], label: str) -> float | None:
    if label not in TRUE_OBJECTS:
        return None
    tx, ty = TRUE_OBJECTS[label]
    return math.hypot(est[0] - tx, est[1] - ty)


def bearing_error_deg(obs: list[dict]) -> dict | None:
    """
    How far each label's assumed bearing is from the true one.

    Reports the SIGNED mean as well as the absolute median, because they say
    different things and only one of them predicts localisation error. Averaging
    and triangulation both tolerate symmetric noise and are wrecked by a
    systematic offset. Asking the model which third of the frame the object was
    in improved the absolute median (14.6 -> 7.6 deg) while doubling the signed
    bias (+5.4 -> -10.6 deg), because it answered "left" almost regardless of the
    truth — and localisation got worse, not better. The absolute median alone
    would have called that a success.
    """
    offs = []
    for o in obs:
        if o["label"] not in TRUE_OBJECTS:
            continue
        tx, ty = TRUE_OBJECTS[o["label"]]
        true_b = math.atan2(ty - o["y"], tx - o["x"])
        offs.append((math.degrees(true_b - o["yaw"]) + 180) % 360 - 180)
    if not offs:
        return None
    return {
        "abs_median": statistics.median(abs(v) for v in offs),
        "abs_mean": statistics.fmean(abs(v) for v in offs),
        "signed_mean": statistics.fmean(offs),
        "n": len(offs),
    }


def report(obs: list[dict]) -> None:
    print(f"{len(obs)} frames carry an object label")
    be = bearing_error_deg(obs)
    if be is not None:
        print(f"bearing error of those labels: |median| {be['abs_median']:.1f} deg, "
              f"|mean| {be['abs_mean']:.1f} deg, signed mean {be['signed_mean']:+.1f} deg")
        print("  (the signed mean is the one that predicts localisation error)\n")

    naive, naive_rng, _ = best_naive(obs)
    corr, rejected = estimate_corroborated(obs)

    # Guessing the middle of the room for everything, the floor any estimator
    # has to clear before it has said anything.
    triv = statistics.median(math.hypot(x, y) for x, y in TRUE_OBJECTS.values())
    print(f"=== TRIVIAL (guess the room centre) ===\n  median error {triv:.2f} m\n")

    for name, est in ((f"NAIVE (trust every label, best fixed range {naive_rng} m)", naive),
                      ("CORROBORATED (>=2 views must agree)", corr)):
        print(f"=== {name} ===")
        errs = []
        for label in sorted(TRUE_OBJECTS):
            if label in est:
                e = error_m(est[label]["pos"], label)
                errs.append(e)
                px, py = est[label]["pos"]
                tx, ty = TRUE_OBJECTS[label]
                print(f"  {label:<11} at ({px:+.2f},{py:+.2f})  true ({tx:+.2f},{ty:+.2f})"
                      f"   error {e:.2f} m   support {est[label]['support']}")
            else:
                print(f"  {label:<11} NOT LOCATED")
        ghosts = [l for l in est if l not in TRUE_OBJECTS]
        for g in ghosts:
            print(f"  {g:<11} GHOST — reported but no such object exists")
        if errs:
            print(f"  located {len(errs)}/4   median error {statistics.median(errs):.2f} m"
                  f"   worst {max(errs):.2f} m")
        print()

    if rejected:
        print("rejected by corroboration (too few agreeing views):")
        for label, r in sorted(rejected.items()):
            print(f"  {label:<11} {r['observations']} raw label(s) -> only "
                  f"{r['support']} agreeing pair(s), needed {MIN_SUPPORT}")


def query(obs: list[dict], text: str) -> None:
    corr, _ = estimate_corroborated(obs)
    hit = next((l for l in sorted(corr, key=len, reverse=True) if l in text.lower()), None)
    if hit is None:
        print(f'"{text}"\n  -> no object in the semantic map matches that.')
        return
    px, py = corr[hit]["pos"]
    e = error_m((px, py), hit)
    print(f'"{text}"')
    print(f"  -> {hit} at ({px:+.2f}, {py:+.2f}) in the map"
          + (f", {e:.2f} m from where it really is" if e is not None else ""))
    print(f"     (drive a Nav2 goal to that pose to go and look)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--results",
                   default=str(HERE / "results/map_room.name_object.llava-phi3.jsonl"))
    p.add_argument("--manifest", default=str(HERE / "data/map_room/manifest.jsonl"))
    p.add_argument("--scenario", default=str(HERE / "scenarios/map_room.json"))
    p.add_argument("--poses", choices=["gt", "slam"], default="gt",
                   help="'slam' uses slam_toolbox's estimate, which is what a real "
                        "robot has; 'gt' the simulator's true pose")
    p.add_argument("--no-side", action="store_true",
                   help="ignore the model's left/centre/right answer")
    p.add_argument("--query", default=None, help="natural-language question")
    a = p.parse_args()

    obs = load_observations(Path(a.results), Path(a.manifest),
                            poses=a.poses, use_side=not a.no_side,
                            scenario=Path(a.scenario))
    if not obs:
        raise SystemExit("no labelled observations yet — has the VLM run finished?")
    print(f"poses: {a.poses}\n")
    if a.query:
        query(obs, a.query)
    else:
        report(obs)


if __name__ == "__main__":
    main()
