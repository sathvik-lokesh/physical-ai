"""
Re-derive every number quoted in p2's README.md from the data, and fail if one moved.

Same reason as p1's version: the numbers in that README were pasted by hand from
scoring runs, and anything that changes the data or the estimator silently
invalidates them. p1's copy has already caught two stale figures of mine, and
p2's write-up corrects two claims that were wrong the first time, so the numbers
here have earned less trust than usual.

    python verify_readme_numbers.py

A failing check means the README and the data disagree. Work out which is wrong.
Do not edit the expected value to match a new number without reading why it moved.
"""

import collections
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from semantic_map import (  # noqa: E402
    TRUE_OBJECTS, best_naive, bearing_error_deg, error_m, load_observations,
)

MANIFEST = HERE / "data/map_room/manifest.jsonl"
SCENARIO = HERE / "scenarios/map_room.json"
PLAIN = HERE / "results/map_room.name_object.llava-phi3.jsonl"
BEARING = HERE / "results/map_room.name_object_bearing.llava-phi3.jsonl"


def frame_accuracy(results: Path) -> dict:
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(MANIFEST) if l.strip()}
    tally = collections.Counter()
    for line in open(results):
        if not line.strip():
            continue
        r = json.loads(line)
        it = meta[r["id"]]
        vis = it["meta"].get("visibility")
        want = (it.get("gt") or {}).get("object_choice")
        got = (r.get("parsed") or {}).get("object_choice")
        tally[(vis, got == want)] += 1
    return {v: (tally[(v, True)], tally[(v, True)] + tally[(v, False)])
            for v in ("clear", "edge", "none")}


def main() -> int:
    checks = []

    def ck(name, got, want, tol=None):
        ok = (abs(got - want) <= tol) if (tol is not None and
                                          isinstance(got, (int, float))) else got == want
        checks.append((name, got, want, ok))

    # --- dataset ---
    items = [json.loads(l) for l in open(MANIFEST) if l.strip()]
    ck("keyframes", len(items), 61)
    ck("pixel-unique keyframes", len({i["meta"]["pixel_sha1"] for i in items}), 60)
    vis = collections.Counter(i["meta"].get("visibility") for i in items)
    ck("frames: object centred", vis["clear"], 9)
    ck("frames: object at edge", vis["edge"], 26)
    ck("frames: empty", vis["none"], 26)

    # --- SLAM quality ---
    errs = [i["meta"]["slam_pos_error_m"] for i in items
            if i["meta"].get("slam_pos_error_m") is not None]
    ck("slam poses present", len(errs), 61)
    ck("slam median pose error (mm)", round(statistics.median(errs) * 1000), 8, tol=1)
    ck("slam max pose error (mm)", round(max(errs) * 1000), 20, tol=1)

    # --- per-frame labelling, plain prompt ---
    acc = frame_accuracy(PLAIN)
    ck("plain: centred correct", acc["clear"], (7, 9))
    ck("plain: edge correct", acc["edge"], (7, 26))
    ck("plain: empty correct", acc["none"], (26, 26))

    # --- per-frame labelling, bearing prompt (it got worse) ---
    accb = frame_accuracy(BEARING)
    ck("bearing: centred correct", accb["clear"], (5, 9))
    ck("bearing: edge correct", accb["edge"], (5, 26))
    ck("bearing: empty correct", accb["none"], (26, 26))

    # --- the side field it could not answer ---
    sides = collections.Counter(
        (json.loads(l).get("parsed") or {}).get("side")
        for l in open(BEARING) if l.strip())
    ck("bearing: said 'left'", sides["left"], 9)
    ck("bearing: said 'right'", sides["right"], 1)
    ck("bearing: said 'centre'", sides["centre"], 0)
    ck("bearing: said 'none'", sides["none"], 51)

    # --- localisation, all on SLAM poses ---
    def run(results, use_side):
        obs = load_observations(results, MANIFEST, poses="slam",
                                use_side=use_side, scenario=SCENARIO)
        est, rng, _ = best_naive(obs)
        return obs, est, rng

    obs_b, est_b, rng_b = run(BEARING, False)
    ck("best naive fixed range (m)", rng_b, 3.0)
    ck("bearing/no-side: chair error (cm)",
       round(error_m(est_b["chair"]["pos"], "chair") * 100), 8, tol=1)
    ck("bearing/no-side: bookshelf error (cm)",
       round(error_m(est_b["bookshelf"]["pos"], "bookshelf") * 100), 22, tol=1)
    ck("bearing/no-side: sofa error (cm)",
       round(error_m(est_b["sofa"]["pos"], "sofa") * 100), 146, tol=2)
    ck("bearing/no-side: table located", "table" in est_b, False)

    obs_p, est_p, _ = run(PLAIN, True)
    ck("plain: labelled frames", len(obs_p), 23)
    ck("plain: table error (m)",
       round(error_m(est_p["table"]["pos"], "table"), 2), 2.02, tol=0.02)

    # --- the metric that actually predicts localisation error ---
    for label, results, side, want in (
        ("plain", PLAIN, True, -13.8),
        ("bearing/side-applied", BEARING, True, -11.5),
        ("bearing/no-side", BEARING, False, +4.5),
    ):
        o = load_observations(results, MANIFEST, poses="slam",
                              use_side=side, scenario=SCENARIO)
        ck(f"signed bearing bias, {label} (deg)",
           round(bearing_error_deg(o)["signed_mean"], 1), want, tol=0.2)

    # --- the trivial floor every estimator must clear ---
    ck("trivial baseline (m)",
       round(statistics.median(math.hypot(x, y) for x, y in TRUE_OBJECTS.values()), 2),
       2.50, tol=0.01)

    failed = 0
    for name, got, want, ok in checks:
        if not ok:
            failed += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got}"
              + ("" if ok else f"   (README says {want})"))
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    if failed:
        print("\nThe README no longer matches the data. Work out which one is wrong.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
