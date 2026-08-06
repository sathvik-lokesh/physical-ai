"""
Derive per-keyframe ground truth from a rollout's event timeline.

Adds a `gt` block to every item in the run's manifest.jsonl, in place:

    {"id": "...", "image": "...", "meta": {...},
     "gt": {"failed": true, "failure_mode": "stuck"}}

Labels come from what the odometry *did*, never from the scenario's `expect`
field. `expect` is only used as a cross-check: if a scenario meant to produce
`stuck` and no frame was ever labelled `stuck`, that is reported loudly, because
the honest response is to fix the scenario — not to relabel the data to match
the intent. Ground truth bent toward what we hoped for makes every downstream
accuracy number meaningless.

See TAXONOMY.md for the label definitions.

Usage:
    python label_run.py data/stuck_wall
    python label_run.py data/*/          # label every run
"""

import argparse
import json
import math
import sys
from pathlib import Path

# --- oscillation thresholds (see TAXONOMY.md) ---
#
# The window must span at least one full oscillation period, or it only ever
# sees half a cycle — which looks like ordinary progress. The `oscillate_spin`
# scenario arcs 8 s left then 8 s right, a 16 s period, so 20 s covers a full
# cycle with margin. A shorter window (10 s was tried first) labelled 1 frame
# out of 11 on a run whose net displacement over the whole rollout was 5 cm.
OSC_WINDOW_S = 20.0      # look this far either side of a frame
OSC_MIN_PATH_M = 0.35    # it has to actually be moving...
OSC_MAX_RADIUS_M = 0.40  # ...while staying inside a small patch of floor...
OSC_MIN_PATH_PER_EXTENT = 3.0  # ...and doubling back, not going straight

# That last ratio is the one that makes this a shape test rather than a size
# test, and it was added after `stuck_gap` exposed the gap. The extent cap alone
# labelled three frames of a robot that drove 0.43 m in a dead-straight line and
# stopped, because a straight run's extent is only ever about half its path
# length, so any straight run shorter than ~0.8 m fits inside a 0.40 m patch.
# The earlier scenarios all travelled further than that and hid the bug.
#
# Measured over all seven runs the two populations are nowhere near each other:
#
#     genuine oscillation (oscillate_spin)   path/extent  6.6 - 12.0
#     straight lines (every other run)       path/extent  1.2 -  2.3
#
# so 3.0 sits in a wide empty band rather than being tuned to a boundary case.
# The extent cap is kept as well: both conditions must hold, which is the
# conservative direction for ground truth. Re-labelling the three pre-existing
# runs with this rule reproduces their old labels exactly, so the published
# results are unaffected.


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stall_onset(events: list[dict]) -> float | None:
    """Time of the first detected stall, if the robot ever stalled."""
    for e in events:
        if e["kind"] == "stall_detected":
            return e["t_s"]
    return None


def stall_cleared_after(events: list[dict], t: float) -> float | None:
    for e in events:
        if e["kind"] == "stall_cleared" and e["t_s"] > t:
            return e["t_s"]
    return None


def nav_failure(events: list[dict]) -> bool:
    return any(e["kind"] in ("goal_failed", "goal_timeout") for e in events)


def is_oscillating(items: list[dict], i: int) -> bool:
    """
    True when the robot covered ground without getting anywhere.

    Measures path length travelled against the *spatial extent* of the window —
    the furthest any sample strays from the window's centroid — rather than the
    displacement between its first and last sample. The decisive test is their
    ratio: covering several times more ground than the patch you covered it in
    is what doubling back means, and a straight line cannot fake it.

    Endpoint displacement was tried first and is phase-sensitive: on a robot
    arcing left then right, two samples 20 s apart can sit at opposite ends of an
    arc and look like real progress, so a run that stayed inside a 0.55 m box
    while driving 4.83 m got exactly one frame labelled. Extent has no such
    dependence on where in the cycle the window happens to land.
    """
    meta = items[i]["meta"]
    t = meta.get("t_s")
    # Ground truth only. Using wheel odometry here would call a wall-pinned
    # robot "oscillating" purely from slip.
    if t is None or meta.get("gt_x") is None:
        return False

    window = [
        it["meta"] for it in items
        if it["meta"].get("t_s") is not None
        and abs(it["meta"]["t_s"] - t) <= OSC_WINDOW_S
        and it["meta"].get("gt_x") is not None
    ]
    if len(window) < 2:
        return False

    path = window[-1]["gt_path_m"] - window[0]["gt_path_m"]
    cx = sum(m["gt_x"] for m in window) / len(window)
    cy = sum(m["gt_y"] for m in window) / len(window)
    radius = max(math.hypot(m["gt_x"] - cx, m["gt_y"] - cy) for m in window)
    if path < OSC_MIN_PATH_M or radius > OSC_MAX_RADIUS_M:
        return False
    # A pinned robot has zero extent and near-zero path; MIN_PATH has already
    # excluded it, so anything reaching here has moved and radius is meaningful.
    return radius > 0 and path / radius >= OSC_MIN_PATH_PER_EXTENT


def label_run(run_dir: Path, dry_run: bool = False) -> dict:
    manifest_path = run_dir / "manifest.jsonl"
    events_path = run_dir / "events.json"
    for p in (manifest_path, events_path):
        if not p.exists():
            raise SystemExit(f"{run_dir}: missing {p.name}")

    items = load_jsonl(manifest_path)
    run = json.loads(events_path.read_text())
    events = run["events"]
    expect = run.get("expect", "")

    onset = stall_onset(events)
    cleared = stall_cleared_after(events, onset) if onset is not None else None
    nav_failed = nav_failure(events)

    counts: dict[str, int] = {}
    for i, item in enumerate(items):
        t = item["meta"].get("t_s")
        mode = "none"

        if onset is not None and t is not None and t >= onset:
            # A stall that later cleared only labels the frames inside it
            if cleared is None or t < cleared:
                mode = "stuck"
        if mode == "none" and is_oscillating(items, i):
            mode = "oscillation"
        if mode == "none" and nav_failed:
            mode = "goal_unreachable"

        item["gt"] = {"failed": mode != "none", "failure_mode": mode}
        counts[mode] = counts.get(mode, 0) + 1

    if not dry_run:
        with open(manifest_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    return {
        "run": run_dir.name,
        "frames": len(items),
        "counts": counts,
        "expect": expect,
        "stall_onset_s": onset,
        "matched_expect": expect in counts if expect else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("runs", nargs="+", help="run directories (each holds manifest.jsonl + events.json)")
    p.add_argument("--dry-run", action="store_true", help="report labels without writing")
    args = p.parse_args()

    problems = []
    for run in args.runs:
        run_dir = Path(run)
        if not run_dir.is_dir():
            continue
        summary = label_run(run_dir, args.dry_run)
        spread = ", ".join(f"{k}={v}" for k, v in sorted(summary["counts"].items()))
        print(f"{summary['run']:<18} {summary['frames']:>3} frames   {spread}")
        if summary["expect"] and not summary["matched_expect"]:
            problems.append(
                f"  {summary['run']}: scenario expected {summary['expect']!r} but no frame "
                f"was labelled that. The scenario needs fixing, not the labels."
            )

    if problems:
        print("\nGROUND TRUTH DOES NOT MATCH INTENT:")
        print("\n".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
