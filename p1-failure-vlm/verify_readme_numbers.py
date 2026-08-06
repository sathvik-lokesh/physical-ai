"""
Re-derive every number quoted in README.md from the data, and fail if one moved.

A negative result is only worth anything if its numbers are exactly right, and
the numbers in that README were pasted by hand from a scoring run. Anything that
changes the dataset — re-recording a rollout, re-labelling, adding a scenario,
rerunning the model — silently invalidates them. This makes that loud instead.

    python verify_readme_numbers.py          # after ANY change to data/ or results/

Checks are literal duplicates of the README's claims on purpose. If a check
fails, decide which is wrong — the claim or the data — and fix that one. Do not
edit the expected value to match a new number without reading why it moved.
"""

import collections
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load():
    gt, sha, run = {}, {}, {}
    for path in sorted(glob.glob(str(HERE / "data" / "*" / "manifest.jsonl"))):
        for line in open(path):
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("gt"):
                continue
            gt[item["id"]] = item["gt"]
            sha[item["id"]] = item["meta"]["pixel_sha1"]
            run[item["id"]] = item["meta"]["run_id"]
    rows = [
        json.loads(line)
        for path in sorted(glob.glob(str(HERE / "results" / "*.jsonl")))
        for line in open(path)
        if line.strip()
    ]
    return gt, sha, run, rows


def main() -> int:
    gt, sha, run, rows = load()
    if not gt or not rows:
        raise SystemExit("no labelled manifests or no results — nothing to verify")

    visual = [r for r in rows if r["task"] == "failure_visual"]
    grounded = [r for r in rows if r["task"] == "failure_grounded"]

    def ans(row, field):
        return (row.get("parsed") or {}).get(field)

    def recall(sel, mode):
        return sum(
            1 for r in sel
            if gt[r["id"]]["failure_mode"] == mode and ans(r, "failure_mode") == mode
        )

    def contradictions(sel):
        return sum(
            1 for r in sel
            if ans(r, "failed") is False and ans(r, "failure_mode") not in (None, "none")
        )

    stuck_ids = [i for i in gt if gt[i]["failure_mode"] == "stuck"]
    walls = ("stuck_wall", "stuck_wall_oblique", "stuck_wall_dim", "stuck_corner")
    wall_groups = collections.Counter(
        sha[i] for i in stuck_ids if run[i] in walls
    )
    shared_sha, shared_n = wall_groups.most_common(1)[0]

    # Identical image + identical prompt must give an identical answer at
    # temperature 0. Only `failure_visual` qualifies: the grounded prompt embeds
    # per-frame telemetry, so its duplicates are not identical queries.
    by_image = collections.defaultdict(set)
    for r in visual:
        by_image[sha[r["id"]]].add(json.dumps(r.get("parsed"), sort_keys=True))

    checks = [
        # dataset shape
        ("rollouts", len({run[i] for i in gt}), 9),
        ("labelled keyframes", len(gt), 103),
        ("evaluations", len(rows), 206),
        ("errors", sum(1 for r in rows if r.get("error")), 0),
        ("answers missing a field", sum(
            1 for r in rows for f in ("failed", "failure_mode") if ans(r, f) is None), 0),
        ("distinct images overall", len(set(sha.values())), 51),
        ("stuck rollouts", len({run[i] for i in stuck_ids}), 7),
        ("stuck frames", len(stuck_ids), 55),
        ("distinct stuck images", len({sha[i] for i in stuck_ids}), 15),
        # the finding that four scenes render one picture
        ("frames in the shared wall image", shared_n, 30),
        ("rollouts sharing that image", len({run[i] for i in gt if sha[i] == shared_sha}), 4),
        # headline recalls
        ("visual: `failed` ever answered True", sum(
            1 for r in visual if ans(r, "failed") is True), 0),
        ("visual: stuck recall", recall(visual, "stuck"), 0),
        ("grounded: stuck recall", recall(grounded, "stuck"), 1),
        ("visual: none recall", recall(visual, "none"), 22),
        ("grounded: none recall", recall(grounded, "none"), 1),
        ("grounded: oscillation recall", recall(grounded, "oscillation"), 10),
        # self-contradiction
        ("visual contradictions", contradictions(visual), 29),
        ("grounded contradictions", contradictions(grounded), 73),
        # the kerb cell
        ("kerb visual answered none", sum(
            1 for r in visual if run[r["id"]] == "stuck_kerb"
            and gt[r["id"]]["failure_mode"] == "stuck" and ans(r, "failure_mode") == "none"), 8),
        ("kerb grounded answered oscillation", sum(
            1 for r in grounded if run[r["id"]] == "stuck_kerb"
            and gt[r["id"]]["failure_mode"] == "stuck"
            and ans(r, "failure_mode") == "oscillation"), 8),
        # determinism
        ("duplicate images given differing answers", sum(
            1 for s in by_image.values() if len(s) > 1), 0),
    ]

    # Per-scenario stuck recall, exactly as tabulated in the README.
    for scenario, n_frames in (
        ("stuck_wall", 7), ("stuck_wall_oblique", 7), ("stuck_wall_dim", 7),
        ("stuck_corner", 9), ("stuck_wall_edge", 8), ("stuck_gap", 9), ("stuck_kerb", 8),
    ):
        got = sum(1 for i in stuck_ids if run[i] == scenario)
        checks.append((f"{scenario}: stuck frames", got, n_frames))

    failed = 0
    for name, got, want in checks:
        mark = "ok  " if got == want else "FAIL"
        if got != want:
            failed += 1
        print(f"  {mark} {name}: {got}" + ("" if got == want else f"   (README says {want})"))

    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    if failed:
        print("\nThe README no longer matches the data. Work out which one is wrong.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
