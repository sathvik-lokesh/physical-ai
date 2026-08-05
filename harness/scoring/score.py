"""
Score vlmrunner results against ground truth and emit a report + charts.

Ground truth lives in the manifest, in a `gt` block per item, so it never
touches the prompt path:

    {"id": "run01_kf07", "image": "...", "meta": {...},
     "gt": {"failed": true, "failure_mode": "stuck"}}

Two accuracies are reported on purpose, and the gap between them is the finding:

  strict   — correct / all items with ground truth. Unparseable answers count
             as wrong. This is the number that belongs in a README.
  answered — correct / items where the model actually produced the field.
             Comparing it to strict tells you whether a model is *wrong* or
             merely *unintelligible*, which are different failure modes.

Usage:
    python -m scoring.score --manifest data/run01/manifest.jsonl \
        --results results/*.jsonl --out results/report
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a truncated tail from a killed run
    return rows


def normalize(value):
    """Compare case-insensitively for text; leave bools and numbers alone."""
    if isinstance(value, str):
        return value.strip().lower()
    return value


def score(manifest_path: Path, result_paths: list[Path]) -> dict:
    gt_by_id = {}
    for item in load_jsonl(manifest_path):
        if item.get("gt"):
            gt_by_id[item["id"]] = item["gt"]
    if not gt_by_id:
        raise SystemExit(f"{manifest_path}: no items carry a 'gt' block — nothing to score")

    rows = []
    for p in result_paths:
        rows.extend(load_jsonl(p))
    if not rows:
        raise SystemExit("no result rows found")

    # (model, task) -> per-field tallies, latencies, error count
    report: dict = {}
    for row in rows:
        key = f"{row.get('model','?')} / {row.get('task','?')}"
        entry = report.setdefault(
            key,
            {
                "model": row.get("model"),
                "task": row.get("task"),
                "n_rows": 0,
                "n_errors": 0,
                "n_scored": 0,
                "latencies_ms": [],
                "fields": defaultdict(
                    lambda: {"n": 0, "answered": 0, "correct": 0, "pairs": []}
                ),
            },
        )
        entry["n_rows"] += 1

        if row.get("error"):
            entry["n_errors"] += 1
        if row.get("wall_ms") is not None:
            entry["latencies_ms"].append(row["wall_ms"])

        gt = gt_by_id.get(row["id"])
        if gt is None:
            continue
        entry["n_scored"] += 1
        parsed = row.get("parsed") or {}

        for field, truth in gt.items():
            stats = entry["fields"][field]
            stats["n"] += 1
            if field in parsed:
                stats["answered"] += 1
                predicted = parsed[field]
                if normalize(predicted) == normalize(truth):
                    stats["correct"] += 1
            else:
                predicted = None
            stats["pairs"].append((truth, predicted))

    # Collapse tallies into final numbers
    for entry in report.values():
        lat = entry.pop("latencies_ms")
        entry["latency"] = (
            {
                "n": len(lat),
                "mean_s": round(statistics.mean(lat) / 1000, 1),
                "median_s": round(statistics.median(lat) / 1000, 1),
                "p90_s": round(sorted(lat)[max(0, int(0.9 * len(lat)) - 1)] / 1000, 1),
                "total_min": round(sum(lat) / 60000, 1),
            }
            if lat
            else None
        )
        fields = {}
        for field, s in entry["fields"].items():
            majority, majority_acc = _majority_baseline(s["pairs"])
            strict = round(s["correct"] / s["n"], 3) if s["n"] else 0.0
            fields[field] = {
                "n": s["n"],
                "answered": s["answered"],
                "coverage": round(s["answered"] / s["n"], 3) if s["n"] else 0.0,
                "correct": s["correct"],
                "strict_accuracy": strict,
                "answered_accuracy": (
                    round(s["correct"] / s["answered"], 3) if s["answered"] else 0.0
                ),
                # A model is only interesting if it beats guessing the commonest
                # label every time. Reporting accuracy without this is how a
                # constant predictor gets written up as a result.
                "majority_label": majority,
                "majority_accuracy": majority_acc,
                "lift_over_majority": round(strict - majority_acc, 3),
                "confusion": _confusion(s["pairs"]),
            }
        entry["fields"] = fields
    return report


def _majority_baseline(pairs: list[tuple]) -> tuple[str, float]:
    """Accuracy of always predicting the commonest ground-truth label."""
    truths = Counter(str(t) for t, _ in pairs)
    if not truths:
        return "", 0.0
    label, count = truths.most_common(1)[0]
    return label, round(count / sum(truths.values()), 3)


def _confusion(pairs: list[tuple]) -> dict | None:
    """gt -> {predicted -> count}, only for small label sets (enums and bools)."""
    labels = {str(t) for t, _ in pairs} | {str(p) for _, p in pairs if p is not None}
    if len(labels) > 8:
        return None  # free-text field; a confusion matrix would be noise
    matrix: dict[str, Counter] = defaultdict(Counter)
    for truth, predicted in pairs:
        matrix[str(truth)][str(predicted) if predicted is not None else "<no answer>"] += 1
    return {k: dict(v) for k, v in matrix.items()}


def print_report(report: dict) -> None:
    for key, entry in sorted(report.items()):
        lat = entry["latency"]
        print(f"\n=== {key} ===")
        print(
            f"  rows {entry['n_rows']}  scored {entry['n_scored']}  "
            f"errors {entry['n_errors']}"
            + (
                f"  |  latency mean {lat['mean_s']}s  median {lat['median_s']}s  "
                f"p90 {lat['p90_s']}s  total {lat['total_min']}min"
                if lat
                else ""
            )
        )
        for field, s in sorted(entry["fields"].items()):
            lift = s["lift_over_majority"]
            verdict = "BEATS" if lift > 0 else ("ties" if lift == 0 else "BELOW")
            print(
                f"  {field:<16} strict {s['strict_accuracy']:.1%} "
                f"({s['correct']}/{s['n']})   "
                f"answered {s['answered_accuracy']:.1%} ({s['correct']}/{s['answered']})   "
                f"coverage {s['coverage']:.1%}"
            )
            print(
                f"  {'':<16} baseline (always {s['majority_label']!r}) "
                f"{s['majority_accuracy']:.1%}  ->  {verdict} baseline by "
                f"{lift:+.1%}"
            )
            if s["confusion"]:
                for truth, preds in sorted(s["confusion"].items()):
                    shown = ", ".join(f"{p}×{c}" for p, c in sorted(preds.items()))
                    print(f"      gt {truth:<20} -> {shown}")


def make_charts(report: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")  # headless: no display on this box
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fields = sorted({f for e in report.values() for f in e["fields"]})
    models = sorted(report)
    if not fields or not models:
        return written

    # Accuracy: strict vs answered, grouped by model, one subplot per field
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4), squeeze=False)
    for ax, field in zip(axes[0], fields):
        strict = [report[m]["fields"].get(field, {}).get("strict_accuracy", 0) for m in models]
        answered = [
            report[m]["fields"].get(field, {}).get("answered_accuracy", 0) for m in models
        ]
        x = range(len(models))
        ax.bar([i - 0.2 for i in x], strict, 0.4, label="strict", color="#2b6cb0")
        ax.bar([i + 0.2 for i in x], answered, 0.4, label="answered-only", color="#a0aec0")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy")
        ax.set_title(field)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "accuracy.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # Latency: mean per model, with the p90 marked — the tail is what hurts here
    lat_models = [m for m in models if report[m]["latency"]]
    if lat_models:
        fig, ax = plt.subplots(figsize=(1.6 * len(lat_models) + 3, 4))
        means = [report[m]["latency"]["mean_s"] for m in lat_models]
        p90s = [report[m]["latency"]["p90_s"] for m in lat_models]
        ax.bar(lat_models, means, color="#2b6cb0", label="mean")
        ax.scatter(lat_models, p90s, color="#e53e3e", zorder=3, label="p90")
        ax.set_ylabel("seconds per item")
        ax.set_title("Inference latency (CPU-only)")
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "latency.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    return written


def main():
    p = argparse.ArgumentParser(description="Score VLM results against manifest ground truth")
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", required=True, nargs="+")
    p.add_argument("--out", default=None, help="dir for summary.json + charts")
    p.add_argument("--no-charts", action="store_true")
    args = p.parse_args()

    report = score(Path(args.manifest), [Path(r) for r in args.results])
    print_report(report)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = out_dir / "summary.json"
        with open(summary, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsummary -> {summary}")
        if not args.no_charts:
            for path in make_charts(report, out_dir):
                print(f"chart   -> {path}")


if __name__ == "__main__":
    main()
