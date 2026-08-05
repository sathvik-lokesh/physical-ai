"""
Resumable batch VLM evaluation: manifest.jsonl + task spec -> results.jsonl.

The non-negotiable property on this hardware (~80 s per image) is that the job
can be killed at any instant — Ctrl-C, OOM killer, closing the laptop — and
restarted without losing or repeating completed work. Every result is appended
and fsync'd the moment it arrives, and a rerun skips whatever is already in the
output file.

Usage:
    python -m vlmrunner.run \
        --manifest ../p1-failure-vlm/data/run01/manifest.jsonl \
        --task     ../p1-failure-vlm/tasks/failure_detect.json \
        --model    llava-phi3 \
        --out      ../p1-failure-vlm/results/run01.llava-phi3.jsonl
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python run.py` as well as `-m vlmrunner.run`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmrunner import ollama_client as oc
from vlmrunner.tasks import Task

_stop = False


def _request_stop(signum, _frame):
    """First signal finishes the in-flight item; a second one is your hard kill."""
    global _stop
    if _stop:
        print("\n[vlmrunner] second signal — exiting now", file=sys.stderr)
        sys.exit(130)
    _stop = True
    print(
        f"\n[vlmrunner] signal {signum} received — will stop after the current item "
        f"(press again to abort immediately)",
        file=sys.stderr,
    )


def read_manifest(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: bad JSON in manifest: {e}") from e
            if "id" not in item:
                raise SystemExit(f"{path}:{lineno}: manifest item has no 'id'")
            items.append(item)
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise SystemExit(f"{path}: duplicate manifest ids: {sorted(dupes)[:5]}")
    return items


def load_done(out_path: Path, model: str, task_name: str, retry_errors: bool) -> set[str]:
    """
    Ids already evaluated in this file for this exact (model, task) pair.

    Keyed on all three so that pointing two models at one output file compares
    them instead of the second silently inheriting the first one's results.
    """
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from a hard kill — just redo that item
            if row.get("model") != model or row.get("task") != task_name:
                continue
            if retry_errors and row.get("error"):
                continue
            done.add(row["id"])
    return done


def resolve_images(item: dict, image_root: Path) -> list[Path]:
    raw = item.get("images") or ([item["image"]] if "image" in item else [])
    paths = []
    for p in raw:
        path = Path(p)
        paths.append(path if path.is_absolute() else image_root / path)
    return paths


def append_row(fh, row: dict) -> None:
    """Append one result and force it to disk before moving on."""
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def run(args) -> int:
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()
    task = Task.load(args.task)

    image_root = Path(args.image_root).resolve() if args.image_root else manifest_path.parent
    items = read_manifest(manifest_path)

    if args.dry_run:
        item = items[0]
        print(f"task    : {task.name} ({task.source})")
        print(f"model   : {args.model}")
        print(f"items   : {len(items)}")
        print(f"images  : {[str(p) for p in resolve_images(item, image_root)]}")
        print(f"\n--- rendered prompt for {item['id']} ---\n{task.prompt_for(item)}")
        return 0

    print(f"[vlmrunner] ollama {oc.version(args.host)} at {args.host}")
    available = oc.installed_models(args.host)
    # `ollama list` reports "moondream:latest"; people type "moondream"
    if args.model not in available and f"{args.model}:latest" not in available:
        raise SystemExit(
            f"model {args.model!r} is not installed. Available: {', '.join(available)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path, args.model, task.name, args.retry_errors)
    todo = [i for i in items if i["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(
        f"[vlmrunner] task={task.name} model={args.model} "
        f"total={len(items)} done={len(done)} todo={len(todo)}"
    )
    if not todo:
        print("[vlmrunner] nothing to do")
        return 0

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    latencies: list[float] = []
    n_err = 0
    started = time.time()

    with open(out_path, "a") as fh:
        for n, item in enumerate(todo, 1):
            if _stop:
                break

            images = resolve_images(item, image_root)
            missing = [str(p) for p in images if not p.exists()]
            row = {
                "id": item["id"],
                "task": task.name,
                "model": args.model,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "images": [str(p.relative_to(image_root)) if p.is_relative_to(image_root)
                           else str(p) for p in images],
                "meta": item.get("meta", {}),
            }

            if missing:
                row["error"] = f"missing image(s): {missing}"
                append_row(fh, row)
                n_err += 1
                print(f"  [{n}/{len(todo)}] {item['id']}: {row['error']}")
                continue

            prompt = task.prompt_for(item)
            encoded = [oc.encode_image(p) for p in images]

            result, err = None, None
            for attempt in range(1, args.retries + 1):
                try:
                    result = oc.generate(
                        args.model,
                        prompt,
                        encoded,
                        host=args.host,
                        json_mode=task.json_mode,
                        timeout=args.timeout,
                        num_predict=task.num_predict,
                        keep_alive=args.keep_alive,
                    )
                    break
                except oc.OllamaError as e:
                    err = str(e)
                    if attempt < args.retries and not _stop:
                        backoff = 5 * attempt
                        print(f"  [{n}/{len(todo)}] {item['id']}: {err} "
                              f"— retry {attempt}/{args.retries - 1} in {backoff}s")
                        time.sleep(backoff)

            if result is None:
                row["error"] = err
                n_err += 1
            else:
                row["raw"] = result["text"]
                row["parsed"] = task.parse(result["text"])
                row["wall_ms"] = result["wall_ms"]
                row["ollama_ms"] = result["ollama_ms"]
                row["prompt_tokens"] = result["prompt_tokens"]
                row["output_tokens"] = result["output_tokens"]
                latencies.append(result["wall_ms"])

            append_row(fh, row)

            avg = sum(latencies) / len(latencies) / 1000 if latencies else 0
            eta = fmt_eta(avg * (len(todo) - n)) if latencies else "?"
            summary = row.get("error") or json.dumps(row.get("parsed", {}))
            print(
                f"  [{n}/{len(todo)}] {item['id']}  "
                f"{row.get('wall_ms', 0)/1000:.0f}s  eta {eta}  {summary[:90]}"
            )

            if args.sleep and n < len(todo) and not _stop:
                time.sleep(args.sleep)

    elapsed = time.time() - started
    print(
        f"[vlmrunner] {'stopped' if _stop else 'finished'}: "
        f"{len(latencies)} ok, {n_err} error, {fmt_eta(elapsed)} elapsed"
        + (f", {sum(latencies)/len(latencies)/1000:.0f}s/item avg" if latencies else "")
    )
    print(f"[vlmrunner] results -> {out_path}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--manifest", required=True, help="JSONL of items to evaluate")
    p.add_argument("--task", required=True, help="task spec JSON")
    p.add_argument("--model", required=True, help="Ollama model tag, e.g. llava-phi3")
    p.add_argument("--out", required=True, help="results JSONL (appended to, resumable)")
    p.add_argument("--image-root", default=None,
                   help="base dir for relative image paths (default: manifest's dir)")
    p.add_argument("--host", default=oc.DEFAULT_HOST)
    p.add_argument("--limit", type=int, default=None, help="stop after N new items")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="per-request timeout in seconds (cold load can exceed 300)")
    p.add_argument("--retries", type=int, default=3, help="attempts per item")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="pause between items, to keep the laptop responsive")
    p.add_argument("--keep-alive", default="5m",
                   help="how long Ollama holds the model; '0' to unload immediately")
    p.add_argument("--retry-errors", action="store_true",
                   help="re-run items whose stored row has an error")
    p.add_argument("--dry-run", action="store_true",
                   help="show the rendered prompt for the first item and exit")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
