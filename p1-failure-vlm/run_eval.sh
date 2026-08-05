#!/usr/bin/env bash
# Phase B for p1: run the visual-vs-grounded ablation over every labelled run.
#
# This is the overnight job. It is safe to Ctrl-C at any point and rerun — the
# runner resumes per (id, model, task), so you lose at most the item in flight.
#
#   ./run_eval.sh                        # llava-phi3, both tasks, all runs
#   MODELS="llava-phi3 moondream" ./run_eval.sh
#   RUNS="data/stuck_wall" ./run_eval.sh
#
# Do NOT start this while a rollout is recording. One resident VLM is ~3.9 GB and
# the simulator needs the CPU; running both is what wedges this laptop.

set -uo pipefail
cd "$(dirname "$0")"

MODELS="${MODELS:-llava-phi3}"
TASKS="${TASKS:-failure_visual failure_grounded}"
RUNS="${RUNS:-$(find data -mindepth 1 -maxdepth 1 -type d | sort)}"

export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
# vlmrunner/ and scoring/ live in the shared harness, not in this project
export PYTHONPATH="$(cd ../harness && pwd)${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "$RUNS" ]; then
    echo "no runs under data/ — record some first" >&2
    exit 1
fi

mkdir -p results

total=0
for run in $RUNS; do
    [ -f "$run/manifest.jsonl" ] || { echo "skip $run (no manifest)"; continue; }
    grep -q '"gt"' "$run/manifest.jsonl" || {
        echo "skip $run (not labelled — run: python label_run.py $run)"; continue; }
    n=$(wc -l < "$run/manifest.jsonl")
    total=$((total + n))
done

n_combos=$(( $(echo "$MODELS" | wc -w) * $(echo "$TASKS" | wc -w) ))
echo "runs     : $(echo "$RUNS" | wc -w)"
echo "frames   : $total per model/task, $n_combos combination(s)"
echo "estimate : ~$(( total * n_combos * 100 / 60 )) min at ~100 s/frame"
echo

for model in $MODELS; do
    for task in $TASKS; do
        for run in $RUNS; do
            [ -f "$run/manifest.jsonl" ] || continue
            grep -q '"gt"' "$run/manifest.jsonl" || continue
            run_id=$(basename "$run")
            out="results/${run_id}.${task}.${model//:/_}.jsonl"
            echo "=== $run_id | $task | $model ==="
            ../harness/sysguard/capped python3 -m vlmrunner.run \
                --manifest "$run/manifest.jsonl" \
                --task "tasks/${task}.json" \
                --model "$model" \
                --out "$out" || {
                    echo "runner exited non-zero for $out — rerun to resume" >&2
                }
        done
    done
done

echo
echo "done. score with:"
echo "  python3 -m scoring.score --manifest data/<run>/manifest.jsonl \\"
echo "      --results results/<run>.*.jsonl --out results/report"
