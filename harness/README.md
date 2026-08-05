# harness — shared infrastructure for p1..p4

Built once, reused by all four projects. Three pieces plus a safety wrapper:

| | what it does |
|---|---|
| `recorder/` | headless sim rollout → rosbag2 → keyframe PNGs + manifest |
| `vlmrunner/` | resumable batch VLM evaluation → results JSONL |
| `scoring/` | accuracy / coverage / latency + charts |
| `sysguard/` | cgroup CPU+memory ceiling so no job can wedge the laptop |

## The rule everything is built around

**Never run the simulator and the VLM at the same time.**

```
Phase A (sim)    record.py            → rosbag2 + events.json
Phase A'         extract_keyframes.py → frames/*.png + manifest.jsonl
Phase B (infer)  vlmrunner/run.py     → results.jsonl      ← background this
Phase C          scoring/score.py     → summary.json + charts
```

Phase A needs the CPU for physics and software rendering. Phase B needs ~4 GB
of RAM for the model. On an 8.41 GB machine with no GPU they do not coexist —
attempting it is what used to hang the box.

Before starting a sim run, make sure no model is resident:

```bash
ollama ps                                             # should be empty
curl -s localhost:11434/api/generate -d '{"model":"llava-phi3","keep_alive":0}'
```

A loaded `llava-phi3` holds **3.9 GB**. Leaving it resident during a rollout is
the single easiest way to bring the laptop down.

## Verified facts about this machine

Measured while building this, not assumed:

- **Headless camera rendering works.** `gz sim -s -r --headless-rendering` with
  the `Sensors` system on `ogre2` produces real camera frames through Mesa
  llvmpipe with **no `/dev/dri` and no GPU**. First frame lands ~10 s after
  start. This was the make-or-break unknown for p1 and p2.
- **The stock TurtleBot3 has no RGB camera.** `nav2_minimal_tb3_sim`'s waffle
  carries a *depth-only* sensor, and its bridge config forwards **no image topic
  at all**. Nothing RGB reaches ROS out of the box. `recorder/models/waffle_rgbd.sdf.xacro`
  vendors the SDF and adds an RGB sensor; `recorder/launch/rollout.launch.py`
  adds the missing image bridge.
- **The cgroup cap is real.** `sysguard/capped` with `MEM=200M` SIGKILLs a
  500 MB allocation (exit 137) and passes a 100 MB one. It is a ceiling, not a
  suggestion.
- **VLM latency is worse than the planning figure under contention.**
  `PLAN.md` assumed ~80 s/image from `edge-vlm-bench`. Measured here:
  `moondream` 106–157 s, `llava-phi3` **430 s** on the first item while a
  Gazebo test shared the CPU. Budget ~80–150 s/image for a *quiet* machine and
  do not run anything else during Phase B.
- **Model quality tracks `edge-vlm-bench`.** On the same scene-classification
  smoke test, `llava-phi3` answered correctly where `moondream` did not.

## Usage

### Phase A — record a rollout

```bash
source /opt/ros/jazzy/setup.bash
python recorder/record.py ../p1-failure-vlm/scenarios/stuck_wall.json \
    --out-dir ../p1-failure-vlm/data
```

Creates `data/<run_id>/` containing `bag/`, `events.json` (ground truth),
`run.json` (metadata), and `logs/`.

Scenarios are declarative JSON — see `record.py`'s docstring. `"mode": "script"`
drives `/cmd_vel` open-loop with Nav2 out of the picture, which is the reliable
way to *cause* a specific failure; `"mode": "goal"` runs the full Nav2 stack.

### Phase A' — extract keyframes

```bash
python recorder/extract_keyframes.py ../p1-failure-vlm/data/stuck_wall/bag \
    --out ../p1-failure-vlm/data/stuck_wall --run-id stuck_wall
```

Sampling is **by distance travelled** (default every 0.5 m), not by time, with a
stall timeout (default 5 s) so a stuck robot still contributes frames. Sampling
on a clock would fill the manifest with thousands of identical frames and spend
hours of VLM time re-answering the same question.

### Phase B — run the VLM (background this)

```bash
export OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1
sysguard/capped python -m vlmrunner.run \
    --manifest ../p1-failure-vlm/data/stuck_wall/manifest.jsonl \
    --task     ../p1-failure-vlm/tasks/failure_detect.json \
    --model    llava-phi3 \
    --out      ../p1-failure-vlm/results/stuck_wall.llava-phi3.jsonl
```

**Resumability is the point.** Every result is appended and `fsync`'d
immediately, and a rerun skips what is already done, keyed on
`(id, model, task)`. You can Ctrl-C at any moment to reclaim the laptop and lose
at most the item in flight. A truncated final line from a hard kill is tolerated
and that item is simply redone.

Useful flags: `--dry-run` (show the rendered prompt and exit), `--limit N`,
`--retry-errors`, `--sleep S` (keep the machine responsive).

### Phase C — score

```bash
python -m scoring.score \
    --manifest ../p1-failure-vlm/data/stuck_wall/manifest.jsonl \
    --results  ../p1-failure-vlm/results/*.jsonl \
    --out      ../p1-failure-vlm/results/report
```

Ground truth lives in the manifest's per-item `gt` block, so it never touches
the prompt path. Two accuracies are reported and **the gap between them is the
finding**:

- `strict` — correct / all items. Unparseable answers count as wrong. This is
  the number for a README.
- `answered` — correct / items where the field was actually produced.

A model that is *wrong* and a model that is *unintelligible* fail differently,
and only reporting `strict` hides which one you have.

## Task specs

A task is a JSON file, so projects add evaluations without touching harness
code:

```json
{
  "name": "failure_detect",
  "json_mode": true,
  "num_predict": 200,
  "prompt": "The robot was asked to {goal}. Did it fail? Respond as {\"failed\": true}",
  "fields": {
    "failed":       {"type": "bool"},
    "failure_mode": {"type": "enum", "values": ["stuck", "oscillation", "none"]},
    "reason":       {"type": "str"}
  }
}
```

`{placeholder}` spans are filled from the manifest item's `meta`. Literal JSON
braces in the prompt pass through untouched — the renderer deliberately is not
`str.format`, which chokes on exactly the JSON examples these prompts need.

Answer parsing is tolerant by design: code fences, prose around the JSON,
`field: value` text, and enum values mentioned anywhere in the answer are all
recovered, and a completely garbled answer yields `{}` rather than an exception.
Losing one item of a three-hour batch to a parse error is worse than scoring it
as a miss.

## Resource discipline

```bash
sysguard/capped ./anything            # 3 GB, 5 of 8 threads, nice 19
MEM=2G CPU=300% sysguard/capped ...   # tighter
NICE=0 sysguard/capped ...            # foreground work you are waiting on
```

`.wslconfig` is still `memory=12GB` on an 8.41 GB host, which is over-committed
— run `wsl --shutdown` from PowerShell before a heavy session. The per-job caps
above are the mitigation that does not require changing it.
