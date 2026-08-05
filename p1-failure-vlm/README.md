# p1 — Can a small VLM tell that a robot has failed?

A TurtleBot3 is driven into deliberate failure modes in headless Gazebo. The
rollout is recorded to rosbag2, keyframes are extracted, and a small VLM is asked
**offline**: did this fail, and how? Answers are scored against ground truth
derived from the simulator's true pose.

The question is not "can a VLM caption a robot photo". It is whether a 3B-class
model that runs on a laptop CPU contributes anything a telemetry threshold does
not already give you.

**On this dataset, it does not.** Every configuration scored at or below the
accuracy of guessing the commonest label every time.

## Results

`llava-phi3` (3.8B), 32 labelled keyframes, 64 evaluations, 0 errors.

| prompt | field | accuracy | always-guess baseline | lift |
|---|---|---|---|---|
| `failure_visual` | `failed` | 40.6% (13/32) | 59.4% (always `True`) | **−18.8** |
| `failure_visual` | `failure_mode` | 40.6% (13/32) | 40.6% (always `none`) | **±0.0** |
| `failure_grounded` | `failed` | 56.2% (18/32) | 59.4% (always `True`) | **−3.2** |
| `failure_grounded` | `failure_mode` | 31.2% (10/32) | 40.6% (always `none`) | **−9.4** |

Parse coverage was 100% — every answer was well-formed JSON. The model is not
failing to *communicate*; it is failing to *discriminate*.

### The two prompts fail in opposite directions

From the image alone, the model essentially never reports a failure:

```
failure_visual, field `failed`
  gt False  ->  False x13          (13/13 right, trivially)
  gt True   ->  False x19          (0/19  — never once says True)
```

Add telemetry — commanded velocity, distance travelled, elapsed time — and the
bias inverts. It now almost never reports *success*:

```
failure_grounded, field `failure_mode`
  gt none         ->  oscillation x9, stuck x4     (0/13 — never says none)
  gt oscillation  ->  oscillation x10, stuck x2    (10/12)
  gt stuck        ->  oscillation x7               (0/7)
```

Telemetry moved the operating point without adding discrimination. `failed`
improved (40.6% → 56.2%) purely because the prior shifted toward the majority
class; `failure_mode` got *worse* (40.6% → 31.2%).

### `stuck` is never once identified

**0 out of 7, under both prompts.** This is the sharpest result, because those
frames are not subtle — the robot is pressed against a wall and the obstacle
fills the entire 320×240 frame. Asked what it sees, the model reliably answers
`oscillation` with the reasoning *"The robot is moving but not getting
anywhere"* — a fluent, plausible sentence about a robot that is completely
motionless in front of a wall.

That is the failure mode worth naming: the output is not noise, it is confident
and well-formed and wrong. Anything downstream that trusted the `reason` field
would be misled rather than merely uninformed.

## Why this is the expected shape, not a broken setup

It lines up with **MV-RoboBench (2026)**: strong scores on general single-view
spatial benchmarks do not predict success on robotic spatial tasks. The
contribution here is not the negative result itself but the harness that makes it
*measurable* — ground truth that cannot be gamed, a baseline printed next to
every number, and a resumable evaluation that runs overnight on a CPU laptop.

## Honest limitations

These matter more than the headline, and the numbers should not be quoted
without them:

- **32 frames, and they are not independent.** Each class comes from a single
  rollout, so the 7 `stuck` frames are the *same robot in the same pose* imaged 7
  times. "0/7" is closer to "one situation, failed once, confirmed seven times"
  than to seven independent trials. The `stuck` result needs more rollouts from
  varied approach angles, obstacles and lighting before it is a claim about the
  model rather than about this scene.
- **One model, one prompt pair, one seed.** No prompt sensitivity study. A
  different phrasing could plausibly move these numbers, and until that is run,
  "llava-phi3 cannot do this" is not established — only "these two prompts
  cannot".
- **Simulated, untextured scenes.** Flat-shaded Gazebo geometry is far from the
  distribution these models were trained on.
- **No comparison model.** `moondream` and `bakllava` are installed and the
  harness takes `MODELS="..."`; running them is the obvious next step.

## Reproducing

```bash
source /opt/ros/jazzy/setup.bash

# Phase A — record (sim only, no VLM)
python ../harness/recorder/record.py scenarios/stuck_wall.json --out-dir data

# Phase A' — keyframes + manifest
python ../harness/recorder/extract_keyframes.py data/stuck_wall/bag \
    --out data/stuck_wall --run-id stuck_wall

# Phase A'' — ground-truth labels from the event timeline
python label_run.py data/stuck_wall

# Phase B — the VLM (background this; never while the sim runs)
./run_eval.sh

# Phase C — score
cat data/*/manifest.jsonl > /tmp/all.jsonl
PYTHONPATH=../harness python -m scoring.score \
    --manifest /tmp/all.jsonl --results results/*.jsonl --out results/report
```

Runtime: ~2 min per rollout, ~68 s per VLM item on a quiet machine (~70 min for
all 64). `run_eval.sh` is resumable — killing it costs at most the item in
flight.

See `TAXONOMY.md` for the label definitions and how ground truth is derived, and
`../harness/README.md` for the machine-level findings behind the setup.
