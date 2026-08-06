# p1 — Can a small VLM tell that a robot has failed?

A TurtleBot3 is driven into deliberate failure modes in headless Gazebo. The
rollout is recorded to rosbag2, keyframes are extracted, and a small VLM is asked
**offline**: did this fail, and how? Answers are scored against ground truth
derived from the simulator's true pose.

The question is not "can a VLM caption a robot photo". It is whether a 3B-class
model that runs on a laptop CPU contributes anything a telemetry threshold does
not already give you.

**On this dataset, it does not.** Nine rollouts, 103 labelled keyframes, 206
evaluations, 0 errors, 100% parse coverage — and `stuck` is correctly named
**once in 110 attempts**.

## Results

`llava-phi3` (3.8B), 9 rollouts, 103 labelled keyframes, 206 evaluations,
0 errors. 5.5 hours of CPU.

| prompt | field | accuracy | always-guess baseline | lift |
|---|---|---|---|---|
| `failure_visual` | `failed` | 35.0% (36/103) | 65.0% (always `True`) | **−30.0** |
| `failure_visual` | `failure_mode` | 27.2% (28/103) | 53.4% (always `stuck`) | **−26.2** |
| `failure_grounded` | `failed` | 32.0% (33/103) | 65.0% (always `True`) | **−33.0** |
| `failure_grounded` | `failure_mode` | 11.7% (12/103) | 53.4% (always `stuck`) | **−41.7** |

Parse coverage was 100% — every answer was well-formed JSON. The model is not
failing to *communicate*; it is failing to *discriminate*.

### The same numbers, with duplicate frames collapsed

A stalled robot in a static scene renders the same picture over and over, so
frame count overstates the sample. Weighting every pixel-identical image to 1 in
total gives 51 distinct images, and a baseline weighted the same way:

| prompt | field | distinct-image accuracy | baseline | lift |
|---|---|---|---|---|
| `failure_visual` | `failed` | 48.8% | 51.2% | −2.4 |
| `failure_visual` | `failure_mode` | 43.7% | 48.8% | −5.1 |
| `failure_grounded` | `failed` | **54.6%** | 51.2% | **+3.4** |
| `failure_grounded` | `failure_mode` | 20.6% | 48.8% | −28.2 |

One cell of twelve finally clears the baseline, by 3.4 points on 51 images.
That is not a capability; it is the size of the noise. Report it anyway — the
whole point of printing a baseline next to every number is that it also catches
you rounding your own result up.

### `stuck` is essentially never identified

**0 out of 55 from the image alone. 1 out of 55 with telemetry.** Across seven
rollouts, four obstacle geometries, two lighting levels, two approach angles and
15 distinct stall images:

| scenario | what the camera sees at the stall | visual | grounded |
|---|---|---|---|
| `stuck_wall` | wall fills the frame | 0/7 | 0/7 |
| `stuck_wall_oblique` | wall fills the frame (26° approach) | 0/7 | 0/7 |
| `stuck_wall_dim` | wall fills the frame (⅓ light) | 0/7 | 0/7 |
| `stuck_corner` | funnel fills the frame | 0/9 | **1**/9 |
| `stuck_wall_edge` | half wall, half open room | 0/8 | 0/8 |
| `stuck_gap` | blocked either side, bright slot between | 0/9 | 0/9 |
| `stuck_kerb` | clear floor — the kerb is below the frame | 0/8 | 0/8 |

The single success is one frame out of `stuck_corner` under telemetry.

### The two prompts fail in different, equally useless directions

From the image alone the model **never once reports a failure** — not on any
frame, of any scenario, in the entire dataset:

```
failure_visual, field `failed`
  gt False  ->  False x36        (36/36 right, trivially)
  gt True   ->  False x67        (0/67 — a constant predictor)
```

Add telemetry — commanded velocity, distance travelled, elapsed time — and it
starts saying `True`, but not on the right frames, and it stops being able to
say `none` at all:

```
failure_grounded, field `failure_mode`
  gt none         ->  oscillation x22, stuck x13, none x1   (1/36)
  gt oscillation  ->  oscillation x10, stuck x2             (10/12)
  gt stuck        ->  oscillation x54, stuck x1             (1/55)
```

Telemetry moved the operating point without adding discrimination. It collapsed
`none` from 22 correct to 1, and turned 54 of 55 stalls into `oscillation`.

### Most answers contradict themselves

The two fields are produced in one JSON object and are supposed to agree.
Counting answers that set `failed: false` while naming a non-`none` failure
mode:

| prompt | self-contradictory answers |
|---|---|
| `failure_visual` | 29 / 103 (28%) |
| `failure_grounded` | **73 / 103 (71%)** |

A typical grounded answer about a robot pinned motionless against a wall:

```json
{"failed": false, "failure_mode": "oscillation",
 "reason": "The robot is moving but not getting anywhere"}
```

Fluent, well-formed, internally inconsistent, and wrong three ways at once —
the robot is not moving, it is not oscillating, and it has failed. This matters
more than any accuracy number: anything downstream reading either field would be
confidently misled rather than merely uninformed, and a consumer reading *both*
gets an answer that cannot be true.

## The scenario that separates vision from telemetry

`stuck_kerb` is a 0.06 m kerb — tall enough that the waffle's 0.033 m wheels
cannot climb it and its chassis fouls it squarely, low enough to sit entirely
beneath the camera's frame at contact. The robot is immobilised and the image
shows an unobstructed room.

- `failure_visual` answered `none` 8/8: *"The path ahead is clear and the robot
  can keep driving."* Which is **the correct reading of the image.** There is no
  honest way to answer `stuck` from that picture.
- `failure_grounded` answered `oscillation` 8/8: *"The robot is moving but not
  getting anywhere"* — handed telemetry saying the robot travelled zero distance.

This is the cleanest cell in the experiment. The vision-only failure there is
not a model defect; the grounded failure is, because the telemetry it was given
contained the answer.

## Why this is the expected shape, not a broken setup

It lines up with **MV-RoboBench (2026)**: strong scores on general single-view
spatial benchmarks do not predict success on robotic spatial tasks. The
contribution here is not the negative result itself but the harness that makes
it *measurable* — ground truth that cannot be gamed, a baseline printed next to
every number, duplicate frames counted honestly, and a resumable evaluation that
runs overnight on a CPU laptop.

## What building the dataset taught us

Two findings came out of the data rather than the model, and both changed the
experiment:

**A contact frame carries almost no information, and different scenes converge
to the same one.** At the stall the camera is ~5 cm from the obstacle, where its
60° FOV spans about 6 cm — so anything bigger fills the frame completely.
`stuck_wall`, `stuck_wall_oblique`, `stuck_wall_dim` and `stuck_corner` were
built to differ in approach angle, obstacle shape and lighting, and produced
**30 byte-identical frames**. Lighting does change the approach frames (mean
intensity 125 → 110) and converges to the identical image at contact, because a
surface that close is lit by ambient alone. Approach does not vary a stall
frame; only the obstacle's height and width relative to the FOV does. That is
why `stuck_gap`, `stuck_wall_edge` and `stuck_kerb` exist, and why every frame
now carries a `pixel_sha1`.

**The oscillation labeller was wrong, and only a short rollout exposed it.** It
tested whether the robot stayed inside a 0.40 m patch while covering ≥0.35 m —
which a *straight* 0.43 m run also satisfies, because a straight line's extent
is about half its path. Every earlier scenario travelled more than 0.8 m and hid
it. `stuck_gap` travelled 0.43 m and got three frames labelled `oscillation`
for driving in a straight line. The fix is a shape test — path length over
spatial extent — and the two populations are nowhere near each other (genuine
oscillation 6.6–12.0, every straight run 1.2–2.3). Re-labelling the three
pre-existing runs under the new rule reproduces their old labels byte-for-byte,
so the earlier published numbers were unaffected.

## Honest limitations

These matter more than the headline, and the numbers should not be quoted
without them:

- **103 frames are 51 distinct images**, and the `stuck` class is 55 frames of
  only **15**. Four of the seven stuck rollouts collapse to a single shared
  image. The distinct-image table above is the one to quote.
- **One model, one prompt pair, one seed, temperature 0.** No prompt
  sensitivity study. "llava-phi3 cannot do this" is still not established —
  only "these two prompts cannot". Determinism was confirmed as a side effect:
  identical images under an identical prompt returned identical answers every
  time.
- **The class balance changed** when six stuck rollouts were added, so `stuck`
  is now the majority class and the baseline flipped from `none` to `stuck`.
  Lift figures are not comparable to the earlier 32-frame version of this table.
- **Simulated, untextured scenes.** Flat-shaded Gazebo geometry is far from the
  distribution these models were trained on.
- **No comparison model.** `moondream` and `bakllava` are installed and the
  harness takes `MODELS="..."`; running them is the obvious next step.
- **Nav2 is never exercised.** Every scenario is open-loop `script` mode, so
  `goal_unreachable` and `localisation_loss` have no data at all.

## Reproducing

```bash
source /opt/ros/jazzy/setup.bash

# Phase A — record (sim only, no VLM). One scenario per stuck geometry.
for s in nominal_drive oscillate_spin stuck_wall stuck_wall_oblique \
         stuck_wall_dim stuck_corner stuck_wall_edge stuck_gap stuck_kerb; do
    python ../harness/recorder/record.py scenarios/$s.json --out-dir data
    python ../harness/recorder/extract_keyframes.py data/$s/bag \
        --out data/$s --run-id $s
done

# Phase A'' — ground-truth labels from the event timeline
python label_run.py data/*/

# Phase B — the VLM (background this; never while the sim runs)
./run_eval.sh

# Phase C — score. Pass ONE concatenated manifest: duplicate frames are shared
# between runs, so per-run scoring cannot see them.
cat data/*/manifest.jsonl > /tmp/all.jsonl
PYTHONPATH=../harness python -m scoring.score \
    --manifest /tmp/all.jsonl --results results/*.jsonl --out results/report
```

Runtime: ~72 s per rollout, and 5.5 h for all 206 VLM items (85 s/item mean on
`failure_visual`, 109 s on `failure_grounded`, p90 192 s). `run_eval.sh` is
resumable — killing it costs at most the item in flight.

See `TAXONOMY.md` for the label definitions, how ground truth is derived, and
what actually distinguishes the seven `stuck` scenarios; and
`../harness/README.md` for the machine-level findings behind the setup.
