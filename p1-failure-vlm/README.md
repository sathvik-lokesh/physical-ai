# p1 — Can a small VLM tell that a robot has failed?

A TurtleBot3 is driven into deliberate failure modes in headless Gazebo. The
rollout is recorded to rosbag2, keyframes are extracted, and a small VLM is asked
**offline**: did this fail, and how? Answers are scored against ground truth
derived from the simulator's true pose.

The question isn't "can a VLM caption a robot photo". It's whether a 3B-class
model running on a laptop CPU tells you anything a telemetry threshold doesn't
already.

**On this dataset it doesn't.** Nine rollouts, 103 labelled keyframes, 206
evaluations of `llava-phi3` at 100% parse coverage, and `stuck` correctly named
**once in 110 attempts**.

I then ran two more models and two more prompts over the same frames.
`moondream` and `bakllava` fail differently and no better. Rewording the question
moves the numbers a lot without fixing anything. Here's the claim that survived
all of it:

> Across **4 prompts and 3 models**, no configuration beats the
> always-guess-the-commonest-label baseline once duplicate images are counted
> once. Every configuration collapses toward a constant answer. The prompt
> decides *which* constant, not whether the model discriminates.

The largest single effect in the whole study turned out to be a formatting one.
Removing the second output field takes `stuck` recall from 0/55 to 55/55, while
making the model answer `stuck` to 86 of 103 frames.

## Results

`llava-phi3` (3.8B), 9 rollouts, 103 labelled keyframes, 206 evaluations,
0 errors. 5.5 hours of CPU.

| prompt | field | accuracy | always-guess baseline | lift |
|---|---|---|---|---|
| `failure_visual` | `failed` | 35.0% (36/103) | 65.0% (always `True`) | **−30.0** |
| `failure_visual` | `failure_mode` | 27.2% (28/103) | 53.4% (always `stuck`) | **−26.2** |
| `failure_grounded` | `failed` | 32.0% (33/103) | 65.0% (always `True`) | **−33.0** |
| `failure_grounded` | `failure_mode` | 11.7% (12/103) | 53.4% (always `stuck`) | **−41.7** |

Parse coverage was 100%. Every answer was well-formed JSON, so the model isn't
failing to *communicate*. It's failing to *discriminate*.

### The same numbers, with duplicate frames collapsed

A stalled robot in a static scene renders the same picture over and over, so the
frame count overstates the sample. Weighting every pixel-identical image to 1 in
total gives 51 distinct images, with a baseline weighted the same way:

| prompt | field | distinct-image accuracy | baseline | lift |
|---|---|---|---|---|
| `failure_visual` | `failed` | 48.8% | 51.2% | −2.4 |
| `failure_visual` | `failure_mode` | 43.7% | 48.8% | −5.1 |
| `failure_grounded` | `failed` | **54.6%** | 51.2% | **+3.4** |
| `failure_grounded` | `failure_mode` | 20.6% | 48.8% | −28.2 |

One cell of twelve finally clears the baseline, by 3.4 points on 51 images.
That's the size of the noise, not a capability. It's reported anyway. The point
of printing a baseline beside every number is that it also catches you rounding
your own result up.

### `stuck` is essentially never identified, under these two prompts

> **Read "Prompt sensitivity" below before quoting this.** Removing the second
> output field takes `stuck` recall from 0/55 to **55/55** on the very same
> images. What follows is reproducible, but it's a fact about `failure_visual`
> and `failure_grounded`, not about what the model can see.

**0 out of 55 from the image alone. 1 out of 55 with telemetry.** That's across
seven rollouts, four obstacle geometries, two lighting levels, two approach
angles and 15 distinct stall images:

| scenario | what the camera sees at the stall | visual | grounded |
|---|---|---|---|
| `stuck_wall` | wall fills the frame | 0/7 | 0/7 |
| `stuck_wall_oblique` | wall fills the frame (26° approach) | 0/7 | 0/7 |
| `stuck_wall_dim` | wall fills the frame (⅓ light) | 0/7 | 0/7 |
| `stuck_corner` | funnel fills the frame | 0/9 | **1**/9 |
| `stuck_wall_edge` | half wall, half open room | 0/8 | 0/8 |
| `stuck_gap` | blocked either side, bright slot between | 0/9 | 0/9 |
| `stuck_kerb` | clear floor, kerb below the frame | 0/8 | 0/8 |

The single success is one frame of `stuck_corner` under telemetry.

### The two prompts fail in opposite, equally useless directions

From the image alone the model **never once reports a failure**, on any frame of
any scenario in the entire dataset:

```
failure_visual, field `failed`
  gt False  ->  False x36        (36/36 right, trivially)
  gt True   ->  False x67        (0/67, a constant predictor)
```

Add telemetry (commanded velocity, distance travelled, elapsed time) and it
starts saying `True`, just not on the right frames. It also stops being able to
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

Both fields come out of one JSON object and are supposed to agree. Counting
answers that set `failed: false` while naming a non-`none` failure mode:

| prompt | self-contradictory answers |
|---|---|
| `failure_visual` | 29 / 103 (28%) |
| `failure_grounded` | **73 / 103 (71%)** |

Here's a typical grounded answer about a robot pinned motionless against a wall:

```json
{"failed": false, "failure_mode": "oscillation",
 "reason": "The robot is moving but not getting anywhere"}
```

Fluent, well-formed, internally inconsistent, and wrong three ways at once. The
robot isn't moving, it isn't oscillating, and it has failed. This matters more
than any accuracy number. Anything downstream reading either field gets
confidently misled rather than merely uninformed, and anything reading *both*
gets an answer that can't be true.

## Prompt sensitivity, and the number that nearly fooled me

That `0/55` is largely an artefact of the output format. I ran two more
image-only prompts over the same 103 frames, each asking for `failure_mode`
**alone**. No `failed` field, no `reason`, so the self-contradiction becomes
impossible by construction:

- **`mode_only`** takes the same four categories and the same wording, with one
  key removed.
- **`mode_hinted`** additionally describes each category by *how it looks from
  this camera* ("if a surface is so close that you cannot tell what it is, that
  is stuck"). This is deliberately the most favourable prompt I could write.

| prompt (llava-phi3, image only) | per-frame | baseline | per **distinct image** | baseline |
|---|---|---|---|---|
| `failure_visual` (2 fields) | 27.2% | 53.4% (−26.2) | 43.7% | 48.8% (−5.1) |
| `mode_only` (1 field) | **59.2%** | 53.4% (**+5.8**) | 39.4% | 48.8% (−9.4) |
| `mode_hinted` (1 field + visual hints) | 53.4% | 53.4% (±0.0) | 29.2% | 48.8% (−19.6) |

**Dropping one field took `stuck` recall from 0/55 to 55/55.** That's an enormous
prompt effect, and it forces the headline above to be stated as a property of
those two prompts, not of the model's perception.

**`mode_only` didn't learn to see, though.** It reached 55/55 by answering
`stuck` on **86 of 103 frames**. Recall on everything else collapsed: `none`
1/36, `oscillation` 5/12. It beats the per-frame baseline by 5.8 points and sits
9.4 points **below** it once duplicate images are collapsed, because the frames
it gets right are the 30 copies of one picture.

That gap is the entire argument for duplicate-weighted scoring. Written up as
"removing a field lifts llava-phi3 above the baseline for the first time, +5.8",
it would have been the headline of this project. It's an artefact of counting one
image thirty times.

**The maximally helpful prompt did worst.** `mode_hinted` answered `stuck` on
**103 of 103** frames, a pure constant, scoring exactly the baseline per-frame
and −19.6 against it per image. Describing what `stuck` looks like didn't teach
the model to recognise it. It just told the model which constant to pick.

So the honest claim is narrower than "the model can't see", and stronger than a
single prompt pair could support:

> Across four prompts and three models, **no configuration beats the
> always-guess-the-commonest-label baseline once duplicate images are counted
> once.** The prompt changes which constant the model gravitates to.

## Three models, three different constants

`moondream` (1.7B) and `bakllava` (7B) ran on the same frames. None of the three
discriminates. Each collapses to a constant and states it fluently:

| model | what it always answers | `failed` accuracy | vs baseline |
|---|---|---|---|
| `llava-phi3`, image only | `failed: false`, never reports a failure on any frame | 35.0% | −30.0 |
| `moondream`, both prompts | `failed: true`, on every frame | 65.0% | **±0.0** |
| `bakllava`, image only | `failed: true`, and `stuck` whenever it answers a mode | 63.1% | **±0.0** |

The two models that "tie" the baseline tie it because **they are the baseline**,
a constant `true` predictor on a dataset that's 65% failures. That's the entire
reason a baseline gets printed beside every number in this harness.

Three details sharpen the point:

- **`moondream` scored identically under both prompts.** 65.0% / 0.0%, the same
  confusion matrix frame for frame. Adding telemetry to the prompt changed
  *nothing at all*, which is hard to explain if the input is being read.
- **`moondream` never produced a `failure_mode` field** in **309** attempts
  across three prompts, including `mode_only`, which asks for that one key and
  nothing else. 0% coverage throughout. It can't be scored on the mode task at
  all, and the prompt pass shows that's intrinsic. Asking differently doesn't fix
  it. That failure is unintelligibility, where `llava-phi3`'s is
  wrongness, and calling it "0% accuracy" would be the wrong claim.
- **`bakllava` looks like the one success and is the worst trap here.** Its
  `stuck` recall is 32/41, best of any model. But every one of the 72 mode
  answers it produced was `stuck`. Precision is 32/72. Weighted by distinct
  image it scores **7.1% against a 50.9% baseline**, the lowest number in the
  project. Quoting its recall alone would have been the most misleading sentence
  in this writeup.

### `bakllava` does not fit this machine

It's reported on 84 of 103 frames, one prompt only, because the run was killed.
At 4.7 GB it reached **5.5 GB resident on an 8.41 GB box**, left 144 MB free, and
pinned four cores for hours (17.5 h of CPU across 4.4 h elapsed) without
finishing an item. Where it did finish, latency was 185 s mean and 356 s p90,
already 2.5x `llava-phi3`.

Treat that as a deployment result, not an inconvenience. On 8 GB of RAM
with no GPU, a 7B VLM doesn't just run slowly, it stops making progress.
`run_eval.sh` now takes `TIMEOUT` (default 600 s) so a wedged item gets abandoned
instead of consuming a night.

## The scenario that separates vision from telemetry

`stuck_kerb` is a 0.06 m kerb. Tall enough that the waffle's 0.033 m wheels can't
climb it and its chassis fouls it squarely, low enough to sit entirely beneath
the camera's frame at contact. The robot is immobilised and the image shows an
unobstructed room.

- `failure_visual` answered `none` 8/8: *"The path ahead is clear and the robot
  can keep driving."* Which is **the correct reading of that image.** There's no
  honest way to answer `stuck` from it.
- `failure_grounded` answered `oscillation` 8/8: *"The robot is moving but not
  getting anywhere"*, having been handed telemetry saying the robot travelled
  zero distance.

This is the cleanest cell in the experiment. The vision-only failure isn't a
model defect. The grounded one is, because the telemetry it was given contained
the answer.

## Why this is the expected shape, not a broken setup

It lines up with **MV-RoboBench (2026)**: strong scores on general single-view
spatial benchmarks don't predict success on robotic spatial tasks. What this repo
adds is the harness that makes the failure *measurable*. Ground truth that can't
be gamed, a baseline printed next to every number, duplicate frames counted
honestly, and a resumable evaluation that runs overnight on a CPU laptop.

## What building the dataset taught me

Two findings came out of the data itself, and both changed the experiment.

**A contact frame carries almost no information, and different scenes converge to
the same one.** At the stall the camera sits ~5 cm from the obstacle, where its
60° FOV spans about 6 cm, so anything bigger fills the frame completely.
`stuck_wall`, `stuck_wall_oblique`, `stuck_wall_dim` and `stuck_corner` were
built to differ in approach angle, obstacle shape and lighting. They produced
**30 byte-identical frames**. Lighting does change the approach frames (mean
intensity 125 → 110) and then converges to the identical image at contact,
because a surface that close is lit by ambient alone. Approach doesn't vary a
stall frame. Only the obstacle's height and width relative to the FOV do. That's
why `stuck_gap`, `stuck_wall_edge` and `stuck_kerb` exist, and why every frame
now carries a `pixel_sha1`.

**The oscillation labeller was wrong, and only a short rollout exposed it.** It
tested whether the robot stayed inside a 0.40 m patch while covering ≥0.35 m. A
*straight* 0.43 m run satisfies that too, because a straight line's extent is
about half its path length. Every earlier scenario travelled more than 0.8 m and
hid the bug. Then `stuck_gap` travelled 0.43 m and got three frames labelled
`oscillation` for driving in a straight line. The fix is a shape test, path
length over spatial extent, and the two populations sit nowhere near each other
(genuine oscillation 6.6–12.0, every straight run 1.2–2.3). Re-labelling the
three pre-existing runs under the new rule reproduces their old labels
byte-for-byte, so the earlier published numbers were unaffected.

## Honest limitations

These matter more than the headline, and the numbers shouldn't be quoted without
them:

- **103 frames are 51 distinct images**, and the `stuck` class is 55 frames of
  only **15**. Four of the seven stuck rollouts collapse to one shared image.
  The distinct-image table is the one to quote.
- **Four prompts, one seed, temperature 0.** Four is enough to show the results
  are prompt-dominated. It isn't enough to have searched the space. A prompt that
  induces genuine discrimination hasn't been ruled out, only shown not to be any
  of these, including the one written specifically to help. Determinism was
  confirmed as a side effect: identical images under an identical prompt returned
  identical answers every time.
- **The hinted prompt is unfair to one scenario.** `mode_hinted` tells the model
  that a surface too close to identify means `stuck`. True for the wall-type
  scenes, actively wrong for `stuck_kerb`, where the robot is pinned and the view
  is clear floor. It answered `stuck` everywhere regardless, so nothing hinged on
  it, but a better prompt study wouldn't carry the flaw.
- **`bakllava` is reported on 84 of 103 frames, one prompt only.** Its run was
  killed after it wedged the machine. Those numbers are directional, and the
  distinct-image figures in particular rest on 41 images.
- **Three models isn't "small VLMs in general".** All three are Llava-family or
  Llava-adjacent and share training lineage, so their agreement is weaker
  evidence than three independent architectures would be.
- **The class balance changed** when six stuck rollouts were added. `stuck` is now
  the majority class and the baseline flipped from `none` to `stuck`, so lift
  figures aren't comparable to the earlier 32-frame version of this table.
- **Simulated, untextured scenes.** Flat-shaded Gazebo geometry is a long way from
  the distribution these models were trained on.
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
python3 label_run.py data/*/

# Phase B — the VLM (background this; never while the sim runs)
./run_eval.sh
MODELS="moondream bakllava" ./run_eval.sh
MODELS="llava-phi3" TASKS="mode_only mode_hinted" ./run_eval.sh

# Phase C — score. Pass ONE concatenated manifest: duplicate frames are shared
# between runs, so per-run scoring cannot see them. --task-dir keeps "the task
# never asked for this field" distinct from "the model could not produce it".
cat data/*/manifest.jsonl > /tmp/all.jsonl
PYTHONPATH=../harness python -m scoring.score \
    --manifest /tmp/all.jsonl --results results/*.jsonl \
    --task-dir tasks --out results/report

# Check every number in this README still matches the data
python3 verify_readme_numbers.py
```

Runtime is about 72 s per rollout, and 5.5 h for the 206 `llava-phi3` items
(85 s/item mean on `failure_visual`, 109 s on `failure_grounded`, p90 192 s).
`run_eval.sh` is resumable, so killing it costs at most the item in flight.

See `TAXONOMY.md` for the label definitions, how ground truth is derived, and
what actually distinguishes the seven `stuck` scenarios. See
`../harness/README.md` for the machine-level findings behind the setup.
