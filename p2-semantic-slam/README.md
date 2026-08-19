# p2 — Language-queryable semantic SLAM

`slam_toolbox` builds the metric map. A small VLM labels what the camera sees on
keyframes, each label is anchored to the pose it was seen from, and the result is
queryable: *"where did you last see the chair?"* returns a coordinate.

**It half works, and the half that fails is the interesting half.** Two of four
objects are placed to within 22 cm using nothing but a 3B VLM and a laser map.
The other two are placed badly or not at all, and the reason is measurable: a
label that names a whole image tells you what is there but not *where*, and
every attempt to extract a direction from it failed.

## Results

`llava-phi3` (3.8B), one 12.27 m mapping circuit, 61 keyframes, 0 errors.
Object positions are authored constants in the world file and the objects are
`static`, so ground truth is exact. **All numbers below use slam_toolbox's
estimated poses, not the simulator's.**

| object | localisation error | from |
|---|---|---|
| chair | **0.08 m** | 5 labels |
| bookshelf | **0.22 m** | 3 labels |
| sofa | 1.46 m | 2 labels |
| table | not located | 0 labels |

Guessing the middle of the room for everything gives 2.50 m, so chair and
bookshelf are a real result and the sofa is barely better than nothing. Under
the first prompt the table *was* located, at 2.02 m — worse than the trivial
guess, and the confusion matrix says why: five of its seven labels are actually
frames of the bookshelf, misread. **Label precision converts directly into
metres of localisation error.**

With n=4 objects, do not read a median. The spread is the result.

### Closing the loop: question in, motion out

`results/demo.gif` is the robot's own camera while it drives to an object it was
asked about in English. Nothing in the path is hardcoded to the chair; change the
question and the goal moves.

```
$ python make_goto_scenario.py --query "where did you last see the chair"
target  : chair believed at world (+2.54, +0.04), 0.05 m from truth
goal    : world (+1.54, +0.02) facing +1 deg
          map   (-0.04, +1.48) facing -179 deg
```

| | |
|---|---|
| drove | 1.58 m in 31.8 s |
| stopped | 0.96 m from the real chair (1.00 m standoff requested) |
| final heading | -5.4 deg, with the chair at -10 deg |

The pose it navigates on is slam_toolbox's, not the simulator's.

**It does not use Nav2, and that is a hardware result.** The full Nav2 stack is
fifteen non-composed processes; brought up alongside a software-rendered Gazebo
on four cores it drove the load average to **21**, starved the driver to 2.8% of
a CPU, and never moved the robot. Composition, which would have made it one
process, could not load its own nodes on this machine either. What runs instead
is slam_toolbox alone plus a turn-then-go controller reading `map -> base_footprint`
from TF, which holds the load near 2.

That is honest but it is a real limitation: a proportional controller works here
only because the floor between the patrol ring and the objects is empty. It
would not survive a cluttered room, and planning is exactly what Nav2 would have
provided.

### The map is not the problem

| | |
|---|---|
| mapped room | **6.90 x 6.85 m** (true 7.0 x 7.0) |
| slam_toolbox pose error | **median 8 mm, max 20 mm** over 12.27 m |
| open-loop drift, for contrast | 0.21 m |

Swapping the simulator's true pose for slam_toolbox's estimate changes object
localisation by **3 cm**. Localisation error is two orders of magnitude below
the bearing error that actually limits this system, so the SLAM half is solved
and the perception half is not.

### Labelling

61 frames, 0 errors, median **175 s/frame** (p1 measured 85 s on the same model
and image size; a room full of textured furniture costs roughly double a flat
wall).

| frames | correct |
|---|---|
| object centred | 7/9 |
| object at frame edge | 7/26 |
| **empty** | **26/26** |

It never once invented an object in an empty room. What it does instead is
misname objects that are half out of frame: bookshelf -> table x5, chair -> sofa
x3, with 9 more edge objects missed entirely.

## Why triangulation failed

The plan was that a label plus a pose gives a bearing, so several bearings
should intersect at the object, and disagreement should filter bad labels. That
is not what happened.

| estimator | located | error |
|---|---|---|
| trivial (guess room centre) | 4/4 | 2.50 m |
| naive (average, fixed range) | 3/4 | 0.08 / 0.22 / 1.46 m |
| corroborated (>=3 agreeing ray pairs) | 1/4 | 0.05 m |

Read that last row carefully. Corroboration is not more accurate: the only object
it places is the chair, and naive places the same chair at 0.08 m. It is more
*selective*, and it pays for that with three quarters of the map. Three
measurements explain why it cannot do better:

- **A label is not a bearing.** It says the object is somewhere in a 60 deg cone.
  Taking the camera axis as the direction is wrong by a median of 23 deg and up
  to 75 deg. Two rays each 23 deg off do not meet near the object.
- **It is starved of parallax.** The route spins 360 deg at each of four corners,
  so most same-class observation pairs share a position and carry no
  triangulation information at all. Four corners is not four baselines. Given
  only the frames that are both centred and correctly labelled, corroboration
  locates nothing whatsoever.
- **Averaging tolerates what triangulation cannot.** The naive estimator survives
  symmetric bearing noise because errors cancel. Ray intersection turns the same
  noise into a wrong crossing point.

### Asking the model where the object is made it worse

The obvious fix is to ask which third of the frame the object occupies, turning
a +-30 deg cone into +-10 deg. It backfired:

| config | labels | \|median\| bearing err | **signed bias** | best error |
|---|---|---|---|---|
| original prompt | 23 | 23.0 deg | **-13.8 deg** | 0.22 / 0.55 / 0.86 / 2.02 m |
| + side, applied | 10 | 8.6 deg | **-11.5 deg** | 0.38 m, 1 object |
| + side, ignored | 10 | 15.0 deg | **+4.5 deg** | 0.08 / 0.22 / 1.46 m |

**It answered `left` on 9 of 10 frames and `centre` on none of them**, including
when the object was genuinely 3, 7 and 14 degrees to the *right*. Applying that
as a correction injects a systematic rotation, so the signed bias doubled. Using
the same run and simply throwing its position answers away is the best
configuration in the whole project.

Adding the extra field also degraded identification, from 7/9 to 5/9 on centred
frames and 7/26 to 5/26 at the edges, and made the model far more conservative:
23 labelled frames became 10. That conservatism is why its surviving labels are
cleaner, which is a real effect and not the one that was being tested.

### The metric that predicts localisation error is the signed bias

This is the part worth keeping. Across the three configurations, signed bearing
bias tracks localisation error almost exactly (-13.8 deg -> 0.70 m, -11.5 deg ->
0.58 m, +4.5 deg -> 0.22 m), while absolute median bearing error points the
wrong way: its best value, 8.6 deg, produces worse positions than its worst.

Averaging and triangulation are both destroyed by systematic offset and
comfortable with symmetric noise. I projected this experiment's benefit using
absolute median error, and that projection was wrong for exactly this reason.
`semantic_map.py` now reports the signed mean, and says which one matters.

## Two claims of mine the data overturned

Recorded because they were both load-bearing at the time.

**"It hallucinates a bookshelf onto a bare wall."** That came from a single frame
in the smoke test, and I used it to motivate the entire corroboration design.
Given 26 genuinely empty frames it answered `nothing` to all 26. One failure out
of one is not a rate.

**The naive baseline was rigged.** It dropped each label at a fixed 2.0 m, and
the true median object range on this route is 1.96 m, so it had been handed the
answer's scale. It is now swept and reported at its best value, which made my own
hypothesis look worse, not better. Losing to a baseline tuned in its own favour
is the honest bar.

## What this says about the approach

A 3B VLM naming whole frames is a good *identifier* and a poor *localiser*. It
is right about what is in the room, right about when the room is empty, and it
cannot tell you where anything is to better than a 60 degree cone. Semantic
mapping that needs metric positions wants a detector that emits a bounding box,
with the VLM used for open-vocabulary naming of the regions the detector finds.

That is a conclusion about task decomposition, and it continues the thesis from
p1: these models fail at spatial judgement while succeeding at recognition, so
build systems that ask them only the second kind of question.

## Reproducing

```bash
source /opt/ros/jazzy/setup.bash
../harness/recorder/models/fetch_furniture.sh      # Fuel models + texture patch

python3 ../harness/recorder/record.py scenarios/map_room.json --out-dir data
python3 ../harness/recorder/extract_keyframes.py data/map_room/bag \
    --out data/map_room --run-id map_room --stall-s 3
python3 add_slam_poses.py                           # map->odom o odom->base_footprint

PYTHONPATH=../harness ../harness/sysguard/capped python3 -m vlmrunner.run \
    --manifest data/map_room/manifest.jsonl --task tasks/name_object.json \
    --model llava-phi3 --timeout 600 \
    --out results/map_room.name_object.llava-phi3.jsonl

python3 semantic_map.py --poses slam                # the numbers above
python3 semantic_map.py --poses slam --query "where did you last see the chair"
python3 render_map.py --estimates results/estimates.json --out results/map.png

# question -> goal -> drive -> gif
python3 save_map.py
python3 make_goto_scenario.py --query "where did you last see the chair"
python3 ../harness/recorder/record.py scenarios/goto_query.json --out-dir data
python3 make_demo_gif.py
python3 verify_readme_numbers.py                    # re-derive every number here
```

`render_map.py --check` verifies the map geometry and the world->map transform
without drawing anything, because a wrong transform puts every marker somewhere
plausible and false.

## Honest limitations

- **Four objects, one room, one route, one model, one seed.** Every number here
  rests on 61 frames. The per-object errors are individual measurements, not a
  distribution.
- **`table` is never localised well** under either prompt, and the camera sits
  15.7 cm off the floor so a table is seen from underneath. Whether that is a
  viewpoint problem or a label-precision problem is not separated here.
- **Simulated, untextured room** apart from the four Fuel objects. Flat walls and
  a flat floor are not what these models were trained on.
- **The demo drives in a straight line, it does not plan.** Nav2 could not be run
  on this hardware (see above), so there is no global planner, no costmap and no
  obstacle avoidance. The controller works because the route is open floor.
- **Corroboration is under-tested, not disproven.** A route designed for parallax
  (many distinct viewpoints, no in-place spins) might change the answer. This one
  was designed to see each object four times, which is not the same thing.
