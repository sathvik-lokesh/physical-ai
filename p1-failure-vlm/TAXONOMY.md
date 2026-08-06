# Failure taxonomy — p1

Five labels. Every rollout is ground-truthed to exactly one of them, derived
from odometry and Nav2's own result — never from what the scenario *intended*,
because scoring a VLM against our intent instead of what happened would measure
the wrong thing.

| label | definition (observable, not intended) | how ground truth is derived |
|---|---|---|
| `none` | the robot made progress and ended where it was asked | goal succeeded, or the script completed with path length ≈ commanded |
| `stuck` | motion was commanded but the robot did not move | `commanding == true` and <2 cm of odometry motion over a 6 s window |
| `oscillation` | the robot kept moving but stopped making net progress | path length grows while net displacement over the window stays near zero |
| `goal_unreachable` | the planner could not produce or complete a path | Nav2 returns `FAILED`, or no plan within the goal timeout |
| `localisation_loss` | the pose estimate diverged from ground truth | AMCL pose vs Gazebo ground-truth pose exceeds a threshold |

`stuck` and `oscillation` are deliberately separated. Both look like "not
arriving," and both are trivially detectable from telemetry — but they look
*different* in the camera image (a wall filling the frame vs. a scene sweeping
back and forth). Whether a VLM can tell them apart from vision alone is the
question p1 exists to answer.

## The ablation

The same keyframes are evaluated under two prompts:

- **`failure_visual`** — the image alone. "Is this robot in trouble?"
- **`failure_grounded`** — the image plus the telemetry that a real monitor
  would already have (commanded velocity, distance travelled, elapsed time).

The gap between them is the result worth reporting. If `failure_grounded` wins
by a wide margin, the honest conclusion is that the VLM is reading the numbers,
not the scene — which is exactly the kind of thing MV-RoboBench (2026) found:
strong single-view spatial scores do not predict success on robotic spatial
tasks. If `failure_visual` holds up, the VLM is genuinely contributing
perception that telemetry does not already provide.

Either outcome is a publishable-shaped finding. Only reporting the better of
the two would not be.

## Scenarios

| scenario | mode | expects | notes |
|---|---|---|---|
| `nominal_drive` | script | `none` | drive along open floor, no obstacle |
| `oscillate_spin` | script | `oscillation` | alternate left/right, net displacement ≈ 0 |
| `stuck_wall` | script | `stuck` | drive straight into a wall and stall |
| `stuck_wall_oblique` | script | `stuck` | the same wall met at 26° instead of head-on |
| `stuck_wall_dim` | script | `stuck` | the same wall at a third of the light |
| `stuck_corner` | script | `stuck` | wedge into a V-shaped funnel |
| `stuck_wall_edge` | script | `stuck` | clip a wall's edge — half the view blocked |
| `stuck_gap` | script | `stuck` | jam in a slot narrower than the robot |
| `stuck_kerb` | script | `stuck` | pinned by a kerb too low to appear in frame |
| `nav_goal_ok` | goal | `none` | Nav2 to a reachable pose (not yet recorded) |
| `nav_goal_blocked` | goal | `goal_unreachable` | Nav2 into an obstacle (not yet recorded) |

Open-loop `script` mode is used wherever the failure must be *caused*
reliably — fighting the planner into a specific failure is far less repeatable
than driving into a wall on purpose. Every `stuck` scenario runs the identical
drive script (settle 3 s, then 0.15 m/s for 45 s) so that the scene is the only
thing that varies between them.

### Why seven `stuck` scenarios, and what actually distinguishes them

The first version of this experiment had one, which meant its seven `stuck`
frames were one robot in one pose imaged seven times. Adding six more was meant
to fix that. Measuring the result showed it only partly does, and the reason is
worth stating because it constrains any experiment of this shape:

**At contact the camera is ~5 cm from the obstacle, where a 60° horizontal FOV
spans about 6 cm of it.** Anything taller and wider than that fills the frame
completely, so approach angle, obstacle shape and lighting are all invisible in
the frame that matters. Measured: `stuck_wall`, `stuck_wall_oblique`,
`stuck_wall_dim` and `stuck_corner` produce **byte-identical** stall frames — 30
frames, one image. Lighting does change the *approach* frames (mean intensity
125 → 110) and converges to the identical image at contact, because a surface
that close is lit by ambient alone.

What varies a stall frame is therefore not how the robot got there but **how
much of the obstacle is in shot**, which is set by its height and width:

| scenario | at contact the camera sees | distinct stall images |
|---|---|---|
| `stuck_wall`, `_oblique`, `_dim`, `stuck_corner` | nothing but the obstacle | 1, shared by all four |
| `stuck_wall_edge` | half obstacle, half open room | several |
| `stuck_gap` | obstacle either side, a bright slot between | several |
| `stuck_kerb` | clear floor and open room — the kerb is below the frame | 1 |

`stuck_kerb` is the scenario worth arguing about. The obstacle is 0.06 m tall,
under the bottom of the frame at contact, so the robot is immobilised and the
image is of an unobstructed room. There is no honest way to answer `stuck` from
that image alone. It is included deliberately: it is the case where telemetry
must win, and a model that scores well on it from vision alone is guessing.
