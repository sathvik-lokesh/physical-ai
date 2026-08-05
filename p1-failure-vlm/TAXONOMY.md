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
| `stuck_wall` | script | `stuck` | drive straight into a wall and stall |
| `nominal_drive` | script | `none` | drive along open floor, no obstacle |
| `oscillate_spin` | script | `oscillation` | alternate left/right, net displacement ≈ 0 |
| `nav_goal_ok` | goal | `none` | Nav2 to a reachable pose |
| `nav_goal_blocked` | goal | `goal_unreachable` | Nav2 to a pose inside an obstacle |

Open-loop `script` mode is used wherever the failure must be *caused*
reliably — fighting the planner into a specific failure is far less repeatable
than driving into a wall on purpose.
