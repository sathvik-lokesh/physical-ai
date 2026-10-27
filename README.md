# physical-ai

Small vision-language models (3B-class, run on a laptop CPU through Ollama) tested on robot and driving questions,
with ground truth from a simulator or a labelled dataset and a trivial baseline printed next to every number.
Each project has its own README with the full results and a script that re-derives every number it quotes.

| project | question | status |
|---|---|---|
| [`p1-failure-vlm`](p1-failure-vlm/) | Can a small VLM tell that a robot has failed? A TurtleBot3 is driven into deliberate failures in headless Gazebo, recorded to rosbag2 and scored against the simulator's true pose. | done: across 4 prompts and 3 models, no configuration beats the always-guess-the-commonest-label baseline once duplicate images are counted once |
| [`p2-semantic-slam`](p2-semantic-slam/) | Language-queryable semantic SLAM: slam_toolbox builds the map, a VLM labels keyframes, and "where did you last see the chair?" returns a coordinate. | done: chair 0.08 m and bookshelf 0.22 m, sofa 1.46 m, table not located |
| [`p3-drivelm-eval`](p3-drivelm-eval/) | Can a 3B VLM read a car's own motion off the road ahead (DriveLM-nuScenes)? | designed and built, not yet run |
| [`harness`](harness/) | Shared recorder, resumable VLM runner, scoring and a cgroup guard so no job can wedge the machine. | used by p1 to p3 |

The short version so far: these models recognise objects well and judge spatial and physical state badly.

`PLAN.md` is the working notebook kept during the project. Where it disagrees with a project README, the README is
the current result.

Licence: MIT.
