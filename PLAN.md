# physical-ai — VLM / Physical AI portfolio program

Four portfolio projects on physical AI, object detection, VLMs and SLAM, sharing one
harness. Planned 2026-07-28/30. **Phase 0 built and Phase 1 complete as of 2026-08-05 —
see the RESUME POINT below, `harness/README.md` and `p1-failure-vlm/README.md`.**

---

## RESUME POINT (start here)

**Status (2026-08-06):** Phase 0 harness built and verified. **Phase 1 is complete and
its `stuck` claim has now been hardened** — 9 rollouts, 103 labelled keyframes, 206
evaluations, 0 errors, written up in `p1-failure-vlm/README.md`. Run
`python3 p1-failure-vlm/verify_readme_numbers.py` (29 checks) after touching the data;
it re-derives every number the README quotes.

**Phase 1 headline:** `llava-phi3` scores **below the always-guess-the-commonest-label
baseline in all four configurations** (lift −30.0, −26.2, −33.0, −41.7 points). It named
`stuck` **once in 110 attempts** — 0/55 from the image, 1/55 with telemetry — across
7 rollouts, 4 obstacle geometries, 2 lighting levels and 2 approach angles. Parse
coverage was 100%, so this is discrimination, not formatting. Worse: **71% of grounded
answers contradict themselves**, setting `failed: false` while naming a failure mode,
typically `{"failed": false, "failure_mode": "oscillation", "reason": "The robot is
moving but not getting anywhere"}` about a robot pinned motionless against a wall.

This matches MV-RoboBench (2026). The contribution is the harness that makes it
measurable: ground truth that cannot be gamed, a trivial baseline printed beside every
number, duplicate frames counted honestly, and resumable overnight CPU evaluation.

**Two findings from hardening the claim, both in `p1-failure-vlm/README.md`:**
- **A contact frame carries almost no information, and different scenes converge to the
  same one.** At the stall the camera is ~5 cm out and its 60° FOV spans ~6 cm, so any
  larger obstacle fills it. Four rollouts built to differ in approach angle, obstacle
  shape and lighting produced **30 byte-identical frames**. Approach does not vary a
  stall frame; only the obstacle's height and width do. Hence `stuck_gap` (slot),
  `stuck_wall_edge` (half-blocked) and `stuck_kerb` (0.06 m kerb below the frame — robot
  pinned, image shows a clear room). Every frame now carries a `pixel_sha1` and
  `scoring/score.py` prints a duplicate-weighted accuracy: 103 frames are 51 images, and
  the 55 `stuck` frames are only **15**.
- **The oscillation labeller was wrong** and only a short rollout exposed it: a straight
  0.43 m run satisfied the "stayed in a 0.40 m patch while covering 0.35 m" test, because
  a straight line's extent is half its path. Fixed with a path/extent shape test
  (oscillation 6.6–12.0, straight lines 1.2–2.3). The three pre-existing runs re-label
  byte-for-byte identically, so the earlier published numbers were unaffected.

**Cross-model and prompt-sensitivity passes are now DONE (2026-08-07, commit `e2270bc`).**
4 prompts x 3 models. All three models collapse to a constant answer and none beats the
baseline per distinct image. Two near-miss headlines were caught by the duplicate
weighting and are recorded in `p1-failure-vlm/README.md`. `bakllava` does not fit this
machine (5.5 GB resident on 8.41 GB, killed at 84/103) and should not be retried here.

**Next actions, in order:**
1. **Push to GitHub.** The repo is local-only with no remote. This is the single thing
   standing between p1 and being usable in an application, and it is ~10 minutes.
2. LinkedIn writeup. The honest hook is the two traps the harness caught, not the
   negative result itself.
3. Add the two Nav2 scenarios (`nav_goal_ok`, `nav_goal_blocked`); everything so far is
   open-loop `script` mode, so Nav2 is installed but never exercised, and
   `goal_unreachable`/`localisation_loss` have no data at all.
4. Then Phase 2 (`p2-semantic-slam`), which reuses the recorder unchanged.

**Findings that changed the plan (see `harness/README.md` for detail):**
- **Headless camera rendering works with no GPU.** `gz sim -s --headless-rendering`
  renders through Mesa llvmpipe with no `/dev/dri`. This was the make-or-break
  unknown for Phases 1–2. First frame ~10 s after start.
- **The stock TurtleBot3 has no RGB camera.** `nav2_minimal_tb3_sim`'s waffle is
  depth-only and its bridge forwards **no image topic at all** — nothing RGB reaches
  ROS out of the box. The harness vendors a patched SDF and adds the missing bridge.
  Unanticipated by this plan, and it would have blocked Phases 1 and 2.
- **VLM latency is worse than the ~80 s/image assumption under contention** —
  `llava-phi3` measured at 430 s/item while a sim shared the CPU, `moondream` at
  106–157 s. This is why the sim and the VLM must never run together.
- **A resident model holds 3.9 GB.** Always `ollama ps` and unload before a sim run.
- **`/odom` cannot be ground truth.** The DiffDrive plugin integrates wheel rotation,
  so a robot pressed against a wall reports travel it never made — measured at
  **6.75 m of odometry against 1.00 m of true motion (5.75 m of pure slip)** on the
  final `stuck_wall` run, while the clean runs agree to within 3 cm. Labels now come from a
  `PosePublisher` ground-truth pose; odometry is kept as the *deceptive* signal the
  VLM is shown, which is the heart of the p1 experiment.
- **`PosePublisher` static mode segfaults gz-sim 8** on this world. It is therefore
  off, meaning a motionless robot publishes nothing — so `drive.py` samples the last
  known pose on a 5 Hz timer. Absence of an update is itself the evidence of no motion.
- **The sim runs at a load-dependent RTF (~0.3–0.6)**, so wall-clock-timed drive steps
  under-travel and vary run to run — the same script produced 7.8 m once and 0.4 m the
  next time. `drive.py` now times every step on `/clock`, which made runs reproducible
  to the second.
- **The RGB camera was mounted outside the robot's collision hull** (0.133 m forward vs
  ~0.072 m), so when pressed against an obstacle it sat *inside* that obstacle and
  rendered the scene beyond it — every `stuck` frame looked like open floor. Caught only
  by eyeballing the frames; it would have silently poisoned the class under test.

**Open questions still unresolved:**
- Alpamayo open dataset: size + licence? Needed before Phase 3.
- Does a LeRobot/SmolVLA sim rollout survive a 5 GB memory cap? Needed before Phase 4.
- `edge-vlm-bench` looks complete (results, charts, findings all present) though memory
  recorded it as "in progress". Phase 3 builds on it — confirm before starting.

---

## Hardware reality

| | |
|---|---|
| CPU | Intel i7-10510U — 4C/8T, 15 W mobile, 2019 |
| Host RAM | **8.41 GB physical** |
| GPU | **None.** No CUDA, and no `/dev/dri` either → no iGPU/OpenVINO-GPU path |
| Disk | 861 GB free on `/`, 114 GB free on `C:` |
| Already working | ROS 2 Jazzy + Nav2 + Gazebo (from `../ros2-navigation-planning`) |
| Ollama models | `moondream` 1.7 GB, `llava-phi3` 2.9 GB, `bakllava` 4.7 GB, + text models |
| systemd | running, cgroup v2 with `cpu` + `memory` controllers → real per-job caps |

Ruled out by hardware: any training/fine-tuning, Isaac Sim/Lab, real-time video
inference, 7B+ VLMs at usable speed, 3D Gaussian Splatting.

Measured baseline from `../nvidia-target/edge-vlm-bench`: `llava-phi3` = 83.3 % accuracy
at 83 s/image; `bakllava` = 77.8 % at 193 s/image. Cold start after model load can exceed
5 minutes. Assume ~80 s/image for planning purposes.

---

## `.wslconfig` is over-committed

**NOTE (2026-08-05): the `memory=5GB` cap proposed below was considered and DECLINED —
do not re-propose it.** `autoMemoryReclaim=gradual` was added instead, and the working
practice is `wsl --shutdown` before a heavy session plus the per-job `sysguard/capped`
ceilings, which were verified to work. The whole Phase 0 + Phase 1 run above was done
this way without a hang. The rest of this section is kept for background.

Current `/mnt/c/Users/sathv/.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=16GB
```

12 GB allocated on an 8.41 GB machine. WSL balloons until Windows is starved and
thrashes its pagefile. `free -h` already shows WSL claiming 7.6 GiB of the 8.41 GB box.

**Replace with:**

```ini
[wsl2]
memory=5GB        # leaves Windows ~3GB → laptop stays usable
swap=16GB
swappiness=10     # prefer dropping cache over swapping the model out
```

Then from PowerShell: `wsl --shutdown`

**Tradeoff:** at 5 GB, `moondream` (1.7 GB) and `llava-phi3` (2.9 GB) fit comfortably;
`bakllava` (4.7 GB) does not. Acceptable — `edge-vlm-bench` already showed llava-phi3
beats bakllava on *both* accuracy and speed.

Keep a second profile at `memory=7GB` for unattended overnight batch runs.

---

## Resource discipline (so it never hangs again)

Per-job hard caps, available because cgroup v2 + systemd are present:

```bash
systemd-run --user --scope -p CPUQuota=500% -p MemoryMax=3G \
  nice -n 19 ./run_eval.sh      # 5 of 8 threads, hard 3 GB ceiling
```

Ollama environment — never hold two models at once:

```bash
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
```

---

## THE ARCHITECTURAL RULE

**Never run the simulator and the VLM at the same time.** Record first, infer later:

```
Phase A (sim):    Gazebo HEADLESS (gz sim -s, no GUI) → rosbag2
Phase B (infer):  rosbag2 → keyframes → VLM → JSONL
```

- The Gazebo **GUI** is what actually kills this machine — software-rendered OpenGL on a
  15 W chip with no `/dev/dri`. Server-only (`gz sim -s`) drops it to a fraction.
- Phase B is pure batch with no GUI → this is the part you background overnight while
  working on something else.
- **Every runner appends JSONL incrementally and resumes from the last completed item.**
  Non-negotiable at ~80 s/image: you must be able to kill any job instantly and reclaim
  the laptop without losing work.

---

## Repo layout

```
~/Hanuman/physical-ai/
├── PLAN.md           # this file
├── harness/          # SHARED — built once in Phase 0
│   ├── recorder/     # headless sim → rosbag2 → keyframe extraction
│   ├── vlmrunner/    # resumable batch VLM eval, JSONL out
│   └── scoring/      # metrics + charts
├── p1-failure-vlm/
├── p2-semantic-slam/
├── p3-alpamayo-eval/
└── p4-smolvla-cpu/
```

---

## Phases

| Phase | Project | Builds | Reuses | Hours |
|---|---|---|---|---|
| **0** | Harness + system tuning | recorder, vlmrunner, scoring | — | ~10 |
| **1** | Robot failure detection | failure scenarios, taxonomy | all | 20–30 |
| **2** | Semantic SLAM | pose-anchored labels, NL query | all + P1 recorder | 20–30 |
| **3** | Alpamayo reasoning eval | reasoning-trace scoring | vlmrunner only | 20–25 |
| **4** | SmolVLA on CPU | LeRobot sim rollouts | scoring | 30–40 |

**Total ≈ 100–135 h** — roughly 3–4 months at 8–10 h/week. The shared harness is why
it isn't ~160 h.

**Sequencing rationale:**
- Phase 1 builds the rollout recorder Phase 2 needs → 1 before 2.
- **Phase 3 needs no simulator at all** → this is the one that genuinely runs in the
  background while you work on Phase 1 or 2. That's the parallelism, not four repos at once.
- Phase 4 is last because it's the only one that might not fit in RAM, and by then it's
  droppable without hurting the portfolio story.

---

### Phase 1 — Robot failure detection & reasoning in ROS 2 ⭐ top pick

Drive TurtleBot3 in headless Gazebo into deliberate failure modes (stuck on obstacle,
localisation loss, goal unreachable, oscillation). Record camera + `/cmd_vel` + costmap
to rosbag2. A small VLM then watches the rollout **offline** and answers: *did this fail,
when, and why?* Score against ground truth you control; produce a failure taxonomy.

- **Why hot:** the AHA / FailSafe-VLM line of 2026 work — VLMs that detect and reason
  about robot failures in natural language and help VLAs recover.
- **Why it fits:** reuses the working ROS 2 stack; 100 % offline inference.
- **Why it's *yours*:** this is the reliability thesis already running through
  `gap-aware-rag` (abstention), `nemoclaw-agent` (agent failure flight recorder) and
  `sentinelops` (self-healing), applied to physical AI. Makes the portfolio read as one
  coherent argument instead of six unrelated repos.
- **Risk:** low.

### Phase 2 — Language-queryable semantic SLAM

`slam_toolbox` builds the metric map. On keyframes, a VLM captions/labels what the camera
sees; each label is anchored to the map pose. Result: a semantic layer you can query in
natural language — *"where did you last see a chair?"* → pose → Nav2 drives there.

- **Why hot:** open-vocabulary semantic SLAM is a live 2026 front (OpenVox, LEGO-SLAM,
  FM-Fusion, OpenMonoGS-SLAM).
- **Why it fits:** VLM annotates *asynchronously on keyframes*, not per frame — which is
  what real systems do, not a hack. 80 s/image is fine when you annotate every ~2 m.
- **Best demo value of anything on the list** — produces a GIF.
- **Risk:** low.

### Phase 3 — Alpamayo reasoning distillation eval (the NVIDIA-facing one)

NVIDIA open-sourced **Alpamayo-1** on Hugging Face: a 10 B chain-of-thought reasoning VLA
for AV that emits trajectories *plus reasoning traces*, with open Physical AI datasets.
10 B won't run here — but *"can a 3B VLM reproduce Alpamayo-grade driving reasoning, and
where does it break?"* is pure offline evaluation.

- Builds directly on `edge-vlm-bench` and `av-scenario-forge`.
- No simulator → the background-friendly phase.
- **Risk:** medium — dataset size and licence unverified.

### Phase 4 — SmolVLA on CPU (stretch)

Run LeRobot's **SmolVLA** (450 M) on a lightweight sim task (PushT / sim-ALOHA) and
characterise where it succeeds and fails.

- **Why hot:** VLA is the single hottest term in the sector; having actually run one is a
  real differentiator.
- **Why plausible:** SmolVLA is explicitly designed to be "deployed on consumer-grade
  GPUs or even CPUs" — layer-skipping in the VLM backbone + asynchronous inference — and
  matches VLAs 10× its size.
- **Risk:** medium-high. RAM is the constraint; LeRobot sim deps are heavy. Droppable.

---

## Why these four — industry evidence (July 2026)

NVIDIA's GTC 2026 position has three legs:
1. **Cosmos 3** — world foundation model unifying synthetic world generation, vision
   reasoning and action simulation.
2. **GR00T N1.7 / N2** — N2 previewed as succeeding at new tasks in new environments
   >2× as often as leading VLA models.
3. **Physical AI Data Factory Blueprint** — open reference architecture for gathering,
   shaping and **assessing** real + simulated data.

Leg 3 is the one nobody builds in a portfolio, and the only one reachable without a GPU.

Market: robotics VC hit **$55.8 B by mid-2026**, ~2× all of 2025; ~$6 B into world-model
companies in Q1 alone. Physical Intelligence raising at $11 B+; Skild AI ~$1.4 B at $14 B+.

From the user's own AI digest (June 2026): Prometheus $12 B at $41 B ("artificial general
engineer for the physical world"); **Generalist AI $400 M at $2 B, NVIDIA-backed**;
**Mecka AI $60 M for robot *training data*** (the bottleneck is data, not models);
Seedcamp €279 M with physical AI a named focus; NVIDIA Jetson T4000 (4× efficiency),
Alpamayo, Siemens Industrial AI OS.

The two findings that define the open gap:
- **MV-RoboBench (2026):** strong scores on general single-view spatial benchmarks do
  **not** predict success on robotic spatial tasks; SOTA remains far below human.
  *Evaluation is broken.*
- **AHA / FailSafe-VLM (2026):** emerging line on VLMs that detect and reason about robot
  failures in natural language.

**Strategy: do not compete on training models. Compete on the evaluation and reliability
infrastructure that GPU-rich teams still need someone to build.** That is the shape of
`edge-vlm-bench` and `gap-aware-rag` already.

---

## Sources

- [NVIDIA & global robotics leaders](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world)
- [GTC 2026 robotics highlights](https://theaiinsider.tech/2026/03/21/10-robotics-highlights-from-nvidia-gtc-2026/)
- [Isaac GR00T](https://developer.nvidia.com/isaac/gr00t)
- [Alpamayo announcement](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development)
- [Robotics VC records mid-2026](https://www.techtimes.com/articles/319037/20260625/robotics-vc-breaks-annual-records-midyear-why-physical-ai-commands-software-multiples.htm)
- [Physical Intelligence raise](https://www.pymnts.com/artificial-intelligence-2/2026/physical-intelligence-seeks-1-billion-as-robotics-interest-grows/)
- [MV-RoboBench](https://arxiv.org/html/2510.19400v2)
- [AHA-VLM](https://aha-vlm.github.io/)
- [FailSafe](https://arxiv.org/pdf/2510.01642)
- [SmolVLA paper](https://huggingface.co/papers/2506.01844) · [model](https://huggingface.co/lerobot/smolvla_base)
- [Open-vocabulary semantic SLAM overview](https://www.emergentmind.com/topics/open-vocabulary-semantic-slam)

---

## Rejected alternatives (don't re-litigate)

- **Open-vocab detection benchmark** (YOLO-World / OWLv2 int8 ONNX vs closed-set YOLO) —
  solid and low-risk, but overlaps `edge-vlm-bench` in shape. Keep as a fallback if
  Phase 3 or 4 dies.
- **MASt3R/DUSt3R vs classical ORB matching** — genuinely interesting, but peak RAM at
  512 px is a real problem under a 5 GB cap.
- **Monocular depth → ROS 2 costmap** (Depth-Anything-V2-Small) — very achievable but
  lower novelty. Good "quick win" if a fifth project is ever wanted.
- Anything requiring training, Isaac Sim/Lab, or real-time inference — hardware-blocked.

---

## Side finding (unrelated to this program)

**The AI daily digest has stopped.** Newest file in the Drive folder is
`AI_Digest_2026-06-22_Evening.txt` — nothing for 5 weeks — and `AI_Digest_Latest.txt` is
stale at June 14. Either both routines are dead or the Google Drive connector dropped.
See `~/.claude/.../memory/project-ai-digest.md` for the routine IDs and troubleshooting steps.
