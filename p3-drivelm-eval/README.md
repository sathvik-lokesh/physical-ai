# p3 — Can a 3B VLM read a car's own motion off the road ahead?

**Status: designed and built, not yet run.** The code is here and validated
end to end; the data it runs on needs one manual step (below). No result in
this file is measured yet, and the two distribution numbers quoted in the
design section are provisional — see *Data integrity* for why.

p1 asked whether a small VLM can tell that a robot is stuck. p2 asked whether it
can say *where* an object is. Both answered the same way: these models recognise
things well and judge spatial and physical state badly. p3 asks the same
question of real driving data, where the ground truth is not simulated and the
scenes are not ours.

## The task

DriveLM-nuScenes labels every keyframe with the ego vehicle's own behaviour,
written as one sentence pair:

> The ego vehicle is going straight. The ego vehicle is driving fast.

That factorises into two labels of very different difficulty, and scoring them
separately is the point of the experiment:

| axis | classes | corpus majority |
|---|---|---|
| `steer` | going straight, slightly steering L/R, steering L/R | **78.9%** (going straight) |
| `speed` | not moving, slowly, normal, fast, very fast | **27.8%** (driving fast) |

**A model that answers "going straight" to everything scores 78.9% on steering.**
That is the trap this project's harness exists to catch — the same shape as p1's
`mode_only` result, which beat the per-frame baseline and lost to it badly once
duplicate images were counted once. `scoring/score.py` prints the majority
baseline next to every accuracy, so the number cannot be quoted without its
baseline standing beside it.

Speed is where discrimination has to show up, because the classes are nearly
uniform. It is also the one that cannot be answered from a single photograph
even in principle: a still frame does not contain a speed.

## The two arms

- **single** — one CAM_FRONT frame. Speed is not recoverable. This arm measures
  what the model does when the evidence is *absent*: fall back on the prior, or
  admit it.
- **pair** — the same frame plus the previous keyframe, with the true gap stated
  in the prompt ("the left image was taken 2.4 seconds before the right one").
  Now speed *is* recoverable, from how far the scene moved. Any gain over the
  single arm is the model actually using temporal evidence.

Keyframe spacing in DriveLM ranges from 0.4 s to 13.25 s (median 2.55 s). A gap
that varies 30-fold across items is a confound, not a variable, so the pair arm
is restricted to gaps of 1.5–3.0 s *and* told the exact figure.

The pair arm runs in two formats — one side-by-side composite image, or two
images in one request — because p1's largest single effect was not the model or
the prompt but the output format: dropping one field from the response schema
moved `stuck` recall from 0/55 to 55/55 on identical images. Assuming format
neutrality here would repeat a mistake already made once in this repo.

## Sampling

`build_manifest.py` takes **one frame per scene** and round-robins across
(speed × straight-or-turning) strata.

One per scene because keyframes within a nuScenes scene are a few seconds apart
on the same stretch of road. They are not byte-identical, so p1's `pixel_sha1`
de-duplication will not flag them, but they are nowhere near independent trials.
p1's lesson was that correlated frames inflate a score silently; here the fix is
to spend the sample budget on distinct scenes instead.

Stratifying moves the class balance away from the corpus prior on purpose — a
200-item random draw would contain about 10 left turns. That is only safe
because `score.py` computes its majority baseline from the manifest that was
actually run, so the comparison stays like-for-like. On a 200-item draw the
steering baseline falls from 78.9% to about 56%, and speed sits near 23%.

## Data integrity — read this before trusting any number here

DriveLM is gated on Hugging Face with auto-approval. An **ungated third-party
mirror** of it also exists, and its copy of `v1_1_train_nus.json` has *exactly*
the official byte length — 192,961,829 bytes — and **a different sha256**. Same
size, different content. Every size check in the world passes it.

The design numbers above (78.9% / 27.8%, and the keyframe-gap distribution) were
computed from that mirror before the discrepancy was found, so they are marked
provisional throughout. `fetch_drivelm.sh` verifies every download against the
sha256 the Hugging Face API reports for the official repo, refuses to install a
file that does not match, and records the API's hash beside the file as
`v1_1_train_nus.json.sha256`. `build_manifest.py` checks the file against that
recorded hash a second time, because a manifest built from the wrong bytes would
look entirely normal.

**The second gate refuses when it has nothing to compare against.** If no hash
has been recorded and none is pinned in the source, the builder exits rather
than building — a gate that passes silently when it cannot check is not a gate.
`--record-hash` asks the HF API directly and writes the sidecar, for a QA file
that arrived by some route other than `fetch_drivelm.sh`. `--allow-unverified`
exists only for the synthetic fixture in `selftest.py` and prints a warning;
nothing built under it may be published.

**Nothing gets published from this project until the numbers are re-derived from
the hash-verified official file.**

## Why not Alpamayo

The plan originally specified NVIDIA's Alpamayo / `PhysicalAI-Autonomous-Vehicles`
data, on the theory that its Chain-of-Causation reasoning traces were the ideal
reference. They would have been — 1,450 annotated training clips in a 150 KB
parquet file, with camera chunks at ~2 GB per 100 clips, all of it affordable
here despite the 133 TB total.

The licence makes it unusable for this project, for three independent reasons:

- **§1** grants use *"solely for your internal development of autonomous vehicles
  and automated driving assisted systems using NVIDIA technology"*, and **§4.3**
  forbids any other purpose. A public portfolio evaluation is not that.
- **§4.6** forbids derivative works, distribution, embedding or hosting of any
  part of the dataset.
- **§3** defines NVIDIA Confidential Information to include *"output, and any
  results of benchmarking or other competitive analysis or regression or
  performance data relating to the Dataset."* Publishing benchmark numbers
  measured on this data is precisely what the licence prohibits — and a
  published writeup is the entire deliverable.

DriveLM's language data is CC BY-NC-SA 4.0, which permits exactly what this
project needs. The nuScenes images it points at carry nuScenes' own
non-commercial terms and are **not** redistributable, so this repo publishes
code, manifests and results — never frames.

## Running it

```bash
# once: accept the licence at https://huggingface.co/datasets/OpenDriveLab/DriveLM
export HF_TOKEN=hf_...                  # read token
./fetch_drivelm.sh --images             # 193 MB QA + 3.5 GB images, hash-verified

python3 selftest.py                     # 36 checks, no network, no VLM
python3 build_manifest.py --arm single --n 200
python3 build_manifest.py --arm pair   --n 200 --pair-mode composite

cd ../harness
python3 -m vlmrunner.run \
  --manifest ../p3-drivelm-eval/data/single/manifest.jsonl \
  --task     ../p3-drivelm-eval/tasks/behavior_single.json \
  --model    llava-phi3 \
  --out      ../p3-drivelm-eval/results/single.llava-phi3.jsonl

python3 -m scoring.score \
  --manifest ../p3-drivelm-eval/data/single/manifest.jsonl \
  --results  ../p3-drivelm-eval/results/*.jsonl \
  --out      ../p3-drivelm-eval/results/report
```

## Self-test

`selftest.py` runs the whole chain — QA file to manifest to the runner's
image/prompt resolution to `scoring/score.py` — on a synthetic DriveLM tree it
builds in a temp dir. No network, no VLM, a couple of seconds:

```bash
python3 selftest.py      # 36 checks
```

It exists because a run costs 8–10 hours and the data needs a manual licence
acceptance before a single byte arrives, so "the pieces do not fit together" is
the one failure that must not be discovered afterwards. It asserts the joins,
not the model: that `{dt_s}` reaches the rendered prompt, that the runner can
resolve every path in a manifest built by the builder, that pair gaps stay in
the 1.5–3.0 s band, that an unparseable answer scores as wrong instead of
raising, and — the point of the project — that a predictor answering the
commonest label every time ties the majority baseline *exactly* while the
corpus-prior answer never beats it.

Writing it found three defects in code that had already been "dry-run end to
end": the second hash gate was dead (`OFFICIAL_QA_SHA256` was `None` and the
`--record-hash` flag that was supposed to fill it did not exist), `--qa`
repointed the labels but not the images, and the builder could emit truncated
item ids that collide, which `vlmrunner` rejects outright.

### The two-images arm works, and the two pair formats are not the same experiment

The design above flagged one unknown it could not settle on paper: whether
`llava-phi3`'s projector accepts two images in a single request at all. A
four-item smoke run against `llava-phi3` on synthetic road frames settles it —
it does, and returns well-formed JSON. The token accounting is what makes that
trustworthy rather than merely plausible:

| arm | images | prompt tokens | wall (warm) |
|---|---|---|---|
| pair, composite | 1 | 827 | 92, 122 s |
| pair, two-images | 2 | 1408 | 174 s |

**Budget: run with `--keep-alive` set, never `0`.** The one two-images item that
was run with `--keep-alive 0` took **420 s** against 174 s warm, because the flag
unloads the 3.9 GB model after every request and the next item pays a cold load.
Over 200 items that is the difference between a 10-hour run and a 23-hour one.
The 174 s warm figure is itself above this README's 150–180 s estimate for the
cheaper arm, so budget **~10 h for two-images** and ~6 h for composite, and do
not schedule both in one night.

The 581-token difference is one more full patch grid, so the second image really
is embedded rather than accepted and dropped — which is the failure mode that
would otherwise have produced a clean-looking null result. Both fit inside
`llava-phi3`'s 4096-token context, so neither arm is silently truncated.

It also shows the two pair formats differ in more than layout. A composite is
**one** image to the encoder: both frames share a single patch grid, so each
frame gets half the visual resolution. The two-images arm spends a full grid per
frame. So the comparison is not purely "how the frames are arranged" — it is
also "how many pixels each frame is worth", and a difference between the arms
cannot be attributed to layout alone.

The composite arm carries a second, sharper cost on real data. nuScenes
`CAM_FRONT` frames are 1600x900, so a side-by-side composite is roughly 3.6:1,
and a LLaVA-style encoder takes a square input. Each frame therefore reaches the
model squashed to about half its width. Fine detail — lane markings, the swing
of a kerb — is exactly what the steering label depends on, and it is the first
thing that squash destroys. This is a reason to run both formats, not a reason
to drop either: if the composite arm underperforms, this is the first
explanation to rule out before concluding anything about temporal reasoning. Given p1's finding that format moved
results more than the model did, that distinction belongs in the writeup rather
than in a footnote.

## Cost

p1 measured 85 s per item on simple simulated scenes; p2 measured 175 s on
cluttered ones. Real street scenes are at least as busy as p2's, so budget
~150–180 s per item: **200 items is 8–10 hours per arm**, two to three overnight
runs for the full design. `vlmrunner` appends and resumes after every item, so a
run can be killed at any moment to get the laptop back.

Never run this while anything else heavy is running, and `ollama ps` before
starting — a resident model holds 3.9 GB of the 8.41 GB on this machine.
