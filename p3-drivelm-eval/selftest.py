"""
Run the whole p3 chain on a synthetic DriveLM tree, with no network and no VLM.

p3 costs 8-10 hours per arm on this machine, and the real data needs a manual
licence acceptance before a single byte can be fetched. Discovering that the
manifest, the prompt template or the scorer do not fit together *after* that
wait is the expensive failure. This builds a fake DriveLM whose schema matches
the real one, pushes it through build_manifest -> vlmrunner's manifest/prompt
path -> scoring/score.py, and asserts the joins.

It does not check the model's answers. It checks that an item built here is an
item the runner can resolve, that {dt_s} reaches the prompt, and that a constant
predictor scores exactly the majority baseline -- which is the trap this whole
project exists to catch, so the scorer had better catch it on data where the
answer is known in advance.

    python3 selftest.py          # exits non-zero on the first failure
"""

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "harness"
sys.path.insert(0, str(HARNESS))

from vlmrunner.run import read_manifest, resolve_images  # noqa: E402
from vlmrunner.tasks import Task  # noqa: E402
from scoring.score import score  # noqa: E402

STEER = ["going straight", "slightly steering to the left",
         "slightly steering to the right", "steering to the left",
         "steering to the right"]
SPEED = ["not moving", "driving slowly", "driving with normal speed",
         "driving fast", "driving very fast"]

checks = 0
failures = []


def check(label, got, want):
    global checks
    checks += 1
    if got == want:
        print(f"  ok   {label}: {got}")
    else:
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
        failures.append(label)


def check_that(label, cond, detail=""):
    global checks
    checks += 1
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        failures.append(label)


def build_fake_drivelm(root: Path, n_scenes: int = 40) -> Path:
    """
    A QA file shaped exactly like DriveLM-nuScenes, plus the images it points at.

    Deliberately includes the two things the real file contains that break naive
    parsers: keyframes whose behavior answer does not match the sentence-pair
    template, and keyframe gaps far outside the pair arm's 1.5-3.0 s band.
    """
    from PIL import Image

    rng = random.Random(7)
    # Real nuScenes scene/sample tokens are 32 random hex chars. Generating them
    # that way matters: the builder truncates them for readable item ids, and a
    # fixture with structured tokens would test truncation on the easy case.
    token = lambda: "".join(rng.choice("0123456789abcdef") for _ in range(32))  # noqa: E731
    samples = root / "nuscenes" / "samples" / "CAM_FRONT"
    samples.mkdir(parents=True)

    data = {}
    n_unparseable = 0
    for s in range(n_scenes):
        scene_id = token()
        frames = {}
        t = 1533151600.0 + s * 1000
        for k in range(4):
            # Gaps alternate in and out of the pair arm's band on purpose.
            t += [2.0, 9.5, 2.5, 0.6][k]
            stamp = int(t * 1e6)
            fname = f"n008-2018-08-01__CAM_FRONT__{stamp}.jpg"
            # Distinct pixels per frame: p3 expects no duplicates, and a fixture
            # of identical images would hide it if the scorer stopped noticing.
            img = Image.new("RGB", (64, 36), (s * 5 % 256, k * 60 % 256, 90))
            img.save(samples / fname, quality=92)

            steer = STEER[(s + k) % len(STEER)]
            speed = SPEED[(s * 3 + k) % len(SPEED)]
            answer = f"The ego vehicle is {steer}. The ego vehicle is {speed}."
            if s == 0 and k == 0:  # the malformed one
                answer = "The ego vehicle is reversing carefully."
                n_unparseable += 1
            frames[token()] = {
                "image_paths": {"CAM_FRONT": f"../nuscenes/samples/CAM_FRONT/{fname}"},
                "QA": {"behavior": [{"Q": "What is the ego vehicle doing?",
                                     "A": answer}]},
            }
        data[scene_id] = {"key_frames": frames}

    qa = root / "v1_1_train_nus.json"
    qa.write_text(json.dumps(data))
    return qa


def run_builder(qa: Path, out: Path, extra: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HERE / "build_manifest.py"), "--qa", str(qa),
         "--out", str(out)] + extra,
        capture_output=True, text=True,
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="p3-selftest-"))
    try:
        drivelm = tmp / "drivelm"
        drivelm.mkdir()
        qa = build_fake_drivelm(drivelm)

        print("\n[1] the hash gate")
        # No sidecar and no pinned constant: the gate has nothing to compare
        # against, and must refuse rather than wave the file through.
        r = run_builder(qa, tmp / "nope", ["--arm", "single", "--n", "5"])
        check("refuses an unverified QA file", r.returncode, 3)
        check_that("says why", "no official sha256" in r.stderr, r.stderr[:120])

        # A sidecar with the wrong hash must also be refused.
        Path(str(qa) + ".sha256").write_text("0" * 64 + f"  {qa.name}\n")
        r = run_builder(qa, tmp / "nope", ["--arm", "single", "--n", "5"])
        check("refuses a QA file whose hash does not match", r.returncode, 3)

        # Correct sidecar: this is what fetch_drivelm.sh writes.
        import hashlib
        real = hashlib.sha256(qa.read_bytes()).hexdigest()
        Path(str(qa) + ".sha256").write_text(f"{real}  {qa.name}\n")

        print("\n[2] single arm")
        single = tmp / "single"
        r = run_builder(qa, single, ["--arm", "single", "--n", "20"])
        check("builder exit code", r.returncode, 0)
        if r.returncode:
            print(r.stderr)
            return 1
        items = read_manifest(single / "manifest.jsonl")
        check("items built", len(items), 20)
        check("one frame per scene", len({i["meta"]["scene"] for i in items}), 20)
        check("every item has both gt axes",
              all(set(i["gt"]) == {"steer", "speed"} for i in items), True)
        check("unparseable behavior dropped",
              "1 behavior answers did not parse" in r.stdout, True)
        check_that("pixel_sha1 present on every item",
                   all("pixel_sha1" in i["meta"] for i in items))
        check("all frames distinct",
              len({i["meta"]["pixel_sha1"] for i in items}), len(items))

        # The runner has to be able to find the images from the manifest alone.
        paths = [p for i in items for p in resolve_images(i, single)]
        check_that("runner resolves every image path",
                   all(p.exists() for p in paths),
                   next((str(p) for p in paths if not p.exists()), ""))

        print("\n[3] pair arm, composite")
        pair = tmp / "pair"
        r = run_builder(qa, pair, ["--arm", "pair", "--n", "20",
                                   "--pair-mode", "composite"])
        check("builder exit code", r.returncode, 0)
        pitems = read_manifest(pair / "manifest.jsonl")
        check_that("every pair item carries dt_s",
                   all("dt_s" in i["meta"] for i in pitems))
        check_that("every gap inside the 1.5-3.0 s band",
                   all(1.5 <= i["meta"]["dt_s"] <= 3.0 for i in pitems),
                   sorted({i["meta"]["dt_s"] for i in pitems})[:5])
        from PIL import Image
        composites = [resolve_images(i, pair)[0] for i in pitems]
        check_that("composites written", all(p.exists() for p in composites))
        w, h = Image.open(composites[0]).size
        check_that("composite is two frames wide", w > 2 * h, f"{w}x{h}")

        print("\n[4] the prompt actually gets the gap")
        task = Task.load(HERE / "tasks" / "behavior_pair.json")
        prompt = task.prompt_for(pitems[0])
        check_that("{dt_s} substituted", "{dt_s}" not in prompt)
        check_that("the real gap appears in the prompt",
                   f"{pitems[0]['meta']['dt_s']} seconds" in prompt,
                   prompt[:100])
        single_task = Task.load(HERE / "tasks" / "behavior_single.json")
        check_that("single prompt leaves no placeholder",
                   "{" not in single_task.prompt_for(items[0]).split("Respond as JSON")[0])

        print("\n[5] pair arm, two-images")
        pair2 = tmp / "pair2"
        r = run_builder(qa, pair2, ["--arm", "pair", "--n", "20",
                                    "--pair-mode", "two-images"])
        check("builder exit code", r.returncode, 0)
        p2items = read_manifest(pair2 / "manifest.jsonl")
        check_that("two images per item",
                   all(len(resolve_images(i, pair2)) == 2 for i in p2items))
        check_that("both images exist",
                   all(p.exists() for i in p2items for p in resolve_images(i, pair2)))
        check_that("earlier frame comes first",
                   all(resolve_images(i, pair2)[0] != resolve_images(i, pair2)[1]
                       for i in p2items))

        print("\n[6] the scorer catches a constant predictor")
        # The whole point of p3. Two properties, and they are not the same one:
        #   - always answering the manifest's commonest label must tie the
        #     baseline *exactly*, never beat it;
        #   - always answering "going straight" -- the corpus prior, and what a
        #     model that has learned the prior will do -- must never beat the
        #     baseline either, even though stratified sampling means it is no
        #     longer the commonest label in the manifest that was run.
        from collections import Counter
        top = {axis: Counter(i["gt"][axis] for i in items).most_common(1)[0][0]
               for axis in ("steer", "speed")}

        def constant_results(path, name, answer):
            with open(path, "w") as f:
                for i in items:
                    f.write(json.dumps({
                        "id": i["id"], "task": "behavior_single", "model": name,
                        "meta": i["meta"], "wall_ms": 1000, "parsed": dict(answer),
                    }) + "\n")
            return path

        results = constant_results(tmp / "constant.jsonl", "constant", top)
        entry = score(single / "manifest.jsonl", [results],
                      HERE / "tasks")["constant / behavior_single"]
        check("scored every item", entry["n_scored"], 20)
        for axis in ("steer", "speed"):
            fs = entry["fields"][axis]
            check_that(f"{axis}: majority-label predictor ties the baseline exactly",
                       abs(fs["lift_over_majority"]) < 1e-9,
                       f"lift {fs['lift_over_majority']}")
            check("   baseline label is the manifest's own, not the corpus prior",
                  fs["majority_label"], top[axis])
        check("distinct weighting is a no-op on distinct frames",
              entry["fields"]["steer"]["n_distinct_images"], 20)

        prior = constant_results(tmp / "prior.jsonl", "prior",
                                 {"steer": "going straight", "speed": "driving fast"})
        pentry = score(single / "manifest.jsonl", [prior],
                       HERE / "tasks")["prior / behavior_single"]
        for axis in ("steer", "speed"):
            fs = pentry["fields"][axis]
            check_that(f"{axis}: the corpus-prior answer never beats the baseline",
                       fs["lift_over_majority"] <= 0,
                       f"lift {fs['lift_over_majority']}")

        # And a perfect predictor must clear it, or the metric is broken the
        # other way round.
        perfect = tmp / "perfect.jsonl"
        with open(perfect, "w") as f:
            for i in items:
                f.write(json.dumps({
                    "id": i["id"], "task": "behavior_single", "model": "oracle",
                    "meta": i["meta"], "wall_ms": 1000, "parsed": dict(i["gt"]),
                }) + "\n")
        rep = score(single / "manifest.jsonl", [perfect], HERE / "tasks")
        for axis in ("steer", "speed"):
            check(f"{axis}: oracle scores 100%",
                  rep["oracle / behavior_single"]["fields"][axis]["strict_accuracy"], 1.0)

        print("\n[7] a garbled answer scores as wrong, not as a crash")
        garbled = tmp / "garbled.jsonl"
        with open(garbled, "w") as f:
            for n, i in enumerate(items):
                raw = "I cannot tell from this image." if n % 2 else \
                      '```json\n{"steer": "going straight", "speed": "driving fast"}\n```'
                row = {"id": i["id"], "task": "behavior_single", "model": "garbled",
                       "meta": i["meta"], "raw": raw,
                       "parsed": single_task.parse(raw)}
                f.write(json.dumps(row) + "\n")
        rep = score(single / "manifest.jsonl", [garbled], HERE / "tasks")
        fs = rep["garbled / behavior_single"]["fields"]["steer"]
        check_that("coverage falls below 100% when answers are unparseable",
                   fs["coverage"] < 1.0, f"coverage {fs['coverage']}")
        check_that("strict <= answered when coverage is partial",
                   fs["strict_accuracy"] <= fs["answered_accuracy"])
        check_that("code-fenced JSON still parses",
                   fs["answered"] > 0, f"answered {fs['answered']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
