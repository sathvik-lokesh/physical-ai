#!/usr/bin/env bash
# Fetch the DriveLM-nuScenes QA file and its images from the OFFICIAL repo, and
# verify what arrived against the hashes the Hugging Face API reports.
#
# The verification is not ceremony. An ungated third-party mirror of this
# dataset (`nicolas93/DriveLM`) serves a `v1_1_train_nus.json` with *exactly*
# the official byte length and a different sha256 — i.e. a modified copy that
# every size check in the world would wave through. Numbers derived from it are
# not numbers about DriveLM. Always compare the hash, never the size.
#
# The repo is gated with auto-approval: accept the licence once at
#   https://huggingface.co/datasets/OpenDriveLab/DriveLM
# then export a read token from https://huggingface.co/settings/tokens
#
#   export HF_TOKEN=hf_...
#   ./fetch_drivelm.sh            # QA file only (193 MB)
#   ./fetch_drivelm.sh --images   # also the training images (3.5 GB zip)
#
# Idempotent: a file that is already present and hash-correct is left alone.

set -euo pipefail

REPO="OpenDriveLab/DriveLM"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/data/drivelm"
WANT_IMAGES=0
[[ "${1:-}" == "--images" ]] && WANT_IMAGES=1

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set." >&2
    echo "Accept the licence at https://huggingface.co/datasets/$REPO," >&2
    echo "then: export HF_TOKEN=hf_...  (read token)" >&2
    exit 2
fi

mkdir -p "$DEST"

# Ask the API for the authoritative sha256 of every file in the repo root.
official_sha() {
    curl -sf --max-time 60 -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/api/datasets/$REPO/tree/main" |
    python3 -c "
import json, sys
want = sys.argv[1]
for x in json.load(sys.stdin):
    if x['path'] == want:
        print((x.get('lfs') or {}).get('oid', ''))
        break
" "$1"
}

get() {
    local name="$1" expect actual
    expect="$(official_sha "$name")"
    if [[ -z "$expect" ]]; then
        echo "!! $name: no sha256 from the API (gate not accepted, or bad token)" >&2
        exit 1
    fi

    if [[ -f "$DEST/$name" ]]; then
        actual="$(sha256sum "$DEST/$name" | cut -d' ' -f1)"
        if [[ "$actual" == "$expect" ]]; then
            echo "ok   $name (already present, hash verified)"
            return
        fi
        echo "..   $name present but hash does not match the official file; re-fetching"
    fi

    echo ">>   $name"
    curl -fL --progress-bar -H "Authorization: Bearer $HF_TOKEN" \
        -o "$DEST/$name.part" \
        "https://huggingface.co/datasets/$REPO/resolve/main/$name"

    actual="$(sha256sum "$DEST/$name.part" | cut -d' ' -f1)"
    if [[ "$actual" != "$expect" ]]; then
        echo "!! $name: sha256 mismatch after download — refusing to install it" >&2
        rm -f "$DEST/$name.part"
        exit 1
    fi
    mv "$DEST/$name.part" "$DEST/$name"
    echo "ok   $name (hash verified)"
}

get v1_1_train_nus.json

# The zip lays out `nuscenes/samples/<CAM>/<file>.jpg`, which is what the QA
# file's image_paths point at once their leading `../` is stripped. Once that
# tree exists the zip is redundant, so having deleted it must not trigger a
# 3.5 GB re-download.
if (( WANT_IMAGES )); then
    if [[ -d "$DEST/nuscenes/samples/CAM_FRONT" ]]; then
        echo "ok   images already unpacked at $DEST/nuscenes/samples/"
    else
        get drivelm_nus_imgs_train.zip
        echo ">>   unzipping (this takes a few minutes)"
        unzip -q -n "$DEST/drivelm_nus_imgs_train.zip" -d "$DEST"
        echo "ok   images at $DEST/nuscenes/samples/"
        echo "     the zip may now be deleted; this script will not re-fetch it"
        echo "     while the unpacked tree is there"
    fi
fi

echo
echo "DriveLM language data is CC BY-NC-SA 4.0. The nuScenes images it points at"
echo "carry nuScenes' own non-commercial terms, which do not permit"
echo "redistribution — so results and code go in the repo, images do not."
