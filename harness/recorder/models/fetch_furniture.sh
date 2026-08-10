#!/usr/bin/env bash
# Fetch the Gazebo Fuel furniture p2's semantic world needs, and patch it so it
# actually renders.
#
# Two things make this necessary rather than a plain `gz fuel download`:
#
#   1. Fuel's box-composite models reference their diffuse texture as a REMOTE
#      URL inside the SDF:
#
#        <albedo_map>https://fuel.gazebosim.org/.../bookshelf_diffuse.jpg</albedo_map>
#
#      Nothing fetches that at render time. Under `--headless-rendering` the
#      material silently falls back to flat shading, so `table` and `bookshelf`
#      came out as featureless dark slabs while the mesh-based `chair` and `sofa`
#      rendered correctly. The jpg is sitting in the same directory the whole
#      time; this rewrites the URL to that local filename.
#
#   2. It has to be idempotent and re-runnable, because a re-download restores
#      the remote URLs.
#
# Run once after cloning, before recording any p2 rollout:
#     ./fetch_furniture.sh
#
# Verify by eye afterwards, not by exit code. p1's whole lesson is that a
# rendering fault is invisible in logs: record a rollout and look at the frames.

set -uo pipefail

MODELS=("Chair" "Table" "Bookshelf" "Sofa")
BASE="https://fuel.gazebosim.org/1.0/OpenRobotics/models"
CACHE="${HOME}/.gz/fuel/fuel.gazebosim.org/openrobotics/models"

command -v gz >/dev/null 2>&1 || {
    echo "gz not found. source /opt/ros/jazzy/setup.bash first." >&2
    exit 1
}

echo "fetching ${#MODELS[@]} models into $CACHE"
for m in "${MODELS[@]}"; do
    enc="${m// /%20}"
    if gz fuel download -u "$BASE/$enc" >/dev/null 2>&1; then
        echo "  $m: downloaded"
    else
        # Already cached is the common case, and gz reports it as a failure.
        [ -d "$CACHE/$(echo "$m" | tr '[:upper:]' '[:lower:]')" ] \
            && echo "  $m: already cached" \
            || { echo "  $m: DOWNLOAD FAILED and not cached" >&2; }
    fi
done

echo
echo "rewriting remote albedo_map URLs to local filenames"
patched=0
while IFS= read -r sdf; do
    n=$(grep -c '<albedo_map>http' "$sdf" 2>/dev/null || true)
    [ "${n:-0}" -eq 0 ] && continue
    # Keep one pristine copy so the edit stays auditable.
    [ -f "$sdf.orig" ] || cp "$sdf" "$sdf.orig"
    python3 - "$sdf" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
t = re.sub(r'<albedo_map>https?://[^<]*/([^/<]+)</albedo_map>',
           r'<albedo_map>\1</albedo_map>', t)
open(p, 'w').write(t)
PY
    echo "  $(realpath --relative-to="$CACHE" "$sdf"): $n rewritten"
    patched=$((patched + n))
done < <(find "$CACHE" -name 'model*.sdf' 2>/dev/null)

echo
if [ "$patched" -eq 0 ]; then
    echo "nothing to patch (already local, or models missing)"
else
    echo "patched $patched albedo_map reference(s)"
fi
echo "now record a rollout and LOOK at the frames before trusting them."
