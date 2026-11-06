#!/usr/bin/env bash
# Encode render_isaac.py's PNG frames into <out>/<camera>.mp4 (H.264, the trial's fps), on the host.
#   bash sim/isaac/encode_isaac.sh <trial_dir>/isaac [fps]
set -euo pipefail
out=$1
fps=${2:-30}
for cam in head wrist_left wrist_right; do
    [ -d "$out/$cam" ] || continue
    ffmpeg -v error -y -framerate "$fps" -pattern_type glob -i "$out/$cam/*.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 "$out/$cam.mp4"
    echo "$out/$cam.mp4 ($(ls "$out/$cam" | wc -l) frames)"
done
