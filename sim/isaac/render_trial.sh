#!/usr/bin/env bash
# Render one fold trial with Isaac Sim end to end, on the lab PC:
#   1. export_scene.py (host, conda newton17)  frames.npz -> scene.npz
#   2. render_isaac.py (Isaac Sim 4.5 container) -> <trial>/isaac/<camera>/*.png
#   3. encode_isaac.sh (host ffmpeg)            -> <trial>/isaac/<camera>.mp4
# Extra arguments go to render_isaac.py (colours, lights, --rt-subframes, ...).
#
#   bash sim/isaac/render_trial.sh outputs/style3d_gate/sim_v10/base/trial_00 [--stride 1] [render args]
set -euo pipefail
cd "$(dirname "$0")/../.."
trial=${1%/}
shift
stride=1
if [ "${1:-}" = "--stride" ]; then stride=$2; shift 2; fi
export SG2_ROS_INSTALL=${SG2_ROS_INSTALL:-$HOME/Desktop/jaehoon/sg2_ros}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate newton17
python sim/isaac/export_scene.py "$trial" --stride "$stride"
fps=$(python -c "import numpy as np; print(float(np.load('$trial/scene.npz')['fps']))")
bash sim/isaac/run_isaac.sh sim/isaac/render_isaac.py "/work/$trial/scene.npz" --out "/work/$trial/isaac" "$@" \
    2>&1 | grep -E "stage ready|frame [0-9]+/|wrote|Error|Traceback"
bash sim/isaac/encode_isaac.sh "$trial/isaac" "$fps"
