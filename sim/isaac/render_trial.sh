#!/usr/bin/env bash
# Render one fold trial with Isaac Sim end to end, on the lab PC:
#   1. export_scene.py (host, conda newton17)  frames.npz -> scene.npz
#   2. render_isaac.py (Isaac Sim 4.5 container) -> <trial>/isaac/<camera>/*.png
#   3. encode_isaac.sh (host ffmpeg)            -> <trial>/isaac/<camera>.mp4
# --stride and --robot-base-y (where the SG2 stands, default -0.45; FoldNet++ replays use -0.55) go to
# export_scene.py; extra arguments go to render_isaac.py (colours, lights, --rt-subframes, ...).
#
#   bash sim/isaac/render_trial.sh outputs/style3d_gate/sim_v10/base/trial_00 [--stride 1] [--robot-base-y -0.45] [render args]
set -euo pipefail
cd "$(dirname "$0")/../.."
trial=${1%/}
shift
stride=1
base_y=-0.45
while [ "${1:-}" = "--stride" ] || [ "${1:-}" = "--robot-base-y" ]; do
    if [ "$1" = "--stride" ]; then stride=$2; else base_y=$2; fi
    shift 2
done
export SG2_ROS_INSTALL=${SG2_ROS_INSTALL:-$HOME/Desktop/jaehoon/sg2_ros}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${NEWTON_ENV:-newton17}"   # vai_4090: ~/Desktop/jaehoon/envs/newton17
python sim/isaac/export_scene.py "$trial" --stride "$stride" --robot-base-y "$base_y"
fps=$(python -c "import numpy as np; print(float(np.load('$trial/scene.npz')['fps']))")
bash sim/isaac/run_isaac.sh sim/isaac/render_isaac.py "/work/$trial/scene.npz" --out "/work/$trial/isaac" "$@" \
    2>&1 | grep -E "stage ready|frame [0-9]+/|wrote|Error|Traceback"
bash sim/isaac/encode_isaac.sh "$trial/isaac" "$fps"
