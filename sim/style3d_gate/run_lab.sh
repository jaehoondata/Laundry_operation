#!/usr/bin/env bash
# Style3D cloth gate test on the lab PC. Run inside tmux:
#   tmux new -s style3d_gate 'bash sim/style3d_gate/run_lab.sh; exec bash'
# Waits until the GPU is mostly free (current training run), then runs the gate.
set -euo pipefail
cd "$(dirname "$0")/../.."

ENV_NAME=newton17
MAX_USED_MIB=${MAX_USED_MIB:-8000}
OUT=${OUT:-outputs/style3d_gate/$(date +%m%d_%H%M)}

source ~/anaconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.11
    conda activate "$ENV_NAME"
    pip install "newton==1.7.0" matplotlib imageio imageio-ffmpeg
else
    conda activate "$ENV_NAME"
fi

# PyPI warp-lang is built with CUDA 13 (needs driver >= 580); the lab PC has 570, so use the cu12 build.
if ! python -c "import warp as wp; wp.config.quiet = True; wp.init(); assert wp.is_cuda_available()" 2>/dev/null; then
    WARP_VER=$(python -c "import importlib.metadata as m; print(m.version('warp-lang').split('+')[0])")
    pip install --force-reinstall --no-deps \
        "https://github.com/NVIDIA/warp/releases/download/v${WARP_VER}/warp_lang-${WARP_VER}%2Bcu12-py3-none-manylinux_2_28_x86_64.whl"
fi

while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt "$MAX_USED_MIB" ] && break
    echo "$(date +%H:%M) GPU busy (${used} MiB used), waiting..."
    sleep 120
done

mkdir -p "$OUT"
python sim/style3d_gate/gate_fling.py --device cuda:0 --out "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
