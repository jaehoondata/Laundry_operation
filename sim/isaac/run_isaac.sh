#!/usr/bin/env bash
# Run a Python script inside the Isaac Sim 4.5 container (nvcr.io/nvidia/isaac-sim:4.5.0), headless.
# The repo is mounted at /work and the SG2 ROS install (URDF + meshes) read-only at /sg2_ros.
# Caches persist under ~/docker/isaac-sim so only the first run compiles shaders. Files the
# container writes under /work/outputs are handed back to the host user afterwards.
#
#   bash sim/isaac/run_isaac.sh sim/isaac/render_isaac.py outputs/style3d_gate/sim_v9/base/trial_00
set -euo pipefail
cd "$(dirname "$0")/../.."
script=$1
shift
C=~/docker/isaac-sim
mkdir -p $C/cache/kit $C/cache/ov $C/cache/pip $C/cache/glcache $C/cache/computecache $C/logs $C/data outputs/isaac
docker run --rm --gpus all --network=host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -v $C/cache/kit:/isaac-sim/kit/cache:rw \
    -v $C/cache/ov:/root/.cache/ov:rw \
    -v $C/cache/pip:/root/.cache/pip:rw \
    -v $C/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v $C/cache/computecache:/root/.nv/ComputeCache:rw \
    -v $C/logs:/root/.nvidia-omniverse/logs:rw \
    -v $C/data:/root/.local/share/ov/data:rw \
    -v "$PWD":/work:rw \
    -v "${SG2_ROS_INSTALL:-$HOME/Desktop/jaehoon/sg2_ros}":/sg2_ros:ro \
    --entrypoint bash nvcr.io/nvidia/isaac-sim:4.5.0 \
    -c "cd /work && /isaac-sim/python.sh \"\$0\" \"\$@\"; rc=\$?; chown -R $(id -u):$(id -g) /work/outputs/isaac /work/outputs/style3d_gate /work/outputs/foldnetxx 2>/dev/null; exit \$rc" \
    "$script" "$@"
