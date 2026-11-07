#!/usr/bin/env bash
# Generate sim episodes that reproduce the real laundry_fold_tshirt_arm16 teleop data, render them
# and write a LeRobot dataset, on the lab PC (run inside tmux):
#   1. fold_test.py --sequence teleop: each trial replays one real episode step by step, with the
#      condition that matched the real data best (see self-improvement.md, sim_v15/v16)
#   2. render: Newton (quick, default) or Isaac Sim 4.5 (RENDERER=isaac, ~1 h per episode)
#   3. fidelity.py report against the real dataset, then export_lerobot.py
#
#   bash sim/style3d_gate/run_real_replay.sh NAME TRIALS [SEED]
#   RENDERER=isaac bash sim/style3d_gate/run_real_replay.sh real_replay_01 4 7
#
# Output: outputs/style3d_gate/NAME/trial_XX, NAME/fidelity.txt, and the dataset in
# outputs/lerobot/NAME (repo id local/NAME; nothing is pushed).
set -euo pipefail
cd "$(dirname "$0")/../.."
name=$1
trials=$2
seed=${3:-0}
renderer=${RENDERER:-newton}
real=${REAL_ROOT:-$HOME/.cache/huggingface/lerobot/hub/datasets--jaehoon2da--laundry_fold_tshirt_arm16/snapshots/95a9c41ee4511d4db5bc77502ef2f4cabce39862}
out=outputs/style3d_gate/$name
# best match to the real data so far (sim_v16 e06fast): low cloth-table friction lets the shirt slide
# toward the robot in the hem fold as the real one does (0.03 slides more but can bunch the shirt up),
# and hands move between steps at the real pace
condition=(--table-mu 0.06 --ground-mu 0.06 --approach-speed 0.07)

export SG2_ROS_INSTALL=${SG2_ROS_INSTALL:-$HOME/Desktop/jaehoon/sg2_ros}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate newton17
python sim/style3d_gate/fold_test.py --robot --sequence teleop --record-dt 0.0333333 --no-video \
    --trials "$trials" --seed "$seed" "${condition[@]}" --out "$out" 2>&1 | tee "$out.log"
video_dir=""
for t in "$out"/trial_*; do
    if [ "$renderer" = isaac ]; then
        bash sim/isaac/render_trial.sh "$t"
        video_dir=isaac
    else
        python sim/sg2_newton/render_newton.py "$t" --stills 8 > "$t/render.log" 2>&1
    fi
done

conda activate lerobot061
python sim/sg2_newton/fidelity.py --json "$out/fidelity.json" --real "$real"/data/chunk-000/*.parquet \
    --real-videos "$real/videos/observation.images.head/chunk-000" --real-meta "$real/meta/episodes/chunk-000" \
    -- "$out"/trial_* | tee "$out/fidelity.txt"
python sim/sg2_newton/export_lerobot.py --repo-id "local/$name" --root "outputs/lerobot/$name" \
    ${video_dir:+--video-dir "$video_dir"} "$out"/trial_*
