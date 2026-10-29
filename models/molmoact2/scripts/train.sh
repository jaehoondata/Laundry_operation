#!/usr/bin/env bash
# MolmoAct2 fine-tuning with LeRobot 0.6.1 (policy.type=molmoact2), run in a tmux session.
#
#   DATASET_REPO=jaehoon2da/laundry_fold_tshirt_arm16 DATASET_REV=v0.1 \
#       bash models/molmoact2/scripts/train.sh [extra lerobot-train args...]
#
# Defaults follow the LeRobot MolmoAct2 docs for <200 real-world demos: LoRA on the VLM
# with the action expert fully trainable (~27GiB at bs=16 on H100, fits a 48GB A6000 Ada).
# Starts from MolmoAct2-BimanualYAM (bimanual, absolute joint pose) instead of the base.
# normalize_gripper=true: SG2 gripper joints span 0..1.1 rad, outside the [-1, 1] passthrough
# range MolmoAct2 assumes for un-normalized gripper dims.
#
# Measured on the RTX 6000 Ada (bs=16): ~10.4 s/step with these defaults, ~5.2 s/step with
# action-expert-only training (--policy.action_mode=continuous --policy.enable_lora_vlm=false
# --policy.train_action_expert_only=true).
set -euo pipefail

: "${DATASET_REPO:?set DATASET_REPO, e.g. jaehoon2da/laundry_fold_tshirt_arm16}"
: "${DATASET_REV:?set DATASET_REV, e.g. v0.1}"
CHECKPOINT="${CHECKPOINT:-allenai/MolmoAct2-BimanualYAM}"
# TODO: 실제 장면 설명으로 다듬기. 예시는 allenai/MolmoAct2 norm_stats.json의 metadata_by_tag 참고
SETUP_TYPE="${SETUP_TYPE:-bimanual mobile manipulator folding laundry on a table}"
CHUNK="${CHUNK:-30}"
BATCH="${BATCH:-16}"
STEPS="${STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
JOB_NAME="${JOB_NAME:-molmoact2_$(basename "$DATASET_REPO")_${DATASET_REV}_$(date +%y%m%d_%H%M)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/outputs/$JOB_NAME"

# python -m rather than the accelerate script: a user-site ~/.local/bin/accelerate (system
# Python, no torch) can come before the conda env in PATH.
"$REPO_ROOT/tools/tmux_run.sh" "$JOB_NAME" "$REPO_ROOT/outputs/logs" \
    python -m accelerate.commands.launch --num_processes=1 --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.revision="$DATASET_REV" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --policy.type=molmoact2 \
    --policy.checkpoint_path="$CHECKPOINT" \
    --policy.device=cuda \
    --policy.action_mode=both \
    --policy.inference_action_mode=continuous \
    --policy.enable_lora_vlm=true \
    --policy.chunk_size="$CHUNK" \
    --policy.n_action_steps="$CHUNK" \
    --policy.setup_type="$SETUP_TYPE" \
    --policy.control_mode="absolute joint pose" \
    --policy.image_keys='["observation.images.head","observation.images.wrist_left","observation.images.wrist_right"]' \
    --policy.model_dtype=bfloat16 \
    --policy.num_flow_timesteps=8 \
    --policy.gradient_checkpointing=true \
    --policy.normalize_gripper=true \
    --policy.scheduler_decay_steps="$STEPS" \
    --policy.push_to_hub=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --batch_size="$BATCH" \
    --num_workers=4 \
    --steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --env_eval_freq=-1 \
    --wandb.enable=true \
    --wandb.project=laundry_operation \
    "$@"
