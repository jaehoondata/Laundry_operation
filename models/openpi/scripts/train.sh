#!/usr/bin/env bash
# pi0.5 fine-tuning with LeRobot 0.6.1 (policy.type=pi05), run in a tmux session.
#
#   DATASET_REPO=jaehoondata/laundry_fold_eef DATASET_REV=v0.1 \
#       bash models/openpi/scripts/train.sh
#
# Defaults target one A6000 Ada (48GB): the LeRobot docs size full fine-tuning for
# an 80GB GPU, so this starts with the VLM frozen (action expert only).
# Set TRAIN_EXPERT_ONLY=false to try full fine-tuning (lower BATCH first).
set -euo pipefail

: "${DATASET_REPO:?set DATASET_REPO, e.g. jaehoondata/laundry_fold_eef}"
: "${DATASET_REV:?set DATASET_REV, e.g. v0.1}"
PRETRAINED="${PRETRAINED:-lerobot/pi05_base}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
BATCH="${BATCH:-32}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
JOB_NAME="${JOB_NAME:-pi05_$(basename "$DATASET_REPO")_${DATASET_REV}_$(date +%y%m%d_%H%M)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/outputs/$JOB_NAME"

"$REPO_ROOT/tools/tmux_run.sh" "$JOB_NAME" "$REPO_ROOT/outputs/logs" \
    lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.revision="$DATASET_REV" \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path="$PRETRAINED" \
    --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
    --policy.freeze_vision_encoder="$TRAIN_EXPERT_ONLY" \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --batch_size="$BATCH" \
    --num_workers=8 \
    --steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --wandb.enable=true \
    --wandb.project=laundry_operation
