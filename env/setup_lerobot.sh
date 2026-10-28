#!/usr/bin/env bash
# Ubuntu 학습 환경: Python 3.12 + LeRobot 0.6.1 (pi05, molmoact2 정책 포함)
#
#   bash env/setup_lerobot.sh [env_name]      (default: lerobot061)
#
# torch/lerobot 버전은 VAI_AIWORKER(setup_ubuntu_env.sh)와 같게 맞춘다.
# 그래야 여기서 학습한 체크포인트를 VAI_AIWORKER의 PolicyRuntime이 그대로 불러온다.
set -euo pipefail

ENV_NAME="${1:-lerobot061}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.12 pip
fi
conda activate "$ENV_NAME"

python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX_URL"
python -m pip install "lerobot[training,pi,molmoact2]==0.6.1"

python - <<'EOF'
import torch, lerobot
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("lerobot", lerobot.__version__)
EOF

cat <<EOF

Done. Next (한 번만):
  conda activate $ENV_NAME
  hf auth login        # private 데이터셋 다운로드, gated 모델 접근
  wandb login
  # pi05: https://huggingface.co/google/paligemma-3b-pt-224 에서 라이선스 동의 필요
EOF
