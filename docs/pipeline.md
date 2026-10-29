# 데이터·학습·추론 파이프라인

| 단계 | 장비 | 도구 |
| --- | --- | --- |
| 수집 | Mac (로봇 LAN) | VAI_AIWORKER `real_vr_teleop_record.ipynb` → `~/aiworker_data/<task>` (LeRobot v3.0, 30Hz) |
| 검수 | Mac | VAI_AIWORKER `review_dataset.ipynb` → `<task>_final` |
| 관절 선택 | Mac | `python tools/select_joints.py <task>_final <task>_final_arm16` — lift·head 제외, 16 joints |
| 업로드 | Mac | `python tools/push_dataset.py <task>_final_arm16 --repo-id jaehoondata/<name> --tag vX.Y` (private) |
| 학습 | Ubuntu (A6000 Ada) | `models/<model>/scripts/train.sh` — LeRobot 0.6.1, tmux 세션, wandb |
| 추론 | Ubuntu (로봇 LAN) | VAI_AIWORKER `real_policy_inference.ipynb`, `CHECKPOINT_PATH=outputs/<job>/checkpoints/<step>/pretrained_model`, `EXCLUDED_JOINTS = ["lift_joint", "head_joint1", "head_joint2"]` |

## 처음 한 번

Ubuntu:

```bash
git clone --recursive https://github.com/jaehoondata/Laundry_operation.git   # molmoact2는 GIT_LFS_SKIP_SMUDGE=1 필요
bash env/setup_lerobot.sh
conda activate lerobot061 && hf auth login && wandb login
```

Mac에서 원격 접속: Tailscale(Mac·Ubuntu) + SSH 키 + VS Code Remote-SSH.
학습은 tmux 안에서 돌기 때문에 시작 후 Mac은 꺼도 된다.

## 데이터셋 스키마 (VAI_AIWORKER `episode_recorder.py`)

- `observation.state`, `action`: 19차원 절대 관절 위치 — lift 1, head 2, arm_l 7, arm_r 7, gripper_l 1, gripper_r 1
  - lift·head는 수집 중 고정 → `select_joints.py`로 제외한 16차원(arm_l 7, arm_r 7, gripper 2)으로 학습
- `observation.images.{head, wrist_left, wrist_right}`: h264 영상
- `RECORD_EEF=True`이면 `observation.eef_pose`, `action.eef_pose` 추가
- 로딩 시 `video_backend=pyav`

## 확인 필요

- `observation.eef_pose`를 각 정책이 추가 입력으로 쓰는지, 제외해야 하는지
- pi05·MolmoAct2를 30Hz로 돌릴 때 VAI_AIWORKER `PolicyDeployment`의 추론 지연
