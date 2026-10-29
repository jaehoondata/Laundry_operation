# MolmoAct2

- Fork: https://github.com/jaehoondata/molmoact2
- Upstream: https://github.com/allenai/molmoact2
- `upstream/` (submodule): 참고용. 공식 실험 스크립트·FastAPI 서버 확인
- 학습은 LeRobot 0.6.1의 `policy.type=molmoact2`로 한다 (`scripts/train.sh`).

## Training

```bash
DATASET_REPO=jaehoon2da/laundry_fold_tshirt_arm16 DATASET_REV=v0.1 bash models/molmoact2/scripts/train.sh
```

- 기본값: `allenai/MolmoAct2-BimanualYAM`에서 시작, VLM LoRA + action expert 전체 학습, bs=16 (H100 기준 ~27GiB).
- `SETUP_TYPE`(프롬프트에 들어가는 장면 설명)은 아직 임시 문구.

## Clone note

Some Git LFS objects in the fork return 404 (e.g. `experiments/lerobot/tests/artifacts/`),
so clone/update with LFS smudge disabled:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init models/molmoact2/upstream
```

## Upstream sync

```bash
gh repo sync jaehoondata/molmoact2
GIT_LFS_SKIP_SMUDGE=1 git -C models/molmoact2/upstream pull origin main
git add models/molmoact2/upstream && git commit -m "chore(molmoact2): bump upstream"
```
