# openpi (pi0.5)

- Fork: https://github.com/jaehoondata/openpi
- Upstream: https://github.com/Physical-Intelligence/openpi
- `upstream/` (submodule): 참고용. 공식 설정·정규화 방식 확인
- 학습은 LeRobot 0.6.1의 `policy.type=pi05`로 한다 (`scripts/train.sh`). openpi는 LeRobot 구버전(v2.x 데이터셋)에 고정되어 VAI_AIWORKER의 v3.0 데이터를 바로 읽지 못한다.

## Training

```bash
DATASET_REPO=jaehoondata/laundry_fold_eef DATASET_REV=v0.1 bash models/openpi/scripts/train.sh
```

- 기본값은 A6000 Ada 48GB 기준 action expert만 학습 (`TRAIN_EXPERT_ONLY=true`). LeRobot 문서의 전체 fine-tuning 예시는 80GB GPU 기준.
- `google/paligemma-3b-pt-224` 라이선스 동의 + `hf auth login` 필요.
- STATE/ACTION은 quantile 정규화(q01/q99). LeRobot 0.6.1로 녹화한 데이터셋은 stats에 포함되어 있다.

## Upstream sync

```bash
gh repo sync jaehoondata/openpi
git -C models/openpi/upstream pull origin main
git add models/openpi/upstream && git commit -m "chore(openpi): bump upstream"
```
