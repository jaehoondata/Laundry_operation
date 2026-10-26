# MolmoAct2

- Fork: https://github.com/jaehoondata/molmoact2
- Upstream: https://github.com/allenai/molmoact2
- Code: `upstream/` (submodule)
- Custom scripts: `scripts/`, task configs: `configs/`

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
