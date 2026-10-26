# openpi (pi0.5)

- Fork: https://github.com/jaehoondata/openpi
- Upstream: https://github.com/Physical-Intelligence/openpi
- Code: `upstream/` (submodule)
- Custom scripts: `scripts/`, task configs: `configs/`

## Upstream sync

```bash
gh repo sync jaehoondata/openpi
git -C models/openpi/upstream pull origin main
git add models/openpi/upstream && git commit -m "chore(openpi): bump upstream"
```
