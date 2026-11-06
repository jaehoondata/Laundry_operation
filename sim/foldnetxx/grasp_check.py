"""For each grasp of a T3 replay: the subtask, where the hand closed, and the nearest FoldNet keypoint of the
simulated shirt at that moment (top layer) with its distance. Shows from which step the cloth state stops
matching what the FoldNet++ hand expected (e.g. 'bottom midpoint' grasps landing far from any hem point).

    python sim/foldnetxx/grasp_check.py outputs/foldnetxx/t3/<name>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_initial_shirt import load_mesh  # noqa: E402

d = Path(sys.argv[1])
fr = np.load(d / "frames.npz")
rep = json.loads((d / "replay.json").read_text())
fit = json.loads((d / "fit.json").read_text())
_, _, kp = load_mesh(fit.get("mesh_id", 0))
q, t = fr["q"], fr["t"]
plan = np.load(d / "plan.npz")
for ts, sub, hand, kind, n in rep["events"]:
    if kind != "grasp":
        continue
    i = int(np.searchsorted(t, ts))
    x = q[min(i, len(q) - 1)]
    p = fr["pickers"][min(i + 2, len(q) - 1)][0 if hand == "l" else 1]
    if np.isnan(p).any():
        continue
    best = min(((name, float(np.linalg.norm(x[max(ids, key=lambda j: x[j, 2]), :2] - p[:2]))) for name, ids in kp.items()),
               key=lambda kv: kv[1])
    edge = float(np.linalg.norm(x[:, :2] - p[:2], axis=1).min())
    print(f"{ts:5.1f}s {hand} {n:3d} verts | nearest kp {best[0]:16s} {100 * best[1]:5.1f} cm | nearest cloth {100 * edge:4.1f} cm | {sub[:70]}")
