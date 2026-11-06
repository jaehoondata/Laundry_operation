"""Top-down stills of a cloth trial (frames.npz) for checking the fold: cloth vertices coloured by height,
pickers in red, at evenly spaced times (one panel per FoldNet++ subtask change is too many).

    python sim/foldnetxx/topview.py outputs/foldnetxx/t3/<name> [--n 8]   # -> <dir>/topview.png
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trial", type=Path)
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(a.trial / "frames.npz")
    q, t, ph, pk = d["q"], d["t"], d["phase"], d["pickers"]
    idx = np.linspace(0, len(q) - 1, a.n).astype(int)
    fig, axs = plt.subplots(1, a.n, figsize=(2.6 * a.n, 3.0))
    lo, hi = q[:, :, :2].reshape(-1, 2).min(0) - 0.05, q[:, :, :2].reshape(-1, 2).max(0) + 0.05
    for ax, i in zip(axs, idx):
        ax.scatter(q[i, :, 0], q[i, :, 1], c=q[i, :, 2], s=0.3, cmap="viridis", vmin=0, vmax=0.06)
        if not np.isnan(pk[i]).all():
            ax.scatter(pk[i, :, 0], pk[i, :, 1], c="r", s=12)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_title(f"{t[i]:.0f}s {str(ph[i])[:22]}", fontsize=6); ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(a.trial / "topview.png", dpi=110)
    print("->", a.trial / "topview.png")


if __name__ == "__main__":
    main()
