"""Compare sim trials with the real teleop data on what a policy trained on both would see.

Per arm joint: median and 5-95% range, real vs sim, flagged when the medians differ by more than
--flag rad. Also episode length, hand speed while moving, hand height and the gripper's grasp
pattern (which hands hold cloth, in what order). Sim trials are fold_test.py --robot outputs
(frames.npz with joint/grip); real data are LeRobot parquet files of laundry_fold_tshirt_arm16.

    python sim/sg2_newton/compare_real.py --real path/to/data/chunk-000/*.parquet -- outputs/.../trial_0*
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
from sg2_reach import DEFAULT_URDF, ArmChain, arm_base_pose  # noqa: E402

FPS = 30
CLOSED = 0.55  # gripper_joint1 above this (of 1.1) counts as holding


def grasp_pattern(grip):
    """'LR-L-R-...': hands closed in each run of frames where any hand is closed (>= 10 frames)."""
    c = grip > CLOSED
    edges = np.flatnonzero(np.diff(np.r_[0, c.any(1).astype(int), 0]))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 10:
            out.append("".join(s for k, s in enumerate("LR") if c[a:b, k].mean() > 0.3))
    return "-".join(out)


def hands(q14, chains, base):
    return np.stack([[chains[s].fk(q[7 * k:7 * k + 7], base)[0][:3, 3] for k, s in enumerate("lr")] for q in q14])


def episode_stats(state, chains, base):
    """state [T, 16] (14 joints + 2 grippers in 0..1.1) -> dict of per-episode numbers."""
    p = hands(state[:, :14], chains, base)
    v = np.linalg.norm(np.diff(p, axis=0), axis=2) * FPS
    return {"len_s": len(state) / FPS, "speed": v[v > 0.02], "z": p[:, :, 2].ravel(),
            "pattern": grasp_pattern(state[:, 14:16])}


def load_real(parquets):
    import pandas as pd

    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "episode_index"]) for f in parquets])
    return [np.stack(g["observation.state"].values) for _, g in df.groupby("episode_index")]


def load_sim(trials):
    eps = []
    for t in trials:
        d = np.load(Path(t) / "frames.npz")
        joint = d["joint"].astype(float)
        known = np.nonzero(~np.isnan(joint).any(axis=(1, 2)))[0]
        joint[: known[0]] = joint[known[0]]
        eps.append(np.concatenate([joint[:, 0], joint[:, 1], d["grip"].astype(float) * 1.1], axis=1))
    return eps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", nargs="+", type=Path, required=True)
    p.add_argument("trials", nargs="+", type=Path)
    p.add_argument("--flag", type=float, default=0.3, help="flag joints whose medians differ by more [rad]")
    p.add_argument("--base-y", type=float, default=-0.45)
    p.add_argument("--height", type=float, default=0.55)
    args = p.parse_args()

    base = arm_base_pose(args.base_y, args.height)
    chains = {s: ArmChain(DEFAULT_URDF, s) for s in "lr"}
    data = {"real": load_real(args.real), "sim": load_sim(args.trials)}
    stats = {k: [episode_stats(e, chains, base) for e in eps] for k, eps in data.items()}

    print(f"episodes: real {len(data['real'])}, sim {len(data['sim'])}")
    print("joint   real p5 / p50 / p95       sim p5 / p50 / p95")
    allq = {k: np.concatenate([e[:, :14] for e in eps]) for k, eps in data.items()}
    for j in range(14):
        r, s = (np.percentile(allq[k][:, j], [5, 50, 95]) for k in ("real", "sim"))
        flag = "  <--" if abs(r[1] - s[1]) > args.flag else ""
        print(f"{'lr'[j // 7]}{j % 7 + 1}    {r[0]:+.2f} / {r[1]:+.2f} / {r[2]:+.2f}      "
              f"{s[0]:+.2f} / {s[1]:+.2f} / {s[2]:+.2f}{flag}")
    for name, key, unit in (("episode length", "len_s", "s"), ("hand speed moving", "speed", "m/s"),
                            ("hand height", "z", "m")):
        vals = {k: np.concatenate([np.atleast_1d(e[key]) for e in st]) for k, st in stats.items()}
        print(f"{name:<18} real p10/50/90 {np.percentile(vals['real'], [10, 50, 90]).round(3)}  "
              f"sim {np.percentile(vals['sim'], [10, 50, 90]).round(3)} {unit}")
    for k, st in stats.items():
        pats = {}
        for e in st:
            pats[e["pattern"]] = pats.get(e["pattern"], 0) + 1
        print(f"grasp patterns {k}: " + ", ".join(f"{n}x {pat}" for pat, n in sorted(pats.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
