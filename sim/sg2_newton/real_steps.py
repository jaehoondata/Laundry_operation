"""Per-episode fold steps of the real teleop data, for fold_test.py --sequence teleop to replay.

For every real episode that folds LR-L-R-L (19 of 24 in laundry_fold_tshirt_arm16), and each hand
in each step: where the hand (TCP, sim frame with the arm base 0.45 m behind the cloth center and
0.55 m above the table) closed and opened, when (s from the episode start), how long it held, and the
shape of its path while closed (horizontal progress and height at 11 evenly spaced times). The sim
picks one real episode per trial and reproduces its steps, so the sim's spread of grasp points,
lifts and timings is the real one.

    python sim/sg2_newton/real_steps.py --build path/to/data/chunk-000/*.parquet    # -> STEPS_PATH
    python sim/sg2_newton/real_steps.py                                              # summary
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
from sg2_reach import DEFAULT_URDF, ArmChain, arm_base_pose  # noqa: E402

STEPS_PATH = Path(__file__).resolve().parent / "data" / "real_steps.json"
FPS = 30
CLOSED = 0.55
PATTERN = ["LR", "L", "R", "L"]
N_SAMPLES = 11


def episode_steps(state, p):
    """state [T, 16], p [T, 2, 3] hand positions -> list of steps or None if not LR-L-R-L."""
    c = state[:, 14:16] > CLOSED
    edges = np.flatnonzero(np.diff(np.r_[0, c.any(1).astype(int), 0]))
    segs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 10]
    hands = ["".join(s for k, s in enumerate("LR") if c[a:b, k].mean() > 0.3) for a, b in segs]
    if hands[:4] != PATTERN:
        return None
    steps = []
    for (a, b), h in zip(segs[:4], hands[:4]):
        step = {}
        for s in h:
            k = "LR".index(s)
            on = a + int(np.flatnonzero(c[a:b, k])[0])
            off = a + int(np.flatnonzero(c[a:b, k])[-1])
            path = p[on:off + 1, k]
            t = np.linspace(0, 1, len(path))
            ts = np.linspace(0, 1, N_SAMPLES)
            d = np.linalg.norm(path[-1, :2] - path[0, :2])
            prog = (path[:, :2] - path[0, :2]) @ ((path[-1, :2] - path[0, :2]) / max(d, 1e-6)) / max(d, 1e-6)
            step[s.lower()] = {
                "close_t": on / FPS, "open_t": off / FPS,
                "close_p": path[0].round(4).tolist(), "open_p": path[-1].round(4).tolist(),
                "progress": np.round(np.clip(np.maximum.accumulate(np.interp(ts, t, prog)), 0, 1), 3).tolist(),
                "z": np.round(np.interp(ts, t, path[:, 2]), 4).tolist(),
            }
        steps.append(step)
    return steps


def build(parquets, base_y, height):
    import pandas as pd

    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "episode_index"]) for f in parquets])
    base = arm_base_pose(base_y, height)
    chains = {s: ArmChain(DEFAULT_URDF, s) for s in "lr"}
    episodes = []
    for e, g in df.groupby("episode_index"):
        st = np.stack(g["observation.state"].values)
        p = np.stack([[chains[s].fk(q[7 * k:7 * k + 7], base)[0][:3, 3] for k, s in enumerate("lr")] for q in st[:, :14]])
        steps = episode_steps(st, p)
        if steps:
            episodes.append({"episode": int(e), "length_s": len(st) / FPS, "steps": steps})
    STEPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STEPS_PATH.write_text(json.dumps({"base_y": base_y, "height": height, "pattern": PATTERN,
                                      "episodes": episodes}, indent=1))
    print(f"{len(episodes)} of {df.episode_index.nunique()} episodes are LR-L-R-L -> {STEPS_PATH}")


def load(path=STEPS_PATH):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", nargs="+", type=Path)
    p.add_argument("--base-y", type=float, default=-0.45)
    p.add_argument("--height", type=float, default=0.55)
    args = p.parse_args()
    if args.build:
        build(args.build, args.base_y, args.height)
        return
    d = load()
    for si, h in enumerate(d["pattern"]):
        for s in h.lower():
            cp = np.array([e["steps"][si][s]["close_p"] for e in d["episodes"]])
            op = np.array([e["steps"][si][s]["open_p"] for e in d["episodes"]])
            print(f"step {si} {s}: close {np.median(cp, 0).round(2)} (sd {cp.std(0).round(2)})  "
                  f"open {np.median(op, 0).round(2)} (sd {op.std(0).round(2)})")


if __name__ == "__main__":
    main()
