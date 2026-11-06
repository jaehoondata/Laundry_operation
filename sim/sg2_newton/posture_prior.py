"""Hand orientation and arm posture taken from the real SG2 teleop data, looked up by hand position.

The sim used one fixed grasp orientation (tilted 35 deg, heading from the shoulder) and the idle pose
as the IK reference. Against laundry_fold_tshirt_arm16 that put the hand 34-38 deg off the operators'
orientation at grasps and the wrist (J6) ~0.6 rad further bent, so sim and real taught different
joint motions for the same hand path. Here every IK target takes the orientation and the reference
posture of the real frames whose hand was nearest the target (Gaussian-weighted k nearest, in the
sim world frame), so the arm moves the way the operators moved it.

The prior stores real joint states only; hand positions come from FK with the sim's arm base, so a
change of --robot-base-y / --robot-height moves the prior with the robot.

    python sim/sg2_newton/posture_prior.py --build path/to/data/chunk-000/*.parquet   # writes PRIOR_PATH
    python sim/sg2_newton/posture_prior.py                                              # summary
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
from sg2_reach import DEFAULT_URDF, ArmChain, arm_base_pose  # noqa: E402

PRIOR_PATH = Path(__file__).resolve().parent / "data" / "real_posture.npz"


class PosturePrior:
    def __init__(self, chain: ArmChain, base: np.ndarray, side: str, path: Path = PRIOR_PATH, k: int = 32,
                 sigma: float = 0.03):
        d = np.load(path)
        self.q = d[f"q_{side}"].astype(float)
        frames = [chain.fk(qi, base)[0] for qi in self.q]
        self.p = np.array([t[:3, 3] for t in frames])
        self.rot = np.array([t[:3, :3] for t in frames])
        self.tree = cKDTree(self.p)
        self.k, self.sigma = k, sigma

    def __call__(self, pos):
        """(rotation 3x3, reference joints [7], distance to the nearest real hand) near `pos`."""
        dist, idx = self.tree.query(np.asarray(pos, dtype=float), k=self.k)
        w = np.exp(-0.5 * ((dist - dist[0]) / self.sigma) ** 2)
        w /= w.sum()
        # chordal mean of the neighbours' rotations
        u, _, vt = np.linalg.svd(np.einsum("n,nij->ij", w, self.rot[idx]))
        rot = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        return rot, w @ self.q[idx], float(dist[0])


def build(parquets, stride: int):
    import pandas as pd

    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "episode_index"]) for f in parquets])
    st = np.stack(df["observation.state"].values)[::stride]
    PRIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRIOR_PATH, q_l=st[:, :7].astype(np.float32), q_r=st[:, 7:14].astype(np.float32),
                        source=np.array([str(Path(f).name) for f in parquets]))
    print(f"{df.episode_index.nunique()} episodes, {len(st)} frames (every {stride}) -> {PRIOR_PATH}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", nargs="+", type=Path, help="real LeRobot data parquet files")
    p.add_argument("--stride", type=int, default=3, help="keep every n-th frame")
    p.add_argument("--base-y", type=float, default=-0.45)
    p.add_argument("--height", type=float, default=0.55)
    args = p.parse_args()
    if args.build:
        build(args.build, args.stride)
        return
    base = arm_base_pose(args.base_y, args.height)
    for side in "lr":
        prior = PosturePrior(ArmChain(DEFAULT_URDF, side), base, side)
        lo, hi = np.percentile(prior.p, [5, 95], axis=0)
        print(f"{side}: {len(prior.q)} frames, hand x {lo[0]:+.2f}..{hi[0]:+.2f} y {lo[1]:+.2f}..{hi[1]:+.2f} "
              f"z {lo[2]:.2f}..{hi[2]:.2f} (5-95%, sim world)")


if __name__ == "__main__":
    main()
