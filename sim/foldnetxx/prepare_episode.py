"""Turn one FoldNet++ episode's folding segment into an SG2 plan for the cloth replay (t3_replay.py).

Runs where h5py is available (conda base on the lab PC). Writes <out>/plan.npz:
  q      [N, 2, 7]  SG2 arm joints (l, r): a min-jerk approach from the real idle pose, then the
                    folding segment solved with the posture-keeping IK (t1_kinematics.py: FoldNet++
                    EEF positions, real-SG2 hand rotations, idle hands parked)
  tcp    [N, 2, 3]  SG2 TCP positions from FK of q: the pickers follow these, so a grasp happens
                    where the SG2 hand really is
  grip   [N, 2]     1 closed / 0 open (FoldNet++ gripper > 0.5; open during the approach)
  dt     [N-1]      step durations: FoldNet++'s 0.2 s, stretched where a joint would move faster
                    than the real teleop maximum (0.84 rad/s)
  subtask [N]       FoldNet++ subtask label per step ("approach" for the added approach)
  target [N, 2, 3]  the FoldNet++ EEF positions the IK followed (NaN during the approach), and
  active [N, 2]     whether a hand works the cloth at that step (else it is parked): t3_replay.py
                    --retarget re-solves the IK from these with each grasp moved onto the sim cloth
  fit               the recovered starting shirt (fit_initial_shirt.py): scale, theta, t, side

Grasp heights: in about a quarter of the episodes every grasp closes ~4.5 cm higher than in the rest
(first close z p50 ~ -0.01 m vs ~ +0.033 m, the same in all robots; likely a per-episode table height).
Our table is at z = 0 and the picker only takes cloth from 1.3 cm below the TCP, so those grasps
caught a single vertex. All EEF heights are shifted so that the first two closes (on the flat shirt)
sit at --grasp-z (fit.json "z_shift"), where SG2's closed fingertips touch the table.

    python sim/foldnetxx/prepare_episode.py path/to/episode.h5 --out outputs/foldnetxx/t3/<name>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sg2_newton"))
sys.path.insert(0, str(HERE.parent / "sg2_reach"))
from fit_initial_shirt import fit, load_mesh  # noqa: E402
from t1_kinematics import FLAT, REAL_MAX_SPEED, STEP_DT, active_mask, load_fold_segment, track_posture  # noqa: E402


def min_jerk_joints(q0, q1, n):
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    w = 10 * s**3 - 15 * s**4 + 6 * s**5
    return q0[None] + w[:, None, None] * (q1 - q0)[None]


def failure_keep_mask(episode, n, grip_closed):
    """Steps of the fold segment to keep when FoldNet++'s injected failures and their recoveries are dropped.

    A block is a run of steps with the same subtask. For a block with any is_correct_actions == False step:
      * if the same subtask starts again within the next 6 blocks, with no other grasp subtask in between
        (the usual "wrong grasp -> pull -> Re-grasp ... -> grasp again"), drop everything from the block
        start up to that retry
      * otherwise (corrected in place) drop only the incorrect steps
    A cut is kept only if both grippers are in the same open/closed state right before and right after it.
    Returns (keep [n] bool, list of (first, last, kind) dropped ranges in fold-segment steps)."""
    import h5py

    with h5py.File(episode, "r") as f:
        sub = [s.decode() for s in f["subtasks"][:]]
        ok = f["is_correct_actions"][:].astype(bool)
    k0 = sub.index(FLAT)
    sub, ok = sub[k0:k0 + n], ok[k0:k0 + n]
    starts = [0] + [i for i in range(1, n) if sub[i] != sub[i - 1]]
    ends = starts[1:] + [n]

    def is_grasp(s):
        s = s.lower()
        return "grasp" in s and not s.startswith("re-grasp")

    keep, cuts = np.ones(n, bool), []

    def cut(a, b, kind):  # drop steps a..b-1
        before = grip_closed[a - 1] if a > 0 else np.zeros(2, bool)
        after = grip_closed[b] if b < n else before
        if b > a and (before == after).all() and keep[a:b].all():
            keep[a:b] = False
            cuts.append((a, b - 1, kind))

    for bi, (a, b) in enumerate(zip(starts, ends)):
        if ok[a:b].all() or not keep[a:b].all():
            continue
        retry = None
        for bj in range(bi + 1, min(bi + 7, len(starts))):
            s = sub[starts[bj]]
            if s == sub[a]:
                retry = starts[bj]
                break
            if is_grasp(s):
                break
        if retry is not None:
            cut(a, retry, "retry")
        else:
            i = a
            while i < b:
                if not ok[i]:
                    j = i
                    while j < b and not ok[j]:
                        j += 1
                    cut(i, j, "in-place")
                    i = j
                else:
                    i += 1
    return keep, cuts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-y", type=float, default=-0.45)
    p.add_argument("--height", type=float, default=0.55)
    p.add_argument("--approach-s", type=float, default=3.0, help="idle pose -> first FoldNet++ pose")
    p.add_argument("--mesh-id", type=int, default=0)
    p.add_argument("--grasp-z", type=float, default=0.008,
                   help="TCP height of the first grasps on the flat shirt. FoldNet++ episodes without the table offset "
                        "close at -0.012, but SG2's closed fingertips reach 0.8 cm below its TCP, so 0.008 puts them "
                        "on the table instead of 2 cm into it")
    p.add_argument("--drop-failures", action="store_true",
                   help="drop FoldNet++'s injected wrong actions and the recovery steps that undo them "
                        "(see failure_keep_mask); the shirt fit still uses the original first grasps")
    args = p.parse_args()

    from sg2_ik import IDLE
    from sg2_robot import SG2Arms
    from posture_prior import PRIOR_PATH

    poses, grip_raw, sub = load_fold_segment(args.episode)
    n = len(sub)
    cuts = []
    if args.drop_failures:
        keep, cuts = failure_keep_mask(args.episode, n, grip_raw > 0.5)
        poses = {s: v[keep] for s, v in poses.items()}
        grip_raw, sub = grip_raw[keep], [s for s, k in zip(sub, keep) if k]
        print(f"dropped {n - len(sub)} of {n} fold steps: {cuts}")
        n = len(sub)
    closed = grip_raw > 0.5
    first_z = sorted((k, poses[s][k, 2, 3]) for i, s in enumerate("lr")
                     for k in range(1, n) if closed[k, i] and not closed[k - 1, i])[:2]
    z_shift = float(args.grasp_z - np.mean([z for _, z in first_z])) if first_z else 0.0
    for s in "lr":
        poses[s][:, 2, 3] += z_shift
    q, ok = track_posture(poses, grip_raw, args.base_y, args.height, n, "prior", "park")
    v, _, kp = load_mesh(args.mesh_id)
    shirt = fit(args.episode, v, kp)
    if shirt is None:
        raise SystemExit("no first grasp found: cannot place the shirt")

    m = int(round(args.approach_s / STEP_DT))
    idle = np.stack([IDLE, IDLE])
    q_all = np.concatenate([min_jerk_joints(idle, q[0], m), q])
    grip = np.concatenate([np.zeros((m, 2)), (grip_raw > 0.5).astype(float)])
    subtask = ["approach"] * m + list(sub)
    arms = SG2Arms(base_y=args.base_y, height=args.height, prior=PRIOR_PATH, start_idle=True)
    tcp = np.array([[arms.tcp(s, qq[k]) for k, s in enumerate("lr")] for qq in q_all])
    dq = np.abs(np.diff(q_all, axis=0)).max(axis=(1, 2))
    dt = np.maximum(STEP_DT, dq / REAL_MAX_SPEED)
    dt[:m] = np.maximum(dt[:m], args.approach_s / m)

    args.out.mkdir(parents=True, exist_ok=True)
    target = np.concatenate([np.full((m, 2, 3), np.nan), np.stack([poses[s][:, :3, 3] for s in "lr"], axis=1)])
    active = np.concatenate([np.zeros((m, 2), bool), np.stack([active_mask(grip_raw[:, i]) for i in range(2)], axis=1)])
    np.savez_compressed(args.out / "plan.npz", q=q_all.astype(np.float32), tcp=tcp.astype(np.float32),
                        grip=grip.astype(np.float32), dt=dt.astype(np.float32), subtask=np.array(subtask),
                        ik_ok=np.concatenate([np.ones((m, 2), bool), ok]), target=target, active=active)
    (args.out / "fit.json").write_text(json.dumps({**shirt, "episode": str(args.episode), "base_y": args.base_y,
                                                   "height": args.height, "mesh_id": args.mesh_id, "z_shift": z_shift,
                                                   "dropped_failures": cuts}, indent=1))
    print(f"{args.episode.name}: {n} FoldNet++ steps + {m} approach, {dt.sum():.1f} s "
          f"(FoldNet++ {STEP_DT * (n - 1):.1f} s), IK ok {100 * ok.all(1).mean():.0f}%, "
          f"shirt size {shirt['size']:.3f} m, theta {shirt['theta_deg']:.1f} deg, z shift {100 * z_shift:+.1f} cm -> {args.out}")


if __name__ == "__main__":
    main()
