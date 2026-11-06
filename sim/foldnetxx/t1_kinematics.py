"""T1: can SG2 follow the folding part of FoldNet++ episodes? Kinematics only, no cloth.

For every FoldNet++ episode (.h5) under the given directories, takes the folding segment (from the
subtask "The T-shirt is flattened." to the end), turns both commanded EEF poses (world frame, table
top at z = 0, cloth near the origin, robots on the -y side) into SG2 TCP targets as they are (the
FoldNet++ EEF frame and SG2's end_effector link agree, see sim/sg2_reach/sg2_reach.py), and solves
SG2 IK step by step (5 Hz) with the arm bases where the lab SG2 stands. Per source robot it reports:
  reach      share of steps whose pose SG2 reaches (5 mm, 3 deg), per subtask and per episode
  speed      joint speed between consecutive 5 Hz steps [rad/s] (real teleop max 0.84, URDF 4.8)
  collision  closest arm-arm / arm-torso distance over the segment (check_arm_collision)

    python sim/foldnetxx/t1_kinematics.py outputs/foldnetxx_sample/valid/* --out outputs/foldnetxx/t1.json
"""

import argparse
import json
import multiprocessing
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sg2_reach"))
sys.path.insert(0, str(ROOT / "sg2_newton"))
from sg2_reach import DEFAULT_URDF, ArmChain, arm_base_pose, homogeneous, home_posture, six_to_matrix, solve_ik  # noqa: E402

FLAT = "The T-shirt is flattened."
STEP_DT = 0.2  # FoldNet++ action step
REAL_MAX_SPEED = 0.84  # rad/s, max joint speed in the real SG2 teleop data


def load_fold_segment(path: Path):
    """(poses {side: [N,4,4]}, grip [N,2], subtasks [N]) of the folding part, or None."""
    import h5py

    with h5py.File(path, "r") as f:
        a = f["actions_eef"][:]
        sub = [s.decode() for s in f["subtasks"][:]]
    if FLAT not in sub:
        return None
    k = sub.index(FLAT)
    a, sub = a[k:], sub[k:]
    poses = {s: np.stack([homogeneous(six_to_matrix(r[o + 3:o + 9]), r[o:o + 3]) for r in a])
             for s, o in (("l", 0), ("r", 9))}
    return poses, a[:, 18:20], sub


def track_plain(poses, base, n, rng):
    """Damped least squares from a sampled home posture (sg2_reach.py), restarts when stuck."""
    chains = {s: ArmChain(DEFAULT_URDF, s) for s in "lr"}
    q_path, ok_path = np.zeros((n, 2, 7)), np.zeros((n, 2), bool)
    for ki, (side, chain) in enumerate(chains.items()):
        q0 = home_posture(chain, base, poses[side][0], rng, restarts=10)
        q_prev = q0
        for k in range(n):
            q, ok, _, _ = solve_ik(chain, base, poses[side][k], 0.95 * q_prev + 0.05 * q0)
            if not ok:
                for _ in range(3):
                    q, ok, _, _ = solve_ik(chain, base, poses[side][k], rng.uniform(chain.lower, chain.upper), iters=150)
                    if ok:
                        break
            if ok:
                q_prev = q
            q_path[k, ki], ok_path[k, ki] = q_prev, ok
    return q_path, ok_path


def active_mask(grip, before=8, after=3):
    """Steps where a hand works the cloth: while closed, `before` steps ahead of each close (its
    approach) and `after` steps past each release (moving clear). Elsewhere the hand is idle."""
    closed = grip > 0.5
    act = closed.copy()
    for k in np.flatnonzero(np.diff(closed.astype(int)) == 1) + 1:
        act[max(0, k - before):k] = True
    for k in np.flatnonzero(np.diff(closed.astype(int)) == -1) + 1:
        act[k:k + after] = True
    return act


def track_posture(poses, grip, base_y, height, n, orient, idle="park"):
    """SG2's posture-keeping IK (sg2_robot.py): starts at the real idle pose, keeps postures near the
    real teleop ones (posture prior), and moves at most max_step per 0.2 s step, so no branch flips.
    orient = 'foldnet' tracks the FoldNet++ EEF rotation, 'prior' the rotation real SG2 hands had
    at that position (positions are what grasps need). idle = 'park' sends a hand that is not
    working the cloth (see active_mask) to the real teleop idle pose instead of following the
    source robot's free motion, which on SG2 crosses the other arm in the flip steps."""
    from posture_prior import PRIOR_PATH
    from sg2_ik import IDLE
    from sg2_robot import SG2Arms

    arms = SG2Arms(base_y=base_y, height=height, prior=PRIOR_PATH, start_idle=True)
    q_path, ok_path = np.zeros((n, 2, 7)), np.zeros((n, 2), bool)
    for ki, side in enumerate("lr"):
        ik, q = arms.iks[side], IDLE.copy()
        active = active_mask(grip[:, ki]) if idle == "park" else np.ones(n, bool)
        park = homogeneous(np.eye(3), arms.park(side))
        for k in range(n):
            pose = poses[side][k] if active[k] else park
            t_prior, q_ref = arms.target(side, pose[:3, 3])
            target = (pose if orient == "foldnet" else t_prior, q_ref)
            for _ in range(25 if k == 0 else 1):  # the first pose is reached from idle beforehand
                q, ok, _, _ = arms.solve(ik, side, pose[:3, 3], q, target)
                if ok:
                    break
            q_path[k, ki], ok_path[k, ki] = q, ok
    return q_path, ok_path


def run_episode(job):
    path, base_y, height, seed, ik, orient, idle = job
    seg = load_fold_segment(Path(path))
    if seg is None:
        return None
    poses, grip, sub = seg
    n = len(sub)
    if ik == "plain":
        q_path, ok_path = track_plain(poses, arm_base_pose(base_y, height), n, np.random.default_rng(seed))
    else:
        q_path, ok_path = track_posture(poses, grip, base_y, height, n, orient, idle)
    speed = np.abs(np.diff(q_path, axis=0)).max(axis=2) / STEP_DT  # [n-1, 2]
    # retimed duration: each step takes at least STEP_DT and as long as the real max speed needs
    retime = float(np.maximum(STEP_DT, np.abs(np.diff(q_path, axis=0)).max(axis=(1, 2)) / REAL_MAX_SPEED).sum()
                   / (STEP_DT * (n - 1)))
    from check_arm_collision import clearance

    c = clearance(q_path, (grip > 0.5).astype(float), STEP_DT, check_dt=STEP_DT)
    by_sub = defaultdict(lambda: [0, 0])
    for k, s in enumerate(sub):
        by_sub[s][0] += int(ok_path[k].all())
        by_sub[s][1] += 1
    return {"episode": str(path), "steps": n, "reach": float(ok_path.all(1).mean()),
            "reach_l": float(ok_path[:, 0].mean()), "reach_r": float(ok_path[:, 1].mean()),
            "speed_p99": float(np.percentile(speed, 99)), "speed_max": float(speed.max()),
            "over_real_speed": float((speed > REAL_MAX_SPEED).any(1).mean()),
            "arm_arm_min": c["arm_arm_min"], "arm_body_min": c["arm_body_min"], "retime": retime,
            "by_subtask": dict(by_sub)}


def summarize(results):
    r = [x for x in results if x]
    if not r:
        return {}
    by_sub = defaultdict(lambda: [0, 0])
    for x in r:
        for s, (a, b) in x["by_subtask"].items():
            by_sub[s][0] += a
            by_sub[s][1] += b
    return {
        "episodes": len(r),
        "step_reach": float(np.mean([x["reach"] for x in r])),
        "episodes_fully_reached": float(np.mean([x["reach"] == 1.0 for x in r])),
        "episodes_reach_ge_95": float(np.mean([x["reach"] >= 0.95 for x in r])),
        "speed_p99_median": float(np.median([x["speed_p99"] for x in r])),
        "steps_over_real_speed": float(np.mean([x["over_real_speed"] for x in r])),
        "arm_collision_episodes": float(np.mean([min(x["arm_arm_min"], x["arm_body_min"]) < 0.01 for x in r])),
        "retime_median": float(np.median([x["retime"] for x in r])),
        "worst_subtasks": sorted(((s, a / b, b) for s, (a, b) in by_sub.items()), key=lambda t: t[1])[:6],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("robot_dirs", nargs="+", type=Path, help="one dir per source robot, holding episode .h5 files")
    p.add_argument("--base-y", type=float, default=-0.45, help="SG2 arm_base_link y (lab setup)")
    p.add_argument("--height", type=float, default=0.55, help="SG2 arm_base_link height above the table (lab setup)")
    p.add_argument("--max-episodes", type=int, default=60, help="per source robot")
    p.add_argument("--ik", choices=["posture", "plain"], default="posture",
                   help="posture: SG2 posture-keeping IK with the real-teleop prior; plain: sg2_reach.py DLS")
    p.add_argument("--idle", choices=["park", "follow"], default="park",
                   help="park: a hand not working the cloth waits at the real idle pose; follow: copy the source robot")
    p.add_argument("--orient", choices=["foldnet", "prior"], default="prior",
                   help="hand rotation target: FoldNet++'s, or the real SG2 hands' at that position")
    p.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 4))
    p.add_argument("--out", type=Path, default=Path("outputs/foldnetxx/t1.json"))
    args = p.parse_args()

    report = {}
    with multiprocessing.Pool(args.workers) as pool:
        for d in args.robot_dirs:
            eps = sorted(d.rglob("*.h5"))[: args.max_episodes]
            res = pool.map(run_episode, [(str(e), args.base_y, args.height, i, args.ik, args.orient, args.idle)
                                         for i, e in enumerate(eps)])
            report[d.name] = {"summary": summarize(res), "episodes": [x for x in res if x]}
            s = report[d.name]["summary"]
            if s:
                print(f"{d.name:20s} eps {s['episodes']:3d} | step reach {100 * s['step_reach']:5.1f}% | "
                      f"episodes 100% {100 * s['episodes_fully_reached']:5.1f}%, >=95% {100 * s['episodes_reach_ge_95']:5.1f}% | "
                      f"speed p99 {s['speed_p99_median']:.2f} rad/s, steps > {REAL_MAX_SPEED} {100 * s['steps_over_real_speed']:.1f}% | "
                      f"arm collision {100 * s['arm_collision_episodes']:.0f}% of episodes | "
                      f"retime x{s['retime_median']:.2f}", flush=True)
                print("   hardest subtasks:", [(t[:50], f"{100 * a:.0f}% of {b}") for t, a, b in s["worst_subtasks"]], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
