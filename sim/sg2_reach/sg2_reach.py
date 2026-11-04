"""Can SG2 follow FoldNet++ end-effector trajectories? IK reachability over base/lift placements.

FoldNet++ stores both EEFs as world-frame poses (table top at z=0, robot on the -y side,
facing +y). The FoldNet++ EEF frame approaches along -z and its picker box is thinnest in y,
which matches SG2's end_effector_{l,r}_link (fingers close along y, approach -z), so the
two frames are used as-is.

For each placement (base y, arm_base height above the table) every sampled EEF pose is
solved with damped-least-squares IK, warm-started like FoldNet++:
q_init = 0.95 * q_prev + 0.05 * q0. A pose counts as reached when position error < 5 mm
and rotation error < 3 deg.

    python sim/sg2_reach/sg2_reach.py --episodes path/to/galbot_h5_dir
"""

import argparse
import json
import math
import multiprocessing
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

DEFAULT_URDF = Path.home() / (
    "deploy/VAI_AIWORKER/ros2_ws/install/ffw_description/share/ffw_description/urdf/"
    "ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
)
POS_TOL = 0.005
ROT_TOL = math.radians(3.0)
LIFT_ORIGIN_Z = None  # filled from the URDF (lift_joint origin z)


# ----------------------------------------------------------------------------- kinematics


def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle(axis, angle):
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def homogeneous(rot, pos):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, pos
    return t


class ArmChain:
    """arm_base_link -> arm_{side}_joint1..7 -> end_effector_{side}_link."""

    def __init__(self, urdf: Path, side: str):
        joints = {}
        for j in ET.parse(urdf).getroot().findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            a, lim = j.find("axis"), j.find("limit")
            joints[j.get("name")] = {
                "origin": homogeneous(rpy_to_matrix(*rpy), xyz),
                "axis": np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None,
                "limit": (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None and lim.get("lower") else None,
                "velocity": float(lim.get("velocity")) if lim is not None and lim.get("velocity") else None,
            }
        global LIFT_ORIGIN_Z
        LIFT_ORIGIN_Z = joints["lift_joint"]["origin"][2, 3]
        self.lift_limit = joints["lift_joint"]["limit"]
        self.names = [f"arm_{side}_joint{i}" for i in range(1, 8)]
        self.origins = [joints[n]["origin"] for n in self.names]
        self.axes = [joints[n]["axis"] for n in self.names]
        self.lower = np.array([joints[n]["limit"][0] for n in self.names])
        self.upper = np.array([joints[n]["limit"][1] for n in self.names])
        self.vel_limit = np.array([joints[n]["velocity"] for n in self.names])
        self.tcp = joints[f"end_effector_{side}_joint"]["origin"]

    def fk(self, q, base: np.ndarray):
        """base = world pose of arm_base_link. Returns tcp pose and per-joint (origin, axis) in world."""
        t = base.copy()
        frames = []
        for origin, axis, qi in zip(self.origins, self.axes, q):
            t = t @ origin
            frames.append((t[:3, 3].copy(), t[:3, :3] @ axis))
            t = t @ homogeneous(axis_angle(axis, qi), [0, 0, 0])
        return t @ self.tcp, frames

    def jacobian(self, tcp, frames):
        p = tcp[:3, 3]
        jac = np.zeros((6, 7))
        for i, (o, a) in enumerate(frames):
            jac[:3, i] = np.cross(a, p - o)
            jac[3:, i] = a
        return jac


def rot_error(target, current):
    """Axis-angle vector rotating current onto target (world frame)."""
    r = target @ current.T
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    s = math.sin(angle)
    if s < 1e-6:  # angle ~ pi
        w, vecs = np.linalg.eigh((r + r.T) / 2)
        return vecs[:, -1] * angle
    return v / (2 * s) * angle


def solve_ik(chain: ArmChain, base, target, q_init, iters=60, damping=0.05, rot_weight=0.3):
    q = np.clip(q_init.copy(), chain.lower, chain.upper)
    for _ in range(iters):
        tcp, frames = chain.fk(q, base)
        ep = target[:3, 3] - tcp[:3, 3]
        er = rot_error(target[:3, :3], tcp[:3, :3])
        if np.linalg.norm(ep) < POS_TOL and np.linalg.norm(er) < ROT_TOL:
            return q, True, np.linalg.norm(ep), np.linalg.norm(er)
        jac = chain.jacobian(tcp, frames)
        jac[3:] *= rot_weight
        err = np.concatenate([ep, er * rot_weight])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
        q = np.clip(q + dq, chain.lower, chain.upper)
    tcp, _ = chain.fk(q, base)
    ep = np.linalg.norm(target[:3, 3] - tcp[:3, 3])
    er = np.linalg.norm(rot_error(target[:3, :3], tcp[:3, :3]))
    return q, (ep < POS_TOL and er < ROT_TOL), ep, er


# ----------------------------------------------------------------------------- data


def six_to_matrix(six):
    # FoldNet++ stores R[:3, :2].flatten() (row-major), not the two columns concatenated
    c = six.reshape(3, 2)
    x = c[:, 0] / np.linalg.norm(c[:, 0])
    y = c[:, 1] - x * (x @ c[:, 1])
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def load_episodes(ep_dir: Path, n: int, stride: int):
    eps = []
    for p in sorted(ep_dir.glob("*.h5"))[:n]:
        with h5py.File(p, "r") as f:
            a = f["proprios_eef"][::stride]
            sub = [s.decode() for s in f["subtasks"][::stride]]
        poses = {}
        for side, off in (("l", 0), ("r", 9)):
            poses[side] = np.stack([homogeneous(six_to_matrix(r[off + 3:off + 9]), r[off:off + 3]) for r in a])
        eps.append({"name": p.stem, "poses": poses, "subtasks": sub})
    return eps


# ----------------------------------------------------------------------------- sweep


def arm_base_pose(base_y: float, height: float) -> np.ndarray:
    """SG2 faces +y (yaw 90 deg); arm_base_link sits `height` above the table top."""
    return homogeneous(rpy_to_matrix(0, 0, math.pi / 2), [0.0, base_y, height])


def home_posture(chain: ArmChain, base, target, rng, restarts=30):
    best, best_err = None, np.inf
    for _ in range(restarts):
        q0 = rng.uniform(chain.lower, chain.upper)
        q, ok, ep, er = solve_ik(chain, base, target, q0, iters=150)
        if ep + 0.3 * er < best_err:
            best, best_err = q, ep + 0.3 * er
    return best


def evaluate(chains, episodes, base, rng, step_dt, restarts=5):
    """tracked = solved from the FoldNet++ warm start; reached = solvable at all (warm or restarts).
    A gap between the two means the pose is in the workspace but the trajectory breaks there.
    speeds = per-step max joint speed between consecutive reached poses [rad/s]."""
    stats = {s: {"tracked": 0, "reached": 0, "total": 0, "speeds": [], "by_subtask": {}} for s in chains}
    for ep in episodes:
        for side, chain in chains.items():
            poses = ep["poses"][side]
            q0 = home_posture(chain, base, poses[0], rng)
            q_prev = q0
            for k, target in enumerate(poses):
                st = stats[side]
                st["total"] += 1
                bs = st["by_subtask"].setdefault(ep["subtasks"][k], [0, 0])
                bs[1] += 1
                q, ok, _, _ = solve_ik(chain, base, target, 0.95 * q_prev + 0.05 * q0)
                if ok:
                    st["tracked"] += 1
                else:
                    for _ in range(restarts):
                        q, ok, _, _ = solve_ik(chain, base, target, rng.uniform(chain.lower, chain.upper), iters=150)
                        if ok:
                            break
                if ok:
                    st["reached"] += 1
                    bs[0] += 1
                    if k:
                        st["speeds"].append(float(np.abs(q - q_prev).max() / step_dt))
                    q_prev = q
    return stats


def run_cell(cell):
    """One (base_y, height) placement; module-level so multiprocessing can pickle it."""
    args, by, h = cell
    chains = {s: ArmChain(args.urdf, s) for s in ("l", "r")}
    episodes = load_episodes(args.episodes, args.n_episodes, args.stride)
    rng = np.random.default_rng(args.seed)
    st = evaluate(chains, episodes, arm_base_pose(by, h), rng, args.stride * 0.2)
    lift = h + args.table_height - LIFT_ORIGIN_Z
    for s in st.values():
        sp = np.array(s.pop("speeds") or [0.0])
        s["speed_p99"] = float(np.percentile(sp, 99))
        s["over_speed"] = int((sp > args.speed_limit).sum())
    return {"base_y": by, "height": h, "lift": lift,
            "lift_in_range": bool(chains["l"].lift_limit[0] <= lift <= chains["l"].lift_limit[1]), "stats": st}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, required=True, help="dir of FoldNet++ episode .h5 files")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--stride", type=int, default=2, help="use every k-th action step")
    p.add_argument("--base-y", type=float, nargs="+", default=[-0.9, -0.8, -0.7, -0.6, -0.5])
    p.add_argument("--height", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   help="arm_base_link height above the table top [m]")
    p.add_argument("--table-height", type=float, default=0.86,
                   help="only used to report the lift value (lab table model, approximate)")
    p.add_argument("--speed-limit", type=float, default=4.8, help="joint speed limit [rad/s] (URDF)")
    p.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    p.add_argument("--out", type=Path, default=Path("outputs/sg2_reach/sweep.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    chains = {s: ArmChain(args.urdf, s) for s in ("l", "r")}
    episodes = load_episodes(args.episodes, args.n_episodes, args.stride)
    n_poses = sum(len(e["poses"]["l"]) for e in episodes)
    print(f"{len(episodes)} episodes, {n_poses} poses per arm, lift range {chains['l'].lift_limit}, "
          f"lift origin {LIFT_ORIGIN_Z:.3f} m above base_link")

    cells = [(args, by, h) for by in args.base_y for h in args.height]
    print(f"{'base_y':>7} {'height':>7} {'lift':>6} {'reach L%':>9} {'reach R%':>9} {'track L%':>9} "
          f"{'track R%':>9} {'p99 rad/s':>10} {'>limit':>7}")
    with multiprocessing.Pool(args.workers) as pool:
        results = []
        for r in pool.imap(run_cell, cells):
            st = r["stats"]
            pct = {f"{key}_{s}": 100 * st[s][key] / st[s]["total"] for key in ("reached", "tracked") for s in "lr"}
            print(f"{r['base_y']:>7.2f} {r['height']:>7.2f} {r['lift']:>5.2f}{' ' if r['lift_in_range'] else '!'} "
                  f"{pct['reached_l']:>9.1f} {pct['reached_r']:>9.1f} {pct['tracked_l']:>9.1f} {pct['tracked_r']:>9.1f} "
                  f"{max(st['l']['speed_p99'], st['r']['speed_p99']):>10.2f} "
                  f"{st['l']['over_speed'] + st['r']['over_speed']:>7}", flush=True)
            results.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": {k: str(v) for k, v in vars(args).items()}, "results": results}, indent=1))
    best = max(results, key=lambda r: min(r["stats"]["l"]["reached"] / r["stats"]["l"]["total"],
                                          r["stats"]["r"]["reached"] / r["stats"]["r"]["total"]))
    print(f"\nbest: base_y={best['base_y']} height={best['height']} (lift {best['lift']:.2f} at table "
          f"{args.table_height} m) -> {args.out}")
    for side in ("l", "r"):
        worst = sorted(best["stats"][side]["by_subtask"].items(), key=lambda kv: kv[1][0] / kv[1][1])[:5]
        print(f"  {side} hardest subtasks:", [(s[:45], f"{a}/{b}") for s, (a, b) in worst])


if __name__ == "__main__":
    main()
