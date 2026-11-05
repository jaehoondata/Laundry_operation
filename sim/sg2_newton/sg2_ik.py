"""Posture-keeping IK for one SG2 arm, for generating smooth 5 Hz joint trajectories.

Ideas taken from the robot's teleop IK (deploy VR_teleoperation/package/posture_ik.py, read only):
weighted pull toward a reference posture (twist joints J3/J5 and wrist J6/J7 held hardest), an
elbow-side limit relative to that reference, and a bounded per-step joint change. Unlike the plain
DLS in sg2_reach.py there are no random restarts, so the arm never jumps to another IK branch;
a pose it cannot reach smoothly is reported as a failure for the caller to replan.

Kinematics come from sg2_reach.ArmChain (matches Newton's FK of the URDF to <0.001 mm).

    python sim/sg2_newton/sg2_ik.py --map                          # reachable table workspace
    python sim/sg2_newton/sg2_ik.py --episodes path/to/galbot_h5_dir --tilt 35
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
from sg2_reach import DEFAULT_URDF, POS_TOL, ROT_TOL, ArmChain, arm_base_pose, load_episodes, rot_error  # noqa: E402,F401


# J1..J7; twist (J3, J5) and wrist (J6, J7) deviations cost the most
POSTURE_WEIGHTS = np.array([0.25, 0.25, 4.0, 0.25, 4.0, 2.5, 2.5])
# robot "idle" pose from VR_teleoperation/qpos/sg2_joint_qpos.json: elbow bent 90 deg
IDLE = np.array([0.0, 0.0, 0.0, -math.pi / 2, 0.0, 0.0, 0.0])


class PostureIK:
    def __init__(self, chain: ArmChain, q_ref=IDLE, max_step=0.5, elbow_limit=math.radians(85),
                 posture_gain=0.05, damping=0.05, rot_weight=0.3, iters=60):
        self.chain = chain
        self.q_ref = np.clip(np.asarray(q_ref, dtype=float), chain.lower, chain.upper)
        self.max_step, self.elbow_limit = max_step, elbow_limit
        self.posture_gain, self.damping, self.rot_weight, self.iters = posture_gain, damping, rot_weight, iters
        self.w_inv = 1.0 / POSTURE_WEIGHTS
        self._ref = None  # (reach axis, elbow side) of q_ref, in the arm-base frame

    @staticmethod
    def _elbow_geometry(frames):
        """(unit shoulder->wrist axis, unit elbow side off that axis); side is None near a straight arm."""
        shoulder, elbow, wrist = frames[1][0], frames[3][0], frames[5][0]
        axis = wrist - shoulder
        axis /= np.linalg.norm(axis) + 1e-12
        radial = (elbow - shoulder) - axis * ((elbow - shoulder) @ axis)
        r = np.linalg.norm(radial)
        return axis, (None if r < 0.02 else radial / r)

    @staticmethod
    def _rotation_between(a, b):
        """Smallest rotation taking unit vector a onto unit vector b (None if opposite)."""
        v, c = np.cross(a, b), float(a @ b)
        if c < -0.999:
            return None
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + k + k @ k / (1 + c)

    def elbow_angle(self, q, base) -> float:
        """Elbow-side deviation from the reference, after carrying the reference along with the reach.

        Reaching further rotates the shoulder-wrist axis and the elbow side with it; comparing raw side
        vectors would reject ordinary far reaches, so the reference side is first rotated by the
        minimal rotation that maps the reference reach axis onto the current one.
        """
        if self._ref is None:
            _, frames = self.chain.fk(self.q_ref, np.eye(4))
            self._ref = self._elbow_geometry(frames)
        ref_axis, ref_side = self._ref
        _, frames = self.chain.fk(q, np.eye(4))
        axis, side = self._elbow_geometry(frames)
        rot = self._rotation_between(ref_axis, axis)
        if side is None or ref_side is None or rot is None:
            return math.pi
        carried = rot @ ref_side
        return math.acos(max(-1.0, min(1.0, float(side @ carried))))

    def solve(self, base, target, q_prev):
        """Track `target` (4x4 world TCP pose) starting from q_prev. Returns (q, ok, pos_err, rot_err)."""
        c = self.chain
        q = q_prev.copy()
        lo = np.maximum(c.lower, q_prev - self.max_step)
        hi = np.minimum(c.upper, q_prev + self.max_step)
        for _ in range(self.iters):
            tcp, frames = c.fk(q, base)
            ep = target[:3, 3] - tcp[:3, 3]
            er = rot_error(target[:3, :3], tcp[:3, :3])
            if np.linalg.norm(ep) < POS_TOL and np.linalg.norm(er) < ROT_TOL:
                break
            jac = c.jacobian(tcp, frames)
            jac[3:] *= self.rot_weight
            err = np.concatenate([ep, er * self.rot_weight])
            # weighted damped least squares, posture pull in the task null space. Joints pinned at a
            # bound and pushed outward are frozen and the step recomputed, so the others take over
            # (the idle pose sits exactly on the J2 limit; plain clipping stalled there).
            w_inv = self.w_inv.copy()
            for _pass in range(3):
                jw = jac * w_inv
                inv = np.linalg.inv(jac @ jw.T + self.damping**2 * np.eye(6))
                dq = jw.T @ inv @ err
                null = np.eye(7) - jw.T @ inv @ jac
                dq += (null @ (self.posture_gain * (self.q_ref - q))) * (w_inv > 0)
                blocked = ((q <= lo + 1e-9) & (dq < 0)) | ((q >= hi - 1e-9) & (dq > 0))
                if not (blocked & (w_inv > 0)).any():
                    break
                w_inv[blocked] = 0.0
            q = np.clip(q + dq, lo, hi)
        tcp, _ = c.fk(q, base)
        pe = float(np.linalg.norm(target[:3, 3] - tcp[:3, 3]))
        re = float(np.linalg.norm(rot_error(target[:3, :3], tcp[:3, :3])))
        self.last_elbow = self.elbow_angle(q, base)
        ok = pe < POS_TOL and re < ROT_TOL and self.last_elbow < self.elbow_limit
        return q, ok, pe, re


def sg2_grasp_rotation(tilt_deg: float, heading=(0.0, 1.0)) -> np.ndarray:
    """TCP orientation real SG2 teleop uses near the table: approach forward (+y world, the robot
    faces +y) tilted down by tilt_deg, TCP y (finger closing axis) toward the robot's left (world -x).
    Recorded teleop (Oct 2026) near the table: 32-42 deg median tilt, TCP y along the robot's left;
    FoldNet++ Galbot data is nearly horizontal (~13 deg).
    heading (world xy) turns the approach about the vertical, e.g. from the shoulder toward the target.
    """
    t = math.radians(tilt_deg)
    hx, hy = np.asarray(heading, dtype=float) / np.linalg.norm(heading)
    z = -np.array([hx * math.cos(t), hy * math.cos(t), -math.sin(t)])  # TCP z = -approach
    y = np.array([-hy, hx, 0.0])  # heading turned +90 deg: the robot's left for a forward reach
    x = np.cross(y, z)
    return np.stack([x / np.linalg.norm(x), y, z], axis=1)


SHOULDER_X = {"l": -0.2275, "r": 0.2275}  # world x of each shoulder (J1->J2 offset, robot faces +y)


def workspace_map(args):
    """Print, per arm and height, which table points the posture IK reaches from idle with the
    teleop-style grasp (tilted --tilt deg down, heading from the shoulder toward the point)."""
    base = arm_base_pose(args.base_y, args.height)
    xs = np.round(np.arange(-0.4, 0.41, 0.1), 2)
    ys = np.round(np.arange(-0.3, 0.41, 0.1), 2)
    for side in "lr":
        ik = PostureIK(ArmChain(args.urdf, side), max_step=10.0, iters=300)
        n_ok = 0
        print(f"== {side} arm (arm base at y={args.base_y}, {args.height} m above table), P = reached")
        for z in (0.03, 0.15, 0.3):
            print(f"  z={z:.2f}  x: " + " ".join(f"{x:+.1f}" for x in xs))
            for y in ys[::-1]:
                row = ""
                for x in xs:
                    rot = sg2_grasp_rotation(args.tilt or 35.0, (x - SHOULDER_X[side], y - args.base_y))
                    target = np.eye(4)
                    target[:3, :3], target[:3, 3] = rot, (x, y, z)
                    ok = ik.solve(base, target, ik.q_ref)[1]
                    n_ok += ok
                    row += "  P  " if ok else "  .  "
                print(f"   y={y:+.1f}{row}")
        print(f"  {side}: {n_ok}/{3 * len(xs) * len(ys)} reached")


def evaluate(args):
    episodes = load_episodes(args.episodes, args.n_episodes, 1)
    if args.tilt is not None:  # keep FoldNet++ positions, use the SG2 teleop grasp orientation
        rot = sg2_grasp_rotation(args.tilt)
        for ep in episodes:
            for poses in ep["poses"].values():
                poses[:, :3, :3] = rot
    base = arm_base_pose(args.base_y, args.height)
    stats = {}
    for side in "lr":
        ik = PostureIK(ArmChain(args.urdf, side), max_step=args.max_step)
        reached = total = 0
        speeds, fails_by_sub, why = [], {}, {"pose": 0, "elbow": 0}
        for ep in episodes:
            poses = ep["poses"][side]
            # approach the first pose from idle without the per-step bound
            q, ok, _, _ = PostureIK(ik.chain, max_step=10.0, iters=300).solve(base, poses[0], ik.q_ref)
            for k, target in enumerate(poses):
                q_new, ok, pe, re = ik.solve(base, target, q)
                total += 1
                if k:
                    speeds.append(float(np.abs(q_new - q).max() / 0.2))
                if ok:
                    reached += 1
                else:
                    sub = ep["subtasks"][k][:45]
                    fails_by_sub[sub] = fails_by_sub.get(sub, 0) + 1
                    why["pose" if (pe >= POS_TOL or re >= ROT_TOL) else "elbow"] += 1
                q = q_new  # a real arm still moves toward the target when it cannot reach it
        sp = np.array(speeds or [0.0])
        stats[side] = (reached, total, float(np.percentile(sp, 99)), float(sp.max()), int((sp > args.speed_limit).sum()),
                       why, sorted(fails_by_sub.items(), key=lambda kv: -kv[1])[:4])
    for side, (r, t, p99, mx, over, why, fails) in stats.items():
        print(f"{side}: tracked {100 * r / t:.1f}% ({r}/{t}) fail causes {why} | joint speed p99 {p99:.2f} "
              f"max {mx:.2f} rad/s, over {args.speed_limit} rad/s: {over} | top failures {fails}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, help="FoldNet++ episode .h5 dir (tracking evaluation)")
    p.add_argument("--map", action="store_true", help="print the reachable table workspace instead")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--base-y", type=float, default=-0.45)
    p.add_argument("--height", type=float, default=0.55, help="arm_base_link above the table top [m]")
    p.add_argument("--max-step", type=float, default=0.5, help="max joint change per 0.2 s action step [rad]")
    p.add_argument("--speed-limit", type=float, default=4.8)
    p.add_argument("--tilt", type=float, default=None,
                   help="replace data orientations with SG2 teleop-style grasp tilted down this many degrees")
    args = p.parse_args()
    if args.map:
        workspace_map(args)
    elif args.episodes:
        evaluate(args)
    else:
        p.error("give --episodes or --map")


if __name__ == "__main__":
    main()
