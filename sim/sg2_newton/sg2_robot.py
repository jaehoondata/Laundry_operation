"""Kinematic SG2 that turns picker waypoints into joint trajectories and the TCP positions it reaches.

The cloth never touches the arm (FoldNet++ attachment grasp); the robot only decides where the
pickers actually go. Each 0.2 s waypoint is solved with PostureIK using the teleop-style grasp
(heading from the shoulder toward the point, tilted down), so unreachable or too-fast targets show up
as TCP tracking error instead of being silently followed.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
from sg2_ik import DEFAULT_URDF, IDLE, SHOULDER_X, PostureIK, sg2_grasp_rotation  # noqa: E402
from sg2_reach import ArmChain, arm_base_pose  # noqa: E402

SIDES = ("l", "r")
# where an arm waits during the other arm's one-hand move (|x| from the center, y ahead of
# arm_base_link, z above the table): the median hand of the idle arm in the teleop data
# (laundry_fold_tshirt_arm16, while the other hand holds cloth: l (-0.25, 0.04, 0.17), r (0.31, 0.03, 0.18)
# with the arm base 0.45 m behind the cloth center)
PARK = (0.30, 0.48, 0.17)


def min_jerk(points, step):
    """Resample a dense path [M, 3] into 0.2 s waypoints that start and stop smoothly (minimum-jerk
    timing along the arc length): same mean speed as `step` per waypoint, peak 1.875x. Teleop hands
    accelerate and slow down like this (real moving-hand speed p10/50/90 0.03/0.07/0.16 m/s; a
    constant speed gave 0.05/0.07/0.09)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    n = max(1, math.ceil(s[-1] / step))
    tau = np.linspace(0.0, 1.0, n + 1)
    targets = s[-1] * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    return np.stack([np.interp(targets, s, points[:, k]) for k in range(3)], axis=1)


class SG2Arms:
    def __init__(self, base_y=-0.45, height=0.55, tilt=35.0, max_step=0.5, urdf=DEFAULT_URDF, prior=None,
                 start_idle=False):
        self.base = arm_base_pose(base_y, height)
        self.base_y, self.tilt = base_y, tilt
        self.iks = {s: PostureIK(ArmChain(urdf, s), max_step=max_step) for s in SIDES}
        # hand orientation + reference posture from the real teleop data (posture_prior.py); None keeps
        # the fixed tilted grasp with the idle pose as reference
        self.priors = None
        if prior is not None:
            from posture_prior import PosturePrior

            self.priors = {s: PosturePrior(self.iks[s].chain, self.base, s, prior) for s in SIDES}
        self.q = {s: IDLE.copy() for s in SIDES}
        # False: the first approach jumps from idle to above the first grasp without the per-step joint
        # bound; True: the arms start in the idle pose and move in from there (every real episode
        # starts exactly at idle, 8.2 s median before the first grasp)
        self.ready = start_idle
        self.traj = []  # (phase, {side: q}) per 0.2 s step, for the dataset's joint actions

    def target(self, side, pos):
        """(4x4 TCP target, IK reference posture) for a hand position."""
        t = np.eye(4)
        t[:3, 3] = pos
        if self.priors is not None:
            t[:3, :3], q_ref, _ = self.priors[side](pos)
            return t, q_ref
        heading = (pos[0] - SHOULDER_X[side], pos[1] - self.base_y)
        t[:3, :3] = sg2_grasp_rotation(self.tilt, heading)
        return t, IDLE

    def solve(self, ik, side, pos, q, target=None):
        target, q_ref = self.target(side, pos) if target is None else target
        ik.q_ref = np.clip(np.asarray(q_ref, dtype=float), ik.chain.lower, ik.chain.upper)
        ik._ref = None  # the elbow-side reference follows the posture reference
        return ik.solve(self.base, target, q)

    def path_targets(self, side, path, sigma_steps=3.0):
        """Targets along a path with the prior's orientation and reference posture low-passed over
        the waypoints (Gaussian, sigma in 0.2 s steps): raw nearest-neighbour lookups change from
        step to step and made the wrist jitter up to 2 rad/s (real joints stay under 0.84 rad/s)."""
        raw = [self.target(side, p) for p in path]
        if self.priors is None or len(path) < 3:
            return raw
        rots = np.array([t[:3, :3] for t, _ in raw])
        refs = np.array([r for _, r in raw])
        k = np.arange(len(path))
        out = []
        for i in k:
            w = np.exp(-0.5 * ((k - i) / sigma_steps) ** 2)
            w /= w.sum()
            u, _, vt = np.linalg.svd(np.einsum("n,nij->ij", w, rots))
            t = raw[i][0].copy()
            t[:3, :3] = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
            out.append((t, w @ refs))
        return out

    def tcp(self, side, q=None):
        return self.iks[side].chain.fk(self.q[side] if q is None else q, self.base)[0][:3, 3]

    def _track(self, side, path, q0):
        """Solve a position path [N, 3] from q0. Returns joint path, reached TCP path, ok flags.
        A NaN path (one-hand move, this arm idle) holds q0."""
        ik, q = self.iks[side], q0.copy()
        if np.isnan(path).all():
            hold = ik.chain.fk(q0, self.base)[0][:3, 3]
            return np.repeat(q0[None], len(path), 0), np.repeat(hold[None], len(path), 0), np.ones(len(path), bool)
        qs, reached, oks = [], [], []
        for p, target in zip(path, self.path_targets(side, path)):
            q, ok, _, _ = self.solve(ik, side, p, q, target)
            qs.append(q)
            reached.append(ik.chain.fk(q, self.base)[0][:3, 3])
            oks.append(ok)
        return np.array(qs), np.array(reached), np.array(oks)

    def assign(self, waypoints):
        """Picker->arm mapping for waypoints [N, 2, 3]: try both, keep the smaller total tracking error."""
        best = None
        for order in (("l", "r"), ("r", "l")):
            err = 0.0
            for k, side in enumerate(order):
                if np.isnan(waypoints[:, k]).all():
                    continue
                _, reached, _ = self._track(side, waypoints[:, k], self.q[side])
                err += float(np.linalg.norm(reached - waypoints[:, k], axis=1).sum())
            if best is None or err < best[0]:
                best = (err, order)
        return best[1]

    def realize(self, waypoints, phase, order=None):
        """Move both arms along waypoints [N, 2, 3]; commit the joint state. Returns reached TCP
        positions [N, 2, 3] (picker k follows arm order[k]), the order, per-arm stats, and the
        joint path [N, 2, 7] in (l, r) order."""
        order = order or self.assign(waypoints)
        reached = np.zeros_like(waypoints)
        stats = {}
        for k, side in enumerate(order):
            qs, rp, oks = self._track(side, waypoints[:, k], self.q[side])
            reached[:, k] = rp
            prev = np.vstack([self.q[side], qs[:-1]])
            idle = np.isnan(waypoints[:, k]).all()
            stats[side] = {
                "ik_ok": float(oks.mean()),
                "track_err_max": 0.0 if idle else float(np.linalg.norm(rp - waypoints[:, k], axis=1).max()),
                "joint_speed_max": float(np.abs(qs - prev).max() / 0.2),
            }
            self.q[side] = qs[-1]
            stats[side]["_qs"] = qs
        for i in range(len(waypoints)):
            self.traj.append((phase, {s: stats[s]["_qs"][i] for s in order}))
        joints = np.stack([stats[s]["_qs"] for s in SIDES], axis=1)  # [N, 2, 7] in (l, r) order
        for s in order:
            stats[s].pop("_qs")
        return reached, order, stats, joints

    def park(self, side):
        return np.array([PARK[0] * (-1 if side == "l" else 1), self.base_y + PARK[1], PARK[2]])

    def approach(self, grasp_points, order, phase, hover=0.05, step=0.03, transit=0.12, hold=False, wait_park=True):
        """Move each arm (cloth untouched) from its TCP up to `transit` above the table, over its grasp
        point and down onto it. In a one-hand move the idle arm goes up and out to its park pose,
        clear of the working arm (the IK has no collision term); with `hold` it stays where it is
        (e.g. a hand already holding cloth while the other one comes in). `wait_park` holds the
        working arm until the idle one has parked; teleop operators move both at once."""
        if not self.ready:  # episode start: put the arms in a ready pose above their first grasp
            for k, side in enumerate(order):
                if np.isnan(grasp_points[k]).any():
                    continue
                pos = grasp_points[k] + [0.0, 0.0, hover]
                ik = PostureIK(self.iks[side].chain, max_step=10.0, iters=300)
                self.q[side] = self.solve(ik, side, pos, self.target(side, pos)[1])[0]
            self.ready = True
        def polyline(points):
            out = [points[0][None]]
            for a, b in zip(points[:-1], points[1:]):
                n = max(1, math.ceil(np.linalg.norm(b - a) / 0.002))
                out.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
            return min_jerk(np.vstack(out), step)

        paths = []
        for k, side in enumerate(order):
            start = self.tcp(side)
            if np.isnan(grasp_points[k]).any():  # idle arm in a one-hand move: park it
                goal = self.park(side)
                if hold or np.linalg.norm(goal - start) < 0.01:
                    paths.append(np.full((2, 3), np.nan))
                    continue
                z = max(start[2], goal[2])
                paths.append(polyline([start, np.array([*start[:2], z]), goal]))
                continue
            goal = grasp_points[k]
            z = max(start[2], goal[2] + hover, transit)
            paths.append(polyline([start, np.array([*start[:2], z]), np.array([*goal[:2], z]), goal]))
        # one-hand move: the working arm waits until the idle arm has parked (moving both at once,
        # from hands 6 cm apart after fold3, swept one wrist through the other)
        parking = [k for k in range(len(order)) if np.isnan(grasp_points[k]).any() and not np.isnan(paths[k]).all()]
        if parking and wait_park:
            wait = len(paths[parking[0]]) - 1
            for k in range(len(order)):
                if k not in parking and not np.isnan(paths[k]).all():
                    paths[k] = np.vstack([np.repeat(paths[k][:1], wait, axis=0), paths[k]])
        n = max(len(p) for p in paths)
        wps = np.stack([np.vstack([p, np.repeat(p[-1:], n - len(p), axis=0)]) for p in paths], axis=1)
        return self.realize(wps, phase, order)

    def joint_trajectory(self):
        """(phases [T], q [T, 2, 7] in l, r order) for the dataset."""
        phases = [p for p, _ in self.traj]
        q = np.array([[qd.get(s, np.full(7, np.nan)) for s in SIDES] for _, qd in self.traj])
        return phases, q
