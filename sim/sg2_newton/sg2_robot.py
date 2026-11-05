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
# where an arm waits during the other arm's one-hand move: out on its own side, in front of the
# torso (|x| from the center, y ahead of arm_base_link, z above the table); IK error ~4 mm
PARK = (0.35, 0.35, 0.18)


class SG2Arms:
    def __init__(self, base_y=-0.45, height=0.55, tilt=35.0, max_step=0.5, urdf=DEFAULT_URDF):
        self.base = arm_base_pose(base_y, height)
        self.base_y, self.tilt = base_y, tilt
        self.iks = {s: PostureIK(ArmChain(urdf, s), max_step=max_step) for s in SIDES}
        self.q = {s: IDLE.copy() for s in SIDES}
        self.ready = False  # first approach jumps from idle without the per-step joint bound
        self.traj = []  # (phase, {side: q}) per 0.2 s step, for the dataset's joint actions

    def target(self, side, pos):
        heading = (pos[0] - SHOULDER_X[side], pos[1] - self.base_y)
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = sg2_grasp_rotation(self.tilt, heading), pos
        return t

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
        for p in path:
            q, ok, _, _ = ik.solve(self.base, self.target(side, p), q)
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

    def approach(self, grasp_points, order, phase, hover=0.05, step=0.03, transit=0.12):
        """Move each arm (cloth untouched) from its TCP up to `transit` above the table, over its grasp
        point and down onto it. In a one-hand move the idle arm goes up and out to its park pose,
        clear of the working arm (the IK has no collision term)."""
        if not self.ready:  # episode start: put the arms in a ready pose above their first grasp
            for k, side in enumerate(order):
                if np.isnan(grasp_points[k]).any():
                    continue
                above = self.target(side, grasp_points[k] + [0.0, 0.0, hover])
                self.q[side] = PostureIK(self.iks[side].chain, max_step=10.0, iters=300).solve(
                    self.base, above, self.iks[side].q_ref)[0]
            self.ready = True
        def polyline(points):
            out = [points[0][None]]
            for a, b in zip(points[:-1], points[1:]):
                n = max(1, math.ceil(np.linalg.norm(b - a) / step))
                out.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
            return np.vstack(out)

        paths = []
        for k, side in enumerate(order):
            start = self.tcp(side)
            if np.isnan(grasp_points[k]).any():  # idle arm in a one-hand move: park it
                goal = self.park(side)
                if np.linalg.norm(goal - start) < 0.01:
                    paths.append(np.full((2, 3), np.nan))
                    continue
                z = max(start[2], goal[2])
                paths.append(polyline([start, np.array([*start[:2], z]), goal]))
                continue
            goal = grasp_points[k]
            z = max(start[2], goal[2] + hover, transit)
            paths.append(polyline([start, np.array([*start[:2], z]), np.array([*goal[:2], z]), goal]))
        n = max(len(p) for p in paths)
        wps = np.stack([np.vstack([p, np.repeat(p[-1:], n - len(p), axis=0)]) for p in paths], axis=1)
        return self.realize(wps, phase, order)

    def joint_trajectory(self):
        """(phases [T], q [T, 2, 7] in l, r order) for the dataset."""
        phases = [p for p, _ in self.traj]
        q = np.array([[qd.get(s, np.full(7, np.nan)) for s in SIDES] for _, qd in self.traj])
        return phases, q
