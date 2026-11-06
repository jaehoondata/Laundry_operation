"""Folding gate test: does the Style3D cloth keep multi-layer folds through a FoldNet++-style sequence?

Starts from a flat shirt with the collar along -x or +x, robot on the -y side. The sequence follows
the FoldNet++ folding rules as described in the paper (written from the description, not from
FoldNet code):
  fold1: near (-y) side onto the centerline     drag1: pull the stack toward the robot
  fold2: far (+y) side onto the centerline      drag2: push it back
  fold3: collar half onto the hem half           flip:  turn the stack over its far x edge
  drag3: slide it sideways
Grasps reuse the gate test's attachment pickers. Trials cycle through the four FoldNet++ initial
configurations: collar left/right x front/back face up.
--sequence teleop instead folds the way the SG2 teleop demonstrations do (collar at +y, away from the
robot): the arms start in the idle pose; half (hem onto collar, both hands), side_l / side_r (each
side in toward the center, one hand each), half2 (left half over the right, left hand), sometimes
tidy (right hand slides the shirt aside), then both arms retract. Trials alternate front/back face up.
Heights, speeds, release points and arm postures come from the real data (see the --help groups).

Per step: strain, table penetration, footprint (coverage of the flat shirt), placement error of the
folded keypoints, and spring-back (footprint growth between release and settling = a fold reopening).

    python sim/style3d_gate/fold_test.py --out outputs/style3d_gate/fold
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from gate_fling import (
    ACTION_DT,
    ClothSim,
    Diverged,
    health,
    linear_path,
    load_mesh,
    make_parser,
    per_triangle_panel,
    projected_area,
    quat_rotate,
    render_video,
    sync_paths,
    unique_edges,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_newton"))

CONFIGS = [(-1, False), (1, False), (-1, True), (1, True)]  # (collar x direction, back face up)
# language labels per step, in the style of the FoldNet++ subtask annotations
SUBTASKS = {
    "fold1": "Fold the half of the T-shirt near the robot onto the centerline.",
    "drag1": "Pull the T-shirt toward the robot.",
    "fold2": "Fold the far half of the T-shirt onto the centerline.",
    "drag2": "Push the T-shirt back to the center.",
    "fold3": "Fold the collar half of the T-shirt onto the hem half.",
    "flip": "Lift the folded edge and flip the T-shirt over.",
    "drag3": "Slide the folded T-shirt aside.",
    "half": "Fold the hem of the T-shirt up onto the collar with both hands.",
    "side_l": "Fold the left side of the T-shirt in toward the center.",
    "side_r": "Fold the right side of the T-shirt in toward the center.",
    "half2": "Fold the left half of the T-shirt over the right half.",
    "tidy": "Slide the folded T-shirt to the right.",
    "sleeve_l": "Fold the left sleeve in.",
    "sleeve_r": "Fold the right sleeve in.",
    "quarter_l": "Fold the left side of the T-shirt to the center.",
    "quarter_r": "Fold the right side of the T-shirt to the center.",
    "final": "Fold the T-shirt in half.",
}


# Hand paths of the real teleop folds (laundry_fold_tshirt_arm16, the 19 episodes that go LR-L-R-L;
# median over episodes, sampled at 11 evenly spaced times while the gripper is closed): progress =
# horizontal fraction of the way to the goal, z = TCP height [m], T = seconds of motion (closed time
# minus ~0.8 s of gripper travel). The hem fold lifts almost straight up first and only then swings
# forward; the last fold lifts in place for 60% of the time before crossing over.
REAL_PROFILES = {
    "half": {"progress": [0.0, 0.02, 0.19, 0.34, 0.34, 0.34, 0.46, 0.69, 0.85, 0.98, 1.0],
             "z": [0.053, 0.054, 0.116, 0.247, 0.369, 0.391, 0.311, 0.245, 0.2, 0.147, 0.117], "T": 11.3},
    "side_l": {"progress": [0.0, 0.0, 0.03, 0.19, 0.4, 0.61, 0.77, 0.87, 0.93, 0.98, 1.0],
               "z": [0.044, 0.048, 0.068, 0.121, 0.147, 0.166, 0.171, 0.156, 0.151, 0.129, 0.127], "T": 5.8},
    "side_r": {"progress": [0.0, 0.0, 0.0, 0.08, 0.36, 0.58, 0.7, 0.82, 0.91, 1.0, 1.0],
               "z": [0.031, 0.037, 0.068, 0.121, 0.155, 0.142, 0.126, 0.127, 0.112, 0.075, 0.079], "T": 6.2},
    "half2": {"progress": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47, 0.87, 0.87, 1.0],
              "z": [0.062, 0.07, 0.088, 0.102, 0.135, 0.166, 0.181, 0.179, 0.14, 0.104, 0.088], "T": 7.7},
}


def carry_profile(h: dict, start_t: float, post_close: float) -> dict:
    """A real hand's recorded path (real_steps.json entry) from start_t (s) to its release. In a
    two-hand step the hand that closed first mostly waits for the other one; the sim carries only
    once both hold, so its carry follows the part of the path after the second close and lasts as
    long as that part did."""
    hold = h["open_t"] - h["close_t"]
    knots = np.linspace(0.0, 1.0, len(h["z"]))
    frac = np.clip((start_t - h["close_t"]) / max(hold, 1e-6), 0.0, 0.95)
    ts = np.linspace(frac, 1.0, len(h["z"]))
    prog = np.interp(ts, knots, h["progress"])
    prog = (prog - prog[0]) / (1.0 - prog[0]) if prog[0] < 0.999 else np.linspace(0.0, 1.0, len(ts))
    return {"progress": prog.tolist(), "z": np.interp(ts, knots, h["z"]).tolist(),
            "T": max(1.0, h["open_t"] - max(start_t, h["close_t"]) - post_close - 0.3)}


def profile_path(start: np.ndarray, goal: np.ndarray, prof: dict, n_steps: int) -> np.ndarray:
    """Waypoints every action step that follow a real hand profile from start to goal: horizontal
    progress and height over time as recorded, the height shifted to start at the grasp and to end
    at the goal's release height."""
    from scipy.interpolate import PchipInterpolator

    knots = np.linspace(0.0, 1.0, len(prof["z"]))
    prog = PchipInterpolator(knots, np.maximum.accumulate(prof["progress"]))
    zr = PchipInterpolator(knots, prof["z"])
    t = np.linspace(0.0, 1.0, max(2, n_steps) + 1)
    f = prog(t)
    rel = zr(t) - prof["z"][0]
    end_fix = (goal[2] - start[2]) - (prof["z"][-1] - prof["z"][0])
    z = start[2] + rel + f * end_fix
    xy = start[:2] + f[:, None] * (goal[:2] - start[:2])
    return np.column_stack([xy, z])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def axis_quat(axis, angle):
    s = math.sin(angle / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)])


def arc_path(start: np.ndarray, goal: np.ndarray, lift_ratio: float, clearance: float, step: float,
             smooth: bool = False) -> np.ndarray:
    """Half-circle-like path from start to goal (fold motion), resampled every `step` metres
    (smooth: same mean spacing with minimum-jerk timing, as hands move)."""
    th = np.linspace(0.0, math.pi, 400)
    c = (start + goal) / 2
    u = start - c
    r = np.linalg.norm(u[:2])
    pts = c + np.outer(np.cos(th), u)
    # end heights may differ (placing onto a stack): blend them, then add the arc lift
    pts[:, 2] = start[2] + (goal[2] - start[2]) * (1 - np.cos(th)) / 2 + np.sin(th) * (lift_ratio * r + clearance)
    if smooth:
        from sg2_robot import min_jerk  # robot runs only (sg2_robot needs the SG2 URDF)

        return min_jerk(pts, step)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    targets = np.linspace(0, s[-1], max(1, math.ceil(s[-1] / step)) + 1)
    return np.stack([np.interp(targets, s, pts[:, k]) for k in range(3)], axis=1)


class FoldEpisode:
    def __init__(self, args, sim: ClothSim, faces, keypoints, flat_area, log):
        self.args, self.sim, self.faces, self.kp = args, sim, faces, keypoints
        self.flat_area, self.log = flat_area, log
        self.edges = unique_edges(faces)
        self.rest_len = np.linalg.norm(sim.rest[self.edges[:, 0]] - sim.rest[self.edges[:, 1]], axis=1)
        self.t_wall = time.time()
        # metres per 0.2 s action step; one speed factor per trial (log["speed_scale"])
        scale = log.get("speed_scale", 1.0)
        self.carry_step = args.carry_speed * scale * ACTION_DT
        self.approach_step = args.approach_speed * scale * ACTION_DT
        self.robot = None
        self.rng = np.random.default_rng(args.seed + 1000 + log["trial"])  # per-trial sequence choices
        if args.robot:  # pickers follow the TCP an SG2 IK actually reaches
            from sg2_robot import SG2Arms

            from posture_prior import PRIOR_PATH

            prior = {"none": None, "real": PRIOR_PATH}.get(args.posture_prior, args.posture_prior)
            self.robot = SG2Arms(base_y=args.robot_base_y, height=args.robot_height, tilt=args.grasp_tilt, prior=prior,
                                 start_idle=args.sequence in ("teleop", "neat"))

    def top(self, name: str) -> int:
        q = self.sim.q()
        return max(self.kp[name], key=lambda i: q[i, 2])

    def sleeve_cuff(self, hand: str) -> int:
        """Top-layer vertex at the middle of the sleeve cuff on the robot's `hand` side (l = -x)."""
        q = self.sim.q()
        side = sorted("lr", key=lambda k: q[self.top(f"{k}_armpit"), 0])["lr".index(hand)]
        tip = (q[self.top(f"{side}_sleeve_top")] + q[self.top(f"{side}_sleeve_bottom")]) / 2
        return self.top_near(tip[:2])

    def top_near(self, xy) -> int:
        """Top-most vertex within 2 cm (xy) of a point; nearest vertex if none."""
        q = self.sim.q()
        d = np.linalg.norm(q[:, :2] - np.asarray(xy), axis=1)
        near = np.nonzero(d < 0.02)[0]
        return int(near[np.argmax(q[near, 2])]) if len(near) else int(d.argmin())

    def surface_z(self, xy) -> float:
        """Height of the cloth (or table) top within 2 cm of a point."""
        q = self.sim.q()
        near = np.linalg.norm(q[:, :2] - np.asarray(xy), axis=1) < 0.02
        return float(q[near, 2].max()) if near.any() else 0.0

    def coverage(self) -> float:
        return projected_area(self.sim.q(), self.faces) / self.flat_area

    def bbox(self):
        q = self.sim.q()
        return q[:, :2].min(0), q[:, :2].max(0)

    def move(self, name: str, anchors: list[int], goals: list[np.ndarray], kind: str, check: list[str] | None = None,
             arms: tuple[str, str] | None = None, peak: float | None = None, rest_radius: float | None = None,
             release: float | None = None, first: str | None = None, profile=None, absolute: bool = False,
             close_at: dict | None = None):
        """Grasp the anchors, carry them to goals (arc = fold, line = drag), release, settle.
        An anchor/goal of None makes it a one-hand move: that picker (and arm) stays idle.
        `arms` fixes which arm takes each picker (default: the pair the IK tracks best); `peak` sets
        the arc's top height above the table instead of --fold-lift-ratio; `rest_radius` overrides
        --grasp-rest-radius (large = pinch every layer under the picker); `release` is the hand height
        above the cloth at the goal (default --place-clearance). In a two-hand move, `first` ("l"/"r")
        makes that hand come in and close first while the other waits, then the other comes in.
        `profile` (a REAL_PROFILES key, or one profile dict per picker) moves the hands along a real
        teleop path shape and timing; `absolute` takes the goals as they are (no release-height
        adjustment); `close_at` ({"l"/"r": sim time}) waits with the hand at its grasp until then
        before closing, to keep a real episode's timeline. Robot hands (TCP) stay --grip-dz above
        the cloth point they pinch, as the real fingertips sit below the TCP."""
        sim = self.sim
        q = sim.q()
        starts = [np.full(3, np.nan) if i is None else q[i].copy() for i in anchors]
        if kind == "arc" and not absolute:
            # release just above whatever already lies at the goal; placing at the grasp height
            # pushed the pinned flap through the folded layers and into the table
            above = self.args.place_clearance if release is None else release
            goals = [None if g is None else np.array([g[0], g[1], self.surface_z(g[:2]) + above]) for g in goals]

        def path(k, s, g):
            if g is None:
                return np.full((2, 3), np.nan)
            if profile is not None:
                if isinstance(profile, str):  # averaged real shape, per-trial speed factor
                    prof, scale = REAL_PROFILES[profile], self.log.get("speed_scale", 1.0)
                else:  # one real episode's own hand path and timing
                    prof, scale = profile[k], 1.0
                return profile_path(s, g, prof, round(prof["T"] / ACTION_DT / scale))
            if kind == "arc" and peak is not None:
                return arc_path(s, g, 0.0, max(0.0, peak - (s[2] + g[2]) / 2), self.carry_step, self.robot is not None)
            if kind == "arc":
                return arc_path(s, g, self.args.fold_lift_ratio, self.args.fold_clearance, self.carry_step,
                                self.robot is not None)
            # drag: lift a little, slide, keep the height
            lifted = s + [0, 0, self.args.drag_lift]
            pts = np.concatenate([linear_path(s, lifted, self.carry_step), linear_path(lifted, g, self.carry_step)[1:]])
            if self.robot is not None:
                from sg2_robot import min_jerk

                dense = np.concatenate([linear_path(s, lifted, 0.002), linear_path(lifted, g, 0.002)[1:]])
                return min_jerk(dense, self.carry_step)
            return pts

        paths = [path(k, s, g) for k, (s, g) in enumerate(zip(starts, goals))]
        waypoints = sync_paths(paths)
        robot_stats = {}
        joints = None
        if self.robot is not None:
            dz = np.array([0.0, 0.0, self.args.grip_dz])
            order = arms or self.robot.assign(waypoints)
            sides = [side for k, side in enumerate(order) if anchors[k] is not None]
            # stages of (hand targets, hands that close). Staggered two-hand grasp (real hem fold: both
            # hands set off together, the right closes at 8.2 s, the left at 14.3 s): the second hand
            # stops above its grasp until the first has closed, then comes down
            nan = np.full(3, np.nan)
            if first and len(sides) == 2:
                second = next(x for x in sides if x != first)
                kf, ks = order.index(first), order.index(second)
                above = starts[ks] + [0.0, 0.0, self.args.stagger_hover]
                stages = [({kf: starts[kf], ks: above}, [first]), ({ks: starts[ks]}, [second])]
            else:
                stages = [({k: starts[k] for k, side in enumerate(order) if side in sides}, sides)]
            closed = np.zeros(2)  # (l, r) closure
            for gi, (targets, group) in enumerate(stages):
                pts = np.stack([targets.get(k, nan) for k in range(len(order))]) + dz
                reached, _, st, j_app = self.robot.approach(
                    pts, order, f"{name}_approach", step=self.approach_step, transit=self.args.transit_height,
                    hold=gi > 0, wait_park=not self.args.parallel_park)
                reached = reached - dz  # pickers sit at the fingertips
                robot_stats.setdefault("approach", {}).update({s: st[s] for s in group})
                # the cloth keeps simulating (untouched) while the arms move in, so frames and joints
                # share one timeline
                sim.follow(reached, f"{name}_approach", joints=j_app)
                # grasp where the hands actually are; a miss shows up as a large approach track_err
                for k, side in enumerate(order):
                    if side in group:
                        starts[k] = reached[-1, k]
                # real teleop timing: pause, close at the gripper speed limit with the arm still, and
                # the cloth counts as held only once the fingers are shut
                sim.settle(self.args.pre_close, f"{name}_grasp")
                if close_at:
                    wait = max(close_at.get(side, 0.0) for side in group) - sim.t
                    if wait > 0:
                        sim.settle(wait, f"{name}_grasp")
                for side in group:
                    closed["lr".index(side)] = 1.0
                sim.ramp_grip(closed.copy(), self.args.grip_stroke_time, f"{name}_grasp")
            for side in order:
                robot_stats["approach"].setdefault(side, {"ik_ok": 1.0, "track_err_max": 0.0, "joint_speed_max": 0.0})
            waypoints, _, st, joints = self.robot.realize(waypoints + dz, name, order)
            waypoints = waypoints - dz
            robot_stats["carry"] = st
            robot_stats["order"] = list(order)
        info = sim.grasp(np.stack(starts), anchors, rest_radius)
        if self.robot is not None:
            sim.settle(self.args.post_close, f"{name}_grasp")
        sim.follow(waypoints, name, joints=joints)
        sim.release()
        cov_release = self.coverage()
        opening = 0.0
        if self.robot is not None:  # open right on arrival; the cloth is free as the fingers part
            opening = sim.ramp_grip(np.zeros(2), self.args.grip_stroke_time, f"{name}_settle")
        sim.settle(max(0.0, self.args.release_settle - opening), f"{name}_settle")
        q, qd = sim.q(), sim.qd()
        entry = {"name": name, "subtask": SUBTASKS.get(name, name), "t": sim.t,
                 "coverage": self.coverage(), "coverage_at_release": cov_release,
                 "center": q[:, :2].mean(0).round(4).tolist(),
                 **health(q, qd, self.edges, self.rest_len, self.args.contact_radius),
                 "wall_s": time.time() - self.t_wall, **info}
        if robot_stats:
            entry["robot"] = robot_stats
        entry["springback"] = entry["coverage"] / max(cov_release, 1e-6)
        if check:  # placement error of the carried keypoints, measured on the keypoint's top vertex
            entry["place_err"] = [float(np.linalg.norm(q[self.top(k)][:2] - g[:2])) for k, g in zip(check, goals)]
        self.log["phases"].append(entry)
        pe = f" place_err={[round(100 * e, 1) for e in entry['place_err']]}cm" if check else ""
        rb = ""
        if robot_stats:
            c = robot_stats["carry"]
            rb = (f" ik_ok={[round(c[s]['ik_ok'], 2) for s in robot_stats['order']]}"
                  f" track_err={[round(100 * c[s]['track_err_max'], 1) for s in robot_stats['order']]}cm"
                  f" approach_err={[round(100 * robot_stats['approach'][s]['track_err_max'], 1) for s in robot_stats['order']]}cm")
        print(f"  [trial {self.log['trial']}] {name:<6} t={sim.t:6.2f}s cov={entry['coverage']:.3f}{rb} "
              f"springback={entry['springback']:.3f} strain_p99={entry['strain_p99']:.3f} "
              f"below={entry['below_table_frac']:.4f} pinned={info['pinned']}{pe} wall={entry['wall_s']:.0f}s",
              flush=True)

    def run(self, collar_dir: int):
        a, sim = self.args, self.sim
        sim.settle(a.flat_settle, "flat")
        self.log["phases"].append({"name": "flat", "coverage": self.coverage()})
        if a.sequence == "teleop":
            return self.run_replay() if a.replay else self.run_teleop()
        if a.sequence == "neat":
            return self.run_neat()
        q = sim.q()
        near = "l" if q[self.top("l_shoulder"), 1] < q[self.top("r_shoulder"), 1] else "r"
        far = "r" if near == "l" else "l"

        # fold1: near edge (shoulder + hem corner) onto the body centerline
        q = sim.q()
        sh, co = self.top(f"{near}_shoulder"), self.top(f"{near}_corner")
        yc = (q[self.top("l_shoulder"), 1] + q[self.top("r_shoulder"), 1]) / 2
        goals = [np.array([q[i, 0], yc - a.fold_gap, q[i, 2]]) for i in (sh, co)]
        self.move("fold1", [sh, co], goals, "arc", check=[f"{near}_shoulder", f"{near}_corner"])

        # drag1: pull the stack toward the robot by its new near (fold) edge
        lo, hi = self.bbox()
        xs = [q[sh, 0], q[co, 0]]
        anchors = [self.top_near((x, lo[1] + 0.02)) for x in xs]
        q = sim.q()
        self.move("drag1", anchors, [q[i] + [0, -a.drag_dist, a.drag_lift] for i in anchors], "line")

        # fold2: far edge onto the folded near edge's line
        q = sim.q()
        sh2, co2 = self.top(f"{far}_shoulder"), self.top(f"{far}_corner")
        y_meet = (q[self.top(f"{near}_shoulder"), 1] + q[self.top(f"{near}_corner"), 1]) / 2
        goals = [np.array([q[i, 0], y_meet + a.fold_gap, q[i, 2]]) for i in (sh2, co2)]
        self.move("fold2", [sh2, co2], goals, "arc", check=[f"{far}_shoulder", f"{far}_corner"])

        # drag2: push it back by the far (fold) edge
        lo, hi = self.bbox()
        q = sim.q()
        anchors = [self.top_near((x, hi[1] - 0.02)) for x in (q[sh2, 0], q[co2, 0])]
        shift = 0.0
        if a.sequence == "sg2":
            # also slide along x so the fold3 stack (the hem half) ends centered between the arms
            x_mid = (lo[0] + hi[0]) / 2
            x_hem = hi[0] if collar_dir < 0 else lo[0]
            shift = -(x_mid + x_hem) / 2
        self.move("drag2", anchors, [q[i] + [shift, a.drag_dist, a.drag_lift] for i in anchors], "line")

        # fold3: collar half onto the hem half (fold line across x)
        lo, hi = self.bbox()
        yc, w = (lo[1] + hi[1]) / 2, hi[1] - lo[1]
        x_end = lo[0] + 0.03 if collar_dir < 0 else hi[0] - 0.03
        x_mid = (lo[0] + hi[0]) / 2
        if a.sequence == "sg2" and a.fold3_hands == 1:
            # one hand at the collar-edge midpoint: two hands on the same edge put one arm across
            # the body and its gripper into the other forearm (0-8 mm in sim_v2)
            anchors = [self.top_near((x_end, yc)), None]
        else:
            anchors = [self.top_near((x_end, yc + s * w / 4)) for s in (-1, 1)]
        q = sim.q()
        goals = [None if i is None else np.array([2 * x_mid - q[i, 0], q[i, 1], q[i, 2]]) for i in anchors]
        self.move("fold3", anchors, goals, "arc")

        if a.sequence == "sg2":
            # flip with one hand (FoldNet++ Galbot style): the arm on the free-edge side takes the
            # fold-edge midpoint over the far edge, staying inside its own half of the workspace
            lo, hi = self.bbox()
            yc = (lo[1] + hi[1]) / 2
            x_edge, x_far = (lo[0] + 0.02, hi[0]) if collar_dir < 0 else (hi[0] - 0.02, lo[0])
            grip = self.top_near((x_edge, yc))
            q = sim.q()
            self.move("flip", [grip, None], [np.array([2 * x_far - q[grip, 0], q[grip, 1], q[grip, 2]]), None], "arc")
            return

        # flip: lift the folded edge and turn the stack over its opposite x edge
        lo, hi = self.bbox()
        yc, w = (lo[1] + hi[1]) / 2, hi[1] - lo[1]
        # after fold3 the fold line sits on the collar side of the stack
        x_edge, x_far = (lo[0] + 0.02, hi[0]) if collar_dir < 0 else (hi[0] - 0.02, lo[0])
        anchors = [self.top_near((x_edge, yc + s * w / 4)) for s in (-1, 1)]
        q = sim.q()
        goals = [np.array([2 * x_far - q[i, 0], q[i, 1], q[i, 2]]) for i in anchors]
        self.move("flip", anchors, goals, "arc")

        # drag3: slide sideways
        lo, hi = self.bbox()
        yc, w = (lo[1] + hi[1]) / 2, hi[1] - lo[1]
        x_edge = lo[0] + 0.02 if collar_dir > 0 else hi[0] - 0.02
        anchors = [self.top_near((x_edge, yc + s * w / 4)) for s in (-1, 1)]
        q = sim.q()
        self.move("drag3", anchors, [q[i] + [-collar_dir * a.drag_dist, 0, a.drag_lift] for i in anchors], "line")


    def run_teleop(self):
        """The fold the SG2 teleop demonstrations use (laundry_fold_tshirt_arm16: 22 of 24 episodes start
        LR-L-R), with positions and heights from the real hands (median, sim frame):
          half    the right hand takes its hem corner first and holds it while the left hand comes in
                  (real: right closes 6-8 s earlier); both lift the hem to ~0.42 m and let go 0.11-0.13 m
                  up near the collar so it falls over the shoulders
          side_l  the left hand pinches the stack's outer edge right next to the fold (real grasp
                  (-0.32, 0.02)), lifts it ~0.17 m and lets go 0.12-0.17 m up with the hand over the
                  center (real end x +0.06 with the shirt centered at +0.045)
          side_r  the same with the right hand (real (0.43, 0.10) -> (0.09, 0.06))
          half2   the left hand takes the left edge over to the right (real (-0.07, 0.09) -> (0.08..0.15,
                  0.10), peak ~0.2 m): a narrow strip
          tidy    sometimes (10/24) the right hand drags the strip ~0.25 m right (real 0.04 -> 0.24..0.32)
        The other arm waits at its park pose during one-hand folds (real hands >= 0.24 m apart)."""
        a, sim = self.args, self.sim
        q = sim.q()
        names = sorted(("l_corner", "r_corner"), key=lambda k: q[self.top(k), 0])
        corners = [self.top(k) for k in names]
        y_top = max(q[self.top("l_shoulder"), 1], q[self.top("r_shoulder"), 1])
        goals = [np.array([q[i, 0], y_top - a.half_short, q[i, 2]]) for i in corners]
        self.move("half", corners, goals, "arc", arms=("l", "r"), release=a.half_release, first="r",
                  profile="half" if a.real_paths else None, peak=a.half_peak)

        # side folds: the near edge of the stack is now the fold line
        lo, hi = self.bbox()
        y_grab = lo[1] + a.side_grab_dy

        def band(y):
            q = sim.q()
            return q[np.abs(q[:, 1] - y) < 0.015, 0]

        b = band(y_grab)
        xc = (b.min() + b.max()) / 2
        for side, sgn in (("l", -1), ("r", 1)):
            b = band(y_grab)
            x_edge = b.min() if sgn < 0 else b.max()
            grip = self.top_near((x_edge - sgn * 0.02, y_grab))
            q = sim.q()
            goal = np.array([xc + sgn * a.side_inset, q[grip, 1], q[grip, 2]])
            other = "r" if side == "l" else "l"
            # fingers close on the stack's edge: every layer there comes along (the folded hem too)
            self.move(f"side_{side}", [grip, None], [goal, None], "arc", arms=(side, other), peak=a.side_peak,
                      rest_radius=1.0, release=a.side_release, profile=f"side_{side}" if a.real_paths else None)

        # half2: the left hand takes the left edge over the center line
        lo, hi = self.bbox()
        y2 = lo[1] + a.half2_grab_dy
        b = band(y2)
        grip = self.top_near((b.min() + 0.02, y2))
        q = sim.q()
        xm = (b.min() + b.max()) / 2
        goal = np.array([min(2 * xm - q[grip, 0] - a.fold_gap, a.half2_max_x), q[grip, 1], q[grip, 2]])
        self.move("half2", [grip, None], [goal, None], "arc", arms=("l", "r"), peak=a.half2_peak, rest_radius=1.0,
                  release=a.half2_release, profile="half2" if a.real_paths else None)
        if self.rng.random() < a.tidy_prob:
            lo, hi = self.bbox()
            grip = self.top_near(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2))
            goal = sim.q()[grip] + [a.tidy_dist, 0.0, 0.05]
            self.move("tidy", [grip, None], [goal, None], "line", arms=("r", "l"), rest_radius=1.0)
        self.retract()

    def run_replay(self):
        """Reproduce one real teleop episode (picked per trial from sim/sg2_newton/data/real_steps.json,
        built from laundry_fold_tshirt_arm16): for each step of LR-L-R-L every hand closes at the cloth
        nearest to where the real hand closed (in the two side steps: on the sleeve cuff on that side,
        which is what the real wrist cameras show the hand taking), follows that hand's recorded path
        shape and hold time,
        and lets go where the real hand opened (TCP --grip-dz above the pinched cloth). Grasps wait for
        the real episode's close times when the sim gets there early, and the episode ends at the real
        length, so timing, hand positions and lifts follow the real spread episode by episode."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_newton"))
        from real_steps import STEPS_PATH, load

        a, sim = self.args, self.sim
        a.parallel_park = True
        eps = load(a.real_steps or STEPS_PATH)["episodes"]
        ep = eps[int(self.rng.integers(len(eps)))]
        self.log["real_episode"] = ep["episode"]
        dz = np.array([0.0, 0.0, a.grip_dz])
        t0 = 0.0  # the real recording starts with the shirt already flat (sim t=0 is before flat_settle)
        misses = {}
        for si, name in enumerate(("half", "side_l", "side_r", "half2")):
            step = ep["steps"][si]
            sides = [x for x in "lr" if x in step]
            anchors, goals, profs, close_at = [], [], [], {}
            last_close = max(step[x]["close_t"] for x in sides)
            for x in sides:
                h = step[x]
                cp, op = np.array(h["close_p"]), np.array(h["open_p"])
                anchor = self.sleeve_cuff(x) if name in ("side_l", "side_r") and a.side_grasp == "sleeve" \
                    else self.top_near(cp[:2])
                misses[f"{name}_{x}"] = round(float(100 * np.linalg.norm(sim.q()[anchor, :2] - cp[:2])), 1)
                anchors.append(anchor)
                goals.append(op - dz)
                profs.append(carry_profile(h, last_close, a.post_close))
                close_at[x] = t0 + h["close_t"]
            if len(sides) == 1:
                arms = (sides[0], "r" if sides[0] == "l" else "l")
                anchors.append(None)
                goals.append(None)
                profs.append(None)
            else:
                arms = ("l", "r")
            first = min(sides, key=lambda x: step[x]["close_t"]) if len(sides) == 2 else None
            self.move(name, anchors, goals, "arc", arms=arms, first=first, profile=profs, absolute=True,
                      rest_radius=None if name == "half" else 1.0, close_at=close_at if a.real_timing else None)
            if name == a.stop_after:
                break
        self.log["grasp_miss_cm"] = misses  # cloth point taken vs the real hand's close point (xy)
        if a.stop_after:
            self.retract()
            return
        if self.rng.random() < a.tidy_prob:
            lo, hi = self.bbox()
            grip = self.top_near(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2))
            goal = sim.q()[grip] + [a.tidy_dist, 0.0, 0.05]
            self.move("tidy", [grip, None], [goal, None], "line", arms=("r", "l"), rest_radius=1.0)
        self.retract(end_at=t0 + ep["length_s"] if a.real_timing else None)

    def run_neat(self):
        """FoldNet/FoldNet++-style keypoint folds for clean, straight layers: every fold takes the
        grasped point to its mirror image across a fold line and lays it down low.
          half     both hands take the hem corners and lay them on the shoulder line (the right hand
                   first, the hem lifted the way operators lift it: REAL_PROFILES["half"])
          sleeve_l / sleeve_r  each hand folds its sleeve tip in across the body's side edge
          quarter_l / quarter_r  each hand takes its side edge (all layers, mid-length) across the
                   quarter line, so the edges meet at the center
          final    the left hand takes the left fold edge across the center line onto the right
        Hands, postures, speeds and gripper timing are the real ones (posture prior, min-jerk paths)."""
        a, sim = self.args, self.sim
        q = sim.q()
        # --- half: hem corners onto the shoulders (mirror across the body's mid line)
        names = sorted(("l_corner", "r_corner"), key=lambda k: q[self.top(k), 0])
        corners = [self.top(k) for k in names]
        shoulders = sorted(("l_shoulder", "r_shoulder"), key=lambda k: q[self.top(k), 0])
        # the shoulder line is past the arms' low reach (y <= ~0.17 at 3 cm up): let go above it,
        # higher, so the hem falls the rest of the way (as operators do)
        goals = [np.array([q[c, 0], min(q[self.top(sh), 1] - a.fold_gap, a.neat_reach_y), q[c, 2]])
                 for c, sh in zip(corners, shoulders)]
        self.move("half", corners, goals, "arc", arms=("l", "r"), first="r",
                  profile="half" if a.real_paths else None, release=a.neat_half_release)

        # --- sleeves: tip across the body's side edge (the armpit's x)
        q = sim.q()
        sides = sorted("lr", key=lambda k: q[self.top(f"{k}_armpit"), 0])  # mesh side on the robot's left first
        for arm, mesh_side, sgn in (("l", sides[0], -1), ("r", sides[1], 1)):
            q = sim.q()
            tip = (q[self.top(f"{mesh_side}_sleeve_top")] + q[self.top(f"{mesh_side}_sleeve_bottom")]) / 2
            grip = self.top_near(tip[:2])
            x_line = q[self.top(f"{mesh_side}_armpit"), 0] - sgn * a.neat_sleeve_inset
            goal = np.array([2 * x_line - q[grip, 0], q[grip, 1], q[grip, 2]])
            other = "r" if arm == "l" else "l"
            self.move(f"sleeve_{arm}", [grip, None], [goal, None], "arc", arms=(arm, other), rest_radius=1.0,
                      release=a.neat_release)

        # --- quarters: each side edge across its quarter line, edges meet at the center
        lo, hi = self.bbox()
        y_mid = lo[1] + a.neat_grab_frac * (hi[1] - lo[1])

        def band():
            q = sim.q()
            return q[np.abs(q[:, 1] - y_mid) < 0.015, 0]

        b = band()
        xc, half_w = (b.min() + b.max()) / 2, (b.max() - b.min()) / 2
        for arm, sgn in (("l", -1), ("r", 1)):
            b = band()
            x_edge = b.min() if sgn < 0 else b.max()
            grip = self.top_near((x_edge - sgn * 0.015, y_mid))
            q = sim.q()
            x_line = xc + sgn * half_w / 2
            goal = np.array([2 * x_line - q[grip, 0] - sgn * a.fold_gap, q[grip, 1], q[grip, 2]])
            other = "r" if arm == "l" else "l"
            self.move(f"quarter_{arm}", [grip, None], [goal, None], "arc", arms=(arm, other), rest_radius=1.0,
                      release=a.neat_release)

        # --- final: left fold edge across the center line
        b = band()
        grip = self.top_near((b.min() + 0.015, y_mid))
        q = sim.q()
        xm = (b.min() + b.max()) / 2
        goal = np.array([min(2 * xm - q[grip, 0] - a.fold_gap, a.neat_final_max_x), q[grip, 1], q[grip, 2]])
        self.move("final", [grip, None], [goal, None], "arc", arms=("l", "r"), rest_radius=1.0,
                  release=a.neat_release)
        self.retract()

    def retract(self, end_at=None):
        """Both hands back out to the idle-arm pose, then hold: operators end an episode like this
        (real: 8.8 s median from the last release to the end of the recording). `end_at` holds until
        that sim time (a real episode's length) instead of --end-hold."""
        if self.robot is None:
            return
        reached, _, _, joints = self.robot.approach(np.full((2, 3), np.nan), ("l", "r"), "retract",
                                                    step=self.approach_step, transit=self.args.transit_height)
        self.sim.follow(reached, "retract", joints=joints)
        hold = self.args.end_hold if end_at is None else max(self.args.end_hold, end_at - self.sim.t)
        self.sim.settle(hold, "retract")
        self.log["phases"].append({"name": "retract", "t": self.sim.t, "coverage": self.coverage()})


def verdict(log) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    if "diverged" in log:
        return [f"diverged at t={log['diverged']['t']:.2f}s in {log['diverged']['phase']}"], warns
    for p in log["phases"]:
        if "strain_p99" not in p:
            continue
        if p["strain_p99"] > 1.3:
            fails.append(f"{p['name']}: strain_p99 {p['strain_p99']:.2f}")
        if p["below_table_frac"] > 0.01:
            fails.append(f"{p['name']}: {100 * p['below_table_frac']:.1f}% under table")
        if p["springback"] > 1.1:
            warns.append(f"{p['name']}: footprint grew {100 * (p['springback'] - 1):.0f}% after release")
        if max(p.get("place_err", [0])) > 0.04:
            warns.append(f"{p['name']}: placement error {100 * max(p['place_err']):.1f} cm")
    c = log.get("arm_clearance")
    if c:
        if c["arm_arm_min"] < c["limit"]:
            fails.append(f"arms {1000 * c['arm_arm_min']:.0f} mm apart (limit {1000 * c['limit']:.0f} mm)")
        if c["arm_body_min"] < c["limit"]:
            fails.append(f"arm {1000 * c['arm_body_min']:.0f} mm from the torso (limit {1000 * c['limit']:.0f} mm)")
    final = log["phases"][-1].get("coverage", 1.0)
    if final > 0.4:
        warns.append(f"final footprint {final:.2f} of flat (expected ~0.25)")
    return fails, warns


def run_trial(args, trial, verts0, faces, keypoints):
    rng = np.random.default_rng(args.seed + trial)
    collar_dir, back_up = CONFIGS[trial % len(CONFIGS)]
    if args.sequence in ("teleop", "neat"):  # collar away from the robot (+y), front up, as in the teleop data
        # (a flipped mesh does not settle flat: its inflated layers bunch up before any grasp)
        collar_dir, back_up = 0, False
    if args.sequence in ("teleop", "neat"):
        scale = args.teleop_scale * rng.uniform(1 - args.teleop_scale_jitter, 1 + args.teleop_scale_jitter)
    else:
        scale = args.scale_target * rng.uniform(0.9, 1.1)
    verts = verts0 * scale
    if args.sequence in ("teleop", "neat"):  # the real shirt is wider for its length than the FoldNet mesh
        verts = verts * [args.teleop_width, 1.0, 1.0]
    panel, panel_idx = per_triangle_panel(verts, faces)
    flat_area = projected_area(verts, faces)
    rest = verts
    if args.sequence in ("teleop", "neat") and args.init_gap > 0:
        # the mesh's front layer floats ~5 cm over the back one; dropped from there it can hit the back
        # layer hard enough to bounce up and crumple before any grasp (real episodes start flat), so
        # start it just above (the Style3D rest shape comes from `panel`, computed above)
        z = verts[:, 2] - verts[:, 2].min()
        verts = verts.copy()
        verts[:, 2] = verts[:, 2].min() + z * min(1.0, args.init_gap / max(z.max(), 1e-9))
    # mesh collar points +y; turn it to -x/+x, optionally flipped about its long axis (back face up)
    rot = axis_quat((0, 0, 1), 0.0 if collar_dir == 0 else math.pi / 2 if collar_dir < 0 else -math.pi / 2)
    if back_up:
        rot = quat_mul(rot, axis_quat((0, 1, 0), math.pi))
    rv = quat_rotate(rot, verts)
    pos = np.array([0.0, 0.0, -rv[:, 2].min() + 0.005])
    if args.sequence in ("teleop", "neat"):  # hem where the operators grasp it (rv y min = hem)
        pos[1] = (args.neat_hem_y if args.sequence == "neat" else args.hem_y) - rv[:, 1].min()
        pos[0] = args.shirt_x
    sim = ClothSim(args, verts, faces, panel, panel_idx, pos, rot)
    sim.rest = rest  # strain and grasp distances against the mesh's own shape, not the squashed start
    speed_scale = rng.uniform(1 - args.speed_jitter, 1 + args.speed_jitter)
    log = {"trial": trial, "collar_dir": collar_dir, "back_up": back_up, "scale": scale,
           "speed_scale": speed_scale, "phases": []}
    print(f"  [trial {trial}] collar {['-x', '+y', '+x'][collar_dir + 1]}, {'back' if back_up else 'front'} face up")
    ep = FoldEpisode(args, sim, faces, keypoints, flat_area, log)
    try:
        ep.run(collar_dir)
    except Diverged as e:
        print(f"  [trial {trial}] DIVERGED: {e}", flush=True)
        log["diverged"] = {"t": e.t, "phase": e.phase, "n_bad": len(e.bad_idx)}
    log["sim_time_s"], log["wall_time_s"] = sim.t, time.time() - ep.t_wall

    trial_dir = args.out / f"trial_{trial:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    extra = {}
    if ep.robot is not None:  # 5 Hz joint targets (l, r) incl. approach moves, for the dataset
        phases, robot_q = ep.robot.joint_trajectory()
        extra = {"robot_q": robot_q.astype(np.float32), "robot_phase": np.array(phases)}
    if ep.robot is not None:  # per recorded frame, for rendering and the dataset's observation.state
        extra.update(joint=np.stack(sim.frame_joint).astype(np.float32), grip=np.stack(sim.frame_grip))
    np.savez_compressed(trial_dir / "frames.npz", q=np.stack(sim.frames).astype(np.float32), faces=faces,
                        t=np.array(sim.frame_t), phase=np.array(sim.frame_phase),
                        pickers=np.stack(sim.frame_pickers).astype(np.float32), **extra)
    if ep.robot is not None:  # the IK has no collision term: measure how close the arms came
        from check_arm_collision import clearance

        log["arm_clearance"] = {**clearance(extra["joint"], extra["grip"], args.record_dt), "limit": args.arm_clearance}
        c = log["arm_clearance"]
        print(f"  [trial {trial}] arm clearance: arm-arm {1000 * c['arm_arm_min']:.0f} mm, "
              f"arm-torso {1000 * c['arm_body_min']:.0f} mm", flush=True)
    if not args.no_video:
        render_video(trial_dir / "video.mp4", sim, faces, 1.0 / args.record_dt)
    return log


def main():
    p = make_parser(__doc__)
    p.set_defaults(out=__import__("pathlib").Path("outputs/style3d_gate/fold"))
    p.add_argument("--trials", type=int, default=4, help="cycles through the 4 FoldNet++ initial configurations")
    p.add_argument("--flat-settle", type=float, default=0.5)
    p.add_argument("--fold-gap", type=float, default=0.01, help="stop this short of the fold target line [m]")
    p.add_argument("--fold-lift-ratio", type=float, default=0.6, help="arc height / half the fold distance")
    p.add_argument("--fold-clearance", type=float, default=0.02)
    p.add_argument("--place-clearance", type=float, default=0.01, help="release height above the cloth at the goal")
    p.add_argument("--drag-dist", type=float, default=0.08)
    p.add_argument("--drag-lift", type=float, default=0.01)
    p.add_argument("--sequence", choices=["foldnet", "sg2", "teleop", "neat"], default="foldnet",
                   help="sg2: drag2 also centers the fold3 stack, one-hand flip, no drag3 (fits the SG2 workspace); "
                        "teleop: the SG2 teleop fold (hem onto collar, left and right sides in, left half over); "
                        "neat: FoldNet-style mirrored folds (hem onto shoulders, sleeves in, sides to the "
                        "center, left half over)")
    # teleop-sequence geometry from the real data (laundry_fold_tshirt_arm16, hands in the sim frame
    # with the arm base 0.45 m behind the cloth center): hem grasped at y -0.18; the hands peak at
    # 0.41 m in the hem fold, 0.17 m in the side folds and 0.21 m in the last fold
    # shirt size and placement fitted to the real head camera (calibrated on the idle grippers): the 19
    # real first frames' shirt silhouette is matched (IoU 0.91) by a flat shirt 0.92 m across the
    # sleeves and 0.70 m hem to collar, centered at x 0, with the hem where the operators grasp it
    p.add_argument("--hem-y", type=float, default=-0.20, help="teleop: hem position (y) of the flat shirt")
    p.add_argument("--shirt-x", type=float, default=0.0, help="teleop: shirt center x")
    p.add_argument("--teleop-width", type=float, default=0.94, help="teleop: widen the shirt by this factor")
    p.add_argument("--teleop-scale-jitter", type=float, default=0.05, help="teleop: per-trial size factor 1 +- this")
    p.add_argument("--no-replay", dest="replay", action="store_false",
                   help="teleop: the averaged scripted fold instead of replaying real episodes' steps")
    p.add_argument("--real-steps", type=Path, default=None, help="teleop replay: real step table (real_steps.py)")
    p.add_argument("--init-gap", type=float, default=0.004,
                   help="teleop/neat: start the shirt's layers this far apart [m] (0 = mesh as is, ~5 cm)")
    p.add_argument("--side-grasp", choices=["sleeve", "real"], default="sleeve",
                   help="teleop replay side steps: take the sleeve cuff (what the real wrist cameras show) or "
                        "the cloth nearest to the real close point")
    p.add_argument("--stop-after", choices=["half", "side_l", "side_r"], default=None,
                   help="teleop replay: end after this step (quick tests of one fold)")
    p.add_argument("--no-real-timing", dest="real_timing", action="store_false",
                   help="teleop replay: do not wait for the real episode's grasp times / length")
    p.add_argument("--parallel-park", action="store_true",
                   help="the idle arm parks while the working arm already moves in (teleop replay turns this on: "
                        "real hands overlap, 5.8 s median between one release and the next grasp)")
    p.add_argument("--picker-scale", type=float, default=1.5,
                   help="scale the picker box in x/y (1 = the gate-test 2 x 1.6 cm box; the shirt at teleop size "
                        "has ~1.5 cm vertex spacing, so a 1x box often pinched a single vertex)")
    p.add_argument("--grip-dz", type=float, default=0.03,
                   help="robot hand (TCP) height above the cloth point it pinches [m] (real TCP at grasps: 0.03-0.04)")
    p.add_argument("--teleop-scale", type=float, default=0.83, help="teleop: shirt size (geo-mean of bbox length/width)")
    p.add_argument("--half-peak", type=float, default=0.40, help="teleop: hand peak height in the hem fold [m]")
    p.add_argument("--side-peak", type=float, default=0.18, help="teleop: hand peak height in the side folds [m]")
    p.add_argument("--half2-peak", type=float, default=0.20, help="teleop: hand peak height in the last fold [m]")
    p.add_argument("--no-real-paths", dest="real_paths", action="store_false",
                   help="teleop: arcs (--*-peak) instead of the recorded real hand path shapes")
    p.add_argument("--neat-release", type=float, default=0.02,
                   help="neat: hand height above the cloth where folds let go [m] (low = layers land where placed)")
    p.add_argument("--neat-sleeve-inset", type=float, default=0.02,
                   help="neat: sleeve fold line this far inside the body's side edge [m]")
    p.add_argument("--neat-hem-y", type=float, default=-0.24,
                   help="neat: hem position (y); 6 cm nearer than the teleop setup so the folded stack stays "
                        "inside the arms' low reach (y <= ~0.17)")
    p.add_argument("--neat-reach-y", type=float, default=0.17, help="neat: hem-fold hands go no further than this y")
    p.add_argument("--neat-half-release", type=float, default=0.08, help="neat: hem-fold release height [m]")
    p.add_argument("--neat-final-max-x", type=float, default=0.12,
                   help="neat: the left hand's last-fold goal stays left of this x (its reach at y ~0.1)")
    p.add_argument("--neat-grab-frac", type=float, default=0.4,
                   help="neat: side/final folds grasp at this fraction of the stack from its near (fold) edge")
    p.add_argument("--stagger-hover", type=float, default=0.10,
                   help="teleop: in the hem fold the left hand waits this far above its corner until the right closes [m]")
    p.add_argument("--half2-max-x", type=float, default=0.15, help="teleop: the last fold's hand goal stays left of this x [m]")
    p.add_argument("--tidy-prob", type=float, default=10 / 24,
                   help="teleop: chance of a last right-hand drag of the folded shirt (10 of 24 real episodes)")
    p.add_argument("--tidy-dist", type=float, default=0.25, help="teleop: that drag's length to the right [m]")
    p.add_argument("--half-release", type=float, default=0.10,
                   help="teleop: hand height above the cloth where the hem fold lets go [m] (real ~0.12 m TCP)")
    p.add_argument("--half-short", type=float, default=0.10,
                   help="teleop: the hem fold lets go this far short of the shoulder line [m]")
    p.add_argument("--side-release", type=float, default=0.10,
                   help="teleop: hand height above the cloth where side folds let go [m] (real TCP 0.12-0.17 m)")
    p.add_argument("--half2-release", type=float, default=0.06,
                   help="teleop: hand height above the cloth where the last fold lets go [m] (real TCP 0.08-0.11 m)")
    p.add_argument("--side-grab-dy", type=float, default=0.03,
                   help="teleop: side folds grasp this far past the fold line (the stack's near edge) [m]")
    p.add_argument("--side-inset", type=float, default=0.0,
                   help="teleop: side-fold hand goal this far short of the shirt center [m] (real ~0)")
    p.add_argument("--half2-grab-dy", type=float, default=0.06, help="teleop: last fold grasp past the fold line [m]")
    p.add_argument("--fold3-hands", type=int, choices=[1, 2], default=1,
                   help="sg2 sequence: fold3 with one hand (collar-edge midpoint) or two (edge quarter points)")
    p.add_argument("--robot", action="store_true", help="drive the pickers through SG2 IK (sim/sg2_newton)")
    p.add_argument("--robot-base-y", type=float, default=-0.45, help="arm_base_link y; cloth center is y=0")
    p.add_argument("--robot-height", type=float, default=0.55, help="arm_base_link above the table (lift 0)")
    p.add_argument("--grasp-tilt", type=float, default=35.0,
                   help="grasp approach below horizontal [deg] (only with --posture-prior none)")
    p.add_argument("--posture-prior", default="real",
                   help="hand orientation and arm posture from the real teleop data (sim/sg2_newton/posture_prior.py): "
                        "'real' = the bundled prior, a path to another prior, or 'none' for the fixed tilted grasp")
    # speeds and heights from the real teleop data (laundry_fold_tshirt_arm16): moving-hand speed
    # median 0.07-0.08 m/s holding cloth, 0.058 m/s empty (minimum-jerk timing puts the median ~5%
    # above these means); open hand ~11 cm above grasp level
    p.add_argument("--carry-speed", type=float, default=0.07, help="mean hand speed while holding cloth [m/s]")
    p.add_argument("--approach-speed", type=float, default=0.055, help="mean hand speed with an empty hand [m/s]")
    p.add_argument("--end-hold", type=float, default=2.0, help="teleop: hold after retracting the arms [s]")
    p.add_argument("--speed-jitter", type=float, default=0.2, help="per-trial speed factor in 1 +- this")
    p.add_argument("--transit-height", type=float, default=0.12, help="empty-hand travel height above the table [m]")
    # gripper timing from the real robot: deploy teleop config gripper_speed_deg_s = 60 (0 <-> 1.1 rad
    # in 1.05 s; the HF data shows 0.97-1.13 s), and in the HF data operators wait 0.2 s (median)
    # before closing and 0.33 s after the fingers shut before moving off
    p.add_argument("--grip-stroke-time", type=float, default=1.1 / math.radians(60.0),
                   help="seconds for a full gripper close or open")
    p.add_argument("--pre-close", type=float, default=0.2, help="pause at the grasp point before closing [s]")
    p.add_argument("--post-close", type=float, default=0.33, help="pause after closing before carrying [s]")
    p.add_argument("--arm-clearance", type=float, default=0.01,
                   help="FAIL a trial whose arms come closer than this to each other or the torso [m]")
    args = p.parse_args()
    if args.quick:
        args.sim_dt, args.iterations, args.linear_iterations = 4e-3, 4, 6
        args.flat_settle, args.release_settle, args.record_dt = 0.2, 0.3, 0.2
        args.trials = min(args.trials, 1)

    verts, faces, keypoints = load_mesh(args)
    args.out.mkdir(parents=True, exist_ok=True)
    logs = []
    for trial in range(args.trials):
        log = run_trial(args, trial, verts, faces, keypoints)
        log["fails"], log["warns"] = verdict(log)
        status = "FAIL: " + "; ".join(log["fails"]) if log["fails"] else "STABLE"
        if log["warns"]:
            status += "  (WARN: " + "; ".join(log["warns"]) + ")"
        print(f"  [trial {trial}] realtime x{log['sim_time_s'] / log['wall_time_s']:.3f}  {status}", flush=True)
        logs.append(log)

    cfg = {k: str(v) for k, v in vars(args).items()}
    (args.out / "metrics.json").write_text(json.dumps({"config": cfg, "trials": logs}, indent=2))
    n_stable = sum(not log["fails"] for log in logs)
    print(f"\n{n_stable}/{len(logs)} trials stable, {sum(bool(log['warns']) for log in logs)} with warnings. "
          f"metrics -> {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
