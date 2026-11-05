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
    LINEAR_STEP,
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
}


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


def arc_path(start: np.ndarray, goal: np.ndarray, lift_ratio: float, clearance: float) -> np.ndarray:
    """Half-circle-like path from start to goal (fold motion), resampled at LINEAR_STEP."""
    th = np.linspace(0.0, math.pi, 400)
    c = (start + goal) / 2
    u = start - c
    r = np.linalg.norm(u[:2])
    pts = c + np.outer(np.cos(th), u)
    # end heights may differ (placing onto a stack): blend them, then add the arc lift
    pts[:, 2] = start[2] + (goal[2] - start[2]) * (1 - np.cos(th)) / 2 + np.sin(th) * (lift_ratio * r + clearance)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    targets = np.linspace(0, s[-1], max(1, math.ceil(s[-1] / LINEAR_STEP)) + 1)
    return np.stack([np.interp(targets, s, pts[:, k]) for k in range(3)], axis=1)


class FoldEpisode:
    def __init__(self, args, sim: ClothSim, faces, keypoints, flat_area, log):
        self.args, self.sim, self.faces, self.kp = args, sim, faces, keypoints
        self.flat_area, self.log = flat_area, log
        self.edges = unique_edges(faces)
        self.rest_len = np.linalg.norm(sim.rest[self.edges[:, 0]] - sim.rest[self.edges[:, 1]], axis=1)
        self.t_wall = time.time()
        self.robot = None
        if args.robot:  # pickers follow the TCP an SG2 IK actually reaches
            from sg2_robot import SG2Arms

            self.robot = SG2Arms(base_y=args.robot_base_y, height=args.robot_height, tilt=args.grasp_tilt)

    def top(self, name: str) -> int:
        q = self.sim.q()
        return max(self.kp[name], key=lambda i: q[i, 2])

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

    def move(self, name: str, anchors: list[int], goals: list[np.ndarray], kind: str, check: list[str] | None = None):
        """Grasp the anchors, carry them to goals (arc = fold, line = drag), release, settle.
        An anchor/goal of None makes it a one-hand move: that picker (and arm) stays idle."""
        sim = self.sim
        q = sim.q()
        starts = [np.full(3, np.nan) if i is None else q[i].copy() for i in anchors]
        if kind == "arc":
            # release just above whatever already lies at the goal; placing at the grasp height
            # pushed the pinned flap through the folded layers and into the table
            goals = [None if g is None else np.array([g[0], g[1], self.surface_z(g[:2]) + self.args.place_clearance])
                     for g in goals]

        def path(s, g):
            if g is None:
                return np.full((2, 3), np.nan)
            if kind == "arc":
                return arc_path(s, g, self.args.fold_lift_ratio, self.args.fold_clearance)
            # drag: lift a little, slide, keep the height
            lifted = s + [0, 0, self.args.drag_lift]
            return np.concatenate([linear_path(s, lifted, LINEAR_STEP), linear_path(lifted, g, LINEAR_STEP)[1:]])

        paths = [path(s, g) for s, g in zip(starts, goals)]
        waypoints = sync_paths(paths)
        robot_stats = {}
        joints = None
        if self.robot is not None:
            order = self.robot.assign(waypoints)
            reached_grasp, _, st, j_app = self.robot.approach(np.stack(starts), order, f"{name}_approach")
            robot_stats["approach"] = st
            # the cloth keeps simulating (untouched) while the arms move in, so frames and joints
            # share one timeline
            sim.follow(reached_grasp, f"{name}_approach", joints=j_app)
            # grasp where the hands actually are; a miss shows up as a large approach track_err
            starts = [reached_grasp[-1, k] for k in range(2)]
            waypoints, _, st, joints = self.robot.realize(waypoints, name, order)
            robot_stats["carry"] = st
            robot_stats["order"] = list(order)
            sim.cur_grip = np.array([anchors[order.index(side)] is not None for side in ("l", "r")])
        info = sim.grasp(np.stack(starts), anchors)
        sim.follow(waypoints, name, joints=joints)
        sim.release()
        sim.cur_grip = np.zeros(2, dtype=bool)
        cov_release = self.coverage()
        sim.settle(self.args.release_settle, f"{name}_settle")
        q, qd = sim.q(), sim.qd()
        entry = {"name": name, "subtask": SUBTASKS.get(name, name), "t": sim.t,
                 "coverage": self.coverage(), "coverage_at_release": cov_release,
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
        anchors = [self.top_near((x_end, yc + s * w / 4)) for s in (-1, 1)]
        q = sim.q()
        goals = [np.array([2 * x_mid - q[i, 0], q[i, 1], q[i, 2]]) for i in anchors]
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
    final = log["phases"][-1].get("coverage", 1.0)
    if final > 0.4:
        warns.append(f"final footprint {final:.2f} of flat (expected ~0.25)")
    return fails, warns


def run_trial(args, trial, verts0, faces, keypoints):
    rng = np.random.default_rng(args.seed + trial)
    collar_dir, back_up = CONFIGS[trial % len(CONFIGS)]
    scale = args.scale_target * rng.uniform(0.9, 1.1)
    verts = verts0 * scale
    panel, panel_idx = per_triangle_panel(verts, faces)
    flat_area = projected_area(verts, faces)
    # mesh collar points +y; turn it to -x/+x, optionally flipped about its long axis (back face up)
    rot = axis_quat((0, 0, 1), math.pi / 2 if collar_dir < 0 else -math.pi / 2)
    if back_up:
        rot = quat_mul(rot, axis_quat((0, 1, 0), math.pi))
    rv = quat_rotate(rot, verts)
    pos = np.array([0.0, 0.0, -rv[:, 2].min() + 0.005])
    sim = ClothSim(args, verts, faces, panel, panel_idx, pos, rot)
    log = {"trial": trial, "collar_dir": collar_dir, "back_up": back_up, "scale": scale, "phases": []}
    print(f"  [trial {trial}] collar {'-x' if collar_dir < 0 else '+x'}, {'back' if back_up else 'front'} face up")
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
    p.add_argument("--sequence", choices=["foldnet", "sg2"], default="foldnet",
                   help="sg2: drag2 also centers the fold3 stack, one-hand flip, no drag3 (fits the SG2 workspace)")
    p.add_argument("--robot", action="store_true", help="drive the pickers through SG2 IK (sim/sg2_newton)")
    p.add_argument("--robot-base-y", type=float, default=-0.45, help="arm_base_link y; cloth center is y=0")
    p.add_argument("--robot-height", type=float, default=0.55, help="arm_base_link above the table (lift 0)")
    p.add_argument("--grasp-tilt", type=float, default=35.0, help="grasp approach below horizontal [deg]")
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
