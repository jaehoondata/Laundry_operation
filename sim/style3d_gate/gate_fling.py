"""Gate test: can Newton's Style3D cloth solver drop and fling a FoldNet T-shirt stably?

No robot here. Two kinematic pickers stand in for the grippers, using the FoldNet++
attachment grasp: when a picker closes, cloth vertices inside its box are pinned to it.

Per trial:
  1. hang the shirt from its highest vertex, lower it to z=0.6 m, release, let it settle
  2. for each fling: grasp a keypoint pair, lift, stretch, cosine fling (FoldNet++ eq.), release
Trial 0 is a flat-on-table sanity check (coverage should stay ~1, strain ~1).

Outputs under --out: metrics.json (all trials), trial_XX/frames.npz, trial_XX/video.mp4
"""

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import style3d

FOLDNET_URL = "https://huggingface.co/datasets/Bowie375/FoldNet/resolve/main/mesh/tshirt_sp/{mesh_id}/{name}"

ACTION_DT = 0.2  # FoldNet++ policy step (5 Hz)
LINEAR_STEP = 0.03  # m per action step for non-fling moves
FLING_STEP = 0.09  # m per action step for the fling
# Picker box in the EEF frame (z = approach direction), FoldNet++ Table
PICKER_BOX = np.array([[-0.01, 0.01], [-0.008, 0.008], [-0.013, 0.030]])
FLING_PAIRS = [("l_shoulder", "r_shoulder"), ("l_corner", "r_corner")]


# ----------------------------------------------------------------------------- mesh


def fetch_mesh(mesh_id: int, cache: Path) -> Path:
    d = cache / "tshirt_sp" / str(mesh_id)
    d.mkdir(parents=True, exist_ok=True)
    for name in ["mesh.obj", "mesh_info.json"]:
        if not (d / name).exists():
            urllib.request.urlretrieve(FOLDNET_URL.format(mesh_id=mesh_id, name=name), d / name)
    return d


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Keep the raw vertex order: FoldNet keypoint indices point into it.
    verts, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def weld(verts: np.ndarray, faces: np.ndarray, keypoints: dict, tol: float = 1e-6):
    """Merge coincident vertices. FoldNet panels (front/back/sleeves/collar) are separate
    pieces whose seam vertices are duplicated; unwelded, the shirt falls apart."""
    key = np.round(verts / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    faces = inv[faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    keypoints = {k: sorted({int(inv[i]) for i in v}) for k, v in keypoints.items()}
    return verts[first], faces, keypoints


def per_triangle_panel(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten each triangle isometrically into 2D for the Style3D rest shape.

    The default (vertex XY) gives back-panel triangles negative area, which Newton drops.
    Local axes follow world X projected onto the triangle so weft/warp stay consistent.
    """
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ref = np.tile([1.0, 0.0, 0.0], (len(faces), 1))
    ref[np.abs(n[:, 0]) > 0.9] = [0.0, 1.0, 0.0]
    e1 = ref - (ref * n).sum(1, keepdims=True) * n
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)

    def to2d(p):
        return np.stack([((p - a) * e1).sum(1), ((p - a) * e2).sum(1)], axis=1)

    panel = np.stack([np.zeros((len(faces), 2)), to2d(b), to2d(c)], axis=1).reshape(-1, 2)
    return panel, np.arange(3 * len(faces))


def unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


# ----------------------------------------------------------------------------- metrics


def projected_area(q: np.ndarray, faces: np.ndarray, cell: float = 0.005) -> float:
    a, b, c = q[faces[:, 0], :2], q[faces[:, 1], :2], q[faces[:, 2], :2]
    pts = [a, b, c]
    for w in [(1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6),
              (0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5)]:
        pts.append(w[0] * a + w[1] * b + w[2] * c)
    cells = np.unique(np.floor(np.concatenate(pts) / cell).astype(np.int64), axis=0)
    return len(cells) * cell * cell


def health(q, qd, edges, rest_len, radius) -> dict:
    lens = np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1)
    strain = lens / rest_len
    return {
        "nan": bool(not np.isfinite(q).all()),
        "strain_p99": float(np.percentile(strain, 99)),
        "strain_max": float(strain.max()),
        "min_z": float(q[:, 2].min()),
        "below_table_frac": float((q[:, 2] < -radius).mean()),
        "max_speed": float(np.linalg.norm(qd, axis=1).max()),
    }


# ----------------------------------------------------------------------------- trajectories


def linear_path(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    n = max(1, math.ceil(np.linalg.norm(b - a) / step))
    return a + (b - a) * np.linspace(0, 1, n + 1)[:, None]


def cosine_fling_path(x: float, y1: float, z1: float, y2: float, z2: float, step: float) -> np.ndarray:
    # FoldNet++: z = z2 + (z1 - z2) * (cos(pi * (y - y1) / (y2 - y1)) + 1) / 2, resampled by arc length
    y = np.linspace(y1, y2, 2000)
    z = z2 + (z1 - z2) * (np.cos(np.pi * (y - y1) / (y2 - y1)) + 1) / 2
    pts = np.stack([np.full_like(y, x), y, z], axis=1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    targets = np.linspace(0, s[-1], max(1, math.ceil(s[-1] / step)) + 1)
    return np.stack([np.interp(targets, s, pts[:, k]) for k in range(3)], axis=1)


def sync_paths(paths: list[np.ndarray]) -> np.ndarray:
    """Pad paths to equal length (hold last waypoint) -> [steps+1, n_pickers, 3]."""
    n = max(len(p) for p in paths)
    return np.stack([np.concatenate([p, np.repeat(p[-1:], n - len(p), axis=0)]) for p in paths], axis=1)


# ----------------------------------------------------------------------------- simulation


@wp.kernel
def drive_pinned(
    idx: wp.array(dtype=int),
    offset: wp.array(dtype=wp.vec3),
    owner: wp.array(dtype=int),
    targets: wp.array(dtype=wp.vec3),
    target_vel: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = idx[i]
    q[p] = targets[owner[i]] + offset[i]
    qd[p] = target_vel[owner[i]]


class Diverged(RuntimeError):
    def __init__(self, t: float, phase: str, bad_idx: np.ndarray):
        super().__init__(f"NaN at t={t:.2f}s in {phase} ({len(bad_idx)} vertices)")
        self.t, self.phase, self.bad_idx = t, phase, bad_idx


class ClothSim:
    def __init__(self, args, verts, faces, panel, panel_idx, pos, rot):
        self.args = args
        self.dt = args.sim_dt
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if args.solver == "style3d":
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
            k, kb = args.stretch_ke, args.bend_ke
            style3d.add_cloth_mesh(
                builder,
                pos=wp.vec3(*pos),
                rot=wp.quat(*rot),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=verts.tolist(),
                indices=faces.flatten().tolist(),
                panel_verts=panel.tolist(),
                panel_indices=panel_idx.tolist(),
                density=args.density,
                tri_aniso_ke=wp.vec3(k, k, k * args.shear_ratio),
                edge_aniso_ke=wp.vec3(kb, kb, kb),
                particle_radius=args.contact_radius,
            )
        else:  # VBD: StVK membrane + dihedral bending; rest shape is the 3D mesh itself
            builder.add_cloth_mesh(
                pos=wp.vec3(*pos),
                rot=wp.quat(*rot),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=verts.tolist(),
                indices=faces.flatten().tolist(),
                density=args.density,
                tri_ke=args.vbd_tri_ke,
                tri_ka=args.vbd_tri_ka,
                tri_kd=args.vbd_tri_kd,
                edge_ke=args.vbd_edge_ke,
                edge_kd=args.vbd_edge_kd,
                particle_radius=args.contact_radius,
            )
        ground = newton.ModelBuilder.ShapeConfig(mu=args.ground_mu) if hasattr(newton.ModelBuilder, "ShapeConfig") else None
        builder.add_ground_plane(cfg=ground)
        if args.solver == "vbd":
            builder.color()  # VBD solves vertices color by color
        self.model = builder.finalize(device=args.device)
        self.model.soft_contact_ke = args.table_ke
        self.model.soft_contact_kd = args.table_kd
        # VBD uses this one coefficient for both cloth-table and cloth-cloth friction
        self.model.soft_contact_mu = args.table_mu if args.solver == "style3d" else args.vbd_mu
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if args.solver == "style3d":
            self.solver = newton.solvers.SolverStyle3D(
                model=self.model, iterations=args.iterations, linear_iterations=args.linear_iterations
            )
            self.solver.collision.radius = args.contact_radius
            self.solver.rebuild_bvh(self.state_0)
        else:
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=args.iterations,
                particle_enable_self_contact=True,
                particle_self_contact_margin=2 * args.contact_radius,  # same 3 mm thickness as Style3D
                particle_self_contact_gap=args.contact_radius,
            )
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()

        self.rest = verts  # rest shape (before pos/rot), same vertex order as the model
        self.mass_free = self.model.particle_mass.numpy().copy()
        self.n_sub = 0
        self.t = 0.0
        self.pinned = None  # (idx, offset, owner) as warp arrays
        self.targets = wp.zeros(2, dtype=wp.vec3, device=args.device)
        self.target_vel = wp.zeros(2, dtype=wp.vec3, device=args.device)
        self.frames, self.frame_t, self.frame_phase, self.frame_pickers = [], [], [], []
        # robot state per recorded frame (fold_test --robot); NaN / False when no robot is driven
        self.cur_joint = np.full((2, 7), np.nan)  # arm joints, (l, r)
        self.cur_grip = np.zeros(2)  # gripper closure 0 open .. 1 closed, (l, r)
        self.frame_joint, self.frame_grip = [], []
        self.next_record = 0.0
        self.phase = "init"
        self.cur_targets = np.zeros((2, 3))

    def q(self) -> np.ndarray:
        return self.state_0.particle_q.numpy().astype(np.float64)

    def qd(self) -> np.ndarray:
        return self.state_0.particle_qd.numpy().astype(np.float64)

    def grasp(self, picker_pos: np.ndarray, must_include: list[int], rest_radius: float | None = None) -> dict:
        """Pin vertices inside each picker box (top-down grasp) to that picker.

        Vertices farther than --grasp-rest-radius (or `rest_radius`) from the anchor in the rest shape
        are left free: otherwise a picker over a crumpled spot clamps distant cloth regions together
        and spreading the pickers stretches the cloth between them. A large `rest_radius` pinches
        every layer under the picker, as fingers closing on the edge of a folded stack do.
        """
        rest_radius = self.args.grasp_rest_radius if rest_radius is None else rest_radius
        q = self.q()
        idx, off, own = [], [], []
        info = {"pinned": [], "dropped_far": [], "manipulable": []}
        for k, p in enumerate(picker_pos):
            anchor = must_include[k]
            if anchor is None:  # one-hand move: this picker stays open
                info["pinned"].append(0)
                info["dropped_far"].append(0)
                info["manipulable"].append(None)
                continue
            d = q - p
            # FoldNet++ EEF approaches along -z (checked on Galbot grasps), so for a top-down grasp
            # EEF z points up and the box z range maps to world dz directly
            box_xy = PICKER_BOX[:2, 1] * getattr(self.args, "picker_scale", 1.0)
            inside = (
                (np.abs(d[:, 0]) <= box_xy[0])
                & (np.abs(d[:, 1]) <= box_xy[1])
                & (d[:, 2] >= PICKER_BOX[2, 0])
                & (d[:, 2] <= PICKER_BOX[2, 1])
            )
            rest_d = np.linalg.norm(self.rest - self.rest[anchor], axis=1)
            near = rest_d <= rest_radius
            sel = (set(np.nonzero(inside & near)[0].tolist()) | {anchor}) - set(idx)
            idx += sorted(sel)
            off += [q[i] - p for i in sorted(sel)]
            own += [k] * len(sel)
            # FoldNet++ affordance: vertices within 2 cm now must lie within 8 cm in the rest shape
            around = np.linalg.norm(q - q[anchor], axis=1) <= 0.02
            info["pinned"].append(len(sel))
            info["dropped_far"].append(int((inside & ~near).sum()))
            info["manipulable"].append(bool((rest_d[around] <= 0.08).all()))
        # Pin by making the vertices very heavy rather than inactive: Style3D's collision kernels
        # take a harmonic mean of per-vertex diagonals, which is 0/0 = NaN when two inactive
        # vertices touch (e.g. two cloth layers inside one picker box).
        mass = self.mass_free.copy()
        mass[idx] = self.mass_free.max() * self.args.pin_mass_factor
        self.model.particle_mass.assign(mass)
        dev = self.args.device
        self.pinned = (
            wp.array(np.array(idx, dtype=np.int32), dtype=int, device=dev),
            wp.array(np.array(off, dtype=np.float32), dtype=wp.vec3, device=dev),
            wp.array(np.array(own, dtype=np.int32), dtype=int, device=dev),
        )
        self.cur_targets = picker_pos.copy()
        self.targets.assign(picker_pos)  # settle() holds the pickers here without calling follow()
        self.target_vel.zero_()
        return info

    def release(self):
        if self.pinned is not None:
            self._drive(self.state_0)  # hand the picker velocity to the freed vertices
        self.model.particle_mass.assign(self.mass_free)
        self.pinned = None
        self.target_vel.zero_()

    def _drive(self, state):
        idx, off, own = self.pinned
        wp.launch(drive_pinned, dim=len(idx), inputs=[idx, off, own, self.targets, self.target_vel],
                  outputs=[state.particle_q, state.particle_qd], device=self.args.device)

    def _record(self):
        if self.t + 1e-9 >= self.next_record:
            q = self.state_0.particle_q.numpy().copy()
            if not np.isfinite(q).all():
                raise Diverged(self.t, self.phase, np.nonzero(~np.isfinite(q).all(1))[0])
            self.frames.append(q)
            self.frame_t.append(self.t)
            self.frame_phase.append(self.phase)
            self.frame_pickers.append(self.cur_targets.copy() if self.pinned is not None else np.full((2, 3), np.nan))
            self.frame_joint.append(self.cur_joint.copy())
            self.frame_grip.append(self.cur_grip.copy())
            self.next_record += self.args.record_dt

    def substep(self):
        if self.n_sub % self.args.collide_every == 0:
            self.pipeline.collide(self.state_0, self.contacts)
        if self.pinned is not None:
            self._drive(self.state_0)
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.n_sub += 1
        self.t += self.dt
        self._record()

    def settle(self, duration: float, phase: str):
        self.phase = phase
        for _ in range(int(round(duration / self.dt))):
            self.substep()

    def ramp_grip(self, target, stroke_time: float, phase: str) -> float:
        """Move the grippers linearly to `target` (0 open .. 1 closed, per arm) at one full stroke per
        stroke_time seconds while the cloth keeps simulating. Returns the time taken."""
        self.phase = phase
        start, target = self.cur_grip.copy(), np.asarray(target, dtype=float)
        n = int(round(float(np.abs(target - start).max()) * stroke_time / self.dt))
        for s in range(n):
            self.cur_grip = start + (target - start) * (s + 1) / n
            self.substep()
        self.cur_grip = target.copy()
        return n * self.dt

    def follow(self, waypoints: np.ndarray, phase: str, step_dt: float = ACTION_DT, joints=None):
        """Track picker waypoints [N+1, 2, 3] spaced step_dt apart (piecewise linear).
        joints [N+1, 2, 7] (l, r), if given, is interpolated the same way for the recorded frames."""
        self.phase = phase
        n_per = int(round(step_dt / self.dt))
        for w in range(len(waypoints) - 1):
            a, b = waypoints[w], waypoints[w + 1]
            self.target_vel.assign((b - a) / step_dt)
            for s in range(n_per):
                if joints is not None:
                    self.cur_joint = joints[w] + (joints[w + 1] - joints[w]) * (s + 1) / n_per
                self.cur_targets = a + (b - a) * (s + 1) / n_per
                self.targets.assign(self.cur_targets)
                self.substep()


# ----------------------------------------------------------------------------- episode


def random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)  # (x, y, z, w), uniform on SO(3)


def quat_rotate(q, v):
    x, y, z, w = q
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return v @ r.T


def run_trial(args, trial: int, verts0, faces, keypoints, out_dir: Path) -> dict:
    rng = np.random.default_rng(args.seed + trial)
    flat_test = trial == 0
    scale = args.scale_target * (1.0 if flat_test else rng.uniform(0.9, 1.1))
    verts = verts0 * scale
    panel, panel_idx = per_triangle_panel(verts, faces)
    edges = unique_edges(faces)
    rest_len = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    flat_area = projected_area(verts, faces)

    if flat_test:
        rot = np.array([0.0, 0.0, 0.0, 1.0])
        pos = np.array([0.0, 0.0, np.abs(verts[:, 2]).max() + 0.01])
    else:
        rot = random_rotation(rng)
        rv = quat_rotate(rot, verts)
        top = int(np.argmax(rv[:, 2]))
        # hang the top vertex high enough that nothing starts under the table
        hang_z = max(args.hang_z, rv[top, 2] - rv[:, 2].min() + 0.02)
        pos = np.array([0.0, 0.0, hang_z]) - rv[top]

    sim = ClothSim(args, verts, faces, panel, panel_idx, pos, rot)
    log = {"trial": trial, "flat_test": flat_test, "scale": scale, "phases": []}
    t_wall = time.time()

    def checkpoint(name):
        q, qd = sim.q(), sim.qd()
        entry = {"name": name, "t": sim.t, "coverage": projected_area(q, faces) / flat_area,
                 **health(q, qd, edges, rest_len, args.contact_radius), "wall_s": time.time() - t_wall}
        log["phases"].append(entry)
        print(f"  [trial {trial}] {name:<18} t={sim.t:6.2f}s cov={entry['coverage']:.3f} "
              f"strain_p99={entry['strain_p99']:.3f} below={entry['below_table_frac']:.4f} "
              f"vmax={entry['max_speed']:.2f} wall={entry['wall_s']:.0f}s", flush=True)
        return entry

    try:
        run_episode(args, sim, flat_test, top if not flat_test else None, verts, keypoints, checkpoint, log)
    except Diverged as e:
        print(f"  [trial {trial}] DIVERGED: {e}", flush=True)
        pinned = set(log.get("last_pinned", []))
        log["diverged"] = {"t": e.t, "phase": e.phase, "n_bad": len(e.bad_idx),
                           "bad_idx_sample": e.bad_idx[:50].tolist(),
                           "bad_in_pinned": len(pinned & set(e.bad_idx.tolist()))}

    sim_time = sim.t
    log["sim_time_s"] = sim_time
    log["wall_time_s"] = time.time() - t_wall
    log["realtime_factor"] = sim_time / log["wall_time_s"]

    trial_dir = out_dir / f"trial_{trial:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trial_dir / "frames.npz", q=np.stack(sim.frames).astype(np.float32), faces=faces,
                        t=np.array(sim.frame_t), phase=np.array(sim.frame_phase),
                        pickers=np.stack(sim.frame_pickers).astype(np.float32))
    if not args.no_video:
        render_video(trial_dir / "video.mp4", sim, faces, 1.0 / args.record_dt)
    return log


def run_episode(args, sim: ClothSim, flat_test: bool, top, verts, keypoints, checkpoint, log):
    if flat_test:
        sim.settle(args.flat_settle, "flat_settle")
        checkpoint("flat_settle")
    else:
        top_pos = sim.q()[top]
        sim.grasp(np.stack([top_pos, top_pos]), [top, top])  # one hanging point, both pickers on it
        sim.settle(args.hang_time, "hang")
        lower = sync_paths([linear_path(top_pos, np.array([0.0, 0.0, args.hang_z]), LINEAR_STEP)] * 2)
        sim.follow(lower, "lower")
        sim.release()
        sim.settle(args.drop_settle, "drop_settle")
        checkpoint("dropped")

        for f in range(args.flings):
            ka, kb = FLING_PAIRS[f % len(FLING_PAIRS)]
            q = sim.q()
            # each keypoint has a vertex per layer; grab the top-most one
            ia = max(keypoints[ka], key=lambda i: q[i, 2])
            ib = max(keypoints[kb], key=lambda i: q[i, 2])
            if q[ia, 0] > q[ib, 0]:
                ia, ib = ib, ia
            ga, gb = q[ia].copy(), q[ib].copy()
            grasp_info = sim.grasp(np.stack([ga, gb]), [ia, ib])
            print(f"  [trial {log['trial']}] fling{f} grasp {ka}/{kb}: pinned={grasp_info['pinned']} "
                  f"dropped_far={grasp_info['dropped_far']} manipulable={grasp_info['manipulable']}", flush=True)
            log["last_pinned"] = sim.pinned[0].numpy().tolist()
            rest_d = np.linalg.norm(verts[keypoints[ka][0]] - verts[keypoints[kb][0]])
            d = min(args.stretch_ratio * rest_d, args.max_spread)
            z1 = args.lift_z
            sim.follow(sync_paths([linear_path(ga, np.array([ga[0], ga[1], z1]), LINEAR_STEP),
                                   linear_path(gb, np.array([gb[0], gb[1], z1]), LINEAR_STEP)]), f"fling{f}_lift")
            sim.follow(sync_paths([linear_path(np.array([ga[0], ga[1], z1]), np.array([-d / 2, args.fling_y1, z1]), LINEAR_STEP),
                                   linear_path(np.array([gb[0], gb[1], z1]), np.array([d / 2, args.fling_y1, z1]), LINEAR_STEP)]),
                       f"fling{f}_adjust")
            checkpoint(f"fling{f}_lifted")
            sim.follow(sync_paths([cosine_fling_path(-d / 2, args.fling_y1, z1, args.fling_y2, args.fling_z2, FLING_STEP),
                                   cosine_fling_path(d / 2, args.fling_y1, z1, args.fling_y2, args.fling_z2, FLING_STEP)]),
                       f"fling{f}_fling")
            sim.release()
            sim.settle(args.release_settle, f"fling{f}_settle")
            e = checkpoint(f"fling{f}_done")
            e.update({"pair": [ka, kb], **grasp_info})


def verdict(log: dict) -> tuple[list[str], list[str]]:
    """FAIL = solver stability (what this gate is about); WARN = fling effectiveness (tunable)."""
    fails, warns = [], []
    if "diverged" in log:
        d = log["diverged"]
        fails.append(f"diverged at t={d['t']:.2f}s in {d['phase']} ({d['n_bad']} NaN vertices)")
        return fails, warns
    for p in log["phases"]:
        if p["nan"]:
            fails.append(f"{p['name']}: NaN")
        if p["strain_p99"] > 1.3:
            fails.append(f"{p['name']}: strain_p99 {p['strain_p99']:.2f} > 1.3")
        if p["below_table_frac"] > 0.01:
            fails.append(f"{p['name']}: {100 * p['below_table_frac']:.1f}% vertices under table")
    if log["flat_test"]:
        if not 0.95 <= log["phases"][-1]["coverage"] <= 1.1:
            fails.append(f"flat coverage {log['phases'][-1]['coverage']:.2f} outside [0.95, 1.1]")
    else:
        dropped = log["phases"][0]["coverage"]
        final = log["phases"][-1]["coverage"]
        if final <= dropped:
            warns.append(f"fling did not increase coverage ({dropped:.2f} -> {final:.2f})")
    return fails, warns


# ----------------------------------------------------------------------------- video


def render_video(path: Path, sim: ClothSim, faces: np.ndarray, fps: float):
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(10, 5), dpi=80)
    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    for q, t, phase, pk in zip(sim.frames, sim.frame_t, sim.frame_phase, sim.frame_pickers):
        ax_top.clear()
        ax_side.clear()
        zc = q[faces, 2].mean(1)
        ax_top.tripcolor(q[:, 0], q[:, 1], faces, facecolors=zc, vmin=0, vmax=0.4, cmap="viridis")
        ax_top.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), aspect="equal", title=f"top  t={t:5.2f}s  {phase}")
        ax_side.scatter(q[:, 1], q[:, 2], s=0.3, c=q[:, 0], cmap="coolwarm", vmin=-0.4, vmax=0.4)
        if np.isfinite(pk).all():
            ax_top.plot(pk[:, 0], pk[:, 1], "r+", ms=12)
            ax_side.plot(pk[:, 1], pk[:, 2], "r+", ms=12)
        ax_side.axhline(0, color="k", lw=0.5)
        ax_side.set(xlim=(-0.7, 0.7), ylim=(-0.05, 1.0), aspect="equal", title="side (y-z)")
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close()
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def make_parser(description: str) -> argparse.ArgumentParser:
    """Mesh, output and cloth-physics options shared with fold_test.py."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/style3d_gate"))
    p.add_argument("--cache", type=Path, default=Path("outputs/style3d_gate/cache"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--quick", action="store_true", help="short CPU smoke test (coarse dt, short phases)")
    # solver
    p.add_argument("--sim-dt", type=float, default=2e-3,
                   help="FoldNet++ uses 8e-4; 2e-3 gave the same strain/coverage at 2.5x speed")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--linear-iterations", type=int, default=10)
    p.add_argument("--collide-every", type=int, default=5, help="substeps between table contact queries")
    p.add_argument("--record-dt", type=float, default=0.1)
    # FoldNet++ physics (Table: physical parameters)
    p.add_argument("--scale-target", type=float, default=0.57, help="geo-mean of bbox length/width [m]")
    p.add_argument("--density", type=float, default=0.2)
    p.add_argument("--stretch-ke", type=float, default=70.0)
    p.add_argument("--shear-ratio", type=float, default=1.0)
    p.add_argument("--bend-ke", type=float, default=7e-7)
    p.add_argument("--table-mu", type=float, default=0.5,
                   help="cloth friction; Style3D uses sqrt(table_mu * ground_mu) against the table")
    p.add_argument("--ground-mu", type=float, default=1.0, help="table (ground plane) material friction")
    p.add_argument("--table-ke", type=float, default=1e3,
                   help="table contact stiffness; Newton's example value 1e1 let pressed cloth sink 1-3 cm")
    p.add_argument("--table-kd", type=float, default=1e-5)
    p.add_argument("--contact-radius", type=float, default=1.5e-3,
                   help="cloth contact radius (thickness = 2x); 3e-3 puffed folded stacks, FoldNet++ uses 0.8 mm")
    p.add_argument("--pin-mass-factor", type=float, default=1e4, help="pinned vertex mass / max cloth vertex mass")
    p.add_argument("--grasp-rest-radius", type=float, default=0.08,
                   help="only pin vertices within this rest-shape distance of the grasp anchor (FoldNet++ 8 cm)")
    p.add_argument("--release-settle", type=float, default=1.0)
    # VBD alternative (has cloth-cloth friction; Style3D does not). Material units differ from Style3D.
    p.add_argument("--solver", choices=["style3d", "vbd"], default="style3d")
    p.add_argument("--vbd-tri-ke", type=float, default=1e2, help="VBD membrane shear stiffness")
    p.add_argument("--vbd-tri-ka", type=float, default=1e2, help="VBD membrane area stiffness")
    p.add_argument("--vbd-tri-kd", type=float, default=1e-6)
    p.add_argument("--vbd-edge-ke", type=float, default=1e-5, help="VBD bending stiffness")
    p.add_argument("--vbd-edge-kd", type=float, default=1e-7)
    p.add_argument("--vbd-mu", type=float, default=0.7,
                   help="VBD friction, shared by cloth-table and cloth-cloth (FoldNet++: 0.5 and 1.0)")
    return p


def load_mesh(args):
    """Init Warp, fetch and weld the FoldNet T-shirt; verts centered with unit geo-mean extent."""
    wp.init()
    if args.device.startswith("cuda") and not wp.is_cuda_available():
        raise SystemExit("Warp sees no CUDA device; check the Warp init message above (driver vs. wheel CUDA version).")
    mesh_dir = fetch_mesh(args.mesh_id, args.cache)
    verts, faces = load_obj(mesh_dir / "mesh.obj")
    keypoints = json.loads((mesh_dir / "mesh_info.json").read_text())["triangulation"]["keypoint_idx"]
    n_raw = len(verts)
    verts, faces, keypoints = weld(verts, faces, keypoints)
    print(f"welded {n_raw - len(verts)} duplicate seam vertices")
    verts[:, :2] -= (verts[:, :2].max(0) + verts[:, :2].min(0)) / 2
    ext = verts[:, :2].max(0) - verts[:, :2].min(0)
    verts /= math.sqrt(ext[0] * ext[1])  # unit geo-mean, rescaled per trial
    print(f"mesh {args.mesh_id}: {len(verts)} verts, {len(faces)} tris, device={args.device}, dt={args.sim_dt}")
    return verts, faces, keypoints


def main():
    p = make_parser(__doc__)
    p.add_argument("--trials", type=int, default=4, help="trial 0 = flat sanity check")
    p.add_argument("--flings", type=int, default=2)
    # episode
    p.add_argument("--hang-z", type=float, default=0.6)
    p.add_argument("--hang-time", type=float, default=1.0)
    p.add_argument("--drop-settle", type=float, default=2.0)
    p.add_argument("--flat-settle", type=float, default=1.5)
    p.add_argument("--lift-z", type=float, default=0.55)
    p.add_argument("--stretch-ratio", type=float, default=0.95)
    p.add_argument("--max-spread", type=float, default=0.7)
    p.add_argument("--fling-y1", type=float, default=-0.2)
    p.add_argument("--fling-y2", type=float, default=0.25)
    p.add_argument("--fling-z2", type=float, default=0.04,
                   help="fling end height; at 0.02 the pinned cloth pressed through the pile below")
    args = p.parse_args()

    if args.quick:
        args.sim_dt, args.iterations, args.linear_iterations = 4e-3, 4, 6
        args.hang_time, args.drop_settle, args.flat_settle, args.release_settle = 0.2, 0.4, 0.2, 0.2
        args.trials, args.flings, args.record_dt = min(args.trials, 2), 1, 0.2

    verts, faces, keypoints = load_mesh(args)

    args.out.mkdir(parents=True, exist_ok=True)
    logs = []
    for trial in range(args.trials):
        log = run_trial(args, trial, verts, faces, keypoints, args.out)
        log["fails"], log["warns"] = verdict(log)
        logs.append(log)
        status = "FAIL: " + "; ".join(log["fails"]) if log["fails"] else "STABLE"
        if log["warns"]:
            status += "  (WARN: " + "; ".join(log["warns"]) + ")"
        print(f"  [trial {trial}] realtime x{log['realtime_factor']:.3f}  {status}", flush=True)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.out / "metrics.json").write_text(json.dumps({"config": cfg, "trials": logs}, indent=2))
    n_stable = sum(not log["fails"] for log in logs)
    n_warn = sum(bool(log["warns"]) for log in logs)
    print(f"\n{n_stable}/{len(logs)} trials stable, {n_warn} with fling warnings. metrics -> {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
