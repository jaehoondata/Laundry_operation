"""T3: replay a FoldNet++ folding segment with SG2 and simulate the cloth (Newton Style3D).

Reads <plan_dir>/plan.npz and fit.json from prepare_episode.py. The FoldNet T-shirt mesh is scaled,
turned and placed where fit.json puts it (flat, front up, layers --init-gap apart), then the pickers
follow the SG2 TCP positions step by step for the planned step durations. A hand grasps when its
gripper closes (FoldNet++ picker box at the TCP, anchored on the top-most cloth vertex within 2 cm)
and lets go when it opens; each hand grasps and releases on its own. Physics are FoldNet++'s
(gate_fling.py defaults) with the cloth-table friction at FoldNet++'s 0.5 (Style3D uses
sqrt(table_mu * ground_mu), so both are set to 0.5).

--retarget keypoint (option A): the cloth here does not end up exactly where FoldNet++'s cloth was, so
replaying the hand paths as they are closes later grasps in the air. With --retarget, before each
grasp the hand's target is moved onto the simulated shirt and the IK is re-solved online:
  * subtasks that name a keypoint ("corner tip", "shoulder tip", "bottom midpoint"): the nearest
    such FoldNet keypoint (top-layer vertex) where it is now
  * other grasps: the original point if it lies on the cloth at least --edge-margin inside its
    top-down outline, else the nearest point that does
The offset (target - original) is blended in over the approach (min-jerk, up to 8 steps, from the
hand's previous release) and kept while the hand carries and until its next grasp, so the motion's
shape and timing stay FoldNet++'s. Step durations are stretched again where the re-solved joints
would move faster than the real teleop maximum.

Writes <plan_dir>/frames.npz in fold_test.py's format (q, faces, t, phase, pickers, joint, grip), so
sim/sg2_newton/render_newton.py, sim/isaac/render_trial.sh and export_lerobot.py work on it.

    python sim/foldnetxx/t3_replay.py outputs/foldnetxx/t3/<name> [--record-dt 0.0333333] [--retarget keypoint]
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "style3d_gate"))
from gate_fling import ClothSim, load_mesh, make_parser, per_triangle_panel, projected_area  # noqa: E402


def axis_quat(axis, angle):
    s = math.sin(angle / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)])  # (x, y, z, w)


def rebuild_pins(sim, idx, off, own):
    import warp as wp

    if len(idx) == 0:
        sim.pinned = None
        sim.model.particle_mass.assign(sim.mass_free)
        return
    mass = sim.mass_free.copy()
    mass[idx] = sim.mass_free.max() * sim.args.pin_mass_factor
    sim.model.particle_mass.assign(mass)
    dev = sim.args.device
    sim.pinned = (wp.array(np.asarray(idx, np.int32), dtype=int, device=dev),
                  wp.array(np.asarray(off, np.float32), dtype=wp.vec3, device=dev),
                  wp.array(np.asarray(own, np.int32), dtype=int, device=dev))


def current_pins(sim):
    if sim.pinned is None:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0, int)
    return tuple(a.numpy() for a in sim.pinned)


def top_near(sim, xy, radius=0.02):
    q = sim.q()
    d = np.linalg.norm(q[:, :2] - xy, axis=1)
    near = np.nonzero(d < radius)[0]
    return int(near[np.argmax(q[near, 2])]) if len(near) else int(d.argmin())


def grasp_hand(sim, hand, targets):
    """Pin cloth in `hand`'s picker box, keeping the other hand's pins."""
    idx0, off0, own0 = current_pins(sim)
    keep = own0 != hand
    anchors = [top_near(sim, targets[hand, :2]) if k == hand else None for k in range(2)]
    info = sim.grasp(targets, anchors)
    idx1, off1, own1 = current_pins(sim)
    rebuild_pins(sim, np.concatenate([idx0[keep], idx1]), np.concatenate([off0[keep], off1]),
                 np.concatenate([own0[keep], own1]))
    return info["pinned"][hand]


def release_hand(sim, hand):
    idx, off, own = current_pins(sim)
    if not len(idx):
        return
    sim._drive(sim.state_0)  # hand the picker velocity to the freed vertices
    keep = own != hand
    rebuild_pins(sim, idx[keep], off[keep], own[keep])
    if sim.pinned is None:
        sim.target_vel.zero_()


KEYPOINT_SUBTASKS = (("tip", ("l_corner", "r_corner", "l_shoulder", "r_shoulder")),
                     ("bottom midpoint", ("spine_bottom_f", "spine_bottom_b")))
APPROACH_STEPS = 8  # t1_kinematics.active_mask(before=8): a hand leaves its park pose 8 steps before closing


def footprint(xy, cell, margin):
    """Top-down cloth outline on a `cell` grid (vertices + triangle-edge midpoints, small gaps closed),
    eroded by `margin`: (inside mask, grid origin)."""
    from scipy import ndimage

    lo = xy.min(0) - 2 * margin
    ij = np.floor((xy - lo) / cell).astype(int)
    grid = np.zeros(ij.max(0) + int(4 * margin / cell) + 2, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_fill_holes(ndimage.binary_closing(grid, iterations=2))
    return ndimage.binary_erosion(grid, iterations=max(1, int(round(margin / cell)))), lo


class Retarget:
    """Per-hand xy offsets that move each FoldNet++ grasp onto the simulated shirt (see module doc)."""

    def __init__(self, sim, faces, kp, plan, args):
        self.sim, self.faces, self.kp, self.args = sim, faces, kp, args
        self.target, self.active, self.grip, self.sub = plan["target"], plan["active"], plan["grip"] > 0.5, plan["subtask"]
        self.d = np.zeros((2, 2))
        self.ramp = [None, None]
        self.log = []
        n = len(self.grip)
        self.closes = [[k for k in range(n) if self.grip[k, h] and (k == 0 or not self.grip[k - 1, h])] for h in range(2)]
        self.opens = [[k for k in range(1, n) if not self.grip[k, h] and self.grip[k - 1, h]] for h in range(2)]

    def goal(self, h, k):
        """Where hand h should close at step k on the cloth as it is now, and how it was chosen."""
        p = self.target[k, h, :2]
        x = self.sim.q()
        sub = str(self.sub[k])
        for key, names in KEYPOINT_SUBTASKS:
            if key in sub:
                pts = {n: x[max(self.kp[n], key=lambda i: x[i, 2]), :2] for n in names}
                name = min(pts, key=lambda n: np.linalg.norm(pts[n] - p))
                return pts[name], name
        mid = x[self.faces].mean(1)[:, :2]
        edges = np.concatenate([x[self.faces[:, [a, b]]].mean(1)[:, :2] for a, b in ((0, 1), (1, 2), (2, 0))])
        inside, lo = footprint(np.concatenate([x[:, :2], mid, edges]), self.args.footprint_cell, self.args.edge_margin)
        c = self.args.footprint_cell
        ij = np.floor((p - lo) / c).astype(int)
        if (ij >= 0).all() and (ij < inside.shape).all() and inside[ij[0], ij[1]]:
            return p, "on cloth"
        cells = np.argwhere(inside)
        if not len(cells):
            return p, "no cloth"
        pts = lo + (cells + 0.5) * c
        return pts[np.argmin(np.linalg.norm(pts - p, axis=1))], "snapped inside"

    def offset(self, k):
        """Offsets [2, 2] to apply at step k; plans each hand's next grasp when its approach starts."""
        for h in range(2):
            nxt = [g for g in self.closes[h] if g >= k]
            if nxt:
                g = nxt[0]
                prev = max([o for o in self.opens[h] if o < g], default=0)
                start = max(prev, g - APPROACH_STEPS, int((self.sub != "approach").argmax()))
                if k == start and np.isfinite(self.target[g, h]).all():
                    goal, how = self.goal(h, g)
                    d_to = goal - self.target[g, h, :2]
                    self.ramp[h] = (k, g, self.d[h].copy(), d_to)
                    self.log.append({"step": g, "hand": "lr"[h], "subtask": str(self.sub[g]), "how": how,
                                     "planned_at": k, "shift_cm": round(100 * float(np.linalg.norm(d_to)), 1)})
            r = self.ramp[h]
            if r and r[0] <= k <= r[1]:
                s = 1.0 if r[1] == r[0] else (k - r[0]) / (r[1] - r[0])
                w = 10 * s**3 - 15 * s**4 + 6 * s**5
                self.d[h] = r[2] + w * (r[3] - r[2])
        return self.d.copy()


def main():
    p = make_parser(__doc__)
    p.add_argument("plan_dir", type=Path)
    p.add_argument("--init-gap", type=float, default=0.004)
    p.add_argument("--picker-scale", type=float, default=2.0,
                   help="scale the FoldNet++ picker box in xy: at 2 x 1.6 cm it holds about one vertex per layer of "
                        "this mesh (~1.6 cm spacing), and most grasps pinned a single vertex")
    p.add_argument("--flat-settle", type=float, default=0.5)
    p.add_argument("--end-settle", type=float, default=1.0)
    p.add_argument("--retarget", choices=("none", "keypoint"), default="none",
                   help="keypoint: move each grasp onto the simulated shirt and re-solve the IK online")
    p.add_argument("--edge-margin", type=float, default=0.015, help="retarget: grasp at least this far inside the cloth outline")
    p.add_argument("--footprint-cell", type=float, default=0.005)
    p.set_defaults(record_dt=1.0 / 30.0, table_mu=0.5, ground_mu=0.5)
    args = p.parse_args()

    plan = np.load(args.plan_dir / "plan.npz")
    fit = json.loads((args.plan_dir / "fit.json").read_text())
    args.mesh_id = fit.get("mesh_id", args.mesh_id)
    verts0, faces, keypoints = load_mesh(args)
    verts = verts0 * fit["scale"]
    panel, panel_idx = per_triangle_panel(verts, faces)
    rest = verts
    z = verts[:, 2] - verts[:, 2].min()
    verts = verts.copy()
    verts[:, 2] = verts[:, 2].min() + z * min(1.0, args.init_gap / max(z.max(), 1e-9))
    rot = axis_quat((0, 0, 1), math.radians(fit["theta_deg"]))
    pos = np.array([fit["t"][0], fit["t"][1], -verts[:, 2].min() + 0.005])
    sim = ClothSim(args, verts, faces, panel, panel_idx, pos, rot)
    sim.rest = rest

    q, tcp, grip, dt, sub = (plan[k].astype(float) if k != "subtask" else plan[k] for k in ("q", "tcp", "grip", "dt", "subtask"))
    q, tcp, dt = q.copy(), tcp.copy(), dt.copy()
    rt = None
    if args.retarget != "none":
        sys.path.insert(0, str(HERE.parent / "sg2_newton"))
        sys.path.insert(0, str(HERE.parent / "sg2_reach"))
        from posture_prior import PRIOR_PATH
        from sg2_robot import SG2Arms
        from t1_kinematics import REAL_MAX_SPEED, STEP_DT

        arms = SG2Arms(base_y=fit["base_y"], height=fit["height"], prior=PRIOR_PATH, start_idle=True)
        park = {s: arms.park(s) for s in "lr"}
        rt = Retarget(sim, faces, keypoints, plan, args)

        def resolve(k):
            """Re-solve step k's joints from step k-1 with the retarget offsets (one IK step, as planned)."""
            d = rt.offset(k)
            for h, s in enumerate("lr"):
                if not rt.active[k, h]:
                    pos = park[s]
                else:
                    pos = rt.target[k, h].copy()
                    pos[:2] += d[h]
                t_prior, q_ref = arms.target(s, pos)
                q[k, h], _, _, _ = arms.solve(arms.iks[s], s, pos, q[k - 1, h], (t_prior, q_ref))
                tcp[k, h] = arms.tcp(s, q[k, h])
            dt[k - 1] = max(dt[k - 1], STEP_DT, float(np.abs(q[k] - q[k - 1]).max()) / REAL_MAX_SPEED)
    sim.cur_joint = q[0].astype(float)
    sim.settle(args.flat_settle, "flat")
    flat_area = projected_area(sim.q(), faces)
    events = []
    first = int((sub != "approach").argmax())
    for k in range(len(q) - 1):
        if rt is not None and k == first:
            resolve(k)  # the planned first FoldNet++ pose may already move a little
        if rt is not None and k + 1 > first:
            resolve(k + 1)
        for h in range(2):
            was = grip[k - 1, h] if k else 0.0
            if grip[k, h] > 0.5 and was <= 0.5:
                events.append((float(sim.t), str(sub[k]), "lr"[h], "grasp", grasp_hand(sim, h, tcp[k])))
            elif grip[k, h] <= 0.5 and was > 0.5:
                release_hand(sim, h)
                events.append((float(sim.t), str(sub[k]), "lr"[h], "release", 0))
        sim.cur_grip = grip[k]
        n_dt = max(1, int(round(float(dt[k]) / args.sim_dt)))
        sim.follow(tcp[k:k + 2], str(sub[k]), step_dt=n_dt * args.sim_dt, joints=q[k:k + 2])
    for h in range(2):
        release_hand(sim, h)
    sim.cur_grip = np.zeros(2)
    sim.settle(args.end_settle, "end")
    final_area = projected_area(sim.q(), faces)

    np.savez_compressed(args.plan_dir / "frames.npz", q=np.stack(sim.frames).astype(np.float32), faces=faces,
                        t=np.array(sim.frame_t), phase=np.array(sim.frame_phase),
                        pickers=np.stack(sim.frame_pickers).astype(np.float32),
                        joint=np.stack(sim.frame_joint).astype(np.float32), grip=np.stack(sim.frame_grip))
    summary = {"sim_s": sim.t, "flat_area": flat_area, "final_area": final_area,
               "final_footprint": final_area / max(flat_area, 1e-9),
               "pinned_per_grasp": [e[4] for e in events if e[3] == "grasp"], "events": events,
               "retarget": args.retarget, "retarget_log": rt.log if rt else [], "duration_s": float(dt.sum())}
    (args.plan_dir / "replay.json").write_text(json.dumps(summary, indent=1))
    print(f"{args.plan_dir.name}: {sim.t:.1f} s sim, final footprint {summary['final_footprint']:.2f} of flat, "
          f"vertices per grasp {summary['pinned_per_grasp']}")


if __name__ == "__main__":
    main()
