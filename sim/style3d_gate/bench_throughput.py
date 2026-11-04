"""Throughput benchmark for the Style3D cloth solver: N shirts in one model, eager vs CUDA graph.

Each shirt is dropped from a random orientation onto the table (self-contact heavy, no
pickers), spaced apart so they never touch. Reports simulated seconds per wall second,
per shirt and aggregate.

Caveat: all shirts share one PCG solve, so its step sizes are global. With a fixed
iteration count each shirt is solved a little less accurately than when run alone.
Shirt 0 starts from the same pose for every N, so its final coverage (cov0) should stay
close to the N=1 value if batching does not hurt fidelity.

    python sim/style3d_gate/bench_throughput.py --envs 1 4 16 32 64
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import style3d

from gate_fling import fetch_mesh, load_obj, per_triangle_panel, projected_area, quat_rotate, random_rotation, weld

SPACING = 1.6  # m between shirt centers; shirts are ~0.7 m across


def build(args, verts, faces, n_envs, seed):
    rng = np.random.default_rng(seed)
    panel, panel_idx = per_triangle_panel(verts, faces)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    cols = math.ceil(math.sqrt(n_envs))
    for i in range(n_envs):
        rot = random_rotation(rng)
        rv = quat_rotate(rot, verts)
        center = np.array([(i % cols) * SPACING, (i // cols) * SPACING, 0.0])
        pos = center + [0.0, 0.0, 0.05 - rv[:, 2].min()] - [*rv[:, :2].mean(0), 0.0]
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
            tri_aniso_ke=wp.vec3(args.stretch_ke, args.stretch_ke, args.stretch_ke),
            edge_aniso_ke=wp.vec3(args.bend_ke, args.bend_ke, args.bend_ke),
            particle_radius=args.contact_radius,
        )
    builder.add_ground_plane()
    model = builder.finalize(device=args.device)
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 1e1, 1e-5, 0.5
    model.set_gravity((0.0, 0.0, -9.81))
    solver = newton.solvers.SolverStyle3D(model=model, iterations=args.iterations,
                                          linear_iterations=args.linear_iterations)
    solver.collision.radius = args.contact_radius
    state_0, state_1 = model.state(), model.state()
    solver.rebuild_bvh(state_0)
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    return model, solver, [state_0, state_1], control, pipeline, contacts


def run(args, verts, faces, n_envs, use_graph) -> dict:
    model, solver, states, control, pipeline, contacts = build(args, verts, faces, n_envs, args.seed)
    k = args.collide_every  # even, so the state swap returns to the same buffers after a block

    def block():
        pipeline.collide(states[0], contacts)
        for s in range(k):
            src, dst = states[s % 2], states[(s + 1) % 2]
            src.clear_forces()
            solver.step(src, dst, control, contacts, args.sim_dt)

    graph = None
    if use_graph:
        block()  # load modules before capture
        with wp.ScopedCapture(device=args.device) as cap:
            block()
        graph = cap.graph

    def advance(n_blocks):
        for _ in range(n_blocks):
            if graph is not None:
                wp.capture_launch(graph)
            else:
                block()

    block_dt = k * args.sim_dt
    advance(max(1, int(args.warmup / block_dt)))
    wp.synchronize_device(args.device)
    n_blocks = max(1, int(args.seconds / block_dt))
    t0 = time.perf_counter()
    advance(n_blocks)
    wp.synchronize_device(args.device)
    wall = time.perf_counter() - t0

    sim = n_blocks * block_dt
    q = states[0].particle_q.numpy()
    nv = len(verts)
    finite = bool(np.isfinite(q).all())
    cov0 = projected_area(q[:nv], faces) / projected_area(verts, faces) if finite else float("nan")
    return {"envs": n_envs, "graph": use_graph, "sim_s": sim, "wall_s": wall,
            "rt_per_env": sim / wall, "rt_aggregate": n_envs * sim / wall, "finite": finite, "cov_env0": cov0}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    p.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    p.add_argument("--seconds", type=float, default=1.0, help="simulated seconds to time")
    p.add_argument("--warmup", type=float, default=0.2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mesh-id", type=int, default=0)
    p.add_argument("--cache", type=Path, default=Path("outputs/style3d_gate/cache"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-dt", type=float, default=2e-3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--linear-iterations", type=int, default=10)
    p.add_argument("--collide-every", type=int, default=6, help="substeps per block (even)")
    p.add_argument("--scale-target", type=float, default=0.57)
    p.add_argument("--density", type=float, default=0.2)
    p.add_argument("--stretch-ke", type=float, default=70.0)
    p.add_argument("--bend-ke", type=float, default=7e-7)
    p.add_argument("--contact-radius", type=float, default=3e-3)
    args = p.parse_args()
    if args.collide_every % 2:
        p.error("--collide-every must be even")

    wp.config.quiet = True
    wp.init()
    if args.device.startswith("cuda") and not wp.is_cuda_available():
        raise SystemExit("Warp sees no CUDA device; check the Warp init message above (driver vs. wheel CUDA version).")
    mesh_dir = fetch_mesh(args.mesh_id, args.cache)
    verts, faces = load_obj(mesh_dir / "mesh.obj")
    keypoints = json.loads((mesh_dir / "mesh_info.json").read_text())["triangulation"]["keypoint_idx"]
    verts, faces, _ = weld(verts, faces, keypoints)
    verts[:, :2] -= (verts[:, :2].max(0) + verts[:, :2].min(0)) / 2
    ext = verts[:, :2].max(0) - verts[:, :2].min(0)
    verts *= args.scale_target / math.sqrt(ext[0] * ext[1])

    modes = {"eager": [False], "graph": [True], "both": [False, True]}[args.mode]
    if not args.device.startswith("cuda"):
        modes = [False]
    print(f"{len(verts)} verts/shirt, dt={args.sim_dt}, iters={args.iterations}x{args.linear_iterations}, "
          f"{args.seconds}s timed per config")
    print(f"{'envs':>5} {'mode':>6} {'wall[s]':>8} {'rt/env':>8} {'rt total':>9} {'finite':>7} {'cov0':>6}")
    for n in args.envs:
        for g in modes:
            r = run(args, verts, faces, n, g)
            print(f"{n:>5} {'graph' if g else 'eager':>6} {r['wall_s']:>8.1f} {r['rt_per_env']:>8.4f} "
                  f"{r['rt_aggregate']:>9.3f} {str(r['finite']):>7} {r['cov_env0']:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
