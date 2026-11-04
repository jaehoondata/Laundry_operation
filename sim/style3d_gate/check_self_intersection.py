"""Count cloth self-intersections in gate_fling.py outputs (frames.npz).

An edge counts as intersecting when its segment passes through a triangle that shares
no vertex with it. Runs on CPU; no GPU needed.

    python sim/style3d_gate/check_self_intersection.py outputs/style3d_gate/<run>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

MAX_HITS_PER_EDGE = 16  # skips over the fan of triangles around the edge's own vertices


@wp.kernel
def edge_hits(
    mesh: wp.uint64,
    pts: wp.array(dtype=wp.vec3),
    edges: wp.array(dtype=wp.vec2i),
    tris: wp.array(dtype=wp.vec3i),
    out: wp.array(dtype=int),
):
    e = wp.tid()
    i0 = edges[e][0]
    i1 = edges[e][1]
    a = pts[i0]
    d = pts[i1] - a
    length = wp.length(d)
    out[e] = 0
    if length < 1.0e-9:
        return
    direction = d / length
    t_end = length * (1.0 - 1.0e-4)
    t0 = float(0.0)
    for _k in range(MAX_HITS_PER_EDGE):
        if t0 >= t_end:
            break
        q = wp.mesh_query_ray(mesh, a + direction * t0, direction, t_end - t0)
        if not q.result:
            break
        tri = tris[q.face]
        shared = tri[0] == i0 or tri[1] == i0 or tri[2] == i0 or tri[0] == i1 or tri[1] == i1 or tri[2] == i1
        if not shared:
            out[e] = 1
            break
        t0 = t0 + q.t + 1.0e-6


def unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def analyze(npz: Path, device: str) -> dict:
    d = np.load(npz)
    q, faces, t, phase = d["q"], d["faces"].astype(np.int32), d["t"], d["phase"]
    edges = unique_edges(faces).astype(np.int32)
    pts = wp.array(q[0], dtype=wp.vec3, device=device)
    mesh = wp.Mesh(points=pts, indices=wp.array(faces.flatten(), dtype=wp.int32, device=device))
    edges_wp = wp.array(edges, dtype=wp.vec2i, device=device)
    tris_wp = wp.array(faces, dtype=wp.vec3i, device=device)
    out = wp.zeros(len(edges), dtype=int, device=device)

    counts = np.zeros(len(q), dtype=np.int64)
    for i in range(len(q)):
        pts.assign(q[i])
        mesh.refit()
        wp.launch(edge_hits, dim=len(edges), inputs=[mesh.id, pts, edges_wp, tris_wp], outputs=[out], device=device)
        counts[i] = int(out.numpy().sum())

    per_phase = {}
    for p in dict.fromkeys(phase.tolist()):
        m = phase == p
        per_phase[p] = {"max": int(counts[m].max()), "mean": float(counts[m].mean()), "end": int(counts[m][-1])}
    worst = int(counts.argmax())
    return {
        "n_edges": len(edges),
        "first_frame": int(counts[0]),
        "last_frame": int(counts[-1]),
        "max": int(counts.max()),
        "max_at": {"t": float(t[worst]), "phase": str(phase[worst])},
        "per_phase": per_phase,
        "counts": counts.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    wp.config.quiet = True
    wp.init()
    report = {}
    for npz in sorted(args.run_dir.glob("trial_*/frames.npz")):
        r = analyze(npz, args.device)
        report[npz.parent.name] = r
        print(f"{npz.parent.name}: {r['n_edges']} edges | intersecting edges first={r['first_frame']} "
              f"max={r['max']} (t={r['max_at']['t']:.1f}s {r['max_at']['phase']}) last={r['last_frame']}")
        for ph, s in r["per_phase"].items():
            print(f"    {ph:<16} max={s['max']:<5} mean={s['mean']:<8.1f} end={s['end']}")
    (args.run_dir / "self_intersection.json").write_text(json.dumps(report, indent=1))
    print(f"-> {args.run_dir / 'self_intersection.json'}")


if __name__ == "__main__":
    main()
