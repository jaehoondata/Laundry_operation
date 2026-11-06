"""T2: recover the flat shirt a FoldNet++ folding segment starts from, using where the grippers first close.

Every FoldNet++ folding segment opens with "Use the left gripper to grasp the corner tip of the clothing,
and the right gripper to grasp the shoulder tip.": both grippers close on the hem corner and the shoulder
of the same (near, -y) side of a flat shirt. Their EEF positions at closing pin that side, so a FoldNet
T-shirt mesh (flat, top layer keypoints) is placed by a 2D similarity: scale from the corner-to-shoulder
distance, rotation from its direction, translation from the corner; the side (mesh l_* or r_*, i.e.
front or back up) is the one that puts the body on the far (+y) side.

Which gripper took the corner is not fixed (the "left/right gripper" words do not say): in about half
of the episodes of every robot it is gripper 1. The later "grasp the bottom midpoint" closes at the hem
end of the folded strip, so the gripper whose first close lies on that end took the hem corner. A
two-point fit always matches its two points, so the bottom-midpoint grasp is also the check: its
distance to the fitted hem midpoint (spine_bottom_f of the flat shirt) along the first-grasp axis: the
"pull toward / away from the robot" steps slide the shirt across that axis, not along it, so a right fit
puts it within a few cm, a flipped one a shirt length (~0.5 m) away.

Checks (no cloth state is stored, so these are indirect):
  size       fitted shirt size (geo-mean of the flat bbox) vs FoldNet++'s 0.57 m x U(0.9, 1.1)
  centerline the next step ("Place the lifted part onto the centerline of the T-shirt.") releases at
             the shirt's lengthwise centerline: fitted centerline y vs the release y of both grippers

    python sim/foldnetxx/fit_initial_shirt.py outputs/foldnetxx_sample/valid/galbot_one_golf --out fit.json
"""

import argparse
import glob
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLAT = "The T-shirt is flattened."
FIRST = "Use the left gripper to grasp the corner tip of the clothing, and the right gripper to grasp the shoulder tip."
PLACE1 = "Place the lifted part onto the centerline of the T-shirt."
MIDPOINT = "Use the left gripper to grasp the bottom midpoint of the T-shirt."
FOLDNETXX_SIZE = (0.57 * 0.9, 0.57 * 1.1)


def load_mesh(mesh_id=0, cache=ROOT.parent / "outputs/style3d_gate/cache"):
    """FoldNet T-shirt mesh as fold_test.py uses it (welded, centered, unit geo-mean xy extent).
    Same steps as gate_fling.load_obj / weld / load_mesh, without importing warp (not needed here)."""
    d = Path(cache) / "tshirt_sp" / str(mesh_id)
    verts, faces = [], []
    for line in (d / "mesh.obj").read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
    v, f = np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    kp = json.loads((d / "mesh_info.json").read_text())["triangulation"]["keypoint_idx"]
    _, first, inv = np.unique(np.round(v / 1e-6).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    kp = {k: sorted({int(inv[i]) for i in idx}) for k, idx in kp.items()}
    v, f = v[first], inv[f]
    v[:, :2] -= (v[:, :2].max(0) + v[:, :2].min(0)) / 2
    ext = v[:, :2].max(0) - v[:, :2].min(0)
    v /= math.sqrt(ext[0] * ext[1])
    return v, f, kp


def kp_xy(v, kp, name):
    return v[max(kp[name], key=lambda i: v[i, 2]), :2]


def similarity(src_a, src_b, dst_a, dst_b):
    """2D scale s, rotation R, translation t with s R src + t: src_a -> dst_a, src_b -> dst_b."""
    ms, md = src_b - src_a, dst_b - dst_a
    s = np.linalg.norm(md) / np.linalg.norm(ms)
    th = math.atan2(md[1], md[0]) - math.atan2(ms[1], ms[0])
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return s, rot, dst_a - s * rot @ src_a, th


def episode_events(path):
    import h5py

    with h5py.File(path, "r") as h:
        a = h["actions_eef"][:]
        sub = [s.decode() for s in h["subtasks"][:]]
    k0 = sub.index(FLAT)
    a, sub = a[k0:], sub[k0:]
    closed = a[:, 18:20] > 0.5
    first = [k for k, s in enumerate(sub) if s == FIRST]
    # first close of each gripper inside the first grasp subtask (or right after it)
    close = {}
    for g, off in ((0, 0), (1, 9)):
        ks = [k for k in range(first[0], min(len(sub), first[-1] + 15)) if closed[k, g]]
        close[g] = a[ks[0], off:off + 3] if ks else None
    place = [k for k, s in enumerate(sub) if s == PLACE1]
    release = None
    if place:
        ks = [k for k in range(place[0], len(sub)) if not closed[k].any()]
        release = a[ks[0] - 1] if ks else a[place[-1]]
    mid = None
    for k in range(1, len(sub)):
        if sub[k] == MIDPOINT:
            g = [g for g in range(2) if closed[k, g] and not closed[k - 1, g]]
            if g:
                mid = a[k, 9 * g[0]:9 * g[0] + 3]
                break
    return close, release, mid


def fit(path, v, kp):
    close, release, mid_grasp = episode_events(path)
    if close[0] is None or close[1] is None:
        return None
    corner_g = 0
    if mid_grasp is not None:  # the hem end of the strip is where the bottom midpoint is grasped
        ax = close[1][:2] - close[0][:2]
        if (mid_grasp[:2] - close[0][:2]) @ ax / (ax @ ax) > 0.5:
            corner_g = 1
    corner_o, shoulder_o = close[corner_g][:2], close[1 - corner_g][:2]
    best = None
    for side in "lr":
        s, rot, t, th = similarity(kp_xy(v, kp, f"{side}_corner"), kp_xy(v, kp, f"{side}_shoulder"), corner_o, shoulder_o)
        placed = (s * (rot @ v[:, :2].T)).T + t
        far = placed[:, 1].mean() - (corner_o[1] + shoulder_o[1]) / 2  # body must lie on the far side
        if best is None or far > best["far"]:
            best = {"side": side, "scale": s, "theta": th, "t": t, "placed": placed, "far": far}
    placed = best["placed"]
    ext = placed.max(0) - placed.min(0)
    size = math.sqrt(ext[0] * ext[1])
    # lengthwise centerline: halfway between the near and far side edges, measured across the shirt
    along = (shoulder_o - corner_o) / np.linalg.norm(shoulder_o - corner_o)
    across = np.array([-along[1], along[0]])
    if across[1] < 0:
        across = -across
    proj = (placed - corner_o) @ across
    mid = (corner_o + shoulder_o) / 2
    center_y = float(mid[1] + (proj.min() + proj.max()) / 2 * across[1])
    out = {"episode": str(path), "side": best["side"], "scale": float(best["scale"]), "size": size,
           "theta_deg": math.degrees(best["theta"]), "t": best["t"].tolist(),
           "corner": corner_o.tolist(), "shoulder": shoulder_o.tolist(), "centerline_y": center_y,
           "corner_gripper": corner_g}
    if mid_grasp is not None:
        th = best["theta"]
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        hem = best["scale"] * rot @ kp_xy(v, kp, "spine_bottom_f") + best["t"]
        along = (shoulder_o - corner_o) / np.linalg.norm(shoulder_o - corner_o)
        out["midpoint_to_hem_m"] = float(abs((mid_grasp[:2] - hem) @ along))
    if release is not None:
        out["place_release_y"] = [float(release[1]), float(release[10])]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--mesh-id", type=int, default=0)
    p.add_argument("--max-episodes", type=int, default=200)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    v, _, kp = load_mesh(args.mesh_id)
    report = {}
    for d in args.dirs:
        rows = [r for r in (fit(Path(f), v, kp) for f in sorted(glob.glob(str(d / "**/*.h5"), recursive=True))[: args.max_episodes]) if r]
        size = np.array([r["size"] for r in rows])
        dy = np.array([np.mean(r["place_release_y"]) - r["centerline_y"] for r in rows if "place_release_y" in r])
        hem = np.array([r["midpoint_to_hem_m"] for r in rows if "midpoint_to_hem_m" in r])
        inside = np.mean((size >= FOLDNETXX_SIZE[0]) & (size <= FOLDNETXX_SIZE[1]))
        print(f"{d.name:20s} n {len(rows):3d} | size p10/50/90 {np.percentile(size, [10, 50, 90]).round(3)} "
              f"(FoldNet++ {FOLDNETXX_SIZE[0]:.3f}-{FOLDNETXX_SIZE[1]:.3f}; inside {100 * inside:.0f}%) | "
              f"theta p50 {np.median([r['theta_deg'] for r in rows]):.1f} deg | side {sorted(set(r['side'] for r in rows))} | "
              f"place release - centerline y p10/50/90 {np.percentile(dy, [10, 50, 90]).round(3)} m | "
              f"corner by gripper 1 {np.mean([r['corner_gripper'] for r in rows]):.2f} | "
              f"bottom-midpoint grasp to hem p50/90 {np.percentile(hem, [50, 90]).round(3)} m (n {len(hem)})")
        report[d.name] = rows
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
