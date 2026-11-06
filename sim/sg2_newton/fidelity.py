"""How closely sim episodes reproduce the real teleop dataset (the ground truth), one report.

Everything is computed the same way on both sides, from the 16-joint state (14 arm joints +
2 grippers) and, optionally, the head camera:
  joints     per-joint 1-D Wasserstein distance of the state distributions [rad], mean over 14
  hands      hand (TCP, sim frame) position distributions: W1 per axis [cm]; moving speed; height
  episode    length [s]; grasp pattern (which hands hold, in order) share starting LR-L-R-L
  steps      for each step of LR-L-R-L: where each hand closes / opens [cm off the real median],
             when (s from the episode start), how long it holds, how high it lifts
  images     (with --real-videos / sim head.mp4) shirt mask IoU of the first frame (placement and
             size), and how far the shirt's far edge moves toward the robot in the hem fold (the top
             mask row cast onto the table with the calibrated head camera; real 0.24 m)
  cloth      the cloth state, which decides whether sim folds like the real hands do, not just moves
             like them. From the head camera, the same way on both sides:
               after each step: shirt area / first frame, taken at the least occluded frame between
                 that release and the next grasp (largest shirt mask; right after a release the
                 gripper still covers the shirt)
               final frame (arms back): area, aspect, fill (1 = clean rectangle), wrinkle (mean
                 brightness gradient inside the shirt; compare with Isaac renders, --video-dir isaac,
                 since Newton's faceted shading inflates it)
             each must lie in the real episodes' p10-p90 range (PASS / FAIL). Sim only, from the mesh:
             sleeve vertices left outside the body's final footprint (gated: median <= 0.15, the
             image gates alone passed shirts with the sleeves still out), stack height p95, and the
             share of sharply bent edges

    python sim/sg2_newton/fidelity.py --real path/to/data/chunk-000/*.parquet \
        [--real-videos path/to/videos/observation.images.head/chunk-000 --real-meta path/to/meta/episodes/chunk-000] \
        -- outputs/style3d_gate/sim_v9/base/trial_0* [--json report.json] [--video-dir isaac]

Real per-episode image statistics are cached in --real-cache (decoding 24 videos takes minutes).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
from compare_real import CLOSED, FPS, grasp_pattern, hands, load_real, load_sim  # noqa: E402
from sg2_reach import DEFAULT_URDF, ArmChain, arm_base_pose  # noqa: E402

STEPS = ["LR", "L", "R", "L"]


def w1(a, b):
    """1-D Wasserstein distance between two samples (quantile difference)."""
    qs = np.linspace(0.005, 0.995, 199)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def segments(grip):
    """[(start, end, hands)] runs where any gripper is closed (>= 10 frames)."""
    c = grip > CLOSED
    edges = np.flatnonzero(np.diff(np.r_[0, c.any(1).astype(int), 0]))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 10:
            out.append((a, b, "".join(s for k, s in enumerate("LR") if c[a:b, k].mean() > 0.3)))
    return out


def step_events(state, p):
    """Per step of LR-L-R-L (if the episode starts that way): for each hand in it, close/open frame,
    close/open position, max lift."""
    segs = segments(state[:, 14:16])
    if [h for _, _, h in segs[:4]] != STEPS:
        return None
    c = state[:, 14:16] > CLOSED
    ev = []
    for (a, b, h) in segs[:4]:
        e = {}
        for s in h:
            k = "LR".index(s)
            on = a + int(np.flatnonzero(c[a:b, k])[0])
            off = a + int(np.flatnonzero(c[a:b, k])[-1])
            e[s] = {"close_t": on / FPS, "open_t": off / FPS, "hold": (off - on) / FPS,
                    "close_p": p[on, k], "open_p": p[off, k], "lift": float(p[on:off + 1, k, 2].max())}
        ev.append(e)
    return ev


def far_edge_y(img):
    """Table y [m] of the shirt's far edge (top mask row with > 20 px) seen by the head camera at the
    idle pose."""
    from camera_calib import CAMERAS, gl_rotation
    from check_arm_collision import Robot, q_dict
    from sg2_reach import homogeneous

    global _HEAD
    if "_HEAD" not in globals():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac"))
        from export_scene import lift_origin

        base = arm_base_pose(-0.45, 0.55) @ np.linalg.inv(homogeneous(np.eye(3), lift_origin()))
        t = base @ Robot().link_pose(CAMERAS["head"][0], q_dict(np.zeros(16)), {})
        _HEAD = (gl_rotation("head", t[:3, :3]), t[:3, 3])
    from fold_quality import shirt_mask

    rows = np.flatnonzero(shirt_mask(img).sum(1) > 20)
    if not len(rows):
        return float("nan")
    rot, c = _HEAD
    _, w, h, hfov, _, _ = CAMERAS["head"]
    f = (w / 2) / np.tan(np.radians(hfov) / 2)
    d = rot @ np.array([0.0, -(rows[0] - h / 2) / f, -1.0])
    return float(c[1] - c[2] / d[2] * d[1])


def iter_video(mp4):
    """RGB frames of an mp4 with PyAV (lerobot env, decodes the real AV1 videos) or imageio."""
    try:
        import av
    except ImportError:
        import imageio.v2 as imageio

        r = imageio.get_reader(mp4)
        yield from r
        r.close()
        return
    with av.open(str(mp4)) as c:
        for f in c.decode(video=0):
            yield f.to_ndarray(format="rgb24")

CLOTH_GATES = ("final_area", "final_aspect", "final_fill")  # + step areas, + wrinkle with Isaac video
# sleeves must end up folded in: share of sleeve vertices outside the body's final footprint (median
# over trials). A FoldNet-style neat fold of the same mesh gives 0.003; the real shirts end as compact
# rectangles with the sleeves inside.
SLEEVE_OUT_MAX = 0.15


def cloth_video_stats(mp4, segs, n, stride=3):
    """One pass over a head video: first-frame mask, hem-fold slide, least occluded shirt area after
    each step, and the final frame's shape. segs: closed-gripper runs (start, end, hands)."""
    from scipy import ndimage

    from fold_quality import blob_shape, shirt_mask

    hem_end = min(segs[0][1] + 45, n - 1)
    wins = []
    for i in range(4):
        lo = segs[i][1] + 15
        hi = segs[i + 1][0] - 15 if i + 1 < len(segs) else n - 1
        wins.append((lo, max(lo, hi)))
    best = [0] * 4
    first = hem = last = None
    for i, f in enumerate(iter_video(mp4)):
        if i >= n:
            break
        if i == 0:
            first = f
        if i == hem_end:
            hem = f
        last = f
        if i % stride == 0 or i == n - 1:
            inwin = [k for k, (lo, hi) in enumerate(wins) if lo <= i <= hi]
            if inwin:
                a = int(shirt_mask(f).sum())
                for k in inwin:
                    best[k] = max(best[k], a)
    m0 = shirt_mask(first)
    a0 = max(int(m0.sum()), 1)
    m1 = shirt_mask(last)
    _, aspect, fill = blob_shape(m1)
    g = np.hypot(*np.gradient(last.astype(float).mean(2)))
    inner = ndimage.binary_erosion(m1, iterations=3)
    return {"mask0": m0, "slide": far_edge_y(hem if hem is not None else last) - far_edge_y(first),
            "step_area": [b / a0 for b in best], "final_area": float(m1.sum()) / a0,
            "final_aspect": aspect, "final_fill": fill,
            "wrinkle": float(g[inner].mean()) if inner.any() else float("nan")}


def mesh_state(trial):
    """Sim-only cloth checks on the final frame of frames.npz: sleeve vertices outside the body's
    footprint, stack height p95 [cm], share of edges bent more than 60 deg."""
    from scipy.spatial import Delaunay

    d = np.load(trial / "frames.npz")
    q0, q1, faces = d["q"][0], d["q"][-1], d["faces"]
    cx = np.median(q0[:, 0])
    half = np.abs(q0[:, 0] - cx).max()
    sleeve = np.abs(q0[:, 0] - cx) > 0.62 * half
    try:
        out = float((Delaunay(q1[~sleeve, :2]).find_simplex(q1[sleeve, :2]) < 0).mean())
    except Exception:
        out = float("nan")
    tri = q1[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    edges = {}
    for fi, f in enumerate(faces):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edges.setdefault((min(a, b), max(a, b)), []).append(fi)
    pairs = np.array([v for v in edges.values() if len(v) == 2])
    cos = np.einsum("ij,ij->i", nrm[pairs[:, 0]], nrm[pairs[:, 1]])
    return {"sleeve_out": out, "stack_p95_cm": float(100 * np.percentile(q1[:, 2], 95)),
            "bent60": float((cos < 0.5).mean())}


def real_cloth_stats(args, real, meta):
    """Per real episode cloth_video_stats, cached in --real-cache."""
    key = str(args.real_videos.resolve())
    cache = json.loads(args.real_cache.read_text()) if args.real_cache.exists() else {}
    if cache.get("key") == key and len(cache.get("episodes", [])) == len(real):
        eps = cache["episodes"]
        return [dict(e, mask0=np.array(e["mask0"], bool)) if e else None for e in eps]
    col = "videos/observation.images.head/file_index"
    out = []
    for ep, e in enumerate(real):
        mp4 = args.real_videos / f"file-{int(meta[col].iloc[ep]):03d}.mp4"
        segs = segments(e[:, 14:16])
        out.append(cloth_video_stats(mp4, segs, len(e)) if mp4.exists() and len(segs) >= 4 else None)
    args.real_cache.parent.mkdir(parents=True, exist_ok=True)
    args.real_cache.write_text(json.dumps({"key": key, "episodes": [
        dict(x, mask0=x["mask0"].astype(int).tolist()) if x else None for x in out]}, default=float))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", nargs="+", type=Path, required=True, help="real data parquet files")
    p.add_argument("--real-videos", type=Path, help="real head-camera video dir (file-XXX.mp4)")
    p.add_argument("--real-meta", type=Path, help="real meta/episodes dir (episode -> video file)")
    p.add_argument("trials", nargs="+", type=Path, help="sim trial dirs (frames.npz, optional head.mp4)")
    p.add_argument("--json", type=Path)
    p.add_argument("--video-dir", default="", help="sim head video at <trial>/<video-dir>/head.mp4, e.g. isaac")
    p.add_argument("--real-cache", type=Path,
                   default=Path(__file__).resolve().parents[2] / "outputs" / "fidelity_real_cloth.json")
    p.add_argument("--base-y", type=float, default=-0.45)
    p.add_argument("--height", type=float, default=0.55)
    args = p.parse_args()

    base = arm_base_pose(args.base_y, args.height)
    chains = {s: ArmChain(DEFAULT_URDF, s) for s in "lr"}
    real, sim = load_real(args.real), load_sim(args.trials)
    rep = {}

    # --- joints and hands
    allq = {k: np.concatenate([e[:, :14] for e in v]) for k, v in (("real", real), ("sim", sim))}
    jw = [w1(allq["real"][:, j], allq["sim"][:, j]) for j in range(14)]
    rep["joint_w1_mean"] = float(np.mean(jw))
    rep["joint_w1"] = {f"{'lr'[j // 7]}{j % 7 + 1}": round(v, 3) for j, v in enumerate(jw)}
    hp = {k: [hands(e[:, :14], chains, base) for e in v] for k, v in (("real", real), ("sim", sim))}
    allp = {k: np.concatenate(v) for k, v in hp.items()}
    rep["hand_w1_cm"] = {f"{s}_{a}": round(100 * w1(allp["real"][:, k, i], allp["sim"][:, k, i]), 1)
                         for k, s in enumerate("lr") for i, a in enumerate("xyz")}
    for k, v in hp.items():
        sp = np.concatenate([np.linalg.norm(np.diff(x, axis=0), axis=2).ravel() * FPS for x in v])
        rep[f"speed_{k}"] = np.percentile(sp[sp > 0.02], [10, 50, 90]).round(3).tolist()
        rep[f"height_{k}"] = np.percentile(allp[k][:, :, 2], [10, 50, 90]).round(3).tolist()
        rep[f"length_{k}"] = float(np.median([len(e) / FPS for e in (real if k == "real" else sim)]))
        pats = [grasp_pattern(e[:, 14:16]) for e in (real if k == "real" else sim)]
        rep[f"pattern_{k}"] = float(np.mean([pt.startswith("LR-L-R-L") for pt in pats]))

    # --- steps
    ev = {k: [x for x in (step_events(e, pp) for e, pp in zip(eps, hp[k])) if x]
          for k, eps in (("real", real), ("sim", sim))}
    steps = []
    for si, h in enumerate(STEPS):
        for s in h:
            row = {"step": si, "hand": s}
            for key in ("close_p", "open_p"):
                r = np.median([e[si][s][key] for e in ev["real"]], 0)
                m = np.median([e[si][s][key] for e in ev["sim"]], 0) if ev["sim"] else np.full(3, np.nan)
                row[key + "_real"], row[key + "_sim"] = r.round(3).tolist(), m.round(3).tolist()
                row[key + "_err_cm"] = round(float(100 * np.linalg.norm(r - m)), 1)
            for key in ("close_t", "open_t", "hold", "lift"):
                row[key + "_real"] = round(float(np.median([e[si][s][key] for e in ev["real"]])), 2)
                row[key + "_sim"] = round(float(np.median([e[si][s][key] for e in ev["sim"]])), 2) if ev["sim"] else None
            steps.append(row)
    rep["steps"] = steps
    rep["n"] = {"real": len(real), "sim": len(sim), "real_lrlrl": len(ev["real"]), "sim_lrlrl": len(ev["sim"])}

    # --- images and cloth state
    if args.real_videos and args.real_meta:
        import pandas as pd

        meta = pd.concat([pd.read_parquet(f) for f in sorted(args.real_meta.glob("*.parquet"))])
        st = {"real": [x for x in real_cloth_stats(args, real, meta) if x]}
        st["sim"] = []
        for t, e in zip(args.trials, sim):
            mp4 = t / args.video_dir / "head.mp4"
            segs = segments(e[:, 14:16])
            if mp4.exists() and len(segs) >= 4:
                st["sim"].append(cloth_video_stats(mp4, segs, len(e)))
        if st["real"] and st["sim"]:
            mr = np.mean([x["mask0"] for x in st["real"]], 0) > 0.5
            ms = np.mean([x["mask0"] for x in st["sim"]], 0) > 0.5
            rep["first_frame_iou"] = round(float((mr & ms).sum() / max((mr | ms).sum(), 1)), 3)
            rep["hem_slide_real"] = round(float(np.median([x["slide"] for x in st["real"]])), 3)
            rep["hem_slide_sim"] = round(float(np.median([x["slide"] for x in st["sim"]])), 3)
            cloth = {}
            for key in ["step_area_1", "step_area_2", "step_area_3", "step_area_4", "final_area",
                        "final_aspect", "final_fill", "wrinkle"]:
                get = (lambda x, k=int(key[-1]) - 1: x["step_area"][k]) if key.startswith("step") else (lambda x, k=key: x[k])
                r = np.array([get(x) for x in st["real"]], float)
                m = np.array([get(x) for x in st["sim"]], float)
                lo, mid, hi = np.nanpercentile(r, [10, 50, 90])
                sm = float(np.nanmedian(m))
                gated = key in CLOTH_GATES or key.startswith("step") or (key == "wrinkle" and args.video_dir == "isaac")
                cloth[key] = {"real_p10": round(lo, 3), "real_p50": round(mid, 3), "real_p90": round(hi, 3),
                              "sim_p50": round(sm, 3), "sim": np.round(m, 3).tolist(),
                              "pass": bool(lo <= sm <= hi) if gated else None}
            rep["cloth"] = cloth
            rep["cloth_pass"] = all(v["pass"] for v in cloth.values() if v["pass"] is not None)
    mesh = [mesh_state(t) for t in args.trials]
    rep["mesh"] = {k: np.round([m[k] for m in mesh], 3).tolist() for k in mesh[0]}
    rep["sleeves_in"] = bool(np.median(rep["mesh"]["sleeve_out"]) <= SLEEVE_OUT_MAX)
    if "cloth_pass" in rep:
        rep["cloth_pass"] = rep["cloth_pass"] and rep["sleeves_in"]

    # --- print
    print(f"episodes real {rep['n']['real']} (LR-L-R-L {rep['n']['real_lrlrl']}), sim {rep['n']['sim']} "
          f"(LR-L-R-L {rep['n']['sim_lrlrl']})")
    print(f"joint W1 mean {rep['joint_w1_mean']:.3f} rad  per joint {rep['joint_w1']}")
    print(f"hand W1 [cm] {rep['hand_w1_cm']}")
    for k in ("speed", "height", "length", "pattern"):
        print(f"{k:<8} real {rep[k + '_real']}  sim {rep[k + '_sim']}")
    print("step hand | close err cm, open err cm | close t real/sim | open t | hold | lift")
    for r in steps:
        print(f"  {r['step']} {r['hand']}  | {r['close_p_err_cm']:5.1f} {r['open_p_err_cm']:5.1f} | "
              f"{r['close_t_real']:5.1f}/{r['close_t_sim']} | {r['open_t_real']:5.1f}/{r['open_t_sim']} | "
              f"{r['hold_real']:4.1f}/{r['hold_sim']} | {r['lift_real']:.2f}/{r['lift_sim']}")
    if "first_frame_iou" in rep:
        print(f"first-frame shirt IoU {rep['first_frame_iou']}; far edge shift in the hem fold [m] "
              f"real {rep['hem_slide_real']} sim {rep['hem_slide_sim']}")
    if "cloth" in rep:
        print(f"cloth state (real p10 / p50 / p90 | sim median | per trial)   -> {'PASS' if rep['cloth_pass'] else 'FAIL'}")
        for k, v in rep["cloth"].items():
            flag = {True: "pass", False: "FAIL", None: "(not gated)"}[v["pass"]]
            print(f"  {k:<13} {v['real_p10']:6.2f} {v['real_p50']:6.2f} {v['real_p90']:6.2f} | "
                  f"{v['sim_p50']:6.2f} | {v['sim']}  {flag}")
    print(f"mesh (sim only, per trial): sleeve verts outside body {rep['mesh']['sleeve_out']}, "
          f"stack p95 {rep['mesh']['stack_p95_cm']} cm, edges bent > 60 deg {rep['mesh']['bent60']}  "
          f"-> sleeves {'in' if rep['sleeves_in'] else 'OUT (FAIL)'} (median <= {SLEEVE_OUT_MAX})")
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
