"""Stack metrics for T3 replays: how tall the folded stack is after fold 5 and what the flip did.

For each replay dir (frames.npz + replay.json from t3_replay.py):
  h5_p99    cloth height (99th percentile of vertex z, table at z=0) 1 s after the fold-5 release
            ("Place the lifted part near the neckline"); the FoldNet++ paths assume about 3 cm or less
  fp_pre6   top-down footprint (fraction of the flat shirt) just before the flip grasp
  n_flip    cloth vertices pinned by the flip grasp ("midpoint of the folded left side")
  fp_final  final footprint (replay.json); FoldNet++ originals end near 0.13-0.17
  h_final   final cloth height (p99)
Writes <dir>/stack_metrics.json. With several dirs whose parents differ (a sweep laid out as
<sweep>/<config>/<episode>), also prints the per-config medians.

    python sim/foldnetxx/stack_metrics.py outputs/foldnetxx/sweep_phys/*/*
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CELL = 0.005


def projected_area(q, faces, cell=CELL):
    """Same as gate_fling.projected_area (used for replay.json final_footprint), without importing warp."""
    a, b, c = q[faces[:, 0], :2], q[faces[:, 1], :2], q[faces[:, 2], :2]
    pts = [a, b, c]
    for w in [(1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6),
              (0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5)]:
        pts.append(w[0] * a + w[1] * b + w[2] * c)
    return len(np.unique(np.floor(np.concatenate(pts) / cell).astype(np.int64), axis=0)) * cell * cell


def measure(d):
    fr = np.load(d / "frames.npz")
    rep = json.loads((d / "replay.json").read_text())
    q, t = fr["q"], fr["t"]
    if not np.isfinite(q).all():
        return {"nan": True}
    faces = fr["faces"]
    flat = rep["flat_area"]

    def at(ts):
        return q[min(int(np.searchsorted(t, ts)), len(q) - 1)]

    out = {"nan": False, "fp_final": rep["final_footprint"], "h_final": float(np.percentile(q[-1][:, 2], 99))}
    rel5 = [e[0] for e in rep["events"] if e[3] == "release" and "neckline" in e[1]]
    flip = [e for e in rep["events"] if e[3] == "grasp" and "folded left side" in e[1]]
    if rel5:
        out["h5_p99"] = float(np.percentile(at(rel5[0] + 1.0)[:, 2], 99))
    if flip:
        out["fp_pre6"] = projected_area(at(flip[0][0] - 0.1), faces) / flat
        out["n_flip"] = int(flip[0][4])
    (d / "stack_metrics.json").write_text(json.dumps(out, indent=1))
    return out


def fmt(v, cm=False):
    if v is None:
        return "   -  "
    return f"{v * 100:5.1f}cm" if cm else (f"{v:6.2f}" if isinstance(v, float) else f"{v:6d}")


def main():
    dirs = [Path(a) for a in sys.argv[1:] if (Path(a) / "frames.npz").exists()]
    groups = defaultdict(list)
    print(f"{'run':40s} {'h5_p99':>7s} {'fp_pre6':>7s} {'n_flip':>6s} {'fp_final':>8s} {'h_final':>7s}")
    for d in dirs:
        m = measure(d)
        name = f"{d.parent.name}/{d.name}"
        if m["nan"]:
            print(f"{name:40s} NaN in frames")
            continue
        groups[d.parent.name].append(m)
        print(f"{name:40s} {fmt(m.get('h5_p99'), True):>7s} {fmt(m.get('fp_pre6')):>7s} {fmt(m.get('n_flip')):>6s} "
              f"{fmt(m['fp_final']):>8s} {fmt(m['h_final'], True):>7s}")
    if len(groups) > 1:
        print("\nmedian per config")
        print(f"{'config':20s} {'n':>2s} {'h5_p99':>7s} {'fp_pre6':>7s} {'n_flip':>6s} {'fp_final':>8s}")
        for g, ms in sorted(groups.items()):
            med = {k: float(np.median([m[k] for m in ms if k in m])) if any(k in m for m in ms) else None
                   for k in ("h5_p99", "fp_pre6", "n_flip", "fp_final")}
            print(f"{g:20s} {len(ms):2d} {fmt(med['h5_p99'], True):>7s} {fmt(med['fp_pre6']):>7s} "
                  f"{med['n_flip'] if med['n_flip'] is None else int(med['n_flip']):>6} {fmt(med['fp_final']):>8s}")


if __name__ == "__main__":
    main()
