"""Fold quality from the head camera, measured the same way on real and sim episodes.

The shirt is the largest bright, unsaturated blob (white cloth on a wood table, black grippers).
Per episode, from the first and last head frames:
  area   final shirt pixels / first-frame shirt pixels (how far it was folded down)
  aspect long / short side of the final blob's principal-axis box (real: a narrow strip)
  fill   final pixels / that box's area (1 = a clean rectangle, low = ragged or crumpled)

    python sim/sg2_newton/fold_quality.py --pairs real_frames/file-003_first.png real_frames/file-003_last.png ...
    python sim/sg2_newton/fold_quality.py outputs/style3d_gate/sim_v6/tele/trial_0*   # head.mp4 of rendered trials
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

WHITE_MIN = 150  # every channel at least this (0-255)
WHITE_SPREAD = 45  # max - min channel at most this (unsaturated)


def shirt_mask(rgb):
    rgb = rgb[..., :3].astype(int)
    m = (rgb.min(axis=2) >= WHITE_MIN) & (rgb.max(axis=2) - rgb.min(axis=2) <= WHITE_SPREAD)
    m[: int(0.45 * len(m))] = False  # wall/partition band (incl. the real white side wall), never the shirt
    lab, n = ndimage.label(ndimage.binary_opening(m, iterations=2))
    if n == 0:
        return np.zeros_like(m)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    return lab == 1 + int(np.argmax(sizes))


def blob_shape(mask):
    """(pixels, aspect, fill) of a blob via its principal-axis bounding box."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return len(xs), float("nan"), float("nan")
    pts = np.stack([xs, ys], 1).astype(float)
    pts -= pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    proj = pts @ vecs
    ext = proj.max(0) - proj.min(0) + 1
    return len(xs), float(ext.max() / ext.min()), float(len(xs) / (ext[0] * ext[1]))


def score(first, last):
    a0, _, _ = blob_shape(shirt_mask(first))
    a1, aspect, fill = blob_shape(shirt_mask(last))
    return {"area": a1 / max(a0, 1), "aspect": aspect, "fill": fill}


def video_ends(mp4):
    import imageio.v2 as imageio

    if not mp4.exists():  # render_newton.py --only-stills: first and last tiles of head_stills.png
        tiles = imageio.imread(mp4.parent / "head_stills.png")
        return tiles[:, :672], tiles[:, -672:]
    r = imageio.get_reader(mp4)
    frames = [f for f in r]
    r.close()
    return frames[0], frames[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trials", nargs="*", type=Path, help="rendered trial dirs (head.mp4)")
    p.add_argument("--pairs", nargs="+", type=Path, help="first/last image pairs")
    p.add_argument("--save-masks", type=Path, help="write last-frame masks here (for checking the threshold)")
    args = p.parse_args()
    import imageio.v2 as imageio

    items = []
    if args.pairs:
        items += [(a.stem.replace("_first", ""), imageio.imread(a), imageio.imread(b))
                  for a, b in zip(args.pairs[::2], args.pairs[1::2])]
    for t in args.trials:
        items.append((t.name, *video_ends(t / "head.mp4")))
    rows = []
    for name, first, last in items:
        s = score(first, last)
        rows.append(s)
        print(f"{name:<14} area {s['area']:.2f}  aspect {s['aspect']:.2f}  fill {s['fill']:.2f}")
        if args.save_masks:
            args.save_masks.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(args.save_masks / f"{name}_mask.png", (shirt_mask(last) * 255).astype(np.uint8))
    if rows:
        med = {k: float(np.nanmedian([r[k] for r in rows])) for k in rows[0]}
        print(f"{'median':<14} area {med['area']:.2f}  aspect {med['aspect']:.2f}  fill {med['fill']:.2f}  (n={len(rows)})")


if __name__ == "__main__":
    main()
