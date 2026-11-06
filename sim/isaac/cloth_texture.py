"""T-shirt texture and UVs for render_isaac.py (--cloth-texture / --cloth-uv), on the host.

The UVs are the cloth's own flat layout: x, y of every simulated vertex in the trial's first frame (the
shirt lies flat there), so the texture is drawn in metres on the flat shirt. FoldNet keypoints place
the details a plain white T-shirt shows on camera:
  hem and sleeve cuffs  double-needle stitching 1.8 and 2.4 cm in from the edge, a faint fold-over band
  neckline              a rib band (fine stripes, slightly grey) with a stitch line below it
  outline               a soft shadow along the side and shoulder seams
  everywhere            a faint jersey-knit grain
The mean colour stays --cloth-color (linear, as render_isaac.py's flat cloth material uses it).

    python sim/isaac/cloth_texture.py outputs/foldnetxx/t3/<trial>    (writes cloth_tex.png, cloth_uv.npy)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "foldnetxx"))


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def seg_dist(p, a, b):
    """Distance of points p [..., 2] to segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def polyline_dist(p, pts):
    return np.min([seg_dist(p, pts[i], pts[i + 1]) for i in range(len(pts) - 1)], axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trial_dir", type=Path)
    p.add_argument("--cloth-color", default="0.93,0.97,1.0")
    p.add_argument("--px-per-m", type=float, default=2400.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from fit_initial_shirt import load_mesh

    fr = np.load(args.trial_dir / "frames.npz")
    q0, faces = fr["q"][0].astype(float), fr["faces"]
    mesh_id = json.loads((args.trial_dir / "fit.json").read_text()).get("mesh_id", 0)
    _, _, kp = load_mesh(mesh_id)
    xy = q0[:, :2]
    lo = xy.min(0) - 0.02
    size = float((xy.max(0) - lo).max() + 0.02)
    n = int(size * args.px_per_m)
    uv = (xy - lo) / size

    # pixel centres in metres (row 0 = top = high v)
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    P = np.stack([lo[0] + (jj + 0.5) / n * size, lo[1] + (1 - (ii + 0.5) / n) * size], axis=-1)

    mask_img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(mask_img)
    pix = np.c_[uv[:, 0] * n, (1 - uv[:, 1]) * n]
    for f in faces:
        draw.polygon([tuple(pix[i]) for i in f], fill=255)
    mask = np.asarray(mask_img) > 0

    def k(name):
        return xy[kp[name]].mean(0)

    shade = np.ones((n, n))
    rng = np.random.default_rng(args.seed)
    # jersey grain: fine vertical wales plus low-amplitude noise
    shade *= 1 - 0.012 * (0.5 + 0.5 * np.sin(2 * np.pi * P[..., 0] / 0.0016))
    shade *= 1 + 0.01 * rng.standard_normal((n, n))

    def stitches(d, offsets, width=0.0007, dash=0.004, along=None, depth=0.10):
        for o in offsets:
            line = np.abs(d - o) < width
            if along is not None:  # dashed: thread visible every other half-dash
                line &= (np.floor(along / (dash / 2)) % 2) == 0
            shade[line] *= 1 - depth

    # hem: the edge between the two hem corners
    a, b = k("l_corner"), k("r_corner")
    d = seg_dist(P, a, b)
    t = (P - a) @ (b - a) / np.linalg.norm(b - a)
    shade[d < 0.026] *= 0.985  # folded-over hem band
    stitches(d, (0.018, 0.024), along=t)
    # sleeve cuffs
    for s in "lr":
        a, b = k(f"{s}_sleeve_top"), k(f"{s}_sleeve_bottom")
        d = seg_dist(P, a, b)
        t = (P - a) @ (b - a) / np.linalg.norm(b - a)
        shade[d < 0.026] *= 0.985
        stitches(d, (0.018, 0.024), along=t)
    # neckline rib: band along the collar curves (front and back), fine stripes across it
    neck = np.minimum(polyline_dist(P, [k("l_collar"), k("neck_f"), k("r_collar")]),
                      polyline_dist(P, [k("l_collar"), k("neck_b"), k("r_collar")]))
    rib = neck < 0.018
    ang = np.arctan2(P[..., 1] - k("neck_f")[1], P[..., 0] - k("neck_f")[0])
    shade[rib] *= 0.965 * (1 - 0.03 * (0.5 + 0.5 * np.sin(ang[rib] * 260)))
    stitches(neck, (0.021,), along=P[..., 0])
    # seams / outline: soft shadow just inside the outline
    from scipy import ndimage

    edge = ndimage.distance_transform_edt(mask) / args.px_per_m
    shade *= 1 - 0.06 * np.exp(-edge / 0.003) * mask

    base = np.array([float(v) for v in args.cloth_color.split(",")])
    img = linear_to_srgb(shade[..., None] * base[None, None])
    img[~mask] = linear_to_srgb(base)  # outside the shirt: unused, keep the colour for filtering at edges
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(args.trial_dir / "cloth_tex.png")
    np.save(args.trial_dir / "cloth_uv.npy", uv.astype(np.float32))
    print(f"wrote {args.trial_dir / 'cloth_tex.png'} ({n} px, {size:.2f} m) and cloth_uv.npy ({len(uv)} vertices)")


if __name__ == "__main__":
    main()
