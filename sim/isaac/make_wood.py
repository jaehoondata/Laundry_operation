"""Procedural wood-grain texture for the Isaac table top (grain along the image's x axis).

The mean colour is the table albedo that matched the real head camera (render_isaac.py
--table-color); grain contrast and ring spacing are rough guesses from the real frames.

    python sim/isaac/make_wood.py --out sim/isaac/textures/wood.png [--color 0.49,0.37,0.17] [--seed 0]
"""

import argparse
from pathlib import Path

import numpy as np


def wood(size=1024, color=(0.49, 0.37, 0.17), contrast=0.08, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    # low-frequency warp so the grain lines wander, tileable via sin/cos of the coordinates
    warp = sum(rng.uniform(0.002, 0.01) * np.sin(2 * np.pi * (k * x + rng.uniform())) for k in (1, 2, 3))
    warp += 0.004 * np.sin(2 * np.pi * (3 * y + 2 * x))
    rings = np.sin(2 * np.pi * (rng.integers(28, 40) * (y + warp)))
    fine = np.sin(2 * np.pi * (rng.integers(150, 220) * (y + 0.5 * warp))) * 0.3
    noise = rng.normal(0, 0.15, (size, size))
    g = 1 + contrast * (0.7 * rings + fine) + 0.04 * noise
    img = np.clip(np.asarray(color)[None, None, :] * g[..., None], 0, 1)
    return (img ** (1 / 2.2) * 255).astype(np.uint8)  # sRGB


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "textures" / "wood.png")
    p.add_argument("--color", default="0.49,0.37,0.17")
    p.add_argument("--contrast", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from PIL import Image

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(wood(color=[float(v) for v in args.color.split(",")], contrast=args.contrast, seed=args.seed)).save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
