"""Mean colours of fixed head-camera regions in the first frame (arms idle, shirt flat), for matching
renderer lighting/material to the real recordings. Regions (row/col slices of the 672x376 image):
wall above the table, bare table left and right of the shirt, the shirt's middle, the left gripper.

    python sim/isaac/color_regions.py real_first_frames/*.png -- sim_first_frame.png
"""

import sys

import numpy as np

REGIONS = {
    "wall": (slice(40, 140), slice(150, 520)),
    "table": (slice(195, 225), slice(20, 110)),
    "shirt": (slice(250, 300), slice(290, 380)),
    "gripper": (slice(255, 300), slice(150, 175)),
}


def region_means(img):
    return {k: img[r, c, :3].reshape(-1, 3).mean(0) for k, (r, c) in REGIONS.items()}


def main():
    import imageio.v2 as imageio

    args = sys.argv[1:]
    split = args.index("--") if "--" in args else len(args)
    groups = {"real": args[:split], "sim": args[split + 1:]}
    for name, files in groups.items():
        if not files:
            continue
        means = [region_means(imageio.imread(f).astype(float)) for f in files]
        med = {k: np.median([m[k] for m in means], 0).round(0).astype(int).tolist() for k in REGIONS}
        print(f"{name:<5} " + "  ".join(f"{k} {v}" for k, v in med.items()))


if __name__ == "__main__":
    main()
