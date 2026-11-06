"""Head and wrist camera views of the same fold moments: FoldNet++ original, SG2 replay in sim, real SG2.

Shows what the SG2 cameras see when SG2 replays a FoldNet++ episode, next to the original FoldNet++
robot's cameras and a real SG2 teleop episode, at three moments:
  flat     1 s before the first grasp: the shirt lies flat, hands on their way
  grasp    both hands have just closed on the near edge (first fold step)
  placed   the first fold has been laid down and both hands are open again

Three steps, because h5py, PyAV and matplotlib live in different envs:
  conda base (h5py):         camera_compare.py h5-info EPISODE.h5 --out DIR/info.json
  conda lerobot061 (PyAV):   camera_compare.py frames --info DIR/info.json --sim T3_DIR \
                                 --real-root REAL_DATASET --real-episode 0 --out DIR
  any env with matplotlib:   camera_compare.py compose --out DIR
Writes DIR/frames.npz, then DIR/head_compare.png and DIR/wrist_compare.png.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLAT = "The T-shirt is flattened."
FIRST = "Use the left gripper to grasp the corner tip of the clothing, and the right gripper to grasp the shoulder tip."
MOMENTS = ("flat", "grasp", "placed")
TITLES = {"flat": "flat, about to grasp", "grasp": "first grasp (both hands closed)", "placed": "first fold laid down"}


def moments_from_grip(closed, start, lead):
    """grasp = first frame from start with both hands closed, flat = `lead` frames (1 s) before it,
    placed = first frame after the grasp with both hands open."""
    both = np.flatnonzero(closed[start:].all(1))
    g = start + int(both[0]) if len(both) else start
    opened = np.flatnonzero(~closed[g:].any(1))
    return {"flat": max(start, g - lead), "grasp": g, "placed": g + int(opened[0]) if len(opened) else len(closed) - 1}


def h5_info(args):
    import h5py

    with h5py.File(args.episode, "r") as f:
        a = f["actions_eef"][:]
        sub = [s.decode() for s in f["subtasks"][:]]
    k0 = sub.index(FLAT)
    m = moments_from_grip(a[:, 18:20] > 0.5, k0, 5)  # 5 Hz
    vid = args.episode.with_suffix("")
    out = {"episode": str(args.episode), "steps": len(sub), "moments": m,
           "videos": {k: str(vid / f"{k}.mp4") for k in ("head", "wrist_l", "wrist_r")}}
    args.out.write_text(json.dumps(out, indent=1))
    print(out)


def frames_at(mp4, idx):
    """{index: RGB frame} for the given frame indices (PyAV)."""
    import av

    want, got = set(idx), {}
    with av.open(str(mp4)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = f.to_ndarray(format="rgb24")
            if len(got) == len(want):
                break
    return got


def n_frames(mp4):
    import av

    with av.open(str(mp4)) as c:
        return sum(1 for _ in c.decode(video=0))


def frames(args):
    import pandas as pd

    info = json.loads(args.info.read_text())
    rows = {}

    # FoldNet++: video frames map to h5 steps by the frame/step ratio
    ratio = n_frames(info["videos"]["head"]) / info["steps"]
    idx = {k: min(int(round(v * ratio)), int(info["steps"] * ratio) - 1) for k, v in info["moments"].items()}
    rows["FoldNet++ (Galbot, original)"] = {}
    for cam, src in (("head", "head"), ("wrist_left", "wrist_l"), ("wrist_right", "wrist_r")):
        got = frames_at(info["videos"][src], idx.values())
        rows["FoldNet++ (Galbot, original)"][cam] = {k: got[i] for k, i in idx.items()}

    # SG2 replay in sim: frames.npz phase / grip per rendered frame
    d = np.load(args.sim / "frames.npz")
    phase, grip = d["phase"], d["grip"] > 0.5
    start = int(np.flatnonzero(phase != "approach")[0]) if (phase != "approach").any() else 0
    idx = moments_from_grip(grip, start, 30)
    rows["SG2 replay of it (sim)"] = {}
    for cam in ("head", "wrist_left", "wrist_right"):
        got = frames_at(args.sim / f"{cam}.mp4", idx.values())
        rows["SG2 replay of it (sim)"][cam] = {k: got[i] for k, i in idx.items()}

    # real SG2 teleop episode: gripper state from the dataset, video frames from the meta's timestamps
    root = args.real_root
    df = pd.concat(pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"])
                   for f in sorted(glob.glob(str(root / "data/chunk-000/*.parquet"))))
    eps = sorted(df["episode_index"].unique())
    ep = eps[args.real_episode]
    e = df[df["episode_index"] == ep].sort_values("frame_index")
    state = np.stack(e["observation.state"].values)
    sys.path.insert(0, str(HERE.parent / "sg2_newton"))
    from compare_real import CLOSED, FPS

    idx = moments_from_grip(state[:, 14:16] > CLOSED, 0, FPS)
    meta = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/chunk-000/*.parquet"))))
    mrow = meta[meta["episode_index"] == ep].iloc[0]
    real = {}
    for cam in ("head", "wrist_left", "wrist_right"):
        key = f"videos/observation.images.{cam}"
        mp4 = root / f"videos/observation.images.{cam}/chunk-{int(mrow[key + '/chunk_index']):03d}/file-{int(mrow[key + '/file_index']):03d}.mp4"
        off = int(round(float(mrow[key + "/from_timestamp"]) * FPS))
        got = frames_at(mp4, [off + i for i in idx.values()])
        real[cam] = {k: got[off + i] for k, i in idx.items()}
    rows[f"real SG2 teleop (episode {ep})"] = real

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "frames.npz", labels=np.array(list(rows)),
                        **{f"{r}|{cam}|{k}": im for r, (label, views) in enumerate(rows.items())
                           for cam, ims in views.items() for k, im in ims.items()})
    print("wrote", args.out / "frames.npz")


def compose(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.out / "frames.npz")
    labels = list(d["labels"])

    def grid(cams, name, panel_w):
        fig, axes = plt.subplots(len(labels), len(MOMENTS), figsize=(panel_w * len(MOMENTS), 2.9 * len(labels)),
                                 squeeze=False)
        for r, label in enumerate(labels):
            for c, k in enumerate(MOMENTS):
                ims = [d[f"{r}|{cam}|{k}"] for cam in cams]
                h = min(im.shape[0] for im in ims)
                ims = [im if im.shape[0] == h else im[:: max(1, im.shape[0] // h)][:h] for im in ims]
                ax = axes[r, c]
                ax.imshow(np.concatenate(ims, axis=1) if len(ims) > 1 else ims[0])
                ax.set_xticks([]), ax.set_yticks([])
                if r == 0:
                    ax.set_title(TITLES[k], fontsize=11)
                if c == 0:
                    ax.set_ylabel(label, fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out / name, dpi=110, bbox_inches="tight")
        plt.close(fig)

    grid(("head",), "head_compare.png", 4.2)
    grid(("wrist_left", "wrist_right"), "wrist_compare.png", 6.0)
    print("wrote", args.out / "head_compare.png", args.out / "wrist_compare.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("h5-info")
    a.add_argument("episode", type=Path)
    a.add_argument("--out", type=Path, required=True)
    b = sp.add_parser("frames")
    b.add_argument("--info", type=Path, required=True)
    b.add_argument("--sim", type=Path, required=True)
    b.add_argument("--real-root", type=Path, required=True)
    b.add_argument("--real-episode", type=int, default=0, help="position in the sorted real episode list")
    b.add_argument("--out", type=Path, required=True)
    c = sp.add_parser("compose")
    c.add_argument("--out", type=Path, required=True, help="dir with frames.npz")
    args = p.parse_args()
    {"h5-info": h5_info, "frames": frames, "compose": compose}[args.cmd](args)


if __name__ == "__main__":
    main()
