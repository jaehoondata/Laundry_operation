"""Real vs sim head-camera frames at the same fold events, for the real episode a replay trial copied.

For a fold_test.py --sequence teleop trial (metrics.json one level up names its real_episode), takes
the real head video and the sim head.mp4 at: the first frame, each gripper close of the four steps
(+0.3 s) and each release (+1.5 s), and the last frame. Writes a two-row png (real on top).

    python sim/sg2_newton/side_by_side.py --real-root <hf snapshot dir> outputs/style3d_gate/sim_v10/base/trial_00
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_steps import load  # noqa: E402

FPS = 30


def frames_at(mp4, idx, offset=0):
    import av

    want, out = sorted(set(int(i) + offset for i in idx)), {}
    with av.open(str(mp4)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = f.to_ndarray(format="rgb24")
            if i >= want[-1]:
                break
    last = max(out)
    return [out.get(int(i) + offset, out[last]) for i in idx]


def events(steps, length_s):
    t = [0.0]
    for st in steps:
        t.append(min(h["close_t"] for h in st.values()) + 0.3)
        t.append(max(h["open_t"] for h in st.values()) + 1.5)
    return t + [length_s - 0.1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trial", type=Path)
    p.add_argument("--real-root", type=Path, required=True, help="laundry_fold_tshirt_arm16 snapshot dir")
    p.add_argument("--out", type=Path, default=None, help="default: <trial>/real_vs_sim.png")
    p.add_argument("--width", type=int, default=336, help="tile width")
    p.add_argument("--video-dir", default="", help="read <trial>/<video-dir>/head.mp4, e.g. 'isaac'")
    args = p.parse_args()

    import pandas as pd

    idx = int(args.trial.name.split("_")[-1])
    m = json.loads((args.trial.parent / "metrics.json").read_text())
    tr = next(t for t in m["trials"] if t["trial"] == idx)
    ep_id = tr["real_episode"]
    real_ep = next(e for e in load()["episodes"] if e["episode"] == ep_id)

    # sim events from the sim's own gripper record
    d = np.load(args.trial / "frames.npz")
    g = d["grip"] > 0.5
    sim_steps, closed = [], g.any(1)
    edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 10:
            sim_steps.append({"x": {"close_t": float(d["t"][a] - d["t"][0]), "open_t": float(d["t"][b - 1] - d["t"][0])}})
    t_sim = events(sim_steps[:4], len(d["t"]) / FPS)
    t_real = events(real_ep["steps"], real_ep["length_s"])

    meta = pd.concat([pd.read_parquet(f) for f in sorted((args.real_root / "meta/episodes").rglob("*.parquet"))])
    row = meta[meta["episode_index"] == ep_id].iloc[0]
    key = "videos/observation.images.head"
    mp4 = args.real_root / f"videos/observation.images.head/chunk-{int(row[key + '/chunk_index']):03d}/file-{int(row[key + '/file_index']):03d}.mp4"
    start = int(round(float(row[key + "/from_timestamp"]) * FPS))
    real = frames_at(mp4, [round(t * FPS) for t in t_real], offset=start)
    sim = frames_at(args.trial / args.video_dir / "head.mp4", [round(t * FPS) for t in t_sim])

    def tile(im):
        from PIL import Image

        h = round(im.shape[0] * args.width / im.shape[1])
        return np.asarray(Image.fromarray(im).resize((args.width, h)))

    n = min(len(real), len(sim))
    img = np.concatenate([np.concatenate([tile(x) for x in real[:n]], 1),
                          np.concatenate([tile(x) for x in sim[:n]], 1)], 0)
    out = args.out or args.trial / f"real_vs_sim{'_' + args.video_dir if args.video_dir else ''}.png"
    from PIL import Image

    Image.fromarray(img).save(out)
    print(f"episode {ep_id}: real t {np.round(t_real, 1).tolist()}\n  sim t {np.round(t_sim, 1).tolist()} -> {out}")


if __name__ == "__main__":
    main()
