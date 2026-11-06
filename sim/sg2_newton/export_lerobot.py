"""Write rendered sim trials as a LeRobot v3.0 dataset laid out like jaehoon2da/laundry_fold_tshirt_arm16.

Matches the real dataset's schema so the two can be mixed in training:
  fps 30; observation.state / action: 16 joints (arm_l 1-7, arm_r 1-7, gripper_l_joint1,
  gripper_r_joint1); observation.eef_pose / action.eef_pose: 14 (per arm x y z qw qx qy qz);
  observation.images.{head 376x672, wrist_left 240x424, wrist_right 240x424} as h264 video.
Grippers are 0 open / 1.1 closed (what teleop logs show on cloth). action is the joint state
ACTION_LEAD frames ahead: the real action is the teleop command, which the measured state reaches
about 14 frames (0.47 s) later (mean |action[t] - state[t+k]| over 24 real episodes is smallest at
k = 14, 0.0047 rad; 0.036 rad at k = 0).

eef_pose follows the real data exactly (fit on its parquet, 0.0 mm residual): link7 pose with lift at 0,
offset 8 cm along link7 -z, a fixed per-arm rotation, in a base frame 25.4 mm along +x from base_link.

Input trial dirs need frames.npz from fold_test.py --robot --record-dt 0.0333 and the three mp4s
from render_newton.py (or, with --video-dir isaac, from sim/isaac/render_trial.sh). Run in the
lerobot061 env:

    python sim/sg2_newton/export_lerobot.py --repo-id <user>/<name> --root <dir> trial_dir [trial_dir ...]
    python sim/sg2_newton/export_lerobot.py --check-eef path/to/real/file-xxx.parquet   # validate eef_pose
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
from sg2_reach import DEFAULT_URDF, ArmChain, homogeneous  # noqa: E402

FPS = 30
CAMERAS = {"head": (376, 672), "wrist_left": (240, 424), "wrist_right": (240, 424)}
JOINT_NAMES = ([f"arm_l_joint{i}" for i in range(1, 8)] + [f"arm_r_joint{i}" for i in range(1, 8)]
               + ["gripper_l_joint1", "gripper_r_joint1"])
EEF_NAMES = [f"eef_{s}_{c}" for s in "lr" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
GRIPPER_CLOSED = 1.1
ACTION_LEAD = 14  # frames the real teleop command leads the measured joint state
# real-data eef frame relative to link7, and its base relative to base_link (lift 0)
EEF_TOOL = {"l": homogeneous(np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]), [0.0, 0.0, -0.08]),
            "r": homogeneous(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]), [0.0, 0.0, -0.08])}
EEF_BASE_SHIFT = np.array([0.0254, 0.0, 0.0])
ARM_BASE_IN_BASE_LINK = homogeneous(np.eye(3), [-0.0199, 0.0, 1.4316])  # lift_joint origin, lift 0


def quat_wxyz(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        v = np.zeros(3)
        v[i], v[j], v[k] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
        q = np.array([(m[k, j] - m[j, k]) / s, *v])
    return q if q[0] >= 0 else -q


class EefFK:
    def __init__(self, urdf=DEFAULT_URDF):
        self.chains = {s: ArmChain(urdf, s) for s in "lr"}
        for c in self.chains.values():
            c.tcp = np.eye(4)  # FK to link7; the real-data tool frame is applied below

    def __call__(self, q16):
        out = []
        for k, s in enumerate("lr"):
            t, _ = self.chains[s].fk(np.asarray(q16[7 * k:7 * k + 7], dtype=float), ARM_BASE_IN_BASE_LINK)
            e = t @ EEF_TOOL[s]
            out += [*(e[:3, 3] + EEF_BASE_SHIFT), *quat_wxyz(e[:3, :3])]
        return np.asarray(out, dtype=np.float32)


def check_eef(parquet: Path):
    """Compare EefFK against a real dataset parquet (position mm, rotation deg)."""
    import pandas as pd

    df = pd.read_parquet(parquet)
    st, ee = np.stack(df["observation.state"].values), np.stack(df["observation.eef_pose"].values)
    fk = EefFK()
    idx = np.linspace(0, len(df) - 1, min(500, len(df))).astype(int)
    pos, rot = [], []
    for i in idx:
        mine = fk(st[i])
        for k in range(2):
            pos.append(np.linalg.norm(mine[7 * k:7 * k + 3] - ee[i, 7 * k:7 * k + 3]))
            dot = abs(float(np.dot(mine[7 * k + 3:7 * k + 7], ee[i, 7 * k + 3:7 * k + 7])))
            rot.append(math.degrees(2 * math.acos(min(1.0, dot))))
    print(f"{len(idx)} frames: eef position error max {1000 * max(pos):.2f} mm, rotation error max {max(rot):.3f} deg")


def decode(mp4: Path):
    import av

    with av.open(str(mp4)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def trial_frames(trial: Path, lead: int = ACTION_LEAD, video_dir: str = ""):
    d = np.load(trial / "frames.npz")
    if "joint" not in d:
        raise ValueError(f"{trial}: no robot joints (run fold_test.py with --robot)")
    dt = float(np.median(np.diff(d["t"])))
    if abs(dt - 1.0 / FPS) > 1e-3:
        raise ValueError(f"{trial}: frames every {dt:.4f}s; record at {1.0 / FPS:.4f}s (--record-dt) to match {FPS} fps")
    joint = d["joint"].astype(float)
    known = np.nonzero(~np.isnan(joint).any(axis=(1, 2)))[0]
    joint[: known[0]] = joint[known[0]]  # before the first robot move: hold its first pose
    grip = d["grip"].astype(float) * GRIPPER_CLOSED
    state = np.concatenate([joint[:, 0], joint[:, 1], grip], axis=1).astype(np.float32)
    images = {cam: decode(trial / video_dir / f"{cam}.mp4") for cam in CAMERAS}
    n = min(len(state), *(len(v) for v in images.values()))
    state = state[:n]
    action = state[np.minimum(np.arange(n) + lead, n - 1)]  # state `lead` frames on; the end holds
    return state, action, {c: v[:n] for c, v in images.items()}


def trial_ok(trial: Path) -> bool:
    """Skip trials fold_test.py marked as FAIL (metrics.json one level up)."""
    m = trial.parent / "metrics.json"
    if not m.exists():
        return True
    idx = int(trial.name.split("_")[-1])
    t = next((t for t in json.loads(m.read_text())["trials"] if t["trial"] == idx), None)
    return t is None or not t.get("fails")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trials", nargs="*", type=Path)
    p.add_argument("--repo-id", help="dataset id, e.g. <user>/<name> (nothing is pushed)")
    p.add_argument("--root", type=Path, help="local dataset directory (must not exist)")
    p.add_argument("--task", default="fold the t-shirt", help="task string (the real dataset's by default)")
    p.add_argument("--include-failed", action="store_true", help="also export trials marked FAIL")
    p.add_argument("--action-lead", type=int, default=ACTION_LEAD,
                   help="action = joint state this many frames ahead (real teleop lag)")
    p.add_argument("--video-dir", default="",
                   help="read <trial>/<video-dir>/<camera>.mp4, e.g. 'isaac' for sim/isaac/render_trial.sh output")
    p.add_argument("--check-eef", type=Path, help="validate eef_pose against a real dataset parquet and exit")
    args = p.parse_args()
    if args.check_eef:
        check_eef(args.check_eef)
        return
    if not (args.trials and args.repo_id and args.root):
        p.error("trials, --repo-id and --root are required")

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (16,), "names": JOINT_NAMES},
        "observation.state": {"dtype": "float32", "shape": (16,), "names": JOINT_NAMES},
        "observation.eef_pose": {"dtype": "float32", "shape": (14,), "names": EEF_NAMES},
        "action.eef_pose": {"dtype": "float32", "shape": (14,), "names": EEF_NAMES},
        **{f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
           for c, (h, w) in CAMERAS.items()},
    }
    ds = LeRobotDataset.create(args.repo_id, FPS, features, root=args.root, use_videos=True,
                               rgb_encoder=RGBEncoderConfig(vcodec="h264"), image_writer_threads=4,
                               video_backend="pyav")
    fk = EefFK()
    for trial in args.trials:
        if not args.include_failed and not trial_ok(trial):
            print(f"skip {trial} (FAIL in metrics.json)")
            continue
        state, action, images = trial_frames(trial, args.action_lead, args.video_dir)
        for i in range(len(state)):
            frame = {"task": args.task, "observation.state": state[i], "action": action[i],
                     "observation.eef_pose": fk(state[i]), "action.eef_pose": fk(action[i])}
            for cam in CAMERAS:
                frame[f"observation.images.{cam}"] = images[cam][i]
            ds.add_frame(frame)
        ds.save_episode()
        print(f"{trial}: {len(state)} frames")
    ds.finalize()
    print(f"{ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames -> {args.root}")


if __name__ == "__main__":
    main()
