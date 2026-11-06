"""Check SG2 arm-arm and arm-body overlap along recorded joint trajectories (meshes from the URDF).

The fold sim drives the arms kinematically and the IK has no collision term, so nothing stops the
two arms (or an arm and the torso) from passing through each other. This replays joint states and
reports, per frame, the closest surface distance between the left and right arm (links 3-7, gripper,
wrist camera) and between each arm and the torso/head, plus how deep vertices sit inside the other
side's per-link convex hulls (a penetration estimate; hulls over-approximate concave parts).

Inputs: a LeRobot dataset dir (observation.state, 16 joints) or fold_test.py trial dirs (frames.npz).
fold_test.py --robot runs clearance() on every trial and marks it FAIL under --arm-clearance.
On 2026-11-05 the real dataset never brought the arms within 10 cm of each other or 3.5 cm of the
torso; the first sim run collided in every trial during the one-hand flip (idle arm left in place).

    python sim/sg2_newton/check_arm_collision.py ~/Downloads/laundry_fold_tshirt_sim_v0
    python sim/sg2_newton/check_arm_collision.py outputs/style3d_gate/sim_v0/s3d/trial_00
"""

import argparse
import glob
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
from sg2_reach import DEFAULT_URDF, axis_angle, find_ros_install, homogeneous, rpy_to_matrix  # noqa: E402

ARM_LINKS = {s: [f"arm_{s}_link{i}" for i in range(3, 8)] for s in "lr"}
BODY_LINKS = ["arm_base_link", "head_link1", "head_link2"]
CONTACT = 0.005  # surfaces closer than this count as touching
VOXEL = 0.004  # vertex thinning for distance queries (hulls use the full mesh)


def thin(v, voxel=VOXEL):
    _, idx = np.unique(np.floor(v / voxel).astype(np.int64), axis=0, return_index=True)
    return v[np.sort(idx)]


def load_stl(path: Path, scale: float) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * n == len(data):  # binary
        rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
        v = rec["v"].reshape(-1, 3)
    else:  # ascii
        v = np.array([[float(x) for x in ln.split()[1:4]] for ln in data.decode().splitlines()
                      if ln.strip().startswith("vertex")])
    return np.unique(v.astype(float), axis=0) * scale


class Robot:
    def __init__(self, urdf=DEFAULT_URDF, ros=None):
        ros = ros or find_ros_install()
        root = ET.parse(urdf).getroot()
        share = ros / "ffw_description/share/ffw_description"
        self.joints = {}
        for j in root.findall("joint"):
            o, a = j.find("origin"), j.find("axis")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            self.joints[j.find("child").get("link")] = (
                j.get("name"), j.find("parent").get("link"), homogeneous(rpy_to_matrix(*rpy), xyz),
                j.get("type"), np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None)
        self.meshes, self.hulls = {}, {}
        for link in root.findall("link"):
            g = link.find("collision/geometry/mesh")
            if g is None:
                g = link.find("visual/geometry/mesh")
            if g is None:
                continue
            fn = g.get("filename").replace("package://ffw_description/", f"{share}/")
            fn = fn.replace("file:///root/ros2_ws/install/", f"{ros}/")
            fn = fn.replace("${swerve_meshes_dir}", f"{share}/meshes/common/follower/ffw_swerve")
            scale = float(g.get("scale", "1 1 1").split()[0])
            full = load_stl(Path(fn), scale)
            self.hulls[link.get("name")] = ConvexHull(full)
            self.meshes[link.get("name")] = thin(full)

    def link_pose(self, link, q, cache):
        if link in cache:
            return cache[link]
        if link not in self.joints:
            cache[link] = np.eye(4)
            return cache[link]
        name, parent, origin, jtype, axis = self.joints[link]
        t = self.link_pose(parent, q, cache) @ origin
        if jtype in ("revolute", "continuous"):
            t = t @ homogeneous(axis_angle(axis, q.get(name, 0.0)), [0, 0, 0])
        elif jtype == "prismatic":
            t = t @ homogeneous(np.eye(3), axis * q.get(name, 0.0))
        cache[link] = t
        return t

    def side_links(self, side):
        sub = [n for n in self.meshes if n.startswith(f"gripper_{side}_")]
        cam = f"camera_{'left' if side == 'l' else 'right'}_link"
        return ARM_LINKS[side] + sub + ([cam] if cam in self.meshes else [])


def q_dict(state16, head_tilt=0.69):
    q = {"head_joint1": head_tilt}
    for k, s in enumerate("lr"):
        for i in range(7):
            q[f"arm_{s}_joint{i + 1}"] = float(state16[7 * k + i])
        g = float(state16[14 + k]) / 1.1
        for n, v in ((1, 1.1), (2, 1.0), (3, 1.1), (4, 1.0)):
            q[f"gripper_{s}_joint{n}"] = v * g
    return q


def states_from_frames(joint, grip):
    """frames.npz joint [N, 2, 7] (NaN before the first move) and grip [N, 2] -> states [N, 16]."""
    j = np.asarray(joint, dtype=float).copy()
    known = np.nonzero(~np.isnan(j).any(axis=(1, 2)))[0]
    j[: known[0]] = j[known[0]]
    return np.concatenate([j[:, 0], j[:, 1], np.asarray(grip, dtype=float) * 1.1], axis=1)


_ROBOT = None


def clearance(joint, grip, record_dt, check_dt=0.2):
    """Closest arm-arm and arm-torso distance over a trial (0 if any vertex is inside the other
    arm's hulls), checked every check_dt seconds. Used by fold_test.py to FAIL colliding trials."""
    global _ROBOT
    _ROBOT = _ROBOT or Robot()
    r = check(_ROBOT, states_from_frames(joint, grip), max(1, round(check_dt / record_dt)))
    arm_arm = 0.0 if (r["depth"] > 0).any() else float(r["lr"].min())
    return {"arm_arm_min": arm_arm, "arm_body_min": float(min(r["l_body"].min(), r["r_body"].min()))}


def load_states(path: Path):
    """[(episode name, states[N,16], phases[N] or None)]"""
    if (path / "frames.npz").exists():
        d = np.load(path / "frames.npz")
        return [(path.name, states_from_frames(d["joint"], d["grip"]), d["phase"])]
    import pandas as pd

    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(path / "data/*/*.parquet")))])
    return [(f"episode {e}", np.stack(g["observation.state"].values), None) for e, g in df.groupby("episode_index")]


def closest_pairs(robot, q16, n=3):
    """The n closest (left part, right part or torso, distance) at one joint state."""
    q, cache = q_dict(q16), {}
    world = {}
    for name in robot.side_links("l") + robot.side_links("r") + BODY_LINKS:
        t = robot.link_pose(name, q, cache)
        world[name] = robot.meshes[name] @ t[:3, :3].T + t[:3, 3]
    pairs = []
    for a in robot.side_links("l"):
        tree = cKDTree(world[a])
        for b in robot.side_links("r") + BODY_LINKS:
            pairs.append((a, b, float(tree.query(world[b])[0].min())))
    for a in robot.side_links("r"):
        tree = cKDTree(world[a])
        for b in BODY_LINKS:
            pairs.append((a, b, float(tree.query(world[b])[0].min())))
    return sorted(pairs, key=lambda x: x[2])[:n]


def report_phases(robot, name, states, phases, stride, limit):
    """Per phase: closest arm-arm / arm-torso distance, and the closest parts where under `limit`."""
    r = check(robot, states, stride)
    idx = np.arange(0, len(states), stride)
    lr = np.where(r["depth"] > 0, 0.0, r["lr"])
    body = np.minimum(r["l_body"], r["r_body"])
    print(f"{name}")
    for ph in dict.fromkeys(phases[idx]):
        m = phases[idx] == ph
        i_lr, i_b = np.argmin(np.where(m, lr, np.inf)), np.argmin(np.where(m, body, np.inf))
        flag = " <--" if min(lr[i_lr], body[i_b]) < limit else ""
        print(f"  {ph:<18} arm-arm {1000 * lr[i_lr]:6.1f} mm (surface {1000 * r['lr'][m].min():5.1f} mm, "
              f"hull depth {1000 * r['depth'][m].max():4.1f} mm)  arm-torso {1000 * body[i_b]:6.1f} mm{flag}")
        if flag:
            worst = idx[i_lr] if lr[i_lr] <= body[i_b] else idx[i_b]
            for a, b, d in closest_pairs(robot, states[worst]):
                print(f"      frame {worst}: {a} - {b} {1000 * d:.1f} mm")


def check(robot, states, stride):
    hulls = robot.hulls
    sides = {s: robot.side_links(s) for s in "lr"}
    q0, c0 = q_dict(states[0]), {}  # torso and head do not move (lift 0, fixed head tilt)
    body_tree = cKDTree(np.vstack([robot.meshes[n] @ robot.link_pose(n, q0, c0)[:3, :3].T
                                   + robot.link_pose(n, q0, c0)[:3, 3] for n in BODY_LINKS]))
    out = {"lr": [], "l_body": [], "r_body": [], "depth": []}
    for q16 in states[::stride]:
        q, cache = q_dict(q16), {}
        world = {}
        for n in sides["l"] + sides["r"]:
            t = robot.link_pose(n, q, cache)
            world[n] = robot.meshes[n] @ t[:3, :3].T + t[:3, 3]
        pts = {s: np.vstack([world[n] for n in sides[s]]) for s in "lr"}
        out["lr"].append(cKDTree(pts["r"]).query(pts["l"])[0].min())
        out["l_body"].append(body_tree.query(pts["l"])[0].min())
        out["r_body"].append(body_tree.query(pts["r"])[0].min())
        # penetration: left vertices inside right links' hulls (in each link's own frame), and vice versa.
        # Skipped while the arms' closest vertices are over 3 cm apart (a vertex inside the other
        # arm would sit closer than that to its surface samples); this loop is ~95% of the run time.
        depth = 0.0
        for a, b in ((("l", "r"), ("r", "l")) if out["lr"][-1] < 0.03 else ()):
            for n in sides[b]:
                t = robot.link_pose(n, q, cache)
                local = (pts[a] - t[:3, 3]) @ t[:3, :3]
                eq = hulls[n].equations
                s = local @ eq[:, :3].T + eq[:, 3]
                inside = (s <= 0).all(axis=1)
                if inside.any():
                    depth = max(depth, float(-s[inside].max(axis=1).max()))
        out["depth"].append(depth)
    return {k: np.array(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--stride", type=int, default=3, help="check every Nth frame")
    p.add_argument("--phases", action="store_true", help="trial dirs: break down by phase and name the closest parts")
    p.add_argument("--limit", type=float, default=0.01, help="flag phases closer than this [m] (--phases)")
    args = p.parse_args()
    robot = Robot()
    if args.phases:
        for path in args.paths:
            for name, st, ph in load_states(path):
                report_phases(robot, name, st, ph, args.stride, args.limit)
        return
    print(f"{'episode':<24} {'frames':>6} {'L-R min':>8} {'L-R<5mm':>8} {'depth max':>9} {'L-body':>7} {'R-body':>7}")
    for path in args.paths:
        for name, st, _ in load_states(path):
            r = check(robot, st, args.stride)
            touch = (r["lr"] < CONTACT) | (r["depth"] > 0)
            print(f"{name:<24} {len(st):6d} {1000 * r['lr'].min():6.1f}mm {100 * touch.mean():7.1f}% "
                  f"{1000 * r['depth'].max():7.1f}mm {1000 * r['l_body'].min():5.1f}mm {1000 * r['r_body'].min():5.1f}mm")


if __name__ == "__main__":
    main()
