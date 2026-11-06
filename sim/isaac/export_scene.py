"""Bake a fold_test.py --robot trial into one scene file for the Isaac Sim renderer.

The Isaac container only needs numpy + USD: everything kinematic is resolved here, on the host, with
the same URDF and camera frames as render_newton.py:
  link_names / link_verts / link_faces   visual meshes of every SG2 link (URDF scale applied)
  link_T [F, L, 4, 4]                    world pose of every link per frame (arms from the recording,
                                         lift 0, head tilt 0.69, grippers from the recorded closure)
  cam_T  [F, 3, 4, 4]                    camera poses in the USD convention (looks along -z, +y up)
  cloth_q [F, V, 3], cloth_faces         cloth mesh per frame
  t, fps, table / wall geometry, camera intrinsics

    python sim/isaac/export_scene.py outputs/style3d_gate/sim_v9/base/trial_00   # -> trial_00/scene.npz
"""

import argparse
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sg2_reach"))
sys.path.insert(0, str(ROOT / "sg2_newton"))
from camera_calib import CAMERAS, gl_rotation  # noqa: E402
from check_arm_collision import Robot, q_dict, states_from_frames  # noqa: E402
from sg2_reach import DEFAULT_URDF, arm_base_pose, find_ros_install, homogeneous  # noqa: E402

HEAD_TILT = 0.69
TABLE = {"depth": 0.86, "width": 3.0, "thickness": 0.04, "edge_from_base": 0.075}  # render_newton.py
WALL_HEIGHT = 1.2


def load_stl_mesh(path: Path, scale: float):
    """(vertices [N, 3], faces [M, 3]) with shared vertices merged."""
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * n == len(data):
        rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
        tri = rec["v"].reshape(-1, 3, 3).astype(float)
    else:
        v = [[float(x) for x in ln.split()[1:4]] for ln in data.decode().splitlines() if ln.strip().startswith("vertex")]
        tri = np.array(v, dtype=float).reshape(-1, 3, 3)
    flat = np.round(tri.reshape(-1, 3), 7)
    verts, inv = np.unique(flat, axis=0, return_inverse=True)
    return verts * scale, inv.reshape(-1, 3).astype(np.int32)


def visual_meshes(urdf=DEFAULT_URDF, ros=None):
    """{link: (verts, faces, origin 4x4)} from each link's visual mesh."""
    from sg2_reach import rpy_to_matrix

    ros = ros or find_ros_install()
    share = ros / "ffw_description/share/ffw_description"
    out = {}
    for link in ET.parse(urdf).getroot().findall("link"):
        vis = link.find("visual")
        g = None if vis is None else vis.find("geometry/mesh")
        if g is None:
            continue
        fn = g.get("filename").replace("package://ffw_description/", f"{share}/")
        fn = fn.replace("file:///root/ros2_ws/install/", f"{ros}/")
        fn = fn.replace("${swerve_meshes_dir}", f"{share}/meshes/common/follower/ffw_swerve")
        if not fn.lower().endswith(".stl"):
            continue
        o = vis.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        v, f = load_stl_mesh(Path(fn), float(g.get("scale", "1 1 1").split()[0]))
        out[link.get("name")] = (v, f, homogeneous(rpy_to_matrix(*rpy), xyz))
    return out


def lift_origin(urdf=DEFAULT_URDF):
    j = next(j for j in ET.parse(urdf).getroot().findall("joint") if j.get("name") == "lift_joint")
    return [float(v) for v in j.find("origin").get("xyz").split()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trial_dir", type=Path)
    p.add_argument("--robot-base-y", type=float, default=-0.45)
    p.add_argument("--robot-height", type=float, default=0.55)
    p.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="default: <trial_dir>/scene.npz")
    args = p.parse_args()

    d = np.load(args.trial_dir / "frames.npz")
    states = states_from_frames(d["joint"], d["grip"])
    idx = np.arange(0, len(states), args.stride)[: args.max_frames]
    robot = Robot()
    meshes = visual_meshes()
    names = sorted(meshes)
    base = arm_base_pose(args.robot_base_y, args.robot_height) @ np.linalg.inv(homogeneous(np.eye(3), lift_origin()))

    link_T = np.zeros((len(idx), len(names), 4, 4), np.float32)
    cam_T = np.zeros((len(idx), len(CAMERAS), 4, 4), np.float32)
    for fi, i in enumerate(idx):
        q, cache = q_dict(states[i], HEAD_TILT), {}
        for li, n in enumerate(names):
            link_T[fi, li] = base @ robot.link_pose(n, q, cache) @ meshes[n][2]
        for ci, (cam, (link, *_rest)) in enumerate(CAMERAS.items()):
            t = base @ robot.link_pose(link, q, cache)
            cam_T[fi, ci] = homogeneous(gl_rotation(cam, t[:3, :3]), t[:3, 3])
    edge_y = base[1, 3] + TABLE["edge_from_base"]
    out = args.out or args.trial_dir / "scene.npz"
    np.savez_compressed(
        out,
        link_names=np.array(names), link_T=link_T,
        **{f"verts_{k}": meshes[n][0].astype(np.float32) for k, n in enumerate(names)},
        **{f"faces_{k}": meshes[n][1] for k, n in enumerate(names)},
        cam_names=np.array(list(CAMERAS)), cam_T=cam_T,
        cam_res=np.array([[w, h] for _l, w, h, _f, _c, _n in CAMERAS.values()]),
        cam_hfov=np.array([fov for _l, _w, _h, fov, _c, _n in CAMERAS.values()]),
        cam_near=np.array([near for *_x, near in CAMERAS.values()]),
        cloth_q=d["q"][idx].astype(np.float32), cloth_faces=d["faces"].astype(np.int32),
        t=d["t"][idx], fps=np.array(1.0 / float(np.median(np.diff(d["t"]))) / args.stride),
        table=np.array([edge_y, TABLE["depth"], TABLE["width"], TABLE["thickness"], WALL_HEIGHT]),
    )
    print(f"{len(idx)} frames, {len(names)} links -> {out}")


if __name__ == "__main__":
    main()
