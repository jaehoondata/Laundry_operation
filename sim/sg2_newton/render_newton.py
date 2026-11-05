"""Render a fold_test.py --robot trial (frames.npz) from the SG2's three cameras with Newton's raytracer.

Rebuilds the scene from the saved state only (no physics): cloth mesh per frame, SG2 joints and
grippers per frame, a table whose top is the cloth sim's z=0 plane, and the floor. Cameras sit on
the URDF frames of the real ones:
  head        zed_left_camera_optical_frame (ZED Mini, VGA left eye 672x376), head_joint1 = 0.69
  wrist_left  camera_left_link  (D405 color 424x240)
  wrist_right camera_right_link (D405 color 424x240)
Resolutions are the robot's launch defaults and the head tilt is what teleop recordings show; field
of view is from the datasheets until calibrated intrinsics are available.

This is the quick renderer (S4 step 1). The Isaac Sim 4.5 container re-renders the same frames.npz
for the final dataset.

    python sim/sg2_newton/render_newton.py outputs/style3d_gate/fold_sg2/s3d/trial_00
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
import sg2_model  # noqa: E402
from sg2_reach import arm_base_pose, homogeneous  # noqa: E402

HEAD_TILT = 0.69  # head_joint1 in every teleop recording (Oct 2026)
CAMERAS = {  # name: (URDF body, width, height, horizontal FOV deg, frame convention)
    "head": ("zed_left_camera_optical_frame", 672, 376, 87.0, "optical"),
    "wrist_left": ("camera_left_link", 424, 240, 87.0, "ros_link"),
    "wrist_right": ("camera_right_link", 424, 240, 87.0, "ros_link"),
}
# rays start this far in front of the camera frame, past the camera's own housing mesh
# (the head shell's front rim is still in view at 3 cm and gone from 5 cm; the D405 body ~5 mm)
NEAR = {"head": 0.05, "wrist_left": 0.01, "wrist_right": 0.01}
# ROS camera_link (x forward, y left, z up) -> optical (z forward, x right, y down)
LINK_TO_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# optical -> renderer camera frame (looks along -z, +y up)
OPTICAL_TO_GL = np.diag([1.0, -1.0, -1.0])
TABLE = {"depth": 0.75, "width": 1.6, "thickness": 0.04, "edge_from_base": 0.075}  # deploy table.xml (approx.)
COLORS = {"table": (0.82, 0.71, 0.55), "floor": (0.45, 0.45, 0.45), "cloth": (0.20, 0.35, 0.70),
          "camera": (0.15, 0.15, 0.17)}


def quat_from_matrix(m):
    """(x, y, z, w) from a rotation matrix."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


def lift_origin(urdf=sg2_model.URDF):
    """lift_joint origin xyz in base_link (read from the URDF; it carries no rotation)."""
    import xml.etree.ElementTree as ET

    j = next(j for j in ET.parse(urdf).getroot().findall("joint") if j.get("name") == "lift_joint")
    return [float(v) for v in j.find("origin").get("xyz").split()]


def base_link_pose(base_y, height):
    """World pose of base_link given where arm_base_link sits (lift joint at 0)."""
    lift_origin_t = homogeneous(np.eye(3), lift_origin())
    return arm_base_pose(base_y, height) @ np.linalg.inv(lift_origin_t)


def build_scene(frames, args):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    base = base_link_pose(args.robot_base_y, args.robot_height)
    sg2_model.add_sg2(builder, xform=wp.transform(base[:3, 3], quat_from_matrix(base[:3, :3])), with_geometry=True)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=frames["q"][0].tolist(), indices=frames["faces"].flatten().tolist(), density=0.2,
        particle_radius=1e-3, color=COLORS["cloth"],
    )
    # table: top face at z=0 (the cloth sim's ground), robot-side edge just in front of base_link
    edge_y = base[1, 3] + TABLE["edge_from_base"]
    builder.add_shape_box(
        -1, hx=TABLE["width"] / 2, hy=TABLE["depth"] / 2, hz=TABLE["thickness"] / 2,
        xform=wp.transform((0.0, edge_y + TABLE["depth"] / 2, -TABLE["thickness"] / 2), wp.quat_identity()),
        color=COLORS["table"],
    )
    builder.add_ground_plane(height=float(base[2, 3]), color=COLORS["floor"])
    model = builder.finalize(device=args.device)
    # the URDF gives the D405 bodies a red material; the real cameras are dark gray
    bodies = sg2_model.body_index(model)
    cam_bodies = {bodies[n] for n in ("camera_left_link", "camera_right_link") if n in bodies}
    colors, shape_body = model.shape_color.numpy(), model.shape_body.numpy()
    for s in range(len(colors)):
        if shape_body[s] in cam_bodies:
            colors[s] = COLORS["camera"]
    model.shape_color.assign(colors)
    return model


def robot_joint_q(model, joint, grip):
    """Full joint_q for one frame: arms from the recording, lift 0, head tilted, grippers open/closed."""
    qi = sg2_model.joint_q_index(model)
    q = model.joint_q.numpy().copy()
    q[qi["head_joint1"]] = HEAD_TILT
    for k, side in enumerate("lr"):
        for j, name in enumerate(sg2_model.ARM_JOINTS[side]):
            q[qi[name]] = joint[k, j]
        # teleop logs: gripper_joint1 is 0 open / 1.1 (its limit) closed on cloth, rarely in between
        c = 1.0 if grip[k] else 0.0
        for n, v in ((1, 1.1 * c), (2, 1.0 * c), (3, 1.1 * c), (4, 1.0 * c)):
            q[qi[f"gripper_{side}_joint{n}"]] = v
    return q


def camera_transform(body_q_row, convention):
    p, quat = body_q_row[:3], body_q_row[3:7]
    x, y, z, w = quat
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    if convention == "ros_link":
        r = r @ LINK_TO_OPTICAL
    r = r @ OPTICAL_TO_GL
    return wp.transformf(wp.vec3f(*p), wp.quatf(*quat_from_matrix(r)))


def fill_missing_joints(joint):
    """Frames before the first robot move have NaN joints: hold the first known pose."""
    joint = joint.copy()
    first = np.nonzero(~np.isnan(joint).any(axis=(1, 2)))[0]
    if len(first):
        joint[: first[0]] = joint[first[0]]
    return np.nan_to_num(joint)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trial_dir", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--robot-base-y", type=float, default=-0.45)
    p.add_argument("--robot-height", type=float, default=0.55)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stills", type=int, default=6, help="also save this many evenly spaced PNG frames")
    args = p.parse_args()

    import imageio.v2 as imageio

    wp.init()
    d = np.load(args.trial_dir / "frames.npz")
    if "joint" not in d:
        raise SystemExit("frames.npz has no robot joints; rerun fold_test.py with --robot")
    frames = {k: d[k] for k in d.files}
    n = len(frames["q"]) if args.max_frames is None else min(args.max_frames, len(frames["q"]))
    joints, grips = fill_missing_joints(frames["joint"]), frames["grip"]
    fps = 1.0 / float(np.median(np.diff(frames["t"])))

    model = build_scene(frames, args)
    bodies = sg2_model.body_index(model)
    sensor = SensorCamera(model)
    sensor.create_default_light(enable_shadows=True)
    cfg = sensor.default_render_config
    cfg.enable_shadows = True
    cfg.enable_backface_culling = False  # single-layer cloth triangles must show both faces
    state = model.state()

    outs = {}
    for name, (_body, w, h, hfov, _conv) in CAMERAS.items():
        f = w / (2 * math.tan(math.radians(hfov) / 2))
        rays = sensor.compute_camera_rays_pinhole_opencv(w, h, f, f, w / 2, h / 2, device=model.device)
        r = rays.numpy()
        r[..., 0, :] = r[..., 1, :] * NEAR[name]  # origin = near-plane point along each ray
        rays.assign(r)
        img = sensor.allocate_image_color(1, w, h)
        outs[name] = (rays, img, imageio.get_writer(args.trial_dir / f"{name}.mp4", fps=fps, macro_block_size=1), [])
    still_idx = set(np.linspace(0, n - 1, args.stills).astype(int).tolist()) if args.stills else set()

    for i in range(n):
        state.particle_q.assign(frames["q"][i])
        jq = wp.array(robot_joint_q(model, joints[i], grips[i]), dtype=float, device=model.device)
        newton.eval_fk(model, jq, model.joint_qd, state)
        model.bvh_refit_shapes(state)
        model.bvh_refit_particles(state)
        body_q = state.body_q.numpy()
        for name, (body, w, h, _hfov, conv) in CAMERAS.items():
            rays, img, writer, stills = outs[name]
            xf = wp.array([camera_transform(body_q[bodies[body]], conv)], dtype=wp.transformf, device=model.device)
            sensor.update(state, xf, rays, color_image=img)
            rgba = img.numpy()[0].view(np.uint8).reshape(h, w, 4)
            writer.append_data(rgba[..., :3])
            if i in still_idx:
                stills.append(rgba[..., :3].copy())
        if i % 50 == 0:
            print(f"frame {i}/{n} t={frames['t'][i]:.1f}s {frames['phase'][i]}", flush=True)

    for name, (_rays, _img, writer, stills) in outs.items():
        writer.close()
        if stills:
            imageio.imwrite(args.trial_dir / f"{name}_stills.png", np.concatenate(stills, axis=1))
    print(f"wrote {', '.join(f'{c}.mp4' for c in CAMERAS)} ({n} frames, {fps:.1f} fps) to {args.trial_dir}")


if __name__ == "__main__":
    main()
