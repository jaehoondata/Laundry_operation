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
# name: (URDF body, width, height, horizontal FOV deg, frame convention, near) + head correction
from camera_calib import CAMERAS, gl_rotation  # noqa: E402
# rays start this far in front of the camera frame, past the camera's own housing mesh
# (the head shell's front rim is still in view at 3 cm and gone from 5 cm; the D405 body ~5 mm)
NEAR = {name: c[5] for name, c in CAMERAS.items()}
# deploy table.xml (approx.); depth 0.86 puts the far edge where the real head camera sees it, and the
# real table runs past both sides of the image
TABLE = {"depth": 0.86, "width": 3.0, "thickness": 0.04, "edge_from_base": 0.075}
# defaults follow the real recordings: white shirt, light wood table, a black partition right behind
# the table; --cloth-color / --table-color / --wall-color change them (for visual variation)
COLORS = {"table": (0.80, 0.66, 0.46), "floor": (0.45, 0.45, 0.45), "cloth": (0.90, 0.90, 0.90),
          "wall": (0.04, 0.04, 0.04), "camera": (0.15, 0.15, 0.17)}
WALL_HEIGHT = 1.2  # partition behind the table's far edge [m above the table top]


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


def build_scene(frames, args, colors=COLORS):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    base = base_link_pose(args.robot_base_y, args.robot_height)
    sg2_model.add_sg2(builder, xform=wp.transform(base[:3, 3], quat_from_matrix(base[:3, :3])), with_geometry=True)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=frames["q"][0].tolist(), indices=frames["faces"].flatten().tolist(), density=0.2,
        particle_radius=1e-3, color=colors["cloth"],
    )
    # table: top face at z=0 (the cloth sim's ground), robot-side edge just in front of base_link
    edge_y = base[1, 3] + TABLE["edge_from_base"]
    depth, width = args.table_depth, args.table_width
    builder.add_shape_box(
        -1, hx=width / 2, hy=depth / 2, hz=TABLE["thickness"] / 2,
        xform=wp.transform((0.0, edge_y + depth / 2, -TABLE["thickness"] / 2), wp.quat_identity()),
        color=colors["table"],
    )
    far_y = edge_y + depth
    builder.add_shape_box(
        -1, hx=width / 2 + 1.0, hy=0.01, hz=WALL_HEIGHT / 2,
        xform=wp.transform((0.0, far_y + 0.01, WALL_HEIGHT / 2 - TABLE["thickness"]), wp.quat_identity()),
        color=colors["wall"],
    )
    builder.add_ground_plane(height=float(base[2, 3]), color=colors["floor"])
    model = builder.finalize(device=args.device)
    # the URDF gives the D405 bodies a red material; the real cameras are dark gray
    bodies = sg2_model.body_index(model)
    cam_bodies = {bodies[n] for n in ("camera_left_link", "camera_right_link") if n in bodies}
    shape_colors, shape_body = model.shape_color.numpy(), model.shape_body.numpy()
    for s in range(len(shape_colors)):
        if shape_body[s] in cam_bodies:
            shape_colors[s] = COLORS["camera"]
    model.shape_color.assign(shape_colors)
    return model


def robot_joint_q(model, joint, grip):
    """Full joint_q for one frame: arms from the recording, lift 0, head tilted, grippers open/closed."""
    qi = sg2_model.joint_q_index(model)
    q = model.joint_q.numpy().copy()
    q[qi["head_joint1"]] = HEAD_TILT
    for k, side in enumerate("lr"):
        for j, name in enumerate(sg2_model.ARM_JOINTS[side]):
            q[qi[name]] = joint[k, j]
        # grip is the closure 0..1 (older recordings: bool); gripper_joint1 1.1 is fully closed
        c = float(grip[k])
        for n, v in ((1, 1.1 * c), (2, 1.0 * c), (3, 1.1 * c), (4, 1.0 * c)):
            q[qi[f"gripper_{side}_joint{n}"]] = v
    return q


def camera_transform(body_q_row, name):
    p, quat = body_q_row[:3], body_q_row[3:7]
    x, y, z, w = quat
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return wp.transformf(wp.vec3f(*p), wp.quatf(*quat_from_matrix(gl_rotation(name, r))))


def rgb(text):
    v = tuple(float(x) for x in text.split(","))
    if len(v) != 3 or not all(0.0 <= x <= 1.0 for x in v):
        raise argparse.ArgumentTypeError("three comma-separated values in 0..1")
    return v


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
    for k in ("cloth", "table", "wall"):
        p.add_argument(f"--{k}-color", type=rgb, default=COLORS[k], help=f"{k} RGB in 0..1, e.g. 0.9,0.9,0.9")
    p.add_argument("--table-depth", type=float, default=TABLE["depth"], help="table depth (robot to wall) [m]")
    p.add_argument("--table-width", type=float, default=TABLE["width"], help="table width [m]")
    p.add_argument("--only-stills", action="store_true", help="render only the --stills frames (no videos)")
    p.add_argument("--light-dir", type=lambda t: tuple(float(x) for x in t.split(",")), default=(0.0, 0.3, -1.0),
                   help="direction the light travels (default: from above, slightly behind the robot, like the "
                        "lab's ceiling light; Newton's default -1,1,-1 left the cloth's wrinkles dark)")
    p.add_argument("--no-ambient", action="store_true", help="disable ambient light")
    p.add_argument("--shadows", action="store_true",
                   help="cast shadows from the default light (the lab's diffuse ceiling light casts almost none)")
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

    model = build_scene(frames, args, {**COLORS, "cloth": args.cloth_color, "table": args.table_color,
                                       "wall": args.wall_color})
    bodies = sg2_model.body_index(model)
    sensor = SensorCamera(model)
    light = np.asarray(args.light_dir, dtype=float)
    light /= np.linalg.norm(light)
    sensor.create_default_light(enable_shadows=args.shadows, direction=wp.vec3f(*light))
    cfg = sensor.default_render_config
    cfg.enable_shadows = args.shadows
    cfg.enable_ambient_lighting = not args.no_ambient
    cfg.enable_backface_culling = False  # single-layer cloth triangles must show both faces
    state = model.state()

    outs = {}
    for name, (_body, w, h, hfov, _conv, _near) in CAMERAS.items():
        f = w / (2 * math.tan(math.radians(hfov) / 2))
        rays = sensor.compute_camera_rays_pinhole_opencv(w, h, f, f, w / 2, h / 2, device=model.device)
        r = rays.numpy()
        r[..., 0, :] = r[..., 1, :] * NEAR[name]  # origin = near-plane point along each ray
        rays.assign(r)
        img = sensor.allocate_image_color(1, w, h)
        writer = None if args.only_stills else imageio.get_writer(args.trial_dir / f"{name}.mp4", fps=fps,
                                                                    macro_block_size=1)
        outs[name] = (rays, img, writer, [])
    still_idx = set(np.linspace(0, n - 1, args.stills).astype(int).tolist()) if args.stills else set()

    for i in (sorted(still_idx) if args.only_stills else range(n)):
        state.particle_q.assign(frames["q"][i])
        jq = wp.array(robot_joint_q(model, joints[i], grips[i]), dtype=float, device=model.device)
        newton.eval_fk(model, jq, model.joint_qd, state)
        model.bvh_refit_shapes(state)
        model.bvh_refit_particles(state)
        body_q = state.body_q.numpy()
        for name, (body, w, h, _hfov, _conv, _near) in CAMERAS.items():
            rays, img, writer, stills = outs[name]
            xf = wp.array([camera_transform(body_q[bodies[body]], name)], dtype=wp.transformf, device=model.device)
            sensor.update(state, xf, rays, color_image=img)
            rgba = img.numpy()[0].view(np.uint8).reshape(h, w, 4)
            if writer is not None:
                writer.append_data(rgba[..., :3])
            if i in still_idx:
                stills.append(rgba[..., :3].copy())
        if i % 50 == 0:
            print(f"frame {i}/{n} t={frames['t'][i]:.1f}s {frames['phase'][i]}", flush=True)

    for name, (_rays, _img, writer, stills) in outs.items():
        if writer is not None:
            writer.close()
        if stills:
            imageio.imwrite(args.trial_dir / f"{name}_stills.png", np.concatenate(stills, axis=1))
    print(f"wrote {', '.join(f'{c}.mp4' for c in CAMERAS)} ({n} frames, {fps:.1f} fps) to {args.trial_dir}")


if __name__ == "__main__":
    main()
