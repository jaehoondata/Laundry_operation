"""SG2 camera models shared by the Newton and Isaac renderers.

Poses come from the URDF camera frames; the head camera gets a small correction fitted on the real
recordings (laundry_fold_tshirt_arm16): with both arms at the idle pose in every episode's first
frame, the URDF head camera put the grippers ~28 px right of where the real ZED sees them. Fitting
yaw / pitch / roll and the horizontal FOV to the 19 first frames' gripper silhouettes (cables
excluded) raised the silhouette IoU from 0.38 to 0.72 (2026-11-06).
"""

import math

import numpy as np

# name: (URDF frame, width, height, horizontal FOV deg, frame convention, near clip [m])
CAMERAS = {
    "head": ("zed_left_camera_optical_frame", 672, 376, 83.5, "optical", 0.05),
    "wrist_left": ("camera_left_link", 424, 240, 87.0, "ros_link", 0.01),
    "wrist_right": ("camera_right_link", 424, 240, 87.0, "ros_link", 0.01),
}
# extra rotation in the camera's own GL frame (x right, y up, looking along -z): yaw about y,
# pitch about x, roll about z, degrees, applied in that order
CORRECTION_DEG = {"head": (-4.0, 1.75, -0.5)}
LINK_TO_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
OPTICAL_TO_GL = np.diag([1.0, -1.0, -1.0])


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = axis
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def gl_rotation(name, frame_rotation):
    """Camera rotation in the GL convention from the URDF frame's world rotation."""
    conv = CAMERAS[name][4]
    r = frame_rotation @ (LINK_TO_OPTICAL if conv == "ros_link" else np.eye(3)) @ OPTICAL_TO_GL
    yaw, pitch, roll = CORRECTION_DEG.get(name, (0.0, 0.0, 0.0))
    return r @ _rot((0, 1, 0), yaw) @ _rot((1, 0, 0), pitch) @ _rot((0, 0, 1), roll)
