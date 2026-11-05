"""Load the ROBOTIS FFW-SG2 into a Newton ModelBuilder.

The installed URDF mixes three mesh URI styles (package://, ${swerve_meshes_dir}, and an absolute
file:///root/... path from the robot PC). They are rewritten in memory to the local ros2 install;
the deploy tree itself is only read.

Joint naming follows the URDF: lift_joint, head_joint1/2, arm_{l,r}_joint1..7, gripper_{l,r}_joint1..4.
The FoldNet++-aligned TCP is end_effector_{l,r}_link (approach -z, fingers close along y).
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import newton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sg2_reach"))
from sg2_reach import DEFAULT_URDF as URDF  # noqa: E402
from sg2_reach import find_ros_install  # noqa: E402

ROS_INSTALL = find_ros_install()
ARM_JOINTS = {s: [f"arm_{s}_joint{i}" for i in range(1, 8)] for s in "lr"}
TCP_LINKS = {"l": "end_effector_l_link", "r": "end_effector_r_link"}


def resolve_mesh_uris(xml: str, ros_install: Path = ROS_INSTALL) -> str:
    share = ros_install / "ffw_description/share/ffw_description"
    xml = xml.replace("package://ffw_description/", f"{share}/")
    xml = xml.replace("${swerve_meshes_dir}", f"{share}/meshes/common/follower/ffw_swerve")
    return re.sub(r"file:///root/ros2_ws/install/", f"{ros_install}/", xml)


def strip_markers(xml: str) -> str:
    """Drop the red TCP marker boxes on end_effector_{l,r}_link (URDF debug visuals, not on the robot)."""
    root = ET.fromstring(xml)
    for link in root.findall("link"):
        if link.get("name", "").startswith("end_effector_"):
            for el in link.findall("visual") + link.findall("collision"):
                link.remove(el)
    return ET.tostring(root, encoding="unicode")


def strip_geometry(xml: str) -> str:
    """Drop visual/collision elements (kinematics-only model, no mesh loading)."""
    root = ET.fromstring(xml)
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    return ET.tostring(root, encoding="unicode")


def add_sg2(builder: newton.ModelBuilder, xform=None, with_geometry: bool = False, urdf: Path = URDF):
    """Add a fixed-base SG2 (base_link at `xform`). Returns the builder's joint/body count before adding."""
    xml = resolve_mesh_uris(urdf.read_text())
    xml = strip_geometry(xml) if not with_geometry else strip_markers(xml)
    builder.add_urdf(xml, xform=xform, floating=False, enable_self_collisions=False)


def joint_q_index(model: newton.Model) -> dict[str, int]:
    """Joint name -> index into model.joint_q, for joints that own coordinates (fixed joints skipped)."""
    names = model.joint_label if hasattr(model, "joint_label") else model.joint_key
    starts = model.joint_q_start.numpy()
    return {n.split("/")[-1]: int(starts[i]) for i, n in enumerate(names) if starts[i + 1] > starts[i]}


def body_index(model: newton.Model) -> dict[str, int]:
    names = model.body_label if hasattr(model, "body_label") else model.body_key
    return {n.split("/")[-1]: i for i, n in enumerate(names)}
