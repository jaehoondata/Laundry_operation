"""Render a baked fold trial (scene.npz from export_scene.py) with Isaac Sim 4.5's RTX renderer.

Runs inside the nvcr.io/nvidia/isaac-sim:4.5.0 container (numpy + USD only):

    bash sim/isaac/run_isaac.sh sim/isaac/render_isaac.py /work/<trial_dir>/scene.npz --out /work/<trial_dir>/isaac

Builds a USD stage with the SG2 links (URDF visual meshes, posed per frame), the cloth mesh (points
per frame), table, partition wall and floor, lights, and the three robot cameras, then writes
<out>/<camera>/00000.png ... per frame. encode_isaac.sh turns them into <out>/<camera>.mp4 on the host.
Colours and light levels are flags so a sweep can match the real recordings or randomise them.
"""

import argparse
import math
import time
from pathlib import Path

from isaacsim import SimulationApp

p = argparse.ArgumentParser()
p.add_argument("scene", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--max-frames", type=int, default=None)
p.add_argument("--frames", type=str, default=None, help="comma-separated frame indices to render only")
# defaults matched to the real head camera (first frames of the 19 LR-L-R-L episodes, region means
# from sim/isaac/color_regions.py): wall [11 12 15], table [144 125 75], shirt [222 232 240],
# gripper [30 30 34]; rendered with these: wall [11 10 11], table ~[150 132 85], shirt ~[220 226 230],
# gripper [29 30 31] (2026-11-06)
p.add_argument("--cloth-color", default="0.93,0.97,1.0")
p.add_argument("--table-color", default="0.49,0.37,0.17")
p.add_argument("--wall-color", default="0.012,0.012,0.014")
p.add_argument("--robot-color", default="0.045,0.045,0.05")
p.add_argument("--floor-color", default="0.35,0.35,0.35")
p.add_argument("--table-texture", default="wood",
                help="'wood' (generated grain in --table-color), an image path, or 'none' (flat --table-color)")
p.add_argument("--texture-tile", type=float, default=0.8, help="table texture tile size [m]")
p.add_argument("--texture-seed", type=int, default=0)
p.add_argument("--window-bottom", type=float, default=0.52, help="partition top / window bottom above the table [m]")
p.add_argument("--no-windows", dest="windows", action="store_false", help="plain black wall")
p.add_argument("--window-glow", type=float, default=0.7, help="frosted window emission strength")
p.add_argument("--mullion-spacing", type=float, default=0.6)
p.add_argument("--side-wall-x", type=float, default=0.75, help="pale wall on the right at this x (0: none)")
p.add_argument("--side-wall-color", default="0.62,0.63,0.65")
p.add_argument("--dome", type=float, default=120.0, help="dome (ambient) light intensity")
p.add_argument("--rect", type=float, default=4500.0, help="overhead rect light intensity")
p.add_argument("--rect-size", type=float, default=1.5)
p.add_argument("--rect-height", type=float, default=1.8, help="rect light height above the table [m]")
p.add_argument("--rt-subframes", type=int, default=4)
p.add_argument("--normals", action="store_true", help="author smooth vertex normals on the cloth")
p.add_argument("--renderer", default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"])
args, _ = p.parse_known_args()

t0 = time.time()
app = SimulationApp({"headless": True, "renderer": args.renderer})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402


def rgb(text):
    return Gf.Vec3f(*[float(v) for v in text.split(",")])


def material(stage, path, color, roughness=0.6, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def texture_material(stage, path, image, roughness=0.55):
    """UsdPreviewSurface whose diffuse colour comes from an sRGB image via the mesh's `st` UVs."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(image))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def textured_quad(stage, path, x0, x1, y0, y1, z, tile, mat):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y0, z), Gf.Vec3f(x1, y1, z), Gf.Vec3f(x0, y1, z)])
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f((x1 - x0) / tile, 0), Gf.Vec2f((x1 - x0) / tile, (y1 - y0) / tile),
            Gf.Vec2f(0, (y1 - y0) / tile)])
    bind(m.GetPrim(), mat)


def emissive_material(stage, path, color, strength):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(strength * c for c in color)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def mesh(stage, path, verts, faces, mat, double_sided=False):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(verts, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.ravel(), dtype=np.int32)))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(double_sided)
    bind(m.GetPrim(), mat)
    return m


def box(stage, path, center, size, mat):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    x = UsdGeom.XformCommonAPI(c)
    x.SetTranslate(Gf.Vec3d(*center))
    x.SetScale(Gf.Vec3f(*size))
    bind(c.GetPrim(), mat)


def set_matrix(xform_op, t):
    xform_op.Set(Gf.Matrix4d(np.asarray(t, dtype=float).T.tolist()))  # USD row-vector convention


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


d = np.load(args.scene)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
mats = {k: material(stage, f"/World/Looks/{k}", rgb(getattr(args, f"{k}_color")), r)
        for k, r in (("cloth", 0.85), ("table", 0.55), ("wall", 0.9), ("robot", 0.45), ("floor", 0.9))}

# robot links: one Xform per link (pose per frame), mesh in link-visual coordinates
names = [str(n) for n in d["link_names"]]
link_ops = []
for k, n in enumerate(names):
    x = UsdGeom.Xform.Define(stage, f"/World/Robot/{n}")
    link_ops.append(x.AddTransformOp())
    mesh(stage, f"/World/Robot/{n}/mesh", d[f"verts_{k}"], d[f"faces_{k}"], mats["robot"])

# cloth
cloth_faces = d["cloth_faces"]
cloth = mesh(stage, "/World/Cloth", d["cloth_q"][0], cloth_faces, mats["cloth"], double_sided=True)
# authored vertex normals point one way through the single-layer cloth, so a folded-over flap (seen
# from its back) shaded black; --normals keeps them, default lets RTX shade each side itself
cloth_normals = cloth.CreateNormalsAttr() if args.normals else None
if args.normals:
    cloth.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

# table (top at z=0), partition wall behind it, floor at the robot base
edge_y, depth, width, thick, wall_h = [float(v) for v in d["table"]]
box(stage, "/World/Table", (0.0, edge_y + depth / 2, -thick / 2 - 0.001), (width, depth, thick), mats["table"])
if args.table_texture != "none":
    tex = Path(args.table_texture) if args.table_texture != "wood" else args.out / "wood.png"
    if args.table_texture == "wood":  # generated: grain along x, mean colour = --table-color
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from make_wood import wood

        tex.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(wood(color=[float(v) for v in args.table_color.split(",")], seed=args.texture_seed)).save(tex)
    textured_quad(stage, "/World/TableTop", -width / 2, width / 2, edge_y, edge_y + depth, 0.0, args.texture_tile,
                  texture_material(stage, "/World/Looks/tabletop", tex.resolve()))
# black partition up to --window-bottom, frosted windows with black mullions above it (the wrist
# cameras see them; the head camera's view of the wall ends ~0.48 m up), a pale side wall on the right
far = edge_y + depth + 0.01
wb = args.window_bottom
box(stage, "/World/Wall", (0.0, far, wb / 2 - thick / 2), (width + 2.0, 0.02, wb + thick), mats["wall"])
if args.windows:
    glass = emissive_material(stage, "/World/Looks/glass", Gf.Vec3f(0.85, 0.88, 0.92), args.window_glow)
    box(stage, "/World/Windows", (0.0, far + 0.02, wb + 0.6), (width + 2.0, 0.02, 1.2), glass)
    for k, x in enumerate(np.arange(-1.5, 1.51, args.mullion_spacing)):
        box(stage, f"/World/Mullion{k}", (float(x), far + 0.005, wb + 0.6), (0.05, 0.02, 1.2), mats["wall"])
    box(stage, "/World/Transom", (0.0, far + 0.005, wb + 0.45), (width + 2.0, 0.02, 0.04), mats["wall"])
if args.side_wall_x > 0:
    side = material(stage, "/World/Looks/side", rgb(args.side_wall_color), 0.9)
    box(stage, "/World/SideWall", (args.side_wall_x, (edge_y + far) / 2, 0.6), (0.02, far - edge_y, 1.4), side)
box(stage, "/World/Floor", (0.0, 0.0, -0.80), (8.0, 8.0, 0.02), mats["floor"])

# lights: soft ambient + a ceiling panel (the lab is lit evenly from above)
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(args.dome)
rect = UsdLux.RectLight.Define(stage, "/World/Ceiling")
rect.CreateIntensityAttr(args.rect)
rect.CreateWidthAttr(args.rect_size)
rect.CreateHeightAttr(args.rect_size)
UsdGeom.XformCommonAPI(rect).SetTranslate(Gf.Vec3d(0.0, edge_y + 0.4, args.rect_height))  # faces -z by default

# cameras (USD: looks along -z, +y up; cam_T is already in that convention)
cams, cam_ops, annots = [str(c) for c in d["cam_names"]], [], []
for ci, c in enumerate(cams):
    w, h = [int(v) for v in d["cam_res"][ci]]
    cam = UsdGeom.Camera.Define(stage, f"/World/Cameras/{c}")
    f = 18.0
    ha = 2 * f * math.tan(math.radians(float(d["cam_hfov"][ci])) / 2)
    cam.CreateFocalLengthAttr(f)
    cam.CreateHorizontalApertureAttr(ha)
    cam.CreateVerticalApertureAttr(ha * h / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(float(d["cam_near"][ci]), 50.0))
    cam_ops.append(cam.AddTransformOp())
    rp = rep.create.render_product(f"/World/Cameras/{c}", (w, h))
    an = rep.AnnotatorRegistry.get_annotator("rgb")
    an.attach(rp)
    annots.append(an)
    (args.out / c).mkdir(parents=True, exist_ok=True)

rep.orchestrator.set_capture_on_play(False)
n = len(d["t"]) if args.max_frames is None else min(args.max_frames, len(d["t"]))
todo = [int(v) for v in args.frames.split(",")] if args.frames else range(n)
print(f"stage ready in {time.time() - t0:.1f}s; rendering {len(todo)} frames", flush=True)
t1 = time.time()
for k, i in enumerate(todo):
    for li, op in enumerate(link_ops):
        set_matrix(op, d["link_T"][i, li])
    for ci, op in enumerate(cam_ops):
        set_matrix(op, d["cam_T"][i, ci])
    q = d["cloth_q"][i]
    cloth.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(q, dtype=np.float32)))
    if cloth_normals is not None:
        cloth_normals.Set(Vt.Vec3fArray.FromNumpy(vertex_normals(q, cloth_faces).astype(np.float32)))
    rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0)
    for c, an in zip(cams, annots):
        img = np.asarray(an.get_data())
        Image.fromarray(img[..., :3].astype(np.uint8)).save(args.out / c / f"{i:05d}.png")
    if k % 100 == 0:
        print(f"frame {k}/{len(todo)} ({(k + 1) / (time.time() - t1):.1f} fps)", flush=True)
print(f"wrote {len(todo)} frames x {len(cams)} cameras to {args.out} in {time.time() - t1:.0f}s", flush=True)
rep.orchestrator.wait_until_complete()
app.close()
