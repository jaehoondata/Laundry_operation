"""Isaac Sim 4.5 container smoke test: headless RTX render of a cube to a PNG.

    bash sim/isaac/run_isaac.sh sim/isaac/smoke_isaac.py --out outputs/isaac/smoke.png
"""

import argparse
import time

from isaacsim import SimulationApp

p = argparse.ArgumentParser()
p.add_argument("--out", default="/work/outputs/isaac/smoke.png")
args, _ = p.parse_known_args()

t0 = time.time()
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
print(f"SimulationApp up in {time.time() - t0:.1f}s", flush=True)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
import omni.usd  # noqa: E402

stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")
cube = UsdGeom.Cube.Define(stage, "/World/Cube")
cube.GetSizeAttr().Set(0.2)
light = UsdLux.DomeLight.Define(stage, "/World/Dome")
light.GetIntensityAttr().Set(1000.0)
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(0.0, -1.0, 0.4))
UsdGeom.XformCommonAPI(cam).SetRotate(Gf.Vec3f(70.0, 0.0, 0.0))
rp = rep.create.render_product("/World/Cam", (672, 376))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach(rp)
for _ in range(20):
    app.update()
rep.orchestrator.step()
img = np.asarray(annot.get_data())[..., :3]
Image.fromarray(img.astype(np.uint8)).save(args.out)
print(f"saved {args.out} {img.shape} mean {img.mean():.1f}; total {time.time() - t0:.1f}s", flush=True)
app.close()
