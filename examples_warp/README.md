# Warp + OptiX examples

These examples were moved from `warp/examples/core/pyoptix/` into this directory.

Prerequisites:

```bash
pip install -e .. -e ../warp_optix/
```

Then run individual examples, e.g.

```bash
python example_warp_optix_kernels.py
python example_warp_optix_curves.py
python example_warp_optix_motion_blur.py
python example_warp_optix_mixed_geometry.py
python example_warp_optix_tiny_raytracer.py
python example_warp_optix_basic_pathtracing.py
python example_warp_optix_pathtraced_hairball.py
python example_warp_optix_usd_pathtracing.py path/to/scene.usd
```

The hair-ball example procedurally packs thousands of randomized, tapered
helical strands into one native round-linear curve geometry. Use
`--hair-count`, `--segments`, `--hair-length`, `--curl-radius`, and
`--curl-turns` to change its shape; `--seed` makes variations reproducible.

The full PBR path tracer also accepts native OptiX round-linear curves through
the same reusable geometry/instance API as triangle meshes:

```python
import numpy as np

from warp_optix.pathtracing import PathTracerAPI

api = PathTracerAPI()
api.initialize()
material = api.create_pbr_material((0.8, 0.2, 0.05), roughness=0.35, metallic=0.0)

points = np.array(
    ((-1.0, 0.0, 0.0), (0.0, 0.5, 0.0), (1.0, 0.0, 0.0)),
    dtype=np.float32,
)
radii = np.array((0.03, 0.08, 0.03), dtype=np.float32)
curve_id = api.create_curve(points, radii, material_id=material)
instance_id = api.create_instance(curve_id)
api.build_scene()
```

Radii are specified per control point and interpolated linearly along every
segment. By default every consecutive point pair is a segment. Pass explicit
`segment_indices` containing each segment's starting control-point index to
store multiple disjoint strands in one geometry. Curve instances support the
same transforms, visibility updates, material overrides, and TLAS rebuilds as
mesh instances. Curves use the existing PBR material, lighting, alpha,
transmission, guide-buffer, and shadow paths; USD curve import is not required.

The basic path-tracing example automatically downloads its default A Beautiful
Game scene into the platform cache on first use. On Linux this defaults to
`~/.cache/warp_optix/assets/`. Pass `--scene-gltf PATH` to use a local `.gltf`
or `.glb` scene instead.

The USD path-tracing example loads a user-supplied composed USD stage and maps
UsdPreviewSurface plus common NVIDIA OmniPBR/OmniGlass MDL inputs to the
renderer's glTF-style PBR materials. Install OpenUSD alongside the existing
path-tracing image dependencies:

```bash
pip install -e .. -e "../warp_optix[pathtracing,usd]"
python example_warp_optix_usd_pathtracing.py \
  "D:\Sample_Scenes_NVD@10013\Samples\Marbles\Marbles_Assets.usd"
```

DDS images use Pillow's built-in decoder; no separate DDS dependency is
needed. UDIM sets are discovered from `<UDIM>` paths, assembled into grid
atlases, and addressed with normalized texture transforms. OpenUSD composes
referenced USD layers, payloads, and instance proxies before the
loader imports polygon meshes, UVs, normals, material subsets, base color,
normal, ORM, separate roughness/metallic maps, emissive, and glass parameters.
Separate linear maps are packed into glTF's roughness/metallic channels during
parallel texture decoding. The example limits each texture or UDIM tile to
2048 pixels by default. Stitched atlases grow with their tile count, so three
2048-pixel tiles produce a 6144-pixel-wide atlas. Use a smaller value for very
large aggregate stages such as `Marbles_Assets.usd`, or pass
`--max-texture-size 0` to keep full tile resolution.
Pass `--load-usd-environment` to opt into the first composed lat-long
`DomeLight` texture (`--usd-environment-scale` adjusts its intensity). The
stage is composed only once; texture decoding is parallelized and identical
face corners are shared to reduce CPU load, upload size, and BLAS memory.
Both path-tracing examples use Newton ViewerOptix's display defaults (0.68
exposure, 1.08 contrast, and 1.1 saturation), with command-line overrides.
The USD example first uses the camera selected by `UsdRenderSettings` or the
stage's `customLayerData.cameraSettings.boundCamera`; Marbles therefore starts
from `/stage/Overview`. If no suitable perspective camera is authored, it
frames transform-aware world-space geometry bounds. Use `--usd-camera PATH`
to select a camera explicitly or `--no-usd-camera` to force bounds framing.
Extreme flat ground planes are excluded from fallback camera fitting when
other geometry is present, but remain loaded and rendered. Both examples
capture the stable OptiX launch as a CUDA graph by default while keeping
per-frame camera/jitter parameters and DLSS evaluation dynamic; use
`--no-cuda-graphs` only for diagnostics. Frame-level capture is automatically
skipped while DLSS-RR is active because RTX/NGX resource event bookkeeping is
not legal during CUDA stream capture; USD transform/TLAS device batches remain
independently graph-capturable.
The loader triangulates authored polygons but does not tessellate subdivision
surfaces or import camera animation, curves, or point instancers. Other USD
light types are not currently imported.
