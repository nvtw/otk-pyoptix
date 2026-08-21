# warp_optix

Warp addon that adds OptiX support, shipped as part of `otk-pyoptix`.

This package replaces the `warp.optix` Python module that used to live inside
the `warp` source tree. It registers itself with warp on import:

```python
import warp as wp
import warp_optix as wo  # registers OptiX builtins, headers, and entry-point specs
```

Once imported, you can decorate kernels with the OptiX entry-point kinds:

```python
@wo.optix_kernel(wo.OptixKernelType.RAYGEN)
def raygen_program():
    idx = wo.optix_get_launch_index()
    ...
```

## Install

From the otk-pyoptix repo root:

```bash
pip install -e . -e warp_optix/
```

`warp_optix` uses Warp's public addon hooks when they are available. With
vanilla `warp-lang>=1.15`, it falls back to a version-checked private API
adapter for builtin registration, OptiX entry-point generation, and PTX
compilation. The fallback changes only the active Python process; it does not
patch or overwrite the installed Warp package.

## Path-tracing viewer

The example path tracer is also installed as `warp_optix.pathtracing`. Install
its windowing and image extras with:

```bash
pip install -e . -e "warp_optix[pathtracing]"
```

`PathTracingViewer` is a standalone OptiX path-tracing viewer with DLSS Ray
Reconstruction. `PathTracingViewerBackend` exposes a renderer-facing `log_*`
API without importing Newton or any other simulation framework. A framework
can add its own inheritance in a thin wrapper; with current Newton development
branches that wrapper is:

```python
from newton.viewer import ViewerBase
from warp_optix.pathtracing import PathTracingViewerBackend


class ViewerOptix(PathTracingViewerBackend, ViewerBase):
    pass
```

The optional interactive features are installed separately:

```bash
pip install -e . -e "warp_optix[pathtracing,ui,recording]"
```

The viewer follows the earlier hybrid viewer controls:

- WASD or arrow keys move the camera; Q/E move along the model up axis.
- Left drag looks around and the scroll wheel changes field of view.
- Right click and drag picks Newton bodies when Newton is installed.
- Space pauses, Escape closes, and 0-8 select path-tracing debug buffers.
- R starts MP4 recording and T stops it. Recordings default to the system
  Videos directory under `NewtonRecordings/pathtracing_recording_*.mp4`.

Recording packs RGB8 on the GPU, reads back asynchronously through pinned
buffers, and encodes on a worker thread. The automatic encoder probes the
system FFmpeg with a real frame and prefers `h264_nvenc`; it falls back to
`libx264` with its ultrafast low-latency preset. Set `recording_encoder` to
`"h264_nvenc"` or `"libx264"` to override selection. FFmpeg vertically flips
the raw OptiX image into display orientation while encoding.

`register_ui_callback()` adds application controls to the optional ImGui panel.
The panel also exposes rendering statistics, DLSS state, visualization flags,
camera state, debug buffers, picking, pause, and recording controls. Picking is
loaded dynamically from Newton when `set_model()` is called, so importing and
using the standalone viewer does not add a Newton dependency.

The backend accepts Warp arrays for meshes, transforms, colors, and material
parameters. It handles Newton's X/Y/Z up-axis conversion, mesh and instance
caching, visibility updates, and roughness/metallic PBR materials. Its default
physical-sky values and sRGB-to-linear color conversion intentionally match
the earlier hybrid viewer, including the light ground, slight haze, and soft
horizon used for untextured simulation geometry.

The four material values passed to `log_instances()` are roughness, metallic,
U subdivisions, and V subdivisions. Positive subdivision values enable a
procedural UV checker overlay; alternating cells multiply the sampled base
color by `base_color_scale` (default `0.75`). A zero subdivision disables the
overlay. The same `u_subdiv`, `v_subdiv`, and `base_color_scale` controls are
available when creating PBR or glTF materials directly.

The framework-neutral class can also be driven directly through
`log_mesh()`, `log_instances()`, `begin_frame()`, and `end_frame()`. Textured
glTF and USD scenes remain available through `PathTracerAPI`; USD loading uses
the optional `usd-core` extra and maps UsdPreviewSurface and common NVIDIA MDL
inputs to the same internal PBR material representation. Composed DomeLight
HDR textures can be enabled explicitly with
`load_scene_from_usd(..., load_usd_environment=True)`. The framework
`log_mesh` adapter currently maps vertex colors and PBR values but does not
ingest its optional texture argument.

`PathTracerAPI` enables CUDA graph replay for the stable OptiX launch by
default when DLSS-RR is inactive. Launch parameters are written to a persistent device buffer before
each replay, so camera motion, Halton jitter, frame indices, materials, and
TLAS handles remain dynamic. With DLSS-RR active, frame-level capture is
skipped because RTX/NGX resource event queries are forbidden during CUDA
stream capture; USD transform/TLAS device updates remain graph-capturable.
Pass `enable_cuda_graphs=False` to diagnose a driver or
capture compatibility issue; `api.cuda_graph_active` reports successful
capture after the first rendered sample.

Large dynamic arrow sets use one fixed-capacity native-curve geometry rather
than one OptiX instance per arrow. Each arrow is a constant-radius shaft plus
a linearly tapered tip. Inactive slots have zero width, so a changing contact
count does not reallocate buffers or rebuild the whole scene. By default only
the arrow GAS is rebuilt, followed by a TLAS update:

```python
material = api.create_pbr_material((0.9, 0.25, 0.03), 0.65, 0.0)
arrows = api.create_arrow_batch(
    capacity=max_contact_count,
    small_radius=0.002,
    large_radius=0.006,
    tip_length_ratio=0.2,
    material_id=material,
)
api.build_scene()

# CUDA wp.vec3 buffers with at least max_contact_count entries. The one-element
# CUDA int32 count avoids a device-to-host synchronization when contacts vary;
# it is clamped to max_contact_count on-device.
api.update_arrow_batch_device(
    arrows,
    contact_starts_cuda,
    contact_ends_cuda,
    contact_count_cuda,
    material_ids=contact_material_ids_cuda,  # Optional int32 ID per arrow.
    stream=simulation_stream,
)
```

The host equivalent is `update_arrow_batch(arrows, starts, ends)`. Per-arrow
materials use the existing per-curve-primitive material table; the API assigns
the same ID to an arrow's shaft and tip. When updating several dynamic batches,
pass `rebuild_tlas=False` to each update and call `api.rebuild_tlas()` once at
the end.

A fast GAS rebuild is deliberately the default because contact ordering may
shuffle completely between frames. Use `rebuild_gas=False` only when arrow IDs
remain spatially coherent; refitting is correct in either case, but arbitrary
reordering can substantially degrade the refitted BVH's traversal quality.

USD loads also retain a path-addressable transform hierarchy instead of
baking composed transforms into vertices:

```python
api.load_scene_from_usd("scene.usd")
usd = api.usd_scene
body = usd.require_transform("/World/Robot/base")

# Host batch (local 4x4 matrices).
usd.update_local_transforms([body], matrices)

# Zero-staging CUDA batch on a caller-owned Warp stream.
transform_count_cuda = wp.array([capacity], dtype=wp.int32, device="cuda")
usd.update_local_transforms_device(
    transform_count_cuda, body_ids_cuda, local_mat44_cuda, stream=simulation_stream
)
# Newton-style wp.transform + wp.vec3 scale arrays are accepted directly too.
usd.update_local_transform_trs_device(
    transform_count_cuda,
    body_ids_cuda,
    local_poses_cuda,
    local_scales_cuda,
    stream=simulation_stream,
)

# Or decouple hierarchy writes from the OptiX update.
usd.update_local_transforms_device(
    transform_count_cuda,
    body_ids_cuda,
    local_mat44_cuda,
    stream=simulation_stream,
    rebuild_tlas=False,
)
usd.update_tlas(stream=simulation_stream)
```

The ID and transform arrays define a fixed launch capacity. A CUDA kernel may
change the single active-count value inside a captured graph to update any
prefix of those arrays without graph recapture or host synchronization.

`usd.transforms` enumerates a stable `USDTransformHandle` for every composed
transformable prim path. `usd.get_transform(path)` performs a non-throwing
lookup, `usd.get_prim(path)` accesses any prim on the retained OpenUSD stage,
and the CUDA local/world `wp.mat44` arrays are exposed after scene build. A
batch composes hierarchy levels, updates all affected OptiX instance and
motion-vector transforms, and updates the TLAS on the selected CUDA stream.
The device methods and allocation-free TLAS `UPDATE` path are CUDA graph
capture compatible after one warm-up call (which compiles kernels and sizes
the reusable TLAS buffers). Keep the device batch length and array addresses
stable across graph replays; a batch length of one is supported. The NumPy
convenience method performs allocation/upload and is intentionally not the
graph-capture path.

The old hybrid viewer's Vulkan-backed OpenGL transform VBO was specific to its
C# bridge. Compatibility queries remain available, but return unavailable;
dynamic Newton transforms instead use Warp arrays through `log_instances()`
or `update_instance_transforms()` and trigger an OptiX TLAS refit. Transform
matrix construction is vectorized before the retained OptiX instance buffer is
updated. Window presentation uses CUDA/OpenGL interop with Warp's copy fallback
when direct registration is unavailable.

## Layout

- `warp_optix/__init__.py` — public re-exports of the runtime API.
- `warp_optix/_runtime/` — pyoptix-side runtime helpers (formerly `warp/_src/render/optix_*.py`).
- `warp_optix/_builtins.py` — OptiX `add_builtin(...)` registrations.
- `warp_optix/_codegen.py` — `OptixKernelType` enum and codegen entry-point specs.
- `warp_optix/_compat.py` — vanilla Warp private-API fallback.
- `warp_optix/_native/include/warp_optix_builtins.h` — device-side C++ wrappers.
- `warp_optix/_addon.py` — runs on import; wires the above into warp.

## Custom primitives

Custom geometry uses OptiX AABBs plus a Warp intersection program. The runtime
helper accepts either `(N, 6)` AABBs or `(N, 2, 3)` min/max pairs:

```python
gas, gas_buffers = wo.create_custom_primitive_gas(
    optix,
    context,
    aabbs,
    device="cuda",
)

pipeline, sbt, pipeline_resources = wo.create_pipeline_and_sbt(
    optix,
    context,
    ptx,
    raygen_program,
    miss_program,
    closest_hit_program,
    num_payload_values=2,
    num_attribute_values=3,
    device="cuda",
    intersection_entry=intersection_program,
)
```

See `examples_warp/example_warp_optix_motion_blur.py` for temporal sampling
across a two-key triangle deformation.

Inside an `INTERSECTION` kernel, use `wp.optix_get_object_ray_origin()`,
`wp.optix_get_object_ray_direction()`, and `wp.optix_report_intersection()`.
The latter accepts a hit distance, hit kind, and zero to eight `wp.uint32`
attributes. Closest-hit and any-hit kernels can retrieve them with
`wp.optix_get_attribute_0()` through `wp.optix_get_attribute_7()`.

Keep the dictionaries returned by both helpers alive for as long as the GAS,
pipeline, and SBT are in use; they own the corresponding device allocations.

For mixed geometry, build one GAS per geometry type, place them under an IAS
with `create_instance_acceleration_structure()`, and pass one `HitKernel` per
SBT record through `hit_groups`. The pipeline helper infers combined primitive
flags and stores all hit records contiguously. See
`examples_warp/example_warp_optix_mixed_geometry.py` for triangle, native curve,
and analytical custom geometry sharing one pipeline.

`SbtKernelManager` is the single low-level SBT builder used by the convenience
pipeline helper. It keeps records header-only, accepts decorated Warp kernels
or explicit entry names, and returns opaque hit-group handles. Resolve a handle
with `get_sbt_offset()` only when assigning an OptiX instance; users do not need
to pack headers or calculate record strides.

Warp kernels also expose the common OptiX hit context directly: ray time,
flags and visibility mask; instance ID/index; primitive, SBT GAS and GAS
handles; front/back-face tests; and point/vector/normal transforms in both
directions. The names follow Warp's existing snake-case convention, for
example `wp.optix_get_instance_index()` and
`wp.optix_transform_point_from_world_to_object_space()`.

Triangle any-hit and closest-hit kernels can call
`wp.optix_get_triangle_vertex_data()`. It returns a `wp.mat33` whose rows are
the three object-space vertices of the current triangle; this current-hit form
does not require random vertex access or extra launch-parameter arrays.

Acceleration-structure builders accept `compact=True` for static data and a
normal OptiX `build_flags` value for advanced use. For dynamic geometry, build
with `BUILD_FLAG_ALLOW_UPDATE`, update the retained Warp buffer, then refit in
place:

```python
gas, gas_resources = wo.create_triangle_gas(
    optix,
    context,
    vertices,
    indices,
    "cuda",
    build_flags=optix.BUILD_FLAG_ALLOW_UPDATE,
)
gas_resources["d_vertices"].assign(updated_vertices)
gas = wo.refit_acceleration_structure(optix, context, gas_resources)
```

Compaction and update are deliberately mutually exclusive in these helpers:
compaction targets immutable geometry, while update retains the original output
capacity required by OptiX refits.

For vertex motion blur, pass triangle vertices as `(K, N, 3)` motion keys and
enable motion on the pipeline:

```python
gas, gas_resources = wo.create_triangle_gas(
    optix, context, vertex_keys, indices, "cuda", motion_time_range=(0.0, 1.0)
)
pipeline, sbt, pipeline_resources = wo.create_pipeline_and_sbt(
    ...,
    uses_motion_blur=True,
)
```

`create_curve_gas()` builds native round linear, B-spline, Catmull-Rom, and
Bezier curves. Associate its hit record with OptiX's built-in intersector by
setting `HitKernel(builtin_intersection_type=curve_type)`. Curve shaders can
read the current parameter with `wp.optix_get_curve_parameter()`. Curve vertex
and width arrays may also contain motion keys.

Exception, direct-callable, and continuation-callable programs use the same
`optix_kernel()` decorator as other stages. Pass them through
`exception_entry`, `direct_callable_entries`, or
`continuation_callable_entries` when creating the pipeline. The helper packs
their SBT records and computes the additional stack requirements. Callable
handles resolve through `SbtKernelManager.get_callable_index()`.

Callables intentionally use a no-argument, void interface:
`wp.optix_direct_call(index)` or `wp.optix_continuation_call(index)`. They can
read launch parameters and write referenced Warp arrays. Arbitrary typed
callable arguments and returns are outside Warp's current external entry ABI
and are therefore not exposed. Exception programs can use
`wp.optix_get_exception_code()` and detail accessors; user code can raise an
exception with `wp.optix_throw_exception()` and up to eight `wp.uint32`
details.
