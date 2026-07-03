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
@wp.kernel(entry_point=wo.RAYGEN)
def raygen_program():
    idx = wo.optix_get_launch_index()
    ...
```

## Install

From the otk-pyoptix repo root:

```bash
pip install -e warp_optix/
```

The `optix` Python module (the pyoptix C++ binding) must also be installed:

```bash
pip install -e optix/
```

## Layout

- `warp_optix/__init__.py` — public re-exports of the runtime API.
- `warp_optix/_runtime/` — pyoptix-side runtime helpers (formerly `warp/_src/render/optix_*.py`).
- `warp_optix/_builtins.py` — OptiX `add_builtin(...)` registrations.
- `warp_optix/_codegen.py` — `OptixKernelType` enum and codegen entry-point specs.
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

Inside an `INTERSECTION` kernel, use `wp.optix_get_object_ray_origin()`,
`wp.optix_get_object_ray_direction()`, and `wp.optix_report_intersection()`.
The latter accepts a hit distance, hit kind, and zero to eight `wp.uint32`
attributes. Closest-hit and any-hit kernels can retrieve them with
`wp.optix_get_attribute_0()` through `wp.optix_get_attribute_7()`.

Keep the dictionaries returned by both helpers alive for as long as the GAS,
pipeline, and SBT are in use; they own the corresponding device allocations.

For mixed geometry, build one GAS per geometry type, place them under an IAS
with `create_instance_acceleration_structure()`, and pass one `HitKernel` per
SBT record through `hit_groups`. The pipeline helper infers combined triangle
and custom primitive flags and stores all hit records contiguously. See
`examples_warp/example_warp_optix_mixed_geometry.py` for a triangle and a
procedural sphere sharing one pipeline.

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
