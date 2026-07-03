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
