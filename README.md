# warp-optix

Warp ray tracing with bundled OptiX 9.1 bindings and DLSS Ray Reconstruction.
Write ray-generation, hit, miss, and callable programs as Warp kernels, or use
the included path tracer. Applications import `warp_optix`; the native backend
is bundled privately and requires no separate PyOptiX installation.

## Installation

Build from the repository root with Python 3.10+, a CUDA Toolkit and a C++17
compiler. GPU execution requires a compatible NVIDIA driver. OptiX headers and
DLSS SDK 310.9.1 are fetched automatically.

```powershell
pip install .
# Editable development with viewer dependencies:
pip install -e ".[pathtracing,ui,usd]"
```

This builds one platform wheel containing the Python package and native
extension. Extras are `pathtracing`, `ui`, `usd`, `recording`, and `training`.
The nested Python package is no longer an independent build entry point.

## Public API

```python
import warp as wp
import warp_optix as wo
from warp_optix.pathtracing import PathTracerAPI
```

For custom ray tracers, a context owns geometry and pipelines:

```python
# Geometry arrays, PTX, entry kernels, and launch parameters are application inputs.
with wo.Context(device="cuda:0") as context:
    geometry = context.create_triangle_gas(vertices, indices)
    pipeline = context.create_pipeline(
        ptx, raygen, miss, closest_hit,
        num_payload_values=3, num_attribute_values=2,
    )
    # Store geometry.handle in your Warp launch parameters.
    pipeline.launch(params_buffer, width=640, height=480)
```

The context retains geometry buffers, modules, SBTs, and pipelines. Closing
it synchronizes the CUDA device and releases resources. Keep arrays referenced
by launch parameters alive until GPU work completes. Launches are asynchronous
on Warp's current stream, or on an explicitly supplied `wp.Stream`; geometry
builds and refits are synchronous.

See the [API guide](warp_optix/README.md), [migration notes](docs/migration-0.2.md),
and runnable [examples](examples_warp/README.md).

## DLSS builds

DLSS is enabled by default and its runtime library is included in the wheel.
Build alternative variants into separate directories:

```powershell
uv build --wheel --out-dir dist/dlss
uv build --wheel --out-dir dist/no-dlss --config-setting cmake.define.PYOPTIX_ENABLE_DLSS=OFF --config-setting build-dir=build/no-dlss
```

Install only one variant at a time. The disabled variant retains harmless
DLSS stubs. `DLSS_ROOT` selects a local SDK; `PYOPTIX_AUTO_DOWNLOAD_DLSS=OFF`
disables downloading. Existing `CMAKE_ARGS` and DLSS version/hash overrides
remain available. On Windows, `CUDA_PATH` or `CUDA_BIN_DIR` helps locate CUDA DLLs.

## Advanced native interface

```python
from warp_optix import native
print(native.dlss_support_available())
print(native.dlss_rr_available())
```

`native` exposes the bundled binding API and follows its OptiX version. Native
objects use explicit destruction; `Context.native_handle` and
`Pipeline.native_handle` are borrowed and must not be destroyed by callers.
Mixing standalone PyOptiX objects with these bindings, or loading both native
extensions in one process, is unsupported. There is no production `optix` alias.

The distribution version is 0.2.0; the binding version remains 9.1.0.

## Tests and upstream maintenance

Install feature extras, pytest, and the appropriate CuPy wheel. Run the suites
in separate processes:

```powershell
python -m pytest warp_optix/tests
python -m pytest test
```

The second command runs upstream binding tests with a test-only import adapter.
The backend derives from [NVIDIA/otk-pyoptix](https://github.com/NVIDIA/otk-pyoptix).
Its source paths remain stable for upstream merges; see
[maintenance instructions](docs/upstream.md). Binding sources retain their BSD
license; Warp integration sources use Apache 2.0. OptiX and DLSS retain their
NVIDIA license terms. License texts accompany the distribution.

## CUDA graph limitation

OptiX launch capture is currently unsupported. Tests on the RTX 3080 with
driver 610.74 reproduced illegal memory access during graph replay and invalid
resource events during context destruction. `Pipeline.launch` rejects an
active capture before submitting native work. Capture ordinary Warp work in
graphs and submit OptiX launches between those graphs on the same stream.

The renderer retains its `enable_cuda_graphs` argument for caller compatibility,
but requests for OptiX capture use direct asynchronous launches with a logged
explanation. DLSS and ordinary stream ordering remain supported.
