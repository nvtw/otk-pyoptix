# warp-optix

Warp ray tracing with PyOptiX 9.1 and DLSS Ray Reconstruction.
Write ray-generation, hit, miss, and callable programs as Warp kernels, or use
the included path tracer. Applications import `warp_optix`. Installation brings
in `pyoptix==9.1.0` for the standard OptiX bindings and builds a small extension
for DLSS, cooperative vectors, and missing binding fields and fixes.

## Installation

Build from the repository root with Python 3.10+, a CUDA Toolkit and a C++17
compiler. GPU execution requires a compatible NVIDIA driver. OptiX headers and
DLSS SDK 310.9.1 are fetched automatically.

Runtime shader compilation also requires a separately installed OptiX SDK
9.1.0. Set `OPTIX_SDK_INCLUDE_DIR` to that SDK's `include` directory (containing
`optix.h` and `optix_device.h`). OptiX headers are not included in this wheel.

```powershell
pip install .
# Editable development with viewer dependencies:
pip install -e ".[pathtracing,ui,usd]"
```

This builds a platform wheel containing the Python package and supplemental
extension. PyOptiX is a separate dependency, installed automatically. Extras
are `pathtracing`, `ui`, `usd`, `recording`, and `training`.
The nested Python package is no longer an independent build entry point.

For a repository-local uv environment:

```bash
uv sync --python 3.12 --no-default-groups --extra pathtracing --extra ui
uv run --no-sync python examples_warp/example_warp_optix_pathtraced_hairball.py
```

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
That runtime is covered by the NVIDIA RTX SDKs license in `DLSS_LICENSE.txt`,
identified in package metadata as `LicenseRef-NVIDIA-RTX-SDK`. It is not
licensed under Apache 2.0. Public distribution remains subject to the release
review and written SDK-owner approval.
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

`native` exposes PyOptiX's API plus the supplemental bindings. Native
objects use explicit destruction; `Context.native_handle` and
`Pipeline.native_handle` are borrowed and must not be destroyed by callers.
Standard objects are the dependency's actual `optix` types; both packages can
be imported in either order. The supplement adds methods and fields to those
types and repairs module-metadata and continuation-callable handling. Keep
objects on their owning OptiX/CUDA context and obey their lifetimes. No second
copy of the standard bindings or replacement `optix` module is installed.

PyOptiX 9.1.0 has no public native-handle API. The extension therefore uses a
small, version-pinned pybind11 bridge and checks shared type identities and
sizes before accessing them. Builds reject OptiX headers that differ from the
PyOptiX dependency pin; imports reject mismatched compiled SDK versions before
attaching bindings. Both extensions must use a compatible pybind11
major version and C++ ABI. If the import reports an ABI mismatch, rebuild both
packages with pybind11 3.x. See [dependency maintenance](docs/pyoptix-dependency.md).

The distribution version is 0.2.0; the binding version remains 9.1.0.

## Tests and dependency maintenance

Install feature extras and pytest, then run the test suite:

```powershell
python -m pytest warp_optix/tests
```

The suite includes regression coverage for the dependency and supplemental
bindings. The dependency is maintained in
[NVIDIA/otk-pyoptix](https://github.com/NVIDIA/otk-pyoptix); this project no longer
builds a copy of its complete native backend. See
[maintenance instructions](docs/pyoptix-dependency.md). Extracted binding additions
retain their BSD license; Warp integration sources use Apache 2.0. OptiX and DLSS retain their
NVIDIA license terms. License texts accompany the distribution.

Runnable samples are in `examples_warp/`. The inherited PyOptiX examples,
upstream test tree, and legacy requirements file have been removed.

## Licenses and contributions

See [third-party notices](THIRD_PARTY_NOTICES.txt) for component licenses and
source provenance. `src/extension.cpp`, `src/pyoptix_compat.h`, and the inherited
`CMakeLists.txt` retain BSD-3-Clause; NVIDIA-authored Warp integration uses
Apache-2.0. The aggregate package metadata also identifies the DLSS SDK terms.

This project is currently not accepting contributions.
We plan to accept outside contributions in the future. See
[CONTRIBUTING.md](CONTRIBUTING.md) for updates.

Viewer recording invokes a user-installed FFmpeg executable. FFmpeg is not
bundled; its applicable license depends on the user's FFmpeg build.

## CUDA graph limitation

OptiX launch capture is currently unsupported. Tests on the RTX 3080 with
driver 610.74 reproduced illegal memory access during graph replay and invalid
resource events during context destruction. `Pipeline.launch` rejects an
active capture before submitting native work. Capture ordinary Warp work in
graphs and submit OptiX launches between those graphs on the same stream.

The renderer retains its `enable_cuda_graphs` argument for caller compatibility,
but requests for OptiX capture use direct asynchronous launches with a logged
explanation. DLSS and ordinary stream ordering remain supported.
