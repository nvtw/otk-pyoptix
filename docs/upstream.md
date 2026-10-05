# Merging otk-pyoptix changes

Merge the desired upstream commit into develop using the preserved shared
ancestry. Keep `src/main.cpp`, `src/optix`, and the upstream test/examples
trees in their existing locations. Source names retain their upstream origin;
build configuration controls the installed namespace.

Preserve these downstream differences during conflict resolution:

- Root metadata describes warp-optix and its Warp dependency, extras, and
  Apache/BSD license texts. There is one distribution.
- CMake installs native bindings, the loader, headers, and DLSS runtime under
  `warp_optix/_bindings`. The `_optix` initialization name stays unchanged;
  the binding version is independent of the distribution version.
- The loader computes editable header locations from its package name.
  DLL discovery and DLSS feature paths use the private installation directory.
- `test/conftest.py` adapts upstream imports only in a separate pytest process.
  Production code never aliases `optix` or falls back to standalone PyOptiX.

Keep adaptation in the Warp package: `_backend` loads the bundled bindings,
`_api` provides ownership, and `native` exposes the advanced binding API.
Run the Warp and upstream suites separately after merging, build both DLSS
variants, test editable and clean wheel installations, verify header discovery,
and render a DLSS frame. Inspect wheels for accidental top-level optix files.
