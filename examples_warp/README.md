# Warp + OptiX examples

These examples were moved from `warp/examples/core/pyoptix/` into this directory.

Prerequisites:

```bash
pip install -e ../optix/
pip install -e ../warp_optix/
```

Then run individual examples, e.g.

```bash
python example_warp_optix_kernels.py
python example_warp_optix_curves.py
python example_warp_optix_motion_blur.py
python example_warp_optix_mixed_geometry.py
python example_warp_optix_tiny_raytracer.py
python example_warp_optix_basic_pathtracing.py
```

The basic path-tracing example automatically downloads its default A Beautiful
Game scene into the platform cache on first use. On Linux this defaults to
`~/.cache/warp_optix/assets/`. Pass `--scene-gltf PATH` to use a local `.gltf`
or `.glb` scene instead.
