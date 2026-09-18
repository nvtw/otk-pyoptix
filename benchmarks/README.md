# Neural-texture output-width benchmark

Run from the repository root with an existing 16-output `.wnt` asset:

```bash
PYTHONPATH=warp_optix:. .venv/bin/python benchmarks/benchmark_neural_texture_outputs.py material.wnt
```

This is an inference-only experiment; it needs no warp-nn. It truncates the
trained final matrix to 4 or 8 rows, without retraining or changing the public
asset format. Latents, hidden layers, meaningful output values, launch dimensions
and output traffic remain identical within each comparison. The separate
production16 case calls the actual public texel sampler.

Each variant is compiled in an independent OptiX module. An initial shared-module
experiment produced misleading ~2x slowdowns for narrower variants; separately
compiled variants eliminated that difference. Compiler context matters.

Timing excludes compilation, upload and correctness readback. It uses CUDA events,
100 warmup launches per case, then 15 randomized rounds of 200 launches per case.
Both coherent and pseudorandom texel access are tested. Each case writes exactly
three or seven float32 channels per pixel; seven channels from an RGB-trained
asset are only a synthetic arithmetic control, not a material-quality test.
The script reports all round timings, converted buffer sizes and output errors.

## Measured decision (2026-09-18)

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 13.1 reported by
Warp, Warp 1.18.0.dev20260824. Inputs were existing 2048x2048 photograph and
albedo/normal/roughness assets. Times below are median milliseconds for all
4,194,304 evaluations, including stores and (in random cases) coordinate RNG:

| Input / consumed channels | Access | 4 outputs | 8 outputs | 16 outputs | Production 16 |
| --- | --- | ---: | ---: | ---: | ---: |
| Photograph / RGB | Coherent | 0.164 | 0.166 | 0.165 | 0.165 |
| Photograph / RGB | Random | 0.231 | 0.232 | 0.234 | 0.233 |
| Material / 7 channels | Coherent | — | 0.199 | 0.196 | 0.196 |
| Material / 7 channels | Random | — | 0.278 | 0.279 | 0.278 |

All compared outputs were bit-identical in these runs. Repeat/control runs also
showed roughly tied performance; small differences do not establish a useful
speedup. This is a ray-generation microbenchmark, not a full renderer, and does
not establish results for other GPUs.

Converted final-matrix storage was 512 bytes for both 4 and 8 outputs, versus
1024 bytes for 16. Including biases, total GPU parameter savings are only
536 bytes (4 outputs) or 528 bytes (8 outputs); latent storage is unchanged.
Portable FP16 parameter payload savings would be 792 or 528 bytes, respectively.

Decision: keep one public 16-output decoder. The measured benefit does not
justify extra asset architectures, training paths and evaluation entry points.
The benchmark is retained so that this can be revisited on other hardware.
