# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare 4/8/16-output OptiX decoders without changing the public asset format.

Usage:
    PYTHONPATH=warp_optix:. python benchmarks/benchmark_neural_texture_outputs.py image.wnt

The final matrix is truncated, preserving trained outputs exactly. Each case
gets an independent OptiX module to avoid cross-entry-point compiler outlining.
All cases in a comparison write the same channels and number of bytes. Upload,
compilation and correctness checks are outside the GPU-event timing interval.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import warp as wp
import warp_optix as wo

from warp_optix.neural_texture import load_asset, upload_asset
from warp_optix.neural_texture.device import (
    NeuralTextureView,
    Vec32h,
    _hgelu32,
    neural_texture_input_8,
    neural_texture_sample_texel,
)


@wp.struct
class Params:
    texture: NeuralTextureView
    output: wp.array2d(dtype=wp.float32)
    random_access: int


def make_kernels(width: int, consumed: int, production: bool):
    vector_type = wp.types.vector(length=width, dtype=wp.float16)

    @wp.func
    def infer(texture: NeuralTextureView, inputs: Vec32h):
        h0 = Vec32h()
        h1 = Vec32h()
        result = vector_type()
        wp.optix_coop_vec_matmul_bias_fp8_e4m3(
            inputs,
            texture.matrices,
            texture.weight_offsets[0],
            texture.biases,
            texture.bias_offsets[0],
            wp.uint32(0),
            h0,
        )
        h0 = _hgelu32(h0)
        wp.optix_coop_vec_matmul_bias_fp8_e4m3(
            h0,
            texture.matrices,
            texture.weight_offsets[1],
            texture.biases,
            texture.bias_offsets[1],
            wp.uint32(0),
            h1,
        )
        h1 = _hgelu32(h1)
        wp.optix_coop_vec_matmul_bias_fp8_e4m3(
            h1,
            texture.matrices,
            texture.weight_offsets[2],
            texture.biases,
            texture.bias_offsets[2],
            wp.uint32(0),
            result,
        )
        return result

    @wo.optix_kernel(wo.OptixKernelType.RAYGEN, module="unique")
    def raygen(p: Params):
        idx = wp.optix_get_launch_index()
        x = int(idx[0])
        y = int(idx[1])
        dest = y * p.texture.width + x
        if p.random_access != 0:
            state = wp.rand_init(123, dest)
            x = wp.randi(state, 0, p.texture.width)
            y = wp.randi(state, 0, p.texture.height)
        if wp.static(production):
            values = neural_texture_sample_texel(p.texture, x, y)
        else:
            inputs = neural_texture_input_8(
                p.texture.latents,
                p.texture.mip_offsets,
                p.texture.mip_widths,
                p.texture.mip_heights,
                x,
                y,
                p.texture.width,
                p.texture.height,
                0,
                p.texture.latent_level_count,
            )
            values = infer(p.texture, inputs)
        for channel in range(wp.static(consumed)):
            p.output[dest, channel] = wp.float32(values[channel])

    @wo.optix_kernel(wo.OptixKernelType.MISS, module=raygen.module)
    def miss(p: Params):
        _ = p.random_access

    return raygen, miss


def benchmark(args, optix, context, asset, device):
    stream = wp.get_stream(device)
    results = {}
    for consumed in (3, 7):
        output = wp.empty(
            (asset.width * asset.height, consumed), dtype=wp.float32, device=device
        )
        cases = []
        widths = (4, 8, 16) if consumed == 3 else (8, 16)
        for width, production in [*((n, False) for n in widths), (16, True)]:
            label = "production16" if production else str(width)
            kernel, miss = make_kernels(width, consumed, production)
            ptx = wo.compile_warp_module_to_ptx(
                kernel.module,
                "",
                f"outputs_{consumed}_{label}",
                __file__,
                device=device,
            )
            # Experimental upload-only view: do not relax the public format's
            # fixed-16 validation merely to measure hypothetical alternatives.
            last = asset.layers[-1]
            layers = (
                *asset.layers[:-1],
                SimpleNamespace(
                    weights=last.weights[:width].copy(), bias=last.bias[:width].copy()
                ),
            )
            experimental = SimpleNamespace(**{**vars(asset), "layers": layers})
            runtime = upload_asset(experimental, context, optix, device=device)
            pipeline, sbt, keepalive = wo.create_pipeline_and_sbt(
                optix,
                context,
                ptx,
                kernel,
                miss,
                None,
                num_payload_values=0,
                num_attribute_values=0,
                device=device,
            )
            params = Params()
            params.texture = runtime.device_view()
            params.output = output
            buffer = wo.create_launch_params_buffer(Params, device)
            cases.append((label, runtime, pipeline, sbt, keepalive, params, buffer))

        def launch(case):
            wo.launch(
                optix,
                case[2],
                case[3],
                asset.width,
                asset.height,
                case[6],
                stream.cuda_stream,
            )

        for random_access in (0, 1):
            baseline = None
            max_error = 0.0
            for case in cases:
                case[5].random_access = random_access
                wo.write_launch_params(case[6], case[5])
                launch(case)
                actual = output.numpy()
                if baseline is None:
                    baseline = actual
                else:
                    max_error = max(max_error, float(np.max(np.abs(actual - baseline))))
                    np.testing.assert_allclose(actual, baseline, atol=0.002, rtol=0.002)
            # Warm all cases before collecting interleaved, randomized rounds.
            for _ in range(100):
                for case in cases:
                    launch(case)
            wp.synchronize_device(device)
            samples = {case[0]: [] for case in cases}
            for round_index in range(args.rounds):
                for index in np.random.default_rng(round_index).permutation(len(cases)):
                    case = cases[index]
                    start = wp.Event(device, enable_timing=True)
                    end = wp.Event(device, enable_timing=True)
                    stream.record_event(start)
                    for _ in range(args.iterations):
                        launch(case)
                    stream.record_event(end)
                    samples[case[0]].append(
                        wp.get_event_elapsed_time(start, end) / args.iterations
                    )
            key = f"{consumed}ch_" + ("random" if random_access else "coherent")
            results[key] = {
                "max_absolute_error": max_error,
                "variants": {
                    case[0]: {
                        "median_ms": float(np.median(samples[case[0]])),
                        "samples_ms": samples[case[0]],
                        "matrix_bytes": case[1].matrices.size,
                        "last_matrix_bytes": case[1].matrix_sizes[-1],
                        "bias_bytes": case[1].biases.size,
                    }
                    for case in cases
                },
            }
            print(
                key,
                {
                    label: round(float(np.median(times)), 4)
                    for label, times in samples.items()
                },
                flush=True,
            )
        # Context owns API objects; release these completed cases before the next
        # consumer count instead of retaining their compilation state indefinitely.
        for case in cases:
            case[2].destroy()
            for group in case[4]["program_groups"]:
                group.destroy()
            case[4]["module"].destroy()
        del cases
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()
    if args.iterations <= 0 or args.rounds <= 0:
        parser.error("iterations and rounds must be positive")
    device = "cuda:0"
    wp.init()
    with wp.ScopedDevice(device):
        optix = wo.require_optix()
        cuda = wp.get_device(device).context
        handle = cuda.value if hasattr(cuda, "value") else int(cuda)
        context, callback = wo.create_context(optix, handle, log_level=1)
        try:
            asset = load_asset(args.asset)
            results = benchmark(args, optix, context, asset, device)
            print(
                json.dumps(
                    {
                        "gpu": wp.get_device(device).name,
                        "asset": args.asset,
                        "resolution": [asset.width, asset.height],
                        "source_channels": asset.channel_count,
                        "iterations": args.iterations,
                        "rounds": args.rounds,
                        "results": results,
                    },
                    indent=2,
                )
            )
        finally:
            wp.synchronize_device(device)
            context.destroy()


if __name__ == "__main__":
    main()
