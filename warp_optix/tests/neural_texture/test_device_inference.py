# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

import warp as wp  # noqa: E402
import warp_optix as woptix  # noqa: E402
from warp_optix.neural_texture import (  # noqa: E402
    NetworkLayer,
    NeuralTextureAsset,
    TextureChannel,
    pack_latents,
    decode_texel,
    upload_asset,
)
from warp_optix.neural_texture.device import (  # noqa: E402
    NeuralTextureView,
    neural_texture_sample,
    neural_texture_sample_texel,
)


@wp.struct
class NeuralParams:
    output: wp.array(dtype=wp.float32)
    texture: NeuralTextureView


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def neural_raygen(params: NeuralParams):
    value = neural_texture_sample_texel(params.texture, 2, 3)
    uv_value = neural_texture_sample(params.texture, wp.vec2(2.5 / 8.0, 3.5 / 8.0))
    for channel in range(3):
        params.output[channel] = wp.float32(value[channel])
        params.output[channel + 3] = wp.float32(uv_value[channel])


@woptix.optix_kernel(woptix.OptixKernelType.MISS)
def neural_miss(params: NeuralParams):
    _ = params.output


def _asset():
    grid0 = ((np.arange(4 * 4 * 8) % 16) * (2.0 / 15.0) - 1.0).reshape(4, 4, 8)
    grid1 = ((np.arange(2 * 2 * 8) * 3 % 16) * (2.0 / 15.0) - 1.0).reshape(2, 2, 8)
    output_weights = np.zeros((16, 32), dtype=np.float16)
    output_weights[:3, :3] = np.eye(3, dtype=np.float16)
    return NeuralTextureAsset(
        width=8,
        height=8,
        channels=(TextureChannel("color", 0, 3),),
        latent_features=8,
        latent_mips=(pack_latents(grid0), pack_latents(grid1)),
        layers=(
            NetworkLayer(np.eye(32, dtype=np.float16), np.zeros(32, dtype=np.float16)),
            NetworkLayer(np.eye(32, dtype=np.float16), np.zeros(32, dtype=np.float16)),
            NetworkLayer(output_weights, np.zeros(16, dtype=np.float16), "none"),
        ),
    )


def test_neural_texture_device_inference(tmp_path, monkeypatch):
    try:
        optix = woptix.require_optix()
        wp.init()
    except Exception as error:
        pytest.skip(f"OptiX/Warp unavailable: {error}")
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    device_name = "cuda:0"
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp_cache"))
    with wp.ScopedDevice(device_name):
        device = wp.get_device(device_name)
        cuda_context = (
            device.context.value
            if hasattr(device.context, "value")
            else int(device.context)
        )
        context, _ = woptix.create_context(optix, int(cuda_context), log_level=1)
        try:
            flags = context.getProperty(optix.DEVICE_PROPERTY_COOP_VEC)
        except Exception as error:
            pytest.skip(f"cooperative vectors unavailable: {error}")
        if not flags:
            pytest.skip("cooperative vectors unsupported")

        asset = _asset()
        runtime = upload_asset(asset, context, optix, device=device_name)
        wp.synchronize_device(device_name)
        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__),
            "",
            "test_neural_texture",
            __file__,
            device=device_name,
        )
        pipeline, sbt, resources = woptix.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            neural_raygen,
            neural_miss,
            None,
            num_payload_values=0,
            num_attribute_values=0,
            device=device_name,
        )
        output = wp.zeros(6, dtype=wp.float32, device=device_name)
        params = NeuralParams()
        params.output = output
        params.texture = runtime.device_view()
        params_buffer = woptix.create_launch_params_buffer(NeuralParams, device_name)
        woptix.write_launch_params(params_buffer, params)
        woptix.launch(optix, pipeline, sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        expected = decode_texel(asset, 2, 3)
        np.testing.assert_allclose(output.numpy()[:3], expected, atol=3e-3)
        np.testing.assert_allclose(output.numpy()[3:], expected, atol=3e-3)
        assert resources

        single_level = replace(asset, latent_mips=asset.latent_mips[:1])
        single_runtime = upload_asset(single_level, context, optix, device=device_name)
        output.zero_()
        params.texture = single_runtime.device_view()
        woptix.write_launch_params(params_buffer, params)
        woptix.launch(optix, pipeline, sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        expected = decode_texel(single_level, 2, 3)
        np.testing.assert_allclose(output.numpy()[:3], expected, atol=3e-3)
        np.testing.assert_allclose(output.numpy()[3:], expected, atol=3e-3)

        if importlib.util.find_spec("warp_nn") is not None:
            from warp_optix.neural_texture import compress_texture

            x = np.arange(8, dtype=np.float32)[None, :] / 7.0
            y = np.arange(8, dtype=np.float32)[:, None] / 7.0
            image = np.empty((8, 8, 3), dtype=np.float32)
            image[..., 0] = x
            image[..., 1] = y
            image[..., 2] = 0.5 + 0.25 * np.sin(2.0 * np.pi * (x + y))
            trained = compress_texture(
                image,
                steps=2,
                refinement_steps=1,
                batch_size=32,
                device=device_name,
                seed=17,
            ).asset
            trained_runtime = upload_asset(trained, context, optix, device=device_name)
            output.zero_()
            params.texture = trained_runtime.device_view()
            woptix.write_launch_params(params_buffer, params)
            woptix.launch(optix, pipeline, sbt, 1, 1, params_buffer)
            wp.synchronize_device(device_name)
            np.testing.assert_allclose(
                output.numpy()[:3], decode_texel(trained, 2, 3), atol=5e-3
            )
