# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
    decode_image,
    decode_texel,
    pack_latents,
    upload_asset,
)


def _asset():
    latent0 = np.zeros((4, 4, 8), dtype=np.float32)
    latent1 = np.zeros((2, 2, 8), dtype=np.float32)
    return NeuralTextureAsset(
        width=8,
        height=8,
        channels=(TextureChannel("color", 0, 3),),
        latent_features=8,
        latent_mips=(pack_latents(latent0), pack_latents(latent1)),
        layers=(
            NetworkLayer(np.eye(32, dtype=np.float16), np.zeros(32, dtype=np.float16)),
            NetworkLayer(np.eye(32, dtype=np.float16), np.zeros(32, dtype=np.float16)),
            NetworkLayer(
                np.zeros((16, 32), dtype=np.float16),
                np.arange(16, dtype=np.float16),
                "none",
            ),
        ),
    )


def _context():
    try:
        optix = woptix.require_optix()
        wp.init()
    except Exception as error:
        pytest.skip(f"OptiX/Warp unavailable: {error}")
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    device = wp.get_device("cuda:0")
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
    return optix, context


def test_reference_decoder_is_deterministic():
    asset = _asset()
    expected = np.arange(3, dtype=np.float32)
    np.testing.assert_array_equal(decode_texel(asset, 2, 3), expected)
    decoded = decode_image(asset, batch_size=7)
    np.testing.assert_array_equal(decoded[3, 2], expected)


def test_upload_converts_all_layers_to_compact_fp8():
    optix, context = _context()
    conversion_stream = wp.Stream("cuda:0")
    runtime = upload_asset(
        _asset(), context, optix, stream=conversion_stream.cuda_stream
    )
    wp.synchronize_device(runtime.device)

    assert runtime.matrix_element_type == "fp8_e4m3"
    assert len(runtime.weight_offsets) == 3
    assert all(offset % 64 == 0 for offset in runtime.weight_offsets)
    assert all(offset % 16 == 0 for offset in runtime.bias_offsets)
    assert runtime.latents.size == 4 * 4 * 2 + 2 * 2 * 2
    assert runtime.matrices.size == max(
        offset + size
        for offset, size in zip(runtime.weight_offsets, runtime.matrix_sizes)
    )
    assert not hasattr(runtime, "conversion_source")


def test_upload_rejects_unknown_weight_type():
    optix, context = _context()
    with pytest.raises(ValueError, match="matrix_element_type"):
        upload_asset(_asset(), context, optix, matrix_element_type="fp4")
