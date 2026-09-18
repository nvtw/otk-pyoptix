# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("warp_nn", reason="offline compression is an optional extra")

sys.path.insert(0, str(Path(__file__).parents[2]))

from warp_optix.neural_texture import (  # noqa: E402
    compress_texture,
    decode_image,
    load_asset,
    save_asset,
)
from warp_optix.neural_texture.training import _quantize_fp8_e4m3  # noqa: E402


def test_compress_texture_exports_quantized_asset(tmp_path):
    image = np.empty((8, 8, 3), dtype=np.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.5
    image[..., 2] = 0.8

    result = compress_texture(
        image,
        steps=2,
        refinement_steps=1,
        learning_rate=2.0e-2,
        device="cuda:0",
        seed=7,
        channel_name="albedo",
    )

    assert len(result.losses) == 3
    assert np.all(np.isfinite(result.losses))
    assert np.isfinite(result.psnr)
    assert result.asset.channels[0].name == "albedo"
    assert result.asset.metadata["compressor"] == "warp-nn"
    assert result.asset.metadata["weight_type"] == "fp8_e4m3"
    assert result.asset.metadata["batch_size"] == 64

    decoded = decode_image(result.asset, batch_size=7)
    expected_psnr = -10.0 * np.log10(np.mean((decoded - image) ** 2))
    np.testing.assert_allclose(result.psnr, expected_psnr, rtol=1.0e-6)

    path = tmp_path / "compressed.wnt"
    save_asset(path, result.asset)
    loaded = load_asset(path)
    assert loaded.storage_bytes == result.asset.storage_bytes


def test_fp8_projection_uses_optix_tie_to_even_rounding():
    values = np.array([-152.0, -136.0, 136.0, 152.0], dtype=np.float32)
    np.testing.assert_array_equal(
        _quantize_fp8_e4m3(values),
        np.array([-160.0, -128.0, 128.0, 160.0], dtype=np.float32),
    )


def test_compress_texture_maps_more_than_four_outputs():
    image = np.zeros((2, 2, 5), dtype=np.float32)
    result = compress_texture(image, steps=0, refinement_steps=0, seed=9)
    assert [
        (c.name, c.first_channel, c.channel_count) for c in result.asset.channels
    ] == [
        ("texture", 0, 4),
        ("texture_1", 4, 1),
    ]


@pytest.mark.parametrize(
    ("image", "kwargs", "message"),
    [
        (np.empty((0, 2, 3), dtype=np.float32), {}, "dimensions"),
        (
            np.zeros((2, 2, 3), dtype=np.float32),
            {"learning_rate": 0.0},
            "learning_rate",
        ),
        (np.zeros((2, 2, 3), dtype=np.float32), {"batch_size": 0}, "batch_size"),
    ],
)
def test_compress_texture_validates_configuration(image, kwargs, message):
    with pytest.raises(ValueError, match=message):
        compress_texture(image, steps=0, refinement_steps=0, **kwargs)
