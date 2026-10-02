# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

from warp_optix.pathtracing import tonemap
from warp_optix.pathtracing.tonemap import Tonemapper, _adapt_auto_exposure


def test_tonemap_preserves_pixels_and_orientation_on_non_square_image():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    # Distinct channels and corners catch transposition and missing pixels.
    image = np.arange(9 * 17 * 4, dtype=np.float32).reshape(9, 17, 4) / 1000.0
    mapper = Tonemapper(17, 9)
    mapper.is_active = False
    mapper.exposure = 2.0
    mapper.process(wp.array(image, dtype=wp.vec4, device="cuda"))
    expected = np.flipud(image).copy()
    expected[..., :3] *= 2.0
    expected[..., 3] = 1.0
    np.testing.assert_array_equal(mapper.get_numpy(), expected)


def test_debug_normals_preserve_orientation_when_upscaled():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    image = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4) / 100.0
    normal = wp.array(image, dtype=wp.vec4, device="cuda")
    scalar = wp.zeros((3, 5), dtype=wp.float32, device="cuda")
    motion = wp.zeros((3, 5), dtype=wp.vec2, device="cuda")
    mapper = Tonemapper(17, 9)
    mapper.process_debug(
        tonemap.OUTPUT_NORMAL,
        normal,
        scalar,
        motion,
        normal,
        normal,
        normal,
        scalar,
        5,
        3,
    )
    sx = np.arange(17) * 5 // 17
    sy = 2 - np.arange(9) * 3 // 9
    expected = image[sy[:, None], sx[None, :]].copy()
    expected[..., :3] = expected[..., :3] * 0.5 + 0.5
    expected[..., 3] = 1.0
    np.testing.assert_array_equal(mapper.get_numpy(), expected)


def test_auto_exposure_defaults_to_brightening_only(monkeypatch):
    """Keep the renderer-wide automatic exposure floor at baseline."""
    monkeypatch.setattr(tonemap.wp, "zeros", lambda *args, **kwargs: object())

    mapper = Tonemapper(16, 16)

    assert mapper.auto_exposure_min_ev == 0.0


def test_auto_exposure_initializes_from_first_meter_result():
    """Initialize exposure immediately from the first valid luminance meter."""
    measured_luminance = 0.01
    target_luminance = 0.18
    stats = wp.array([np.log2(measured_luminance), 1.0], dtype=wp.float32, device="cpu")
    exposure_ev = wp.zeros(1, dtype=wp.float32, device="cpu")

    wp.launch(
        _adapt_auto_exposure,
        dim=1,
        inputs=[
            stats,
            exposure_ev,
            target_luminance,
            -6.0,
            6.0,
            0.6,
            1.2,
            1.0 / 60.0,
            1,
        ],
        device="cpu",
    )

    np.testing.assert_allclose(exposure_ev.numpy(), [np.log2(18.0)], rtol=1.0e-6)


def test_auto_exposure_can_be_restricted_to_brightening():
    """Keep exposure at baseline when metered luminance exceeds the target."""
    measured_luminance = 2.0
    target_luminance = 0.18
    stats = wp.array([np.log2(measured_luminance), 1.0], dtype=wp.float32, device="cpu")
    exposure_ev = wp.zeros(1, dtype=wp.float32, device="cpu")

    wp.launch(
        _adapt_auto_exposure,
        dim=1,
        inputs=[
            stats,
            exposure_ev,
            target_luminance,
            0.0,
            6.0,
            0.6,
            1.2,
            1.0 / 60.0,
            1,
        ],
        device="cpu",
    )

    np.testing.assert_array_equal(exposure_ev.numpy(), [0.0])
