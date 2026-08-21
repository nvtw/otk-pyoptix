# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import warp as wp

from warp_optix import GLLineOverlay
from warp_optix._runtime.gl_interop import OptixGLInteropViewer


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": 0}, "capacity"),
        ({"capacity": 1, "line_width": 0.0}, "line_width"),
        ({"capacity": 1, "alpha": 1.1}, "alpha"),
    ],
)
def test_gl_line_overlay_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GLLineOverlay(None, **kwargs)


def _uninitialized_cpu_overlay(capacity=2):
    overlay = GLLineOverlay.__new__(GLLineOverlay)
    overlay.capacity = capacity
    overlay.device = wp.get_device("cpu")
    overlay.active_count = -1
    return overlay


def test_zero_count_device_update_changes_active_count_without_mapping_gl():
    overlay = _uninitialized_cpu_overlay()
    starts = wp.empty(0, dtype=wp.vec3, device="cpu")
    ends = wp.empty(0, dtype=wp.vec3, device="cpu")

    overlay.update_device(starts, ends)

    assert overlay.active_count == 0


def test_device_update_validates_count_and_array_lengths_before_mapping_gl():
    overlay = _uninitialized_cpu_overlay(capacity=2)
    starts = wp.empty(2, dtype=wp.vec3, device="cpu")
    short_ends = wp.empty(1, dtype=wp.vec3, device="cpu")

    with pytest.raises(ValueError, match="same number"):
        overlay.update_device(starts, short_ends)
    with pytest.raises(ValueError, match="capacity"):
        overlay.update_device(starts, starts, count=3)
    with pytest.raises(ValueError, match="ends array"):
        overlay.update_device(starts, short_ends, count=2)


def test_viewer_overlay_callback_can_be_replaced_or_disabled():
    viewer = OptixGLInteropViewer.__new__(OptixGLInteropViewer)

    def callback():
        pass

    viewer.set_draw_overlay(callback)
    assert viewer._on_draw_overlay is callback
    viewer.set_draw_overlay(None)
    assert viewer._on_draw_overlay is None
