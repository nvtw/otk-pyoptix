# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

from warp_optix.pathtracing import (
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_SIZE,
    DEFAULT_VIEWER_WIDTH,
    PathTracerAPI,
    PathTracingRenderer,
    PathTracingViewerBackend,
)


def test_default_viewer_size_is_full_hd():
    assert DEFAULT_VIEWER_WIDTH == 1920
    assert DEFAULT_VIEWER_HEIGHT == 1080
    assert DEFAULT_VIEWER_SIZE == (1920, 1080)


def test_high_level_viewers_share_the_full_hd_default():
    constructors = (
        PathTracerAPI.__init__,
        PathTracingRenderer.__init__,
        PathTracingViewerBackend.__init__,
    )

    for constructor in constructors:
        parameters = inspect.signature(constructor).parameters
        assert parameters["width"].default == DEFAULT_VIEWER_WIDTH
        assert parameters["height"].default == DEFAULT_VIEWER_HEIGHT
        assert parameters["backface_culling"].default is True
