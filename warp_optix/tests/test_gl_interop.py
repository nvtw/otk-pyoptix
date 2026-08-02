# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
import time
from types import SimpleNamespace
from unittest import mock

import warp as wp
from warp_optix._runtime.gl_interop import OptixGLInteropViewer


def test_render_stream_completes_before_gl_consumes_pbo():
    events = []

    class _CudaGL:
        def map(self, *, dtype, shape):
            del dtype, shape
            events.append("map")
            return object()

        def unmap(self):
            events.append("unmap")

    viewer = OptixGLInteropViewer.__new__(OptixGLInteropViewer)
    viewer.device = "cuda:0"
    viewer.render_stream = object()
    viewer.cuda_gl = _CudaGL()
    viewer.width = 8
    viewer.height = 4
    viewer.frame_index = 3
    viewer.start_time = time.perf_counter()
    viewer.max_frames = 0
    viewer._render_callback = lambda mapped, frame, elapsed: events.append(
        ("render", mapped, frame, elapsed)
    )
    viewer.window = SimpleNamespace(
        switch_to=lambda: events.append("switch"),
        flip=lambda: events.append("flip"),
    )
    viewer._on_draw = lambda: events.append("draw")

    with (
        mock.patch.object(
            wp, "ScopedDevice", side_effect=lambda _device: nullcontext()
        ),
        mock.patch.object(
            wp,
            "ScopedStream",
            side_effect=lambda _stream, **_kwargs: nullcontext(),
        ),
        mock.patch.object(
            wp, "synchronize_stream", side_effect=lambda _stream: events.append("sync")
        ),
    ):
        viewer._render_frame()

    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "map",
        "render",
        "unmap",
        "sync",
        "switch",
        "draw",
        "flip",
    ]
