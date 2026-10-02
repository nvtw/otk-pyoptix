# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from warp_optix.pathtracing import (
    DenoiserInputs,
    OptixDenoiser,
    PathTracerAPI,
    PathTracingRenderer,
)
from warp_optix.pathtracing import pathtracing_viewer as renderer_module


def _renderer(monkeypatch, mode, *, dlss_available=False):
    renderer = PathTracingRenderer.__new__(PathTracingRenderer)
    renderer.width, renderer.height = 64, 48
    renderer._render_width, renderer._render_height = 64, 48
    renderer.denoiser = mode
    renderer.optix_upscale = False
    renderer._denoiser_explicit = True
    renderer._ctx = object()
    renderer._optix = object()
    renderer._render_stream = object()
    renderer._optix_denoiser = None
    renderer._dlss_enabled = False
    renderer._dlss_init_error = None
    renderer._sync_prev_camera_matrices_to_current = lambda: None
    renderer._destroy_dlss_rr = lambda **kwargs: setattr(
        renderer, "_dlss_enabled", False
    )
    renderer._set_render_resolution = lambda w, h: (
        setattr(renderer, "_render_width", w),
        setattr(renderer, "_render_height", h),
    )

    def init_dlss():
        renderer._dlss_enabled = dlss_available
        renderer._dlss_init_error = None if dlss_available else "DLSS unavailable"

    renderer._init_dlss_rr = init_dlss
    monkeypatch.setattr(renderer_module.wp, "synchronize_stream", lambda stream: None)
    return renderer


def test_auto_prefers_dlss_and_preserves_diagnostic_on_optix_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(
        renderer_module,
        "OptixDenoiser",
        lambda *args, **kwargs: (
            calls.append(kwargs) or SimpleNamespace(close=lambda: None)
        ),
    )
    renderer = _renderer(monkeypatch, "auto", dlss_available=True)
    renderer._init_denoiser()
    assert renderer.active_denoiser == "dlss"
    assert not calls
    renderer._init_dlss_rr = lambda: setattr(
        renderer, "_dlss_init_error", "DLSS unavailable"
    )
    renderer._init_denoiser()
    assert renderer.active_denoiser == "optix"
    assert renderer.denoiser_error == "DLSS unavailable"
    assert len(calls) == 1


def test_auto_falls_back_to_raw_but_explicit_backend_fails(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("OptiX unavailable")

    monkeypatch.setattr(renderer_module, "OptixDenoiser", unavailable)
    renderer = _renderer(monkeypatch, "auto")
    renderer._init_denoiser()
    assert renderer.active_denoiser == "none"
    assert "DLSS unavailable" in renderer.denoiser_error
    assert "OptiX unavailable" in renderer.denoiser_error
    renderer.denoiser = "optix"
    with pytest.raises(RuntimeError, match="OptiX unavailable"):
        renderer._init_denoiser()
    renderer.denoiser = "dlss"
    with pytest.raises(RuntimeError, match="DLSS unavailable"):
        renderer._init_denoiser()
    renderer._denoiser_explicit = False
    renderer._init_denoiser()  # Legacy enable_dlss_rr still permits raw fallback.
    assert renderer.active_denoiser == "none"


def test_failed_switch_restores_previous_backend(monkeypatch):
    renderer = _renderer(monkeypatch, "none")
    monkeypatch.setattr(
        renderer_module,
        "OptixDenoiser",
        lambda *args, **kwargs: SimpleNamespace(close=lambda: None),
    )
    renderer.set_denoiser("optix", optix_upscale=True)
    assert renderer.active_denoiser == "optix"
    assert (renderer._render_width, renderer._render_height) == (32, 24)
    with pytest.raises(RuntimeError, match="DLSS unavailable"):
        renderer.set_denoiser("dlss")
    assert renderer.denoiser == renderer.active_denoiser == "optix"
    assert renderer.optix_upscale is True
    assert renderer._dlss_reset_history is True
    assert renderer.sample_index == 0
    assert renderer._optix_launch_graph is None
    renderer.width = 63
    renderer._init_denoiser()
    assert renderer._render_width == 32


def _gpu_context():
    from warp_optix._runtime.runtime import create_optix_context

    import optix

    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    if not hasattr(optix, "DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X"):
        pytest.skip("Rebuild PyOptiX with temporal/upscaling denoiser bindings")
    return create_optix_context(
        optix, wp.get_device("cuda").context, enable_validation=True
    )


@pytest.mark.parametrize("upscale", [False, True])
def test_optix_gpu_denoises_and_resets_history(upscale):
    context, _logger = _gpu_context()
    rng = np.random.default_rng(42)
    noisy = np.ones((48, 64, 4), dtype=np.float32)
    noisy[..., :3] += rng.normal(0.0, 0.15, (48, 64, 3)).astype(np.float32)
    normals = np.zeros_like(noisy)
    normals[..., 2] = 1.0
    inputs = DenoiserInputs(
        color=wp.array(noisy, dtype=wp.vec4, device="cuda"),
        normal=wp.array(normals, dtype=wp.vec4, device="cuda"),
        albedo=wp.ones((48, 64), dtype=wp.vec4, device="cuda"),
        motion=wp.zeros((48, 64), dtype=wp.vec2, device="cuda"),
    )
    denoiser = OptixDenoiser(context, 64, 48, upscale=upscale)
    try:
        first = denoiser.apply(inputs).numpy()
        scale = 2 if upscale else 1
        assert first.shape == (48 * scale, 64 * scale, 4)
        assert np.isfinite(first).all()
        assert (
            np.mean((first[..., :3] - 1.0) ** 2)
            < np.mean((noisy[..., :3] - 1.0) ** 2) * 0.5
        )
        denoiser.apply(inputs).numpy()
        reset = denoiser.apply(inputs, reset=True).numpy()
        np.testing.assert_allclose(reset, first, rtol=1e-5, atol=1e-5)
    finally:
        denoiser.close()
        context.destroy()


def test_renderer_gpu_switch_resize_and_history():
    context, _logger = _gpu_context()
    context.destroy()
    api = PathTracerAPI(
        width=64, height=48, denoiser="optix", optix_upscale=True, enable_set=False
    )
    try:
        assert api.initialize()
        api.scene.create_cornell_box()
        api.build_scene()
        scene = api.scene
        for _ in range(2):
            api.render_frame()
        assert api.get_frame().shape == (48, 64, 4)
        api.reset_temporal_history()
        api.render_frame()
        api.set_denoiser("none")
        api.render_frame()
        assert api.active_denoiser == "none"
        api.set_denoiser("optix", optix_upscale=False)
        api.render_frame()
        assert api.scene is scene
        assert api.active_denoiser == "optix"
        api.resize(80, 60)
        api.render_frame()
        assert api.get_frame().shape == (60, 80, 4)
        assert np.isfinite(api.get_frame()).all()
        api.set_denoiser("optix", optix_upscale=True)
        api.resize(81, 61)
        api.render_frame()
        assert api.get_frame().shape == (61, 81, 4)
    finally:
        api.close()


def test_renderer_gpu_dlss_quality_survives_backend_switch():
    import optix

    context, _logger = _gpu_context()
    context.destroy()
    if not optix.dlss_rr_available():
        pytest.skip("DLSS RR unavailable")
    api = PathTracerAPI(
        width=128,
        height=96,
        denoiser="dlss",
        dlss_quality="performance",
        enable_set=False,
    )
    try:
        assert api.initialize()
        api.scene.create_cornell_box()
        api.build_scene()
        api.render_frame()
        assert api.active_denoiser == "dlss"
        performance_size = (api.viewer._render_width, api.viewer._render_height)
        api.set_denoiser("optix", optix_upscale=True)
        api.render_frame()
        assert api.active_denoiser == "optix"
        api.set_dlss_quality("native")
        assert api.active_denoiser == "optix"
        api.set_denoiser("dlss")
        api.render_frame()
        assert api.dlss_quality == "native"
        assert (api.viewer._render_width, api.viewer._render_height) == (128, 96)
        assert performance_size != (128, 96)
        assert api.get_frame().shape == (96, 128, 4)
        assert np.isfinite(api.get_frame()).all()
    finally:
        api.close()


@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_optix_gpu_frames_and_camera_advance(cuda_graphs):
    context, _logger = _gpu_context()
    context.destroy()
    api = PathTracerAPI(
        width=96,
        height=64,
        denoiser="optix",
        optix_upscale=True,
        enable_set=False,
        enable_cuda_graphs=cuda_graphs,
    )
    try:
        assert api.initialize()
        api.scene.create_cornell_box()
        api.build_scene()
        frames = []
        for _ in range(5):
            api.render_frame()
            frames.append(api.viewer._color_buffer.numpy().copy())
        assert api.viewer.sample_index == 5
        assert not api.cuda_graph_active
        assert not np.array_equal(frames[-1], frames[-2]), (
            "Ray samples froze after graph capture"
        )
        previous = api.viewer.camera.position.copy()
        api.set_camera_look_at(previous + (1.0, 0.0, 0.0), api.viewer.camera.target)
        api.render_frame()
        moved = api.viewer._color_buffer.numpy()
        motion = api.viewer._motion_buffer.numpy()
        assert not np.array_equal(moved, frames[-1]), "Camera changes did not reach OptiX"
        assert float(np.max(np.abs(motion))) > 0.1, (
            "Camera movement produced no motion vectors"
        )
    finally:
        api.close()
