# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone path-tracing viewer with a renderer-neutral scene logging API."""

from __future__ import annotations

import importlib
import logging
import math
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from warp_optix._runtime.gl_interop import OptixGLInteropViewer

from .pathtracer_api import PathTracerAPI

logger = logging.getLogger(__name__)

_RECORDING_STOP = object()
_NVENC_SUPPORT: dict[str, bool] = {}


@wp.kernel(enable_backward=False)
def _pack_display_rgba8(
    src: wp.array2d(dtype=wp.vec4),
    dst: wp.array(dtype=wp.uint32),
    width: int,
    height: int,
):
    x, y = wp.tid()
    if x >= width or y >= height:
        return

    c = src[y, x]
    r = wp.uint32(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    g = wp.uint32(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    b = wp.uint32(wp.clamp(c[2] * 255.0, 0.0, 255.0))
    a = wp.uint32(255)
    dst[y * width + x] = (
        (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r
    )


@wp.kernel(enable_backward=False)
def _pack_recording_rgb8(
    src: wp.array2d(dtype=wp.vec4),
    dst: wp.array(dtype=wp.uint8),
    width: int,
    height: int,
):
    x, y = wp.tid()
    if x >= width or y >= height:
        return

    c = src[y, x]
    offset = (y * width + x) * 3
    dst[offset] = wp.uint8(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    dst[offset + 1] = wp.uint8(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    dst[offset + 2] = wp.uint8(wp.clamp(c[2] * 255.0, 0.0, 255.0))


@dataclass
class _RecordingReadback:
    device_pixels: wp.array
    host_pixels: wp.array
    ready: wp.Event


def _system_videos_dir() -> Path:
    """Return the desktop-configured Videos directory when available."""
    executable = shutil.which("xdg-user-dir")
    if executable is not None:
        try:
            result = subprocess.run(
                [executable, "VIDEOS"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            path = result.stdout.strip()
            if path:
                return Path(path).expanduser()
        except (OSError, subprocess.SubprocessError):
            pass
    return Path.home() / "Videos"


def _ffmpeg_executable() -> str:
    """Prefer system FFmpeg so distro-provided hardware encoders are visible."""
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    except ImportError as error:
        raise RuntimeError(
            "Recording requires FFmpeg; install warp_optix[recording]"
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def _supports_h264_nvenc(executable: str) -> bool:
    """Probe the driver and encoder with a real one-frame encode."""
    cached = _NVENC_SUPPORT.get(executable)
    if cached is not None:
        return cached
    try:
        result = subprocess.run(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=256x256:rate=1",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        supported = result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        supported = False
    _NVENC_SUPPORT[executable] = supported
    return supported


class _FFmpegVideoWriter:
    """Minimal raw-video pipe with hardware encoding when available."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: int,
        bitrate_mbps: int,
        encoder: str,
    ):
        executable = _ffmpeg_executable()
        if encoder == "auto":
            encoder = "h264_nvenc" if _supports_h264_nvenc(executable) else "libx264"
        if encoder not in {"h264_nvenc", "libx264"}:
            raise ValueError(f"Unsupported recording encoder: {encoder!r}")

        encoder_args = (
            ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll"]
            if encoder == "h264_nvenc"
            else ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"]
        )
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-vf",
            "vflip",
            "-an",
            *encoder_args,
            "-b:v",
            f"{bitrate_mbps}M",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self.encoder = encoder
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def append_data(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg recording pipe is closed")
        try:
            self._process.stdin.write(memoryview(frame).cast("B"))
        except (BrokenPipeError, OSError) as error:
            details = self._read_error()
            raise RuntimeError(f"FFmpeg recording failed: {details}") from error

    def _read_error(self) -> str:
        if self._process.stderr is None:
            return "unknown error"
        return self._process.stderr.read().decode("utf-8", errors="replace").strip()

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
            self._process.stdin = None
        returncode = self._process.wait()
        if returncode != 0:
            raise RuntimeError(f"FFmpeg recording failed: {self._read_error()}")


@dataclass
class _InstanceBatch:
    mesh_name: str
    mesh_id: int
    instance_ids: list[int]
    colors: np.ndarray
    materials: np.ndarray
    hidden: bool = False
    active_count: int = 0


def _as_numpy(value: Any, dtype=None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, wp.array):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _broadcast_rows(
    value: np.ndarray | None, count: int, width: int, default: tuple[float, ...]
) -> np.ndarray:
    if value is None:
        return np.tile(np.asarray(default, dtype=np.float32), (count, 1))
    rows = np.asarray(value, dtype=np.float32).reshape(-1, width)
    if len(rows) == count:
        return rows.copy()
    if len(rows) == 1:
        return np.repeat(rows, count, axis=0)
    raise ValueError(f"Expected 1 or {count} rows with width {width}, got {len(rows)}")


class PathTracingViewerBackend:
    """OptiX/DLSS viewer backend independent of any simulation framework.

    The ``log_*`` surface intentionally accepts Warp arrays and mirrors the
    renderer-facing protocol used by simulation viewers. It can be used
    directly or placed before a framework's viewer base in an MRO.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        title: str = "Warp OptiX Path Tracing",
        fps: int = 0,
        device: str = "cuda",
        headless: bool = False,
        paused: bool = False,
        render_when_paused: bool = False,
        num_frames: int | None = None,
        enable_dlss_rr: bool = True,
        enable_set: bool = True,
        dlss_quality: str = "quality",
        samples_per_frame: int = 1,
        max_bounces: int = 4,
        direct_light_samples: int = 1,
        up_axis: str | int = "Y",
        camera_speed: float = 4.0,
        api: PathTracerAPI | None = None,
        vsync: bool = False,
        max_instances: int = 10000,
        enable_imgui: bool = True,
        enable_picking: bool = True,
        picking_factory: Callable | None = None,
        fallback_to_copy: bool = True,
        recording_writer_factory: Callable | None = None,
        default_ior: float = 1.5,
        default_specular: float = 1.0,
        default_clearcoat: float = 0.0,
        default_clearcoat_roughness: float = 0.1,
    ):
        try:
            super().__init__()
        except TypeError:
            # A structurally compatible base may require its own explicit
            # initialization in the small integration wrapper.
            pass

        self.width = int(width)
        self.height = int(height)
        self.device = wp.get_device(device)
        self.fps = max(0, int(fps))
        self.headless = bool(headless)
        self.paused = bool(paused)
        self.render_when_paused = bool(render_when_paused)
        self.num_frames = num_frames
        self.frame_index = 0
        self.time = 0.0
        self._closed = False
        self._initialized = False
        self._scene_dirty = False
        self._transforms_dirty = False
        self._materials_dirty = False
        self._last_wall_time = time.perf_counter()
        self._warned_texture = False
        self._warned_transform_vbo = False
        self.model = getattr(self, "model", None)
        self.picking_enabled = bool(enable_picking)
        self._picking_factory = picking_factory
        self._picking = None
        self._last_state = None
        self._mouse_x = 0.0
        self._mouse_y = 0.0
        self._mouse_buttons: set[int] = set()

        self._max_instances = max(1, int(max_instances))
        self._ui_callbacks: list[tuple[Callable, str]] = []
        self._imgui_enabled = bool(enable_imgui)
        self._imgui = None
        self._imgui_impl = None
        self._debug_buffer_mode = 0
        self._current_fps = 0.0
        self._fps_frame_count = 0
        self._fps_window_start = time.perf_counter()

        self.recording_fps = 60
        self.recording_bitrate_mbps = 20
        self.recording_frame_skip = 1
        self.recording_encoder = "auto"
        self.recording_buffer_count = 8
        self.recording_dropped_frames = 0
        self.recording_output_path: str | None = None
        self._recording_writer = None
        self._recording_writer_factory = recording_writer_factory
        self._recording_frame_index = 0
        self._recording_path: Path | None = None
        self._recording_thread: threading.Thread | None = None
        self._recording_queue: queue.Queue | None = None
        self._recording_free_slots: queue.SimpleQueue | None = None
        self._recording_stream: wp.Stream | None = None
        self._recording_slots: list[_RecordingReadback] = []
        self._recording_error: BaseException | None = None
        self._recording_submitted_frames = 0
        self._recording_width = 0
        self._recording_height = 0

        self._api = api or PathTracerAPI(
            width=self.width,
            height=self.height,
            enable_dlss_rr=enable_dlss_rr,
            enable_set=enable_set,
            dlss_quality=dlss_quality,
            samples_per_frame=samples_per_frame,
            max_bounces=max_bounces,
            direct_light_samples=direct_light_samples,
        )
        self._presenter = None
        if not self.headless:
            self._presenter = OptixGLInteropViewer(
                width=self.width,
                height=self.height,
                device=str(self.device),
                title=title,
                fps=self.fps,
                on_resize=self._on_resize,
                on_draw_overlay=self._draw_imgui,
                vsync=vsync,
                fallback_to_copy=fallback_to_copy,
            )
            self._presenter.window.push_handlers(self)
            if self._imgui_enabled:
                self._init_imgui()

        self._mesh_ids: dict[str, int] = {}
        self._batches: dict[str, _InstanceBatch] = {}
        self._material_ids: dict[tuple[float, ...], int] = {}
        self._device_transform_batches: dict[
            str, tuple[wp.array, wp.array, wp.array]
        ] = {}
        self._device_material_batches: dict[
            str, tuple[wp.array, wp.array, wp.array]
        ] = {}
        self._default_color = (0.8, 0.8, 0.8)
        self._default_material = (0.5, 0.0, 0.0, 0.0)

        self._default_ior = max(1.0, float(default_ior))
        self._default_specular = float(np.clip(default_specular, 0.0, 1.0))
        self._default_clearcoat = float(np.clip(default_clearcoat, 0.0, 1.0))
        self._default_clearcoat_roughness = float(
            np.clip(default_clearcoat_roughness, 0.001, 1.0)
        )
        self._camera_position = np.array((0.0, 2.0, 8.0), dtype=np.float32)
        self._camera_yaw = 180.0
        self._camera_pitch = 0.0
        self._camera_fov = 45.0
        self._camera_speed = max(0.0, float(camera_speed))
        self._look_sensitivity = 0.1
        self._keys_down: set[int] = set()
        self._user_camera_control = False
        self._global_transform = np.eye(4, dtype=np.float32)
        self._up_axis = 1
        self.set_up_axis(up_axis)

    @staticmethod
    def srgb_to_linear(channel: float) -> float:
        """Convert one sRGB channel to linear light."""
        channel = float(np.clip(channel, 0.0, 1.0))
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    @classmethod
    def srgb_to_linear_rgb(cls, color) -> tuple[float, float, float]:
        """Convert an authored sRGB color to path-tracer linear RGB."""
        return tuple(cls.srgb_to_linear(channel) for channel in color[:3])

    @property
    def tonemap_exposure(self) -> float:
        """Return the linear exposure multiplier."""
        return self._api.tonemap_exposure

    @tonemap_exposure.setter
    def tonemap_exposure(self, value: float) -> None:
        """Set the nonnegative linear exposure multiplier."""
        self._api.tonemap_exposure = value

    @property
    def tonemap_contrast(self) -> float:
        """Return the display contrast multiplier."""
        return self._api.tonemap_contrast

    @tonemap_contrast.setter
    def tonemap_contrast(self, value: float) -> None:
        """Set the nonnegative display contrast multiplier."""
        self._api.tonemap_contrast = value

    @property
    def tonemap_saturation(self) -> float:
        """Return the display saturation multiplier."""
        return self._api.tonemap_saturation

    @tonemap_saturation.setter
    def tonemap_saturation(self, value: float) -> None:
        """Set the nonnegative display saturation multiplier."""
        self._api.tonemap_saturation = value

    @property
    def dlss_quality(self) -> str:
        """Return the selected DLSS input-resolution/quality mode."""
        return self._api.dlss_quality

    @dlss_quality.setter
    def dlss_quality(self, value: str) -> None:
        """Select a DLSS quality mode."""
        self._api.set_dlss_quality(value)

    @property
    def max_bounces(self) -> int:
        """Return the current maximum path depth."""
        return self._api.max_bounces

    @property
    def direct_light_samples(self) -> int:
        """Return the number of direct-light samples per surface hit."""
        return self._api.direct_light_samples

    @property
    def samples_per_frame(self) -> int:
        """Return samples per frame used without DLSS."""
        return self._api.samples_per_frame

    def set_ray_budget(
        self,
        *,
        max_bounces: int | None = None,
        direct_light_samples: int | None = None,
        samples_per_frame: int | None = None,
    ) -> None:
        """Adjust path depth and sampling budgets."""
        self._api.set_ray_budget(
            max_bounces=max_bounces,
            direct_light_samples=direct_light_samples,
            samples_per_frame=samples_per_frame,
        )

    def _ensure_initialized(self):
        if self._initialized:
            return
        if not self._api.initialize():
            raise RuntimeError("Failed to initialize the OptiX path tracer")

        # Keep the neutral lighting defaults used by the reference path-tracing example.
        self._api.set_use_procedural_sky(True)
        self._initialized = True
        self._sync_camera()

    def _qualify_name(self, name: str) -> str:
        qualify = getattr(self, "_qualify", None)
        return qualify(name) if callable(qualify) else str(name)

    def set_up_axis(self, up_axis: str | int):
        axis = (
            "XYZ".index(up_axis.upper()) if isinstance(up_axis, str) else int(up_axis)
        )
        if axis not in (0, 1, 2):
            raise ValueError("up_axis must be X, Y, Z, 0, 1, or 2")
        self._up_axis = axis
        if axis == 2:
            # Physics Z-up -> renderer Y-up: (x, y, z) -> (x, z, -y).
            self._global_transform = np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
        elif axis == 0:
            # Physics X-up -> renderer Y-up: (x, y, z) -> (-y, x, z).
            self._global_transform = np.array(
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
        else:
            self._global_transform = np.eye(4, dtype=np.float32)

    def set_model(self, model):
        """Cooperatively initialize a model supplied by a viewer base class."""
        if model is not None and hasattr(model, "up_axis"):
            self.set_up_axis(model.up_axis)
        if self._mesh_ids or self._batches:
            self.clear()
        parent = getattr(super(), "set_model", None)
        if callable(parent):
            parent(model)
        else:
            self.model = model
        self._initialize_picking(model)

    def _initialize_picking(self, model):
        self._picking = None
        if not self.picking_enabled or model is None:
            return

        factory = self._picking_factory
        if factory is None:
            try:
                from newton._src.viewer.picking import Picking
            except ImportError:
                return
            factory = Picking

        try:
            self._picking = factory(model, pick_stiffness=10000.0, pick_damping=1000.0)
            if hasattr(self._picking, "world_offsets"):
                self._picking.world_offsets = getattr(self, "world_offsets", None)
            if hasattr(self._picking, "visible_worlds_mask"):
                self._picking.visible_worlds_mask = getattr(
                    self, "_visible_worlds_mask", None
                )
        except (TypeError, ValueError) as error:
            logger.warning("Newton picking is unavailable: %s", error)

    def set_world_offsets(self, spacing):
        parent = getattr(super(), "set_world_offsets", None)
        if not callable(parent):
            raise TypeError("World offsets require a simulation viewer base")
        result = parent(spacing)
        if self._picking is not None and hasattr(self._picking, "world_offsets"):
            self._picking.world_offsets = getattr(self, "world_offsets", None)
        return result

    def set_visible_worlds(self, worlds):
        parent = getattr(super(), "set_visible_worlds", None)
        if not callable(parent):
            raise TypeError("Visible worlds require a simulation viewer base")
        result = parent(worlds)
        if self._picking is not None and hasattr(self._picking, "visible_worlds_mask"):
            self._picking.visible_worlds_mask = getattr(
                self, "_visible_worlds_mask", None
            )
        return result

    def log_state(self, state):
        """Cache simulation state for picking, then use the framework logger."""
        self._last_state = state
        parent = getattr(super(), "log_state", None)
        if callable(parent):
            parent(state)

    def _get_or_create_material(self, color, material) -> int:
        color_key = tuple(round(float(v), 2) for v in color[:3])
        roughness = round(float(np.clip(material[0], 0.0, 1.0)), 3)
        metallic = round(float(np.clip(material[1], 0.0, 1.0)), 3)
        key = (
            *color_key,
            roughness,
            metallic,
            self._default_ior,
            self._default_specular,
            self._default_clearcoat,
            self._default_clearcoat_roughness,
        )
        if key in self._material_ids:
            return self._material_ids[key]

        linear_color = self.srgb_to_linear_rgb(color_key)
        material_id = self._api.create_pbr_material(
            linear_color,
            roughness=roughness,
            metallic=metallic,
            ior=self._default_ior,
            specular=self._default_specular,
            clearcoat=self._default_clearcoat,
            clearcoat_roughness=self._default_clearcoat_roughness,
        )
        self._material_ids[key] = material_id
        self._scene_dirty = True
        return material_id

    def log_mesh(
        self,
        name: str,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ):
        """Register or update a mesh prototype."""
        del hidden, backface_culling
        self._ensure_initialized()
        name = self._qualify_name(name)
        if texture is not None and not self._warned_texture:
            logger.warning(
                "Direct log_mesh textures are not yet supported; using the PBR base color"
            )
            self._warned_texture = True

        points_np = np.ascontiguousarray(_as_numpy(points, np.float32).reshape(-1, 3))
        indices_np = np.ascontiguousarray(_as_numpy(indices, np.uint32).reshape(-1, 3))
        normals_np = (
            None
            if normals is None
            else np.ascontiguousarray(_as_numpy(normals, np.float32).reshape(-1, 3))
        )
        uvs_np = (
            None
            if uvs is None
            else np.ascontiguousarray(_as_numpy(uvs, np.float32).reshape(-1, 2))
        )
        base_color = color or self._default_color
        pbr = (
            self._default_material[0] if roughness is None else roughness,
            self._default_material[1] if metallic is None else metallic,
            0.0,
            0.0,
        )
        material_id = self._get_or_create_material(base_color, pbr)

        if name in self._mesh_ids:
            mesh_id = self._mesh_ids[name]
            from .scene import Mesh

            self._api.scene._meshes[mesh_id] = Mesh(
                points_np, indices_np, normals_np, uvs_np, material_id=material_id
            )
        else:
            mesh_id = self._api.create_mesh(
                points_np, indices_np, normals_np, uvs_np, material_id
            )
            self._mesh_ids[name] = mesh_id
        self._scene_dirty = True

    def _instance_matrices(self, xforms, scales) -> np.ndarray:
        xforms_np = _as_numpy(xforms, np.float32).reshape(-1, 7)
        scales_np = _broadcast_rows(
            _as_numpy(scales, np.float32), len(xforms_np), 3, (1.0, 1.0, 1.0)
        )
        count = len(xforms_np)
        matrices = np.zeros((count, 4, 4), dtype=np.float32)
        matrices[:, 3, 3] = 1.0
        matrices[:, :3, 3] = xforms_np[:, :3]

        x, y, z, w = (xforms_np[:, index] for index in range(3, 7))
        matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        matrices[:, 0, 1] = 2.0 * (x * y - w * z)
        matrices[:, 0, 2] = 2.0 * (x * z + w * y)
        matrices[:, 1, 0] = 2.0 * (x * y + w * z)
        matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        matrices[:, 1, 2] = 2.0 * (y * z - w * x)
        matrices[:, 2, 0] = 2.0 * (x * z - w * y)
        matrices[:, 2, 1] = 2.0 * (y * z + w * x)
        matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        matrices[:, :3, :3] *= scales_np[:, None, :]
        return np.matmul(self._global_transform[None, :, :], matrices)

    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms,
        scales,
        colors,
        materials,
        hidden: bool = False,
    ):
        """Create or update an instanced mesh batch."""
        self._ensure_initialized()
        name = self._qualify_name(name)
        mesh = self._qualify_name(mesh)
        if mesh not in self._mesh_ids:
            raise RuntimeError(f"Mesh prototype {mesh!r} has not been logged")

        count = 0 if xforms is None else len(xforms)
        batch = self._batches.get(name)
        if batch is None:
            batch = _InstanceBatch(
                mesh_name=mesh,
                mesh_id=self._mesh_ids[mesh],
                instance_ids=[],
                colors=np.empty((0, 3), dtype=np.float32),
                materials=np.empty((0, 4), dtype=np.float32),
            )
            self._batches[name] = batch
        elif batch.mesh_name != mesh:
            batch.mesh_name = mesh
            batch.mesh_id = self._mesh_ids[mesh]
            for instance_id in batch.instance_ids:
                self._api.scene._instances[instance_id].mesh_index = batch.mesh_id
            self._scene_dirty = True

        previous_count = len(batch.instance_ids)
        while len(batch.instance_ids) < count:
            if len(self._api.scene._instances) >= self._max_instances:
                raise RuntimeError(
                    f"Viewer instance capacity exceeded ({self._max_instances})"
                )
            batch.instance_ids.append(self._api.create_instance(batch.mesh_id))
            self._scene_dirty = True
        instances_added = len(batch.instance_ids) != previous_count

        device_appearance = (
            isinstance(colors, wp.array)
            and colors.device.is_cuda
            or isinstance(materials, wp.array)
            and materials.device.is_cuda
            or name in self._device_material_batches
            and colors is None
            and materials is None
        )
        appearance_changed = False
        if not device_appearance:
            if colors is not None or len(batch.colors) != count:
                updated_colors = _broadcast_rows(
                    _as_numpy(colors, np.float32), count, 3, self._default_color
                )
                if updated_colors.shape != batch.colors.shape or not np.array_equal(
                    updated_colors, batch.colors
                ):
                    batch.colors = updated_colors
                    appearance_changed = True
            if materials is not None or len(batch.materials) != count:
                updated_materials = _broadcast_rows(
                    _as_numpy(materials, np.float32), count, 4, self._default_material
                )
                if (
                    updated_materials.shape != batch.materials.shape
                    or not np.array_equal(updated_materials, batch.materials)
                ):
                    batch.materials = updated_materials
                    appearance_changed = True

        active_ids = batch.instance_ids[:count]
        inactive_ids = batch.instance_ids[count:]
        visibility_changed = (
            instances_added or hidden != batch.hidden or count != batch.active_count
        )
        if visibility_changed:
            self._api.set_instances_visible(active_ids, not hidden)
            self._api.set_instances_visible(inactive_ids, False)
            batch.hidden = bool(hidden)
            batch.active_count = count

        if xforms is not None:
            if isinstance(xforms, wp.array) and xforms.device.is_cuda:
                if not isinstance(scales, wp.array) or not scales.device.is_cuda:
                    raise ValueError("CUDA transforms require CUDA-resident scales")
                cached = self._device_transform_batches.get(name)
                if cached is None or len(cached[0]) != count:
                    instance_ids = wp.array(
                        active_ids, dtype=wp.int32, device=self.device
                    )
                else:
                    instance_ids = cached[0]
                device_batch = (instance_ids, xforms, scales)
                self._device_transform_batches[name] = device_batch
                self._api.set_instance_transform_arrays(
                    instance_ids, xforms, scales, self._global_transform
                )
            else:
                matrices = self._instance_matrices(xforms, scales)
                self._api.set_instance_transform_matrices(active_ids, matrices)
            self._transforms_dirty = True

        if device_appearance:
            cached_materials = self._device_material_batches.get(name)
            if cached_materials is None or len(cached_materials[0]) != count:
                if not isinstance(colors, wp.array) or not isinstance(
                    materials, wp.array
                ):
                    raise ValueError(
                        "The first CUDA appearance update requires colors and materials"
                    )
                material_id_values = []
                default_linear = self.srgb_to_linear_rgb(self._default_color)
                for instance_id in active_ids:
                    material_id = self._api.create_pbr_material(
                        default_linear,
                        roughness=self._default_material[0],
                        metallic=self._default_material[1],
                        ior=self._default_ior,
                        specular=self._default_specular,
                        clearcoat=self._default_clearcoat,
                        clearcoat_roughness=self._default_clearcoat_roughness,
                    )
                    self._api.set_instance_material(instance_id, material_id)
                    material_id_values.append(material_id)
                material_ids = wp.array(
                    material_id_values, dtype=wp.int32, device=self.device
                )
                self._scene_dirty = True
            else:
                material_ids = cached_materials[0]
                colors = cached_materials[1] if colors is None else colors
                materials = cached_materials[2] if materials is None else materials
            if not isinstance(colors, wp.array) or not isinstance(materials, wp.array):
                raise ValueError(
                    "CUDA appearance updates require CUDA colors and materials"
                )
            device_material_batch = (material_ids, colors, materials)
            self._device_material_batches[name] = device_material_batch
            self._api.set_instance_material_arrays(material_ids, colors, materials)

        if not device_appearance and (appearance_changed or instances_added):
            for index, instance_id in enumerate(active_ids):
                material_id = self._get_or_create_material(
                    batch.colors[index], batch.materials[index]
                )
                self._api.set_instance_material(instance_id, material_id)
            self._materials_dirty = True

        if visibility_changed:
            self._transforms_dirty = True

    def log_capsules(
        self, name, mesh, xforms, scales, colors, materials, hidden: bool = False
    ):
        self.log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden)

    def update_instance_transforms(self, name: str, xforms, scales=None):
        """Update an existing batch from Warp arrays without resupplying materials."""
        qualified_name = self._qualify_name(name)
        batch = self._batches.get(qualified_name)
        if batch is None:
            raise KeyError(f"Unknown instance batch {qualified_name!r}")
        self.log_instances(
            qualified_name,
            batch.mesh_name,
            xforms,
            scales,
            None,
            None,
            hidden=False,
        )

    def log_lines(
        self, name, starts, ends, colors, width: float = 0.01, hidden: bool = False
    ):
        """Accept debug lines; ray-traced line geometry is planned separately."""
        del name, starts, ends, colors, width, hidden

    def log_points(self, name, points, radii=None, colors=None, hidden: bool = False):
        """Accept debug points; ray-traced point geometry is planned separately."""
        del name, points, radii, colors, hidden

    def log_array(self, name, array):
        del name, array

    def log_scalar(self, name, value, *, clear: bool = False, smoothing: int = 1):
        del name, value, clear, smoothing

    def apply_forces(self, state):
        """Apply the optional Newton picking force to a simulation state."""
        self._last_state = state
        if self._picking is not None:
            self._picking._apply_picking_force(state)

    def _flush_scene(self):
        if self._scene_dirty:
            self._api.build_scene()
            for (
                material_ids,
                colors,
                properties,
            ) in self._device_material_batches.values():
                self._api.set_instance_material_arrays(material_ids, colors, properties)
            for instance_ids, xforms, scales in self._device_transform_batches.values():
                self._api.set_instance_transform_arrays(
                    instance_ids, xforms, scales, self._global_transform
                )
            if self._device_transform_batches:
                self._api.rebuild_tlas()
        elif self._transforms_dirty:
            self._api.rebuild_tlas()

        if self._materials_dirty and not self._scene_dirty:
            material_ids = np.array(
                [
                    instance.material_id
                    if instance.material_id is not None
                    else self._api.scene._meshes[instance.mesh_index].material_id
                    for instance in self._api.scene._instances
                ],
                dtype=np.uint32,
            )
            self._api.scene.set_instance_material_ids_host(material_ids)

        self._scene_dirty = False
        self._transforms_dirty = False
        self._materials_dirty = False

    def begin_frame(self, time_sec: float):
        parent = getattr(super(), "begin_frame", None)
        if callable(parent):
            parent(time_sec)
        self.time = float(time_sec)

    def _render_to_mapped_buffer(
        self, mapped_image: wp.array, _frame_index: int, _elapsed: float
    ):
        self._api.render_frame()
        wp.launch(
            _pack_display_rgba8,
            dim=(self.width, self.height),
            inputs=[
                self._api.viewer.tonemapped_output,
                mapped_image,
                self.width,
                self.height,
            ],
            device=self.device,
        )

    def end_frame(self):
        if self._closed:
            return
        if self.paused and not self.render_when_paused and self._presenter is not None:
            self._presenter.window.dispatch_events()
        if self.paused and not self.render_when_paused:
            return
        self._ensure_initialized()
        self._flush_scene()
        now = time.perf_counter()
        self._update_camera_from_input(max(0.0, min(now - self._last_wall_time, 0.1)))
        self._last_wall_time = now
        if self._presenter is None:
            self._api.render_frame()
        else:
            self._presenter.render_once(self._render_to_mapped_buffer)
        self.frame_index += 1
        self._update_fps()
        self._record_frame()
        if self.num_frames is not None and self.frame_index >= self.num_frames:
            self.close()

    def render(self):
        """Render and present one frame."""
        self.begin_frame(self.time)
        self.end_frame()

    def run(self):
        """Run the standalone viewer until its window closes."""
        while self.is_running():
            self.begin_frame(time.perf_counter())
            self.end_frame()

    def is_running(self) -> bool:
        if self._closed:
            return False
        if self.num_frames is not None and self.frame_index >= self.num_frames:
            return False
        return self._presenter is None or self._presenter.is_running()

    def is_paused(self) -> bool:
        return self.paused

    def close(self):
        self._closed = True
        self.stop_recording()
        if self._imgui_impl is not None:
            try:
                self._imgui_impl.shutdown()
            except (AttributeError, RuntimeError):
                pass
            self._imgui_impl = None
        self._api.close()
        if self._presenter is not None:
            self._presenter.close()

    def clear(self):
        self._ensure_initialized()
        self._api.clear_scene()
        self._mesh_ids.clear()
        self._batches.clear()
        self._device_transform_batches.clear()
        self._device_material_batches.clear()
        self._material_ids.clear()
        self._scene_dirty = False
        self._transforms_dirty = False
        self._materials_dirty = False

    @property
    def usd_scene(self):
        """Return the retained USD hierarchy from the loaded OptiX scene."""
        return self._api.usd_scene

    def load_scene_from_usd(
        self,
        usd_path: str,
        *,
        clear_existing: bool = True,
        apply_stage_units: bool = True,
        convert_up_axis: bool = True,
        max_texture_size: int | None = None,
        strict_sidedness: bool = False,
        load_usd_environment: bool = False,
        usd_environment_scale: float = 1.0,
        enable_emissive_materials: bool = True,
    ) -> bool:
        """Load a composed USD stage into the OptiX renderer."""
        self._ensure_initialized()
        loaded = self._api.load_scene_from_usd(
            usd_path,
            clear_existing=clear_existing,
            apply_stage_units=apply_stage_units,
            convert_up_axis=convert_up_axis,
            max_texture_size=max_texture_size,
            strict_sidedness=strict_sidedness,
            enable_emissive_materials=enable_emissive_materials,
            load_usd_environment=load_usd_environment,
            usd_environment_scale=usd_environment_scale,
        )
        if loaded and clear_existing:
            self._mesh_ids.clear()
            self._batches.clear()
            self._device_transform_batches.clear()
            self._device_material_batches.clear()
            self._material_ids.clear()
            self._scene_dirty = False
            self._transforms_dirty = False
            self._materials_dirty = False
        return bool(loaded)

    def set_camera_look_at(
        self,
        position,
        target,
        *,
        fov: float | None = None,
        renderer_space: bool = False,
    ) -> None:
        """Set a level camera from an eye and target in physics or renderer space."""
        if self._user_camera_control:
            return
        position = np.asarray(position, dtype=np.float32).reshape(3)
        target = np.asarray(target, dtype=np.float32).reshape(3)
        if renderer_space:
            renderer_to_physics = self._global_transform[:3, :3].T
            position = renderer_to_physics @ position
            target = renderer_to_physics @ target
        direction = target - position
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            raise ValueError("Camera position and target must differ")
        direction /= norm

        if self._up_axis == 0:
            pitch = math.degrees(math.asin(float(np.clip(direction[0], -1.0, 1.0))))
            yaw = math.degrees(math.atan2(float(direction[2]), float(direction[1])))
        elif self._up_axis == 2:
            pitch = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
            yaw = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        else:
            pitch = math.degrees(math.asin(float(np.clip(direction[1], -1.0, 1.0))))
            yaw = math.degrees(math.atan2(float(direction[2]), float(direction[0])))

        self._camera_position = position
        self._camera_pitch = float(np.clip(pitch, -89.0, 89.0))
        self._camera_yaw = (float(yaw) + 180.0) % 360.0 - 180.0
        if fov is not None:
            self._camera_fov = float(np.clip(fov, 5.0, 120.0))
        self._sync_camera()

    def set_camera(self, pos, pitch: float, yaw: float):
        """Set a framework-style camera unless interactive control has taken over."""
        if self._user_camera_control:
            return
        self._camera_position = np.asarray(
            [float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32
        )
        self._camera_pitch = float(np.clip(pitch, -89.0, 89.0))
        self._camera_yaw = (float(yaw) + 180.0) % 360.0 - 180.0
        self._sync_camera()

    def _physics_camera_front(self) -> np.ndarray:
        yaw = math.radians(self._camera_yaw)
        pitch = math.radians(self._camera_pitch)
        cp = math.cos(pitch)
        if self._up_axis == 0:
            return np.array(
                (math.sin(pitch), math.cos(yaw) * cp, math.sin(yaw) * cp),
                dtype=np.float32,
            )
        if self._up_axis == 2:
            return np.array(
                (math.cos(yaw) * cp, math.sin(yaw) * cp, math.sin(pitch)),
                dtype=np.float32,
            )
        return np.array(
            (math.cos(yaw) * cp, math.sin(pitch), math.sin(yaw) * cp), dtype=np.float32
        )

    def _sync_camera(self):
        if not self._initialized:
            return
        rotation = self._global_transform[:3, :3]
        position = rotation @ self._camera_position
        direction = rotation @ self._physics_camera_front()
        self._api.set_camera_look_at(
            position, position + direction, (0.0, 1.0, 0.0), self._camera_fov
        )

    def _update_camera_from_input(self, dt: float):
        if self._presenter is None or not self._keys_down:
            return
        key = self._presenter.pyglet.window.key
        front = self._physics_camera_front()
        up = np.zeros(3, dtype=np.float32)
        up[self._up_axis] = 1.0
        right = np.cross(front, up)
        norm = float(np.linalg.norm(right))
        if norm > 1.0e-6:
            right /= norm
        move = np.zeros(3, dtype=np.float32)
        if key.W in self._keys_down or key.UP in self._keys_down:
            move += front
        if key.S in self._keys_down or key.DOWN in self._keys_down:
            move -= front
        if key.A in self._keys_down or key.LEFT in self._keys_down:
            move -= right
        if key.D in self._keys_down or key.RIGHT in self._keys_down:
            move += right
        if key.Q in self._keys_down:
            move -= up
        if key.E in self._keys_down:
            move += up
        norm = float(np.linalg.norm(move))
        if norm > 1.0e-6:
            speed = self._camera_speed
            if key.LSHIFT in self._keys_down or key.RSHIFT in self._keys_down:
                speed *= 4.0
            self._camera_position += move / norm * speed * dt
            self._user_camera_control = True
            self._sync_camera()

    def on_key_press(self, symbol, _modifiers):
        if self._presenter is None:
            return
        key = self._presenter.pyglet.window.key
        if self._ui_wants_keyboard():
            return
        if symbol == key.SPACE:
            self.paused = not self.paused
        elif symbol == key.ESCAPE:
            self.close()
            return
        elif symbol == key.R:
            if not self.is_recording():
                self.start_recording()
        elif symbol == key.T:
            self.stop_recording()
        else:
            debug_keys = {
                key._1: 2,  # Depth
                key._2: 3,  # Motion
                key._3: 4,  # Normals
                key._4: 6,  # Diffuse
                key._5: 7,  # Specular
                key._6: 1,  # Noisy radiance
                key._7: 8,  # Specular hit distance
                key._8: 5,  # Roughness (OptiX-specific extra)
            }
            if symbol in debug_keys:
                mode = debug_keys[symbol]
                self.set_debug_buffer_mode(
                    0 if self._debug_buffer_mode == mode else mode
                )
            elif symbol in (key._0, key.BACKSPACE):
                self.set_debug_buffer_mode(0)
        self._keys_down.add(symbol)

    def on_key_release(self, symbol, _modifiers):
        self._keys_down.discard(symbol)

    def on_mouse_drag(self, _x, _y, dx, dy, buttons, _modifiers):
        if self._presenter is None:
            return
        if self._ui_wants_mouse():
            return
        mouse = self._presenter.pyglet.window.mouse
        if buttons & mouse.LEFT:
            self._camera_yaw -= float(dx) * self._look_sensitivity
            self._camera_pitch = float(
                np.clip(
                    self._camera_pitch + float(dy) * self._look_sensitivity, -89.0, 89.0
                )
            )
            self._user_camera_control = True
            self._sync_camera()
        if (
            buttons & mouse.RIGHT
            and self.picking_enabled
            and self._picking is not None
            and self._picking.is_picking()
        ):
            origin, direction = self._get_ray_from_mouse(_x, _y)
            self._picking.update(wp.vec3(*origin), wp.vec3(*direction))

    def on_mouse_press(self, x, y, button, _modifiers):
        self._mouse_buttons.add(button)
        if self._presenter is None or self._ui_wants_mouse():
            return
        mouse = self._presenter.pyglet.window.mouse
        if (
            button == mouse.RIGHT
            and self.picking_enabled
            and self._picking is not None
            and self._last_state is not None
        ):
            origin, direction = self._get_ray_from_mouse(x, y)
            self._picking.pick(self._last_state, wp.vec3(*origin), wp.vec3(*direction))

    def on_mouse_release(self, _x, _y, button, _modifiers):
        self._mouse_buttons.discard(button)
        if self._presenter is None:
            return
        if (
            button == self._presenter.pyglet.window.mouse.RIGHT
            and self._picking is not None
        ):
            self._picking.release()

    def on_mouse_motion(self, x, y, _dx, _dy):
        self._mouse_x = float(x)
        self._mouse_y = float(y)

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y):
        if self._ui_wants_mouse():
            return
        self._camera_fov = float(
            np.clip(self._camera_fov - float(scroll_y) * 2.0, 10.0, 120.0)
        )
        self._user_camera_control = True
        self._sync_camera()

    def _get_ray_from_mouse(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Construct a world-space picking ray in the physics coordinate system."""
        width = max(1.0, float(self.width))
        height = max(1.0, float(self.height))
        ndc_x = 2.0 * float(x) / width - 1.0
        ndc_y = 2.0 * float(y) / height - 1.0

        front = self._physics_camera_front()
        front /= max(float(np.linalg.norm(front)), 1.0e-8)
        world_up = np.zeros(3, dtype=np.float32)
        world_up[self._up_axis] = 1.0
        right = np.cross(front, world_up)
        right /= max(float(np.linalg.norm(right)), 1.0e-8)
        camera_up = np.cross(right, front)
        camera_up /= max(float(np.linalg.norm(camera_up)), 1.0e-8)

        half_height = math.tan(math.radians(self._camera_fov) * 0.5)
        direction = (
            front
            + right * ndc_x * (width / height) * half_height
            + camera_up * ndc_y * half_height
        )
        direction /= max(float(np.linalg.norm(direction)), 1.0e-8)
        return self._camera_position.copy(), direction.astype(np.float32)

    def _on_resize(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self._api.resize(self.width, self.height)

    def _update_fps(self):
        self._fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_window_start = now

    def get_fps(self) -> float:
        return float(self._current_fps)

    def register_ui_callback(self, callback: Callable, position: str = "options"):
        """Register an ImGui callback, matching the hybrid viewer API."""
        self._ui_callbacks.append((callback, str(position)))

    def enable_imgui(self, enabled: bool = True):
        self._imgui_enabled = bool(enabled)
        if self._imgui_enabled and self._imgui_impl is None:
            self._init_imgui()

    def _init_imgui(self):
        if self._presenter is None or self._imgui_impl is not None:
            return
        try:
            from imgui_bundle import imgui
            from imgui_bundle.python_backends import pyglet_backend
        except ImportError:
            logger.info(
                "ImGui overlay unavailable; install warp_optix[ui] to enable it"
            )
            return

        try:
            imgui.create_context()
            self._imgui_impl = pyglet_backend.create_renderer(self._presenter.window)
            self._imgui = imgui
            imgui.style_colors_dark()
        except (AttributeError, RuntimeError) as error:
            logger.warning("Failed to initialize ImGui overlay: %s", error)
            self._imgui = None
            self._imgui_impl = None

    def _ui_wants_mouse(self) -> bool:
        if self._imgui is None or self._imgui_impl is None:
            return False
        try:
            return bool(self._imgui.get_io().want_capture_mouse)
        except (AttributeError, RuntimeError):
            return False

    def _ui_wants_keyboard(self) -> bool:
        if self._imgui is None or self._imgui_impl is None:
            return False
        try:
            return bool(self._imgui.get_io().want_capture_keyboard)
        except (AttributeError, RuntimeError):
            return False

    def _draw_imgui(self):
        if not self._imgui_enabled or self._imgui_impl is None:
            return
        imgui = self._imgui
        try:
            self._imgui_impl.process_inputs()
            imgui.new_frame()
            imgui.set_next_window_pos((10.0, 10.0), imgui.Cond_.appearing)
            imgui.set_next_window_size((310.0, 0.0), imgui.Cond_.appearing)
            visible = imgui.begin("OptiX Path Tracing Viewer")
            if isinstance(visible, tuple):
                visible = visible[0]
            if visible:
                imgui.text(f"FPS: {self._current_fps:.1f}")
                imgui.text(f"Meshes: {len(self._mesh_ids)}")
                imgui.text(
                    f"Instances: {sum(len(batch.instance_ids) for batch in self._batches.values())}"
                )
                imgui.text(f"Materials: {len(self._material_ids)}")
                imgui.text(
                    "DLSS RR: active" if self._api.dlss_enabled else "DLSS RR: inactive"
                )

                changed, paused = imgui.checkbox("Pause", self.paused)
                if changed:
                    self.paused = paused
                if hasattr(self, "picking_enabled"):
                    changed, picking = imgui.checkbox(
                        "Enable Picking", self.picking_enabled
                    )
                    if changed:
                        self.picking_enabled = picking
                        if not picking and self._picking is not None:
                            self._picking.release()

                visualization_flags = (
                    ("show_joints", "Show Joints"),
                    ("show_contacts", "Show Contacts"),
                    ("show_particles", "Show Particles"),
                    ("show_springs", "Show Springs"),
                    ("show_com", "Show Center of Mass"),
                    ("show_collision", "Show Collision"),
                    ("show_visual", "Show Visual"),
                )
                for attribute, label in visualization_flags:
                    if hasattr(self, attribute):
                        changed, value = imgui.checkbox(
                            label, bool(getattr(self, attribute))
                        )
                        if changed:
                            setattr(self, attribute, value)

                if self.model is not None:
                    if hasattr(self.model, "world_count"):
                        imgui.text(f"Worlds: {self.model.world_count}")
                    elif hasattr(self.model, "num_worlds"):
                        imgui.text(f"Worlds: {self.model.num_worlds}")

                mode_names = [
                    "Final",
                    "Radiance",
                    "Depth",
                    "Motion",
                    "Normal",
                    "Roughness",
                    "Diffuse",
                    "Specular",
                    "Specular Hit Distance",
                ]
                changed, mode = imgui.combo(
                    "Debug Buffer", self._debug_buffer_mode, mode_names
                )
                if changed:
                    self.set_debug_buffer_mode(mode)

                changed, fov = imgui.slider_float("FOV", self._camera_fov, 20.0, 120.0)
                if changed:
                    self._camera_fov = float(fov)
                    self._sync_camera()
                imgui.text(
                    "Camera: "
                    f"({self._camera_position[0]:.2f}, "
                    f"{self._camera_position[1]:.2f}, "
                    f"{self._camera_position[2]:.2f})"
                )
                imgui.text(
                    f"Yaw {self._camera_yaw:.1f}  Pitch {self._camera_pitch:.1f}"
                )

                if self.is_recording():
                    imgui.text(f"Recording: {self._recording_path}")
                    imgui.text(f"Dropped frames: {self.recording_dropped_frames}")
                    if imgui.button("Stop Recording"):
                        self.stop_recording()
                elif imgui.button("Start Recording"):
                    self.start_recording()

                for callback, _position in self._ui_callbacks:
                    callback(imgui)
            imgui.end()
            imgui.render()
            self._imgui_impl.render(imgui.get_draw_data())
        except (AttributeError, RuntimeError, TypeError) as error:
            logger.warning("Disabling ImGui overlay after rendering error: %s", error)
            self._imgui_enabled = False

    def start_recording(
        self,
        output_path: str | None = None,
        fps: int | None = None,
        bitrate_mbps: int | None = None,
        frame_skip: int | None = None,
        encoder: str | None = None,
    ) -> str:
        """Start buffered MP4 recording using the current tonemapped output."""
        if self._recording_writer is not None:
            return str(self._recording_path)

        actual_fps = max(1, int(fps or self.recording_fps))
        actual_bitrate = max(1, int(bitrate_mbps or self.recording_bitrate_mbps))
        self.recording_frame_skip = max(1, int(frame_skip or self.recording_frame_skip))
        path = Path(
            output_path
            or self.recording_output_path
            or (
                _system_videos_dir()
                / "NewtonRecordings"
                / time.strftime("pathtracing_recording_%Y%m%d_%H%M%S.mp4")
            )
        ).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self._recording_writer_factory is None:
                self._recording_writer = _FFmpegVideoWriter(
                    path,
                    self.width,
                    self.height,
                    actual_fps,
                    actual_bitrate,
                    encoder or self.recording_encoder,
                )
            else:
                self._recording_writer = self._recording_writer_factory(
                    str(path),
                    fps=actual_fps,
                    codec="libx264",
                    bitrate=f"{actual_bitrate}M",
                    macro_block_size=1,
                    output_params=["-vf", "vflip", "-preset", "ultrafast"],
                )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                "Unable to start video recording; install warp_optix[recording]"
            ) from error

        self._recording_path = path.resolve()
        self._recording_frame_index = 0
        self._recording_submitted_frames = 0
        self.recording_dropped_frames = 0
        self._recording_error = None
        self._recording_width = self.width
        self._recording_height = self.height
        buffer_count = max(2, int(self.recording_buffer_count))
        self._recording_queue = queue.Queue(maxsize=buffer_count)
        self._recording_free_slots = queue.SimpleQueue()
        self._recording_slots = []
        self._recording_stream = None
        source = getattr(self._api.viewer, "tonemapped_output", None)
        if self.device.is_cuda and isinstance(source, wp.array):
            self._recording_stream = wp.Stream(self.device)
            pixel_count = self.width * self.height * 3
            for _ in range(buffer_count):
                slot = _RecordingReadback(
                    device_pixels=wp.empty(
                        pixel_count, dtype=wp.uint8, device=self.device
                    ),
                    host_pixels=wp.empty(
                        pixel_count, dtype=wp.uint8, device="cpu", pinned=True
                    ),
                    ready=wp.Event(self.device),
                )
                self._recording_slots.append(slot)
                self._recording_free_slots.put(slot)
            # Compile the packing kernel before recording enters the frame loop.
            wp.launch(
                _pack_recording_rgb8,
                dim=(self.width, self.height),
                inputs=[
                    source,
                    self._recording_slots[0].device_pixels,
                    self.width,
                    self.height,
                ],
                device=self.device,
                stream=self._recording_stream,
            )
            wp.synchronize_stream(self._recording_stream)

        self._recording_thread = threading.Thread(
            target=self._recording_worker,
            name="warp-optix-video-encoder",
            daemon=True,
        )
        self._recording_thread.start()
        selected_encoder = getattr(self._recording_writer, "encoder", "custom")
        logger.info(
            "Recording started: %s (encoder=%s, buffers=%d)",
            self._recording_path,
            selected_encoder,
            buffer_count,
        )
        return str(self._recording_path)

    def _recording_worker(self) -> None:
        """Wait for readbacks and feed FFmpeg without blocking rendering."""
        recording_queue = self._recording_queue
        writer = self._recording_writer
        if recording_queue is None or writer is None:
            return
        try:
            while True:
                item = recording_queue.get()
                if item is _RECORDING_STOP:
                    break
                try:
                    if isinstance(item, _RecordingReadback):
                        wp.synchronize_event(item.ready)
                        frame = item.host_pixels.numpy().reshape(
                            self._recording_height, self._recording_width, 3
                        )
                    else:
                        frame = item
                    if self._recording_error is None:
                        writer.append_data(frame)
                except BaseException as error:
                    if self._recording_error is None:
                        self._recording_error = error
                finally:
                    if isinstance(item, _RecordingReadback):
                        self._recording_free_slots.put(item)
        finally:
            try:
                writer.close()
            except BaseException as error:
                if self._recording_error is None:
                    self._recording_error = error

    def _record_frame(self) -> None:
        if self._recording_writer is None:
            return
        if self._recording_error is not None:
            error = self._recording_error
            self.stop_recording()
            raise RuntimeError("Video recording worker failed") from error
        frame_number = self._recording_frame_index
        self._recording_frame_index += 1
        if frame_number % self.recording_frame_skip != 0:
            return
        if self.width != self._recording_width or self.height != self._recording_height:
            self.recording_dropped_frames += 1
            return

        if self._recording_stream is not None:
            try:
                slot = self._recording_free_slots.get_nowait()
            except queue.Empty:
                self.recording_dropped_frames += 1
                return
            render_stream = getattr(self._api.viewer, "_render_stream", None)
            if render_stream is None:
                render_stream = wp.get_stream(self.device)
            render_done = render_stream.record_event()
            self._recording_stream.wait_event(render_done)
            wp.launch(
                _pack_recording_rgb8,
                dim=(self.width, self.height),
                inputs=[
                    self._api.viewer.tonemapped_output,
                    slot.device_pixels,
                    self.width,
                    self.height,
                ],
                device=self.device,
                stream=self._recording_stream,
            )
            wp.copy(slot.host_pixels, slot.device_pixels, stream=self._recording_stream)
            self._recording_stream.record_event(slot.ready)
            self._recording_queue.put_nowait(slot)
        else:
            frame = self._api.get_frame_uint8()
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            try:
                self._recording_queue.put_nowait(np.ascontiguousarray(frame))
            except queue.Full:
                self.recording_dropped_frames += 1
                return
        self._recording_submitted_frames += 1

    def stop_recording(self) -> None:
        if self._recording_writer is None:
            return
        recording_queue = self._recording_queue
        recording_thread = self._recording_thread
        if recording_queue is not None:
            recording_queue.put(_RECORDING_STOP)
        if recording_thread is not None:
            recording_thread.join()
        error = self._recording_error
        self._recording_writer = None
        self._recording_thread = None
        self._recording_queue = None
        self._recording_free_slots = None
        self._recording_stream = None
        self._recording_slots = []
        logger.info(
            "Recording stopped: %s (frames=%d, dropped=%d)",
            self._recording_path,
            self._recording_submitted_frames,
            self.recording_dropped_frames,
        )
        if error is not None:
            raise RuntimeError("Video recording failed") from error

    def is_recording(self) -> bool:
        return self._recording_writer is not None

    def get_instance_transform_gl_buffer(self) -> int:
        """Compatibility API; OptiX uses CUDA arrays instead of a transform VBO."""
        return 0

    def get_instance_transform_capacity(self) -> int:
        return self._max_instances

    def notify_transforms_updated(self, count: int = -1):
        """Compatibility hook for the old bridge-specific transform VBO API."""
        del count
        if not self._warned_transform_vbo:
            logger.warning(
                "No transform GL buffer is exposed; update Warp transforms with log_instances()"
            )
            self._warned_transform_vbo = True

    def is_gpu_transform_available(self) -> bool:
        """The old Vulkan-backed GL transform buffer is not used by this backend."""
        return False

    def set_sun_direction(self, x: float, y: float, z: float, intensity: float = 1.0):
        self._ensure_initialized()
        direction = self._global_transform[:3, :3] @ np.asarray(
            (x, y, z), dtype=np.float32
        )
        self._api.set_sun_direction(*direction, intensity=float(intensity))

    def reset_temporal_history(self):
        """Discard DLSS reconstruction history after a discontinuous scene change."""
        self._api.reset_temporal_history()

    def set_sky_parameters(self, **kwargs):
        self._ensure_initialized()
        kwargs = dict(kwargs)
        for name in ("ground_color", "night_color"):
            if name in kwargs:
                kwargs[name] = self.srgb_to_linear_rgb(kwargs[name])
        self._api.set_sky_parameters(**kwargs)

    @staticmethod
    def _sun_direction(
        elevation_degrees: float, azimuth_degrees: float
    ) -> tuple[float, float, float]:
        elevation = math.radians(elevation_degrees)
        azimuth = math.radians(azimuth_degrees)
        cos_elevation = math.cos(elevation)
        return (
            cos_elevation * math.sin(azimuth),
            math.sin(elevation),
            cos_elevation * math.cos(azimuth),
        )

    def set_time_of_day(self, preset: str):
        """Apply one of the physical-sky presets used by the hybrid viewer."""
        presets = {
            "dawn": (
                5.0,
                90.0,
                1.0,
                4.0,
                0.5,
                1.2,
                (0.3, 0.2, 0.15),
                1.5,
                (0.001, 0.002, 0.005),
                0.8,
                1.5,
                2.0,
            ),
            "morning": (
                30.0,
                120.0,
                1.0,
                1.0,
                0.1,
                1.0,
                (0.35, 0.3, 0.25),
                1.0,
                (0, 0, 0),
                1.0,
                1.0,
                1.0,
            ),
            "noon": (
                60.0,
                180.0,
                1.0,
                0.0,
                0.0,
                1.0,
                (0.4, 0.4, 0.35),
                1.0,
                (0, 0, 0),
                1.0,
                1.0,
                0.5,
            ),
            "afternoon": (
                45.0,
                240.0,
                1.0,
                0.5,
                0.05,
                1.0,
                (0.4, 0.35, 0.3),
                1.0,
                (0, 0, 0),
                1.0,
                1.0,
                1.0,
            ),
            "sunset": (
                8.0,
                270.0,
                1.0,
                5.0,
                0.6,
                1.3,
                (0.35, 0.2, 0.1),
                2.0,
                (0.001, 0.002, 0.005),
                0.7,
                2.0,
                3.0,
            ),
            "dusk": (
                -5.0,
                280.0,
                1.0,
                3.0,
                0.3,
                1.1,
                (0.15, 0.1, 0.1),
                2.0,
                (0.005, 0.008, 0.015),
                0.3,
                1.5,
                2.0,
            ),
            "night": (
                -30.0,
                0.0,
                0.01,
                0.0,
                0.0,
                0.5,
                (0.02, 0.02, 0.02),
                1.0,
                (0.01, 0.015, 0.025),
                0.0,
                0.0,
                0.0,
            ),
            "midnight": (
                -60.0,
                0.0,
                0.005,
                0.0,
                0.0,
                0.3,
                (0.01, 0.01, 0.01),
                1.0,
                (0.008, 0.012, 0.02),
                0.0,
                0.0,
                0.0,
            ),
        }
        name = str(preset).lower()
        if name == "midday":
            name = "noon"
        if name not in presets:
            raise ValueError(
                f"Unknown sky preset {preset!r}; expected one of {', '.join(presets)}"
            )
        (
            elevation,
            azimuth,
            multiplier,
            haze,
            shift,
            saturation,
            ground,
            blur,
            night,
            disk,
            scale,
            glow,
        ) = presets[name]
        self._api.set_sky_parameters(
            sun_direction=self._sun_direction(elevation, azimuth),
            multiplier=multiplier,
            haze=haze,
            red_blue_shift=shift,
            saturation=saturation,
            ground_color=self.srgb_to_linear_rgb(ground),
            horizon_blur=blur,
            night_color=self.srgb_to_linear_rgb(night),
            sun_disk_intensity=disk,
            sun_disk_scale=scale,
            sun_glow_intensity=glow,
            y_is_up=1,
        )

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        self._api.set_environment_hdr(hdr_path, scaling)

    def set_environment_color(self, color):
        self._api.set_environment_color(self.srgb_to_linear_rgb(color))

    def set_debug_buffer_mode(self, mode: int):
        mode = int(np.clip(mode, 0, 8))
        self._debug_buffer_mode = mode
        self._api.set_debug_buffer_mode(mode)


class PathTracingViewer(PathTracingViewerBackend):
    """Concrete standalone path-tracing viewer."""
