# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone path-tracing viewer with a renderer-neutral scene logging API."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from warp_optix._runtime.gl_interop import OptixGLInteropViewer
from warp_optix._runtime.transform_utils import build_transform_matrix

from .pathtracer_api import PathTracerAPI

logger = logging.getLogger(__name__)


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


@dataclass
class _InstanceBatch:
    mesh_name: str
    mesh_id: int
    instance_ids: list[int]
    colors: np.ndarray
    materials: np.ndarray


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
        fps: int = 60,
        device: str = "cuda",
        headless: bool = False,
        paused: bool = False,
        num_frames: int | None = None,
        enable_dlss_rr: bool = True,
        enable_set: bool = True,
        up_axis: str | int = "Y",
        camera_speed: float = 4.0,
        api: PathTracerAPI | None = None,
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
        self.fps = max(1, int(fps))
        self.headless = bool(headless)
        self.paused = bool(paused)
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

        self._api = api or PathTracerAPI(
            width=self.width,
            height=self.height,
            enable_dlss_rr=enable_dlss_rr,
            enable_set=enable_set,
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
            )
            self._presenter.window.push_handlers(self)

        self._mesh_ids: dict[str, int] = {}
        self._batches: dict[str, _InstanceBatch] = {}
        self._material_ids: dict[tuple[float, float, float, float, float], int] = {}
        self._default_color = (0.8, 0.8, 0.8)
        self._default_material = (0.5, 0.0, 0.0, 0.0)

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

    def _ensure_initialized(self):
        if self._initialized:
            return
        if not self._api.initialize():
            raise RuntimeError("Failed to initialize the OptiX path tracer")

        # Defaults carried over from the hybrid viewer. The lighter ground,
        # slight haze, and soft horizon are important for Newton's untextured
        # simulation geometry.
        self._api.set_use_procedural_sky(True)
        self._api.set_sky_parameters(
            sun_direction=(-0.3, 0.7, 0.5),
            multiplier=1.5,
            saturation=1.0,
            haze=0.03,
            ground_color=(0.7, 0.7, 0.75),
            horizon_blur=0.3,
            sun_disk_intensity=1.0,
            sun_glow_intensity=0.8,
            y_is_up=1,
        )
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

    def _get_or_create_material(self, color, material) -> int:
        color_key = tuple(round(float(v), 2) for v in color[:3])
        roughness = round(float(np.clip(material[0], 0.0, 1.0)), 3)
        metallic = round(float(np.clip(material[1], 0.0, 1.0)), 3)
        key = (*color_key, roughness, metallic)
        if key in self._material_ids:
            return self._material_ids[key]

        linear_color = tuple(self.srgb_to_linear(channel) for channel in color_key)
        material_id = self._api.create_pbr_material(
            linear_color, roughness=roughness, metallic=metallic
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
        matrices = np.empty((len(xforms_np), 4, 4), dtype=np.float32)
        for index, (xform, scale) in enumerate(zip(xforms_np, scales_np)):
            local = build_transform_matrix(xform[:3], xform[3:7], scale)
            matrices[index] = self._global_transform @ local
        return matrices

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

        while len(batch.instance_ids) < count:
            batch.instance_ids.append(self._api.create_instance(batch.mesh_id))
            self._scene_dirty = True

        if colors is not None or len(batch.colors) != count:
            batch.colors = _broadcast_rows(
                _as_numpy(colors, np.float32), count, 3, self._default_color
            )
        if materials is not None or len(batch.materials) != count:
            batch.materials = _broadcast_rows(
                _as_numpy(materials, np.float32), count, 4, self._default_material
            )

        if xforms is not None:
            matrices = self._instance_matrices(xforms, scales)
            for index, instance_id in enumerate(batch.instance_ids):
                active = index < count and not hidden
                self._api.set_instance_visible(instance_id, active)
                if index < count:
                    self._api.set_instance_transform_matrix(
                        instance_id, matrices[index]
                    )
                    material_id = self._get_or_create_material(
                        batch.colors[index], batch.materials[index]
                    )
                    self._api.set_instance_material(instance_id, material_id)
            self._transforms_dirty = True
            self._materials_dirty = True
        else:
            for instance_id in batch.instance_ids:
                self._api.set_instance_visible(instance_id, not hidden)
            self._transforms_dirty = True

    def log_capsules(
        self, name, mesh, xforms, scales, colors, materials, hidden: bool = False
    ):
        self.log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden)

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
        """No-op hook; an integration wrapper may add interactive picking."""
        del state

    def _flush_scene(self):
        if self._scene_dirty:
            self._api.build_scene()
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
        if self.paused and self._presenter is not None:
            self._presenter.window.dispatch_events()
        if self.paused:
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
        self._api.close()
        if self._presenter is not None:
            self._presenter.close()

    def clear(self):
        self._ensure_initialized()
        self._api.clear_scene()
        self._mesh_ids.clear()
        self._batches.clear()
        self._material_ids.clear()
        self._scene_dirty = False
        self._transforms_dirty = False
        self._materials_dirty = False

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
        if (
            self._presenter is not None
            and symbol == self._presenter.pyglet.window.key.SPACE
        ):
            self.paused = not self.paused
        self._keys_down.add(symbol)

    def on_key_release(self, symbol, _modifiers):
        self._keys_down.discard(symbol)

    def on_mouse_drag(self, _x, _y, dx, dy, buttons, _modifiers):
        if self._presenter is None:
            return
        if buttons & self._presenter.pyglet.window.mouse.LEFT:
            self._camera_yaw -= float(dx) * self._look_sensitivity
            self._camera_pitch = float(
                np.clip(
                    self._camera_pitch + float(dy) * self._look_sensitivity, -89.0, 89.0
                )
            )
            self._user_camera_control = True
            self._sync_camera()

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y):
        self._camera_fov = float(
            np.clip(self._camera_fov - float(scroll_y) * 2.0, 10.0, 120.0)
        )
        self._user_camera_control = True
        self._sync_camera()

    def _on_resize(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self._api.resize(self.width, self.height)

    def set_sun_direction(self, x: float, y: float, z: float, intensity: float = 1.0):
        self._ensure_initialized()
        direction = self._global_transform[:3, :3] @ np.asarray(
            (x, y, z), dtype=np.float32
        )
        self._api.set_sun_direction(*direction, intensity=float(intensity))

    def set_sky_parameters(self, **kwargs):
        self._ensure_initialized()
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
            ground_color=ground,
            horizon_blur=blur,
            night_color=night,
            sun_disk_intensity=disk,
            sun_disk_scale=scale,
            sun_glow_intensity=glow,
            y_is_up=1,
        )

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        self._api.set_environment_hdr(hdr_path, scaling)

    def set_environment_color(self, color):
        self._api.set_environment_color(color)

    def set_debug_buffer_mode(self, mode: int):
        self._api.set_debug_buffer_mode(mode)


class PathTracingViewer(PathTracingViewerBackend):
    """Concrete standalone path-tracing viewer."""
