# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-level OptiX API for standalone path tracing usage."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from collections.abc import Iterable

import numpy as np

from warp_optix._runtime.transform_utils import build_transform_matrix

from .pathtracing_viewer import PathTracingViewer as PathTracingRenderer
from .ptx_compiler import get_optix_include_dir
from .scene import Mesh

logger = logging.getLogger(__name__)


class PathTracerAPI:
    """High-level API for driving the OptiX path tracer directly from Python."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        enable_dlss_rr: bool = True,
        enable_set: bool = True,
        enable_cuda_graphs: bool = True,
        dlss_quality: str = "quality",
        samples_per_frame: int = 1,
        max_bounces: int = 4,
        direct_light_samples: int = 1,
    ):
        self.width = int(width)
        self.height = int(height)
        self._viewer = PathTracingRenderer(
            width=self.width,
            height=self.height,
            scene_setup=lambda _scene: None,
            enable_dlss_rr=bool(enable_dlss_rr),
            enable_set=bool(enable_set),
            enable_cuda_graphs=bool(enable_cuda_graphs),
            accumulate_samples=False,
            dlss_quality=dlss_quality,
            samples_per_frame=samples_per_frame,
            max_bounces=max_bounces,
            direct_light_samples=direct_light_samples,
        )
        self._built = False
        self._running = True
        self._init_error: str | None = None

    def _build_init_error_message(self) -> str:
        """Build actionable diagnostics for OptiX initialization failures."""
        optix_available = importlib.util.find_spec("optix") is not None
        optix_sdk_env = os.environ.get("OPTIX_SDK_INCLUDE_DIR")
        optix_sdk_auto = get_optix_include_dir()
        return (
            "PathTracerAPI initialization failed.\n"
            f"- python executable: {sys.executable}\n"
            f"- optix module importable: {optix_available}\n"
            f"- OPTIX_SDK_INCLUDE_DIR: {optix_sdk_env!r}\n"
            f"- auto-detected OptiX include dir: {optix_sdk_auto!r}\n"
            "Ensure `optix` is installed in THIS interpreter and OptiX SDK include path is valid."
        )

    @property
    def viewer(self) -> PathTracingRenderer:
        return self._viewer

    @property
    def scene(self):
        return self._viewer._scene  # internal, initialized by build()

    @property
    def usd_scene(self):
        """Retained path-addressable hierarchy from the last USD load."""
        scene = self.scene
        return None if scene is None else scene.usd_scene

    @property
    def dlss_enabled(self) -> bool:
        """Return whether DLSS Ray Reconstruction initialized successfully."""
        return bool(self._viewer._dlss_enabled)

    @property
    def dlss_init_error(self) -> str | None:
        """Return the DLSS initialization error, if Ray Reconstruction fell back."""
        return self._viewer._dlss_init_error

    @property
    def cuda_graph_active(self) -> bool:
        """Whether the stable OptiX launch has been captured for replay."""
        return self._viewer._optix_launch_graph is not None

    @property
    def dlss_quality(self) -> str:
        """Return the active DLSS quality-mode selection."""
        return self._viewer.dlss_quality

    @property
    def max_bounces(self) -> int:
        """Return the current maximum path depth."""
        return int(self._viewer.max_bounces)

    @property
    def max_compiled_bounces(self) -> int:
        """Return the largest runtime path depth supported by this pipeline."""
        return int(self._viewer._pipeline_max_bounces)

    @property
    def direct_light_samples(self) -> int:
        """Return the direct-light samples evaluated at each surface hit."""
        return int(self._viewer.direct_light_samples)

    @property
    def samples_per_frame(self) -> int:
        """Return the samples rendered per frame when DLSS is disabled."""
        return int(self._viewer.samples_per_frame)

    def set_dlss_quality(self, quality: str) -> None:
        """Select a DLSS quality mode and recreate DLSS resources if needed."""
        self._viewer.set_dlss_quality(quality)

    def set_ray_budget(
        self,
        *,
        max_bounces: int | None = None,
        direct_light_samples: int | None = None,
        samples_per_frame: int | None = None,
    ) -> None:
        """Adjust path depth and sampling budgets."""
        self._viewer.set_ray_budget(
            max_bounces=max_bounces,
            direct_light_samples=direct_light_samples,
            samples_per_frame=samples_per_frame,
        )

    @property
    def linear_depth_output(self):
        """Return the current positive view-space depth buffer."""
        return self._viewer.linear_depth_output

    @property
    def render_resolution(self) -> tuple[int, int]:
        """Return the internal render resolution as ``(width, height)``."""
        return self._viewer.render_resolution

    @property
    def tonemap_exposure(self) -> float:
        """Return the linear exposure multiplier."""
        return float(self._viewer._tonemapper.exposure)

    @tonemap_exposure.setter
    def tonemap_exposure(self, value: float) -> None:
        """Set the nonnegative linear exposure multiplier."""
        self._viewer._tonemapper.exposure = max(0.0, float(value))

    @property
    def tonemap_contrast(self) -> float:
        """Return the display contrast multiplier."""
        return float(self._viewer._tonemapper.contrast)

    @tonemap_contrast.setter
    def tonemap_contrast(self, value: float) -> None:
        """Set the nonnegative display contrast multiplier."""
        self._viewer._tonemapper.contrast = max(0.0, float(value))

    @property
    def tonemap_saturation(self) -> float:
        """Return the display saturation multiplier."""
        return float(self._viewer._tonemapper.saturation)

    @tonemap_saturation.setter
    def tonemap_saturation(self, value: float) -> None:
        """Set the nonnegative display saturation multiplier."""
        self._viewer._tonemapper.saturation = max(0.0, float(value))

    def _require_scene(self):
        """Ensure an initialized scene is available or raise a clear error."""
        if self._viewer._scene is None:
            ok = self.initialize()
            if (not ok) or self._viewer._scene is None:
                raise RuntimeError(self._init_error or self._build_init_error_message())
        return self._viewer._scene

    def initialize(self) -> bool:
        if self._built:
            return True
        self._built = bool(self._viewer.build())
        if not self._built:
            self._init_error = self._build_init_error_message()
        return self._built

    def is_running(self) -> bool:
        return self._running

    def close(self):
        self._running = False
        self._viewer.close()

    def begin_frame(self, time_sec: float):
        # Kept for ViewerBase API compatibility; rendering is driven explicitly.
        _ = float(time_sec)

    def end_frame(self):
        # Rendering is explicit in render_frame(); kept for API parity.
        return None

    def render_frame(self):
        self.initialize()
        self._viewer.render()

    def resize(self, width: int, height: int):
        self.initialize()
        self.width = int(width)
        self.height = int(height)
        self._viewer.resize(self.width, self.height)

    def get_frame(self) -> np.ndarray:
        self.initialize()
        return self._viewer.get_output()

    def get_frame_uint8(self) -> np.ndarray:
        image = np.clip(self.get_frame(), 0.0, 1.0)
        return (image * 255.0).astype(np.uint8)

    def build_scene(self):
        scene = self._require_scene()
        scene.build(self._viewer._optix)
        self._viewer._create_sbt()
        self._viewer._optix_launch_graph = None
        self._viewer._optix_graph_warmed = False
        self._viewer.sample_index = 0
        self._viewer.frame_index = 0
        self._viewer._dlss_reset_history = True
        self._viewer._prev_instance_transforms_valid = False
        self._viewer._sync_prev_camera_matrices_to_current()

    def reset_temporal_history(self):
        """Discard DLSS reconstruction history before a discontinuous scene change."""
        self._viewer._dlss_reset_history = True

    def rebuild_tlas(self):
        scene = self._require_scene()
        # Transform-only updates retain RNG progression and DLSS history; the
        # renderer supplies motion vectors from its previous transform snapshot.
        scene.rebuild_tlas()

    def clear_scene(self):
        self._require_scene().clear()

    def load_scene_from_gltf(
        self,
        gltf_path: str,
        root_transform: np.ndarray | None = None,
        clear_existing: bool = True,
        build_scene: bool = True,
    ) -> bool:
        ok = bool(
            self._require_scene().load_from_gltf(
                gltf_path,
                root_transform=root_transform,
                clear_existing=clear_existing,
            )
        )
        if ok and build_scene:
            self.build_scene()
        return ok

    def load_scene_from_usd(
        self,
        usd_path: str,
        root_transform: np.ndarray | None = None,
        clear_existing: bool = True,
        build_scene: bool = True,
        apply_stage_units: bool = True,
        convert_up_axis: bool = True,
        max_texture_size: int | None = None,
        strict_sidedness: bool = False,
        load_usd_environment: bool = False,
        usd_environment_scale: float = 1.0,
        enable_emissive_materials: bool = True,
    ) -> bool:
        """Load USD geometry and PBR materials through the optional OpenUSD bindings."""
        ok = bool(
            self._require_scene().load_from_usd(
                usd_path,
                root_transform=root_transform,
                clear_existing=clear_existing,
                apply_stage_units=apply_stage_units,
                convert_up_axis=convert_up_axis,
                max_texture_size=max_texture_size,
                strict_sidedness=strict_sidedness,
                enable_emissive_materials=enable_emissive_materials,
            )
        )
        if ok and load_usd_environment:
            environment_path = self._require_scene().usd_environment_path
            if environment_path is None:
                logger.warning(
                    "USD stage contains no supported DomeLight environment texture: %s",
                    usd_path,
                )
            else:
                self.set_environment_hdr(
                    str(environment_path), scaling=float(usd_environment_scale)
                )
        if ok and build_scene:
            self.build_scene()
        return ok

    def load_scene_from_obj(self, obj_path: str) -> bool:
        ok = bool(self._require_scene().load_from_obj(obj_path))
        if ok:
            self.build_scene()
        return ok

    def create_mesh(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        material_id: int = 0,
    ) -> int:
        scene = self._require_scene()
        if scene.materials.count == 0:
            scene.materials.add_diffuse((0.8, 0.8, 0.8))
        mat_id = int(material_id)
        if mat_id < 0 or mat_id >= scene.materials.count:
            mat_id = 0
        mesh = Mesh(
            vertices=np.asarray(positions, dtype=np.float32),
            indices=np.asarray(indices, dtype=np.uint32),
            normals=None if normals is None else np.asarray(normals, dtype=np.float32),
            texcoords=None if uvs is None else np.asarray(uvs, dtype=np.float32),
            material_id=mat_id,
        )
        return int(scene.add_mesh(mesh))

    def create_instance(self, mesh_id: int) -> int:
        return int(self._require_scene().add_instance(int(mesh_id)))

    def set_instance_material(self, instance_id: int, material_id: int | None):
        self._require_scene().set_instance_material(int(instance_id), material_id)

    def set_instance_visible(self, instance_id: int, visible: bool):
        self._require_scene().set_instance_visible(int(instance_id), bool(visible))

    def create_instance_with_transform(
        self,
        mesh_id: int,
        position: Iterable[float],
        rotation_xyzw: Iterable[float],
        scale: float | Iterable[float] = 1.0,
    ) -> int:
        transform = build_transform_matrix(position, rotation_xyzw, scale)
        return int(
            self._require_scene().add_instance(int(mesh_id), transform=transform)
        )

    def set_instance_transform(
        self,
        instance_id: int,
        position: Iterable[float],
        rotation_xyzw: Iterable[float],
        scale: float | Iterable[float] = 1.0,
    ):
        transform = build_transform_matrix(position, rotation_xyzw, scale)
        self._require_scene().set_instance_transform(int(instance_id), transform)

    def set_instance_transform_matrix(self, instance_id: int, matrix: np.ndarray):
        m = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
        self._require_scene().set_instance_transform(int(instance_id), m)

    def set_instance_transform_matrices(
        self, instance_ids: Iterable[int], matrices: np.ndarray
    ):
        self._require_scene().set_instance_transforms_batch(instance_ids, matrices)

    def set_instance_transform_arrays(
        self, instance_ids, xforms, scales, global_transform
    ) -> bool:
        """Update instance transforms from CUDA-resident Warp arrays."""
        return self._require_scene().set_instance_transforms_device(
            instance_ids, xforms, scales, global_transform
        )

    def set_instance_material_arrays(self, material_ids, colors, properties) -> bool:
        """Update compact materials from CUDA-resident Warp arrays."""
        return self._require_scene().set_instance_materials_device(
            material_ids, colors, properties
        )

    def set_instances_visible(self, instance_ids: Iterable[int], visible: bool):
        self._require_scene().set_instances_visible_batch(instance_ids, visible)

    def set_instance_transforms_batch(
        self, instance_ids: Iterable[int], transforms_flat: np.ndarray
    ):
        self.initialize()
        ids = list(instance_ids)
        arr = np.asarray(transforms_flat, dtype=np.float32).reshape(-1, 8)
        for i, instance_id in enumerate(ids):
            if i >= len(arr):
                break
            row = arr[i]
            self.set_instance_transform(
                int(instance_id),
                position=row[0:3],
                rotation_xyzw=row[3:7],
                scale=float(row[7]),
            )

    def create_diffuse_material(self, color: Iterable[float]) -> int:
        return int(
            self._require_scene().materials.add_diffuse(tuple(float(v) for v in color))
        )

    def create_metallic_material(
        self, color: Iterable[float], roughness: float = 0.1
    ) -> int:
        return int(
            self._require_scene().materials.add_metal(
                tuple(float(v) for v in color), float(roughness)
            )
        )

    def create_emissive_material(
        self, color: Iterable[float], intensity: float = 1.0
    ) -> int:
        return int(
            self._require_scene().materials.add_emissive(
                tuple(float(v) for v in color), float(intensity)
            )
        )

    def create_pbr_material(
        self,
        color: Iterable[float],
        roughness: float,
        metallic: float,
        ior: float = 1.5,
        specular: float = 1.0,
        clearcoat: float = 0.0,
        clearcoat_roughness: float = 0.1,
        u_subdiv: float = 0.0,
        v_subdiv: float = 0.0,
        base_color_scale: float = 0.9,
    ) -> int:
        scene = self._require_scene()
        return int(
            scene.materials.add_pbr(
                base_color=tuple(float(v) for v in color),
                roughness=float(roughness),
                metallic=float(metallic),
                ior=max(1.0, float(ior)),
                specular=min(1.0, max(0.0, float(specular))),
                clearcoat=min(1.0, max(0.0, float(clearcoat))),
                clearcoat_roughness=min(1.0, max(0.001, float(clearcoat_roughness))),
                u_subdiv=float(u_subdiv),
                v_subdiv=float(v_subdiv),
                base_color_scale=float(base_color_scale),
            )
        )

    def add_box(
        self, min_pt: Iterable[float], max_pt: Iterable[float], material_id: int
    ) -> int:
        return int(
            self._require_scene().add_box(
                tuple(float(v) for v in min_pt),
                tuple(float(v) for v in max_pt),
                int(material_id),
            )
        )

    def add_sphere(
        self, center: Iterable[float], radius: float, segments: int, material_id: int
    ) -> int:
        return int(
            self._require_scene().add_sphere(
                tuple(float(v) for v in center),
                float(radius),
                int(segments),
                int(material_id),
            )
        )

    def set_camera_look_at(
        self,
        position: Iterable[float],
        target: Iterable[float],
        up: Iterable[float] = (0.0, 1.0, 0.0),
        fov: float = 45.0,
    ):
        self.initialize()
        self._viewer.camera.position = np.asarray(list(position), dtype=np.float32)
        self._viewer.camera.target = np.asarray(list(target), dtype=np.float32)
        self._viewer.camera.up = np.asarray(list(up), dtype=np.float32)
        self._viewer.camera.fov = float(fov)

    def bind_device_camera(
        self,
        positions,
        targets,
        *,
        fov=45.0,
        up=(0.0, 1.0, 0.0),
        camera_transform=None,
    ):
        """Bind graph-written CUDA eye and target arrays to the path tracer."""
        self.initialize()
        self._viewer.bind_device_camera(
            positions, targets, fov=fov, up=up, camera_transform=camera_transform
        )

    def unbind_device_camera(self):
        """Restore the host-driven path tracer camera."""
        self._viewer.unbind_device_camera()

    def set_camera_angles(
        self,
        position: Iterable[float],
        yaw: float,
        pitch: float,
        fov: float = 45.0,
    ):
        self.initialize()
        yaw_rad = np.deg2rad(float(yaw))
        pitch_rad = np.deg2rad(float(pitch))
        direction = np.array(
            [
                np.sin(yaw_rad) * np.cos(pitch_rad),
                np.sin(pitch_rad),
                np.cos(yaw_rad) * np.cos(pitch_rad),
            ],
            dtype=np.float32,
        )
        pos = np.asarray(list(position), dtype=np.float32)
        self.set_camera_look_at(pos, pos + direction, (0.0, 1.0, 0.0), float(fov))

    def set_debug_buffer_mode(self, mode: int):
        self.initialize()
        self._viewer.output_mode = int(mode)

    def set_use_procedural_sky(self, enabled: bool):
        self.initialize()
        if enabled:
            self._viewer._env_map = None

    def set_sun_direction(self, x: float, y: float, z: float, intensity: float = 1.0):
        self.initialize()
        direction = np.array([x, y, z], dtype=np.float32)
        nrm = np.linalg.norm(direction)
        if nrm > 0.0:
            direction = direction / nrm
        self._viewer.sky_sun_direction = (
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
        )
        self._viewer.sky_multiplier = float(intensity)

    def set_sky_parameters(
        self,
        sun_direction: Iterable[float],
        multiplier: float = 1.0,
        haze: float = 0.0,
        red_blue_shift: float = 0.0,
        saturation: float = 1.0,
        horizon_height: float = 0.0,
        ground_color: Iterable[float] = (0.4, 0.4, 0.4),
        horizon_blur: float = 1.0,
        night_color: Iterable[float] = (0.0, 0.0, 0.0),
        sun_disk_intensity: float = 1.0,
        sun_disk_scale: float = 1.0,
        sun_glow_intensity: float = 1.0,
        y_is_up: int = 1,
        grayscale: bool = False,
    ):
        self.initialize()
        direction = np.asarray(tuple(float(v) for v in sun_direction), dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("sun_direction must be nonzero")
        direction /= norm
        self._viewer.sky_sun_direction = tuple(float(v) for v in direction)
        self._viewer.sky_multiplier = float(multiplier)
        self._viewer.sky_haze = float(haze)
        self._viewer.sky_redblueshift = float(red_blue_shift)
        self._viewer.sky_saturation = float(saturation)
        self._viewer.sky_horizon_height = float(horizon_height)
        self._viewer.sky_ground_color = tuple(float(v) for v in ground_color)
        self._viewer.sky_horizon_blur = float(horizon_blur)
        self._viewer.sky_night_color = tuple(float(v) for v in night_color)
        self._viewer.sky_sun_disk_intensity = float(sun_disk_intensity)
        self._viewer.sky_sun_disk_scale = float(sun_disk_scale)
        self._viewer.sky_sun_glow_intensity = float(sun_glow_intensity)
        self._viewer.sky_y_is_up = int(y_is_up)
        self._viewer.sky_grayscale = bool(grayscale)

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        self.initialize()
        self._viewer.set_environment_hdr(hdr_path, scaling=float(scaling))

    def set_environment_color(self, color: Iterable[float]):
        self.initialize()
        self._viewer.set_environment_color(tuple(float(v) for v in color))
