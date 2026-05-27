# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
OptiX Path Tracing Viewer.
Python/OptiX path tracing viewer with DLSS RR support.

This viewer renders a scene using OptiX ray tracing with PBR materials,
displaying raw buffers (radiance, normals, depth, etc.) for debugging.
"""

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import warp as wp
import warp_optix as woptix
from warp_optix._runtime.hit_kernels import HitKernel
from warp_optix._runtime.runtime import create_optix_context
from warp_optix._runtime.sbt import SbtKernelManager

from . import pathtracing_warp_kernels as pwk
from .camera import Camera
from .environment_map import EnvironmentMap
from .scene import Scene
from .tonemap import Tonemapper

try:
    import optix
except ImportError:
    optix = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Initialize warp
wp.init()

logger = logging.getLogger(__name__)


@wp.kernel
def _reset_accum_buffer(accum: wp.array2d(dtype=wp.vec4)):
    x, y = wp.tid()
    accum[y, x] = wp.vec4(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _accumulate_sample(
    sample: wp.array2d(dtype=wp.vec4),
    accum: wp.array2d(dtype=wp.vec4),
    sample_index: int,
):
    x, y = wp.tid()
    s = sample[y, x]
    a = accum[y, x]
    t = 1.0 / float(sample_index + 1)
    accum[y, x] = a + (s - a) * t


class PathTracingViewer:
    """
    OptiX Path Tracing Viewer.

    Renders a scene using hardware ray tracing with PBR materials.
    """

    # Output modes
    OUTPUT_FINAL = 0
    OUTPUT_RADIANCE = 1
    OUTPUT_DEPTH = 2
    OUTPUT_MOTION = 3
    OUTPUT_NORMAL = 4
    OUTPUT_ROUGHNESS = 5
    OUTPUT_DIFFUSE = 6
    OUTPUT_SPECULAR = 7
    OUTPUT_SPEC_HITDIST = 8

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        scene_setup: Optional[Callable[[Scene], None]] = None,
        camera: Optional[Camera] = None,
        accumulate_samples: bool = False,
        samples_per_frame: int = 1,
        max_bounces: int = 4,
        direct_light_samples: int = 1,
        use_halton_jitter: bool = True,
        enable_dlss_rr: bool = True,
        enable_set: bool = True,
    ):
        """
        Initialize the path tracing viewer.

        Args:
            width: Render width
            height: Render height
        """
        self.width = width
        self.height = height
        self._render_width = width
        self._render_height = height
        self.frame_index = 0
        self.sample_index = 0
        self.accumulate_samples = accumulate_samples
        self.samples_per_frame = max(1, int(samples_per_frame))
        self.max_bounces = max(1, int(max_bounces))
        self.direct_light_samples = max(1, int(direct_light_samples))
        self.use_halton_jitter = bool(use_halton_jitter)
        self.enable_dlss_rr = bool(enable_dlss_rr)
        self.enable_set = bool(enable_set)
        self._set_active = False

        # Camera
        if camera is None:
            self.camera = Camera(
                position=(0.0, 0.0, 6.0),
                target=(0.0, 0.0, 0.0),
                fov=45.0,
                aspect_ratio=width / height,
            )
        else:
            self.camera = camera
            self.camera.set_aspect_ratio(width, height)

        # Optional external scene configuration callback
        self._scene_setup = scene_setup

        # Default to path-traced final output.
        self.output_mode = self.OUTPUT_FINAL

        # OptiX state (initialized in build())
        self._optix = None
        self._ctx = None
        self._pipeline = None
        self._sbt = None
        self._ptx = None

        # Scene
        self._scene = None

        # Tonemapper
        self._tonemapper = Tonemapper(width, height)

        # Output buffers
        self._color_buffer = None
        self._accum_buffer = None
        self._normal_roughness_buffer = None
        self._motion_buffer = None
        self._depth_buffer = None
        self._diffuse_buffer = None
        self._specular_buffer = None
        self._spec_hit_dist_buffer = None
        self._dlss_output_buffer = None
        self._instance_transforms_buffer = None
        self._prev_instance_transforms_buffer = None
        self._prev_instance_transforms_valid = False

        # Launch params buffer — cached to avoid per-frame allocation
        self._launch_params_buffer = None
        self._launch_params = None
        self._instance_transform_count = 0

        # CUDA surface objects
        self._color_surface = None
        self._dlss_context = None
        self._dlss_denoiser = None
        self._dlss_color_in_tex = None
        self._dlss_normal_roughness_tex = None
        self._dlss_motion_tex = None
        self._dlss_depth_tex = None
        self._dlss_diffuse_tex = None
        self._dlss_specular_tex = None
        self._dlss_spec_hit_dist_tex = None
        self._dlss_color_out_tex = None
        self._dlss_output_surface = 0
        self._dlss_enabled = False
        self._dlss_reset_history = True
        self._dlss_init_error: str | None = None
        self._dlss_status_reported = False
        self._last_jitter = (0.0, 0.0)

        # Previous-frame camera matrices for motion vectors.
        self._prev_view = None
        self._prev_proj = None
        self._prev_mvp = None
        self._last_accum_view = None
        self._last_accum_proj = None
        self._sync_prev_camera_matrices_to_current()
        self._last_output_mode = self.output_mode

        # Physical sky defaults aligned with the upstream DLSS-RR sample behavior.
        self.sky_rgb_unit_conversion = (1.0 / 80000.0, 1.0 / 80000.0, 1.0 / 80000.0)
        self.sky_multiplier = 1.0
        self.sky_haze = 0.0
        self.sky_redblueshift = 0.0
        self.sky_saturation = 1.0
        self.sky_horizon_height = 0.0
        self.sky_ground_color = (0.4, 0.4, 0.4)
        self.sky_horizon_blur = 1.0
        self.sky_night_color = (0.0, 0.0, 0.0)
        self.sky_sun_disk_intensity = 1.0
        self.sky_sun_direction = (0.0, 1.0, 0.5)
        self.sky_sun_disk_scale = 1.0
        self.sky_sun_glow_intensity = 1.0
        self.sky_y_is_up = 1
        self.env_intensity = (1.0, 1.0, 1.0)
        self.env_rotation = 0.0
        self.use_path_regularization = True
        self.use_psr = True
        self.override_roughness = -1.0
        self.override_metallic = -1.0
        self.bitangent_flip = 1.0

        # Optional HDR environment map (lat-long, RGBA32F).
        self._env_map: EnvironmentMap | None = None

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        """
        Load an HDR environment map from disk.

        The environment map is used for image-based lighting with importance sampling.

        Args:
            hdr_path: Path to HDR file (.hdr format)
            scaling: Intensity multiplier (default 1.0)
        """
        env_map = EnvironmentMap()
        if env_map.load_from_file(hdr_path, scaling=scaling):
            self._env_map = env_map
        else:
            logger.warning("Failed to load HDR environment: %s", hdr_path)

    def set_environment_color(self, color: tuple[float, float, float]):
        """
        Set a uniform color environment (useful for debugging or simple scenes).

        Args:
            color: RGB color values
        """
        env_map = EnvironmentMap()
        if env_map.load_from_color(color):
            self._env_map = env_map

    def clear_environment_map(self):
        """Clear the HDR environment map and use procedural sky only."""
        self._env_map = None

    @property
    def tonemapped_output(self):
        """Return the tonemapped output buffer used for display/extraction."""
        return self._tonemapper.output

    def build(self):
        """Build the OptiX pipeline and scene."""
        logger.info("Initializing OptiX path tracing viewer.")

        if optix is None:
            logger.error("Could not import optix module. Ensure warp.pyoptix and OptiX SDK are installed.")
            return False

        self._optix = optix

        # Create OptiX context
        wp_device = wp.get_device("cuda")
        cu_context = wp_device.context.value if hasattr(wp_device.context, "value") else int(wp_device.context)
        self._ctx, self._optix_logger = create_optix_context(optix, int(cu_context))

        # Build PTX directly from Warp OptiX kernels.
        module = wp.get_module(pwk.__name__)
        self._ptx = woptix.compile_warp_module_to_ptx(
            module=module,
            launch_preamble="",
            module_tag="pathtracing_warp",
            script_dir=str(Path(__file__).parent),
            device="cuda",
        )
        self._set_active = False

        # Create scene
        self._scene = Scene(self._ctx)
        if self._scene_setup is not None:
            self._scene_setup(self._scene)
        else:
            self._scene.create_cornell_box()
        self._scene.build(optix)

        # Create output buffers
        self._create_buffers()
        self._launch_params = pwk.PathtraceLaunchParams()
        self._launch_params_buffer = woptix.create_launch_params_buffer(pwk.PathtraceLaunchParams, device="cuda")
        self._init_dlss_rr()

        self._create_pipeline()
        self._create_sbt()

        logger.info("OptiX path tracing viewer build complete.")
        return True

    @staticmethod
    def _create_cuda_texture_2d(
        height: int, width: int, channels: int, *, surface_access: bool = False
    ) -> wp.Texture2D:
        if channels == 1:
            data = np.zeros((height, width), dtype=np.float32)
        else:
            data = np.zeros((height, width, channels), dtype=np.float32)
        return wp.Texture2D(
            data,
            filter_mode=wp.TextureFilterMode.CLOSEST,
            address_mode=wp.TextureAddressMode.CLAMP,
            device="cuda",
            surface_access=surface_access,
        )

    @staticmethod
    def _half_res(value: int) -> int:
        # Keep dimensions even (where possible) to match common DLSS input expectations.
        v = max(1, int(value) // 2)
        if v > 1 and (v % 2) != 0:
            v -= 1
        return max(1, v)

    def _set_render_resolution(self, render_width: int, render_height: int):
        rw = max(1, int(render_width))
        rh = max(1, int(render_height))
        if rw == self._render_width and rh == self._render_height:
            return
        self._render_width = rw
        self._render_height = rh
        self._create_buffers()
        self.frame_index = 0
        self.sample_index = 0
        self._dlss_reset_history = True

    def _sync_prev_camera_matrices_to_current(self):
        """Initialize previous-frame camera transforms from the current camera pose.

        Mirrors reference first-frame behavior where prevMVP is set to currentMVP to avoid
        spurious large motion vectors after resets/resizes.
        """
        view = self.camera.get_view_matrix().copy()
        proj = self.camera.get_projection_matrix().copy()
        self._prev_view = view
        self._prev_proj = proj
        self._prev_mvp = (view @ proj).astype(np.float32)
        self._last_accum_view = view.copy()
        self._last_accum_proj = proj.copy()

    def _destroy_dlss_rr(self):
        # Surface object lifetime is owned by the Texture2D instance.
        # Clearing references lets texture cleanup release CUDA resources.
        self._dlss_output_surface = 0

        self._dlss_color_in_tex = None
        self._dlss_normal_roughness_tex = None
        self._dlss_motion_tex = None
        self._dlss_depth_tex = None
        self._dlss_diffuse_tex = None
        self._dlss_specular_tex = None
        self._dlss_spec_hit_dist_tex = None
        self._dlss_color_out_tex = None
        self._dlss_output_buffer = None

        if self._dlss_denoiser is not None:
            try:
                self._dlss_denoiser.deinit()
            except Exception as exc:
                logger.warning("Failed to deinitialize DLSS denoiser: %s", exc)
        self._dlss_denoiser = None

        if self._dlss_context is not None:
            try:
                self._dlss_context.deinit()
            except Exception as exc:
                logger.warning("Failed to deinitialize DLSS context: %s", exc)
        self._dlss_context = None
        self._dlss_enabled = False
        self._dlss_status_reported = False
        # If DLSS gets disabled at runtime, restore full-resolution rendering.
        self._set_render_resolution(self.width, self.height)

    def _init_dlss_rr(self):
        self._destroy_dlss_rr()
        self._dlss_init_error = None
        if not self.enable_dlss_rr or self._optix is None:
            return

        required = ("DlssRRContext", "DlssRRInitInfo", "DlssRRResource", "DlssPerfQuality")
        if not all(hasattr(self._optix, name) for name in required):
            logger.info("DLSS RR bindings not present in optix module.")
            return

        try:
            context = self._optix.DlssRRContext()
            context.init()
            if not context.isDlssRRAvailable():
                logger.info("DLSS RR not available on this system.")
                return

            init_info = self._optix.DlssRRInitInfo()
            render_width = self._half_res(self.width)
            render_height = self._half_res(self.height)
            init_info.inputWidth = int(render_width)
            init_info.inputHeight = int(render_height)
            init_info.outputWidth = int(self.width)
            init_info.outputHeight = int(self.height)
            # Prefer an upscaling profile; fallback if binding enum names differ.
            quality_enum = self._optix.DlssPerfQuality
            quality_name = "MAX_QUALITY"
            if not hasattr(quality_enum, quality_name):
                quality_name = "BALANCED" if hasattr(quality_enum, "BALANCED") else "DLAA"
            init_info.quality = getattr(quality_enum, quality_name)
            init_info.preset = self._optix.RayReconstructionHintRenderPreset.DEFAULT
            # Match reference behavior:
            # - MVJittered=false while still passing per-frame jitter to denoise()
            # - lowResolutionMotionVectors=true (motion vectors provided at render resolution)
            init_info.mvJittered = False
            init_info.lowResolutionMotionVectors = True
            init_info.isContentHDR = True
            init_info.depthInverted = False
            init_info.autoExposure = False
            init_info.useHWDepth = False

            # Ask NGX for the optimal input size for the chosen quality mode,
            # for the selected quality mode, and only fallback to half-res on failure.
            if hasattr(context, "querySupportedDlssInputSizes"):
                try:
                    sizes = context.querySupportedDlssInputSizes(int(self.width), int(self.height), init_info.quality)
                    ow = int(getattr(sizes, "optimalWidth", 0))
                    oh = int(getattr(sizes, "optimalHeight", 0))
                    if ow > 0 and oh > 0:
                        render_width = ow
                        render_height = oh
                        init_info.inputWidth = int(render_width)
                        init_info.inputHeight = int(render_height)
                except Exception as exc:
                    logger.warning("Failed to query optimal DLSS input size; using half-res fallback: %s", exc)

            denoiser = context.initDlssRR(init_info)
            self._set_render_resolution(render_width, render_height)

            self._dlss_color_in_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 4)
            self._dlss_normal_roughness_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 4)
            self._dlss_motion_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 2)
            self._dlss_depth_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 1)
            self._dlss_diffuse_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 4)
            self._dlss_specular_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 4)
            self._dlss_spec_hit_dist_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 1)
            self._dlss_color_out_tex = self._create_cuda_texture_2d(self.height, self.width, 4, surface_access=True)
            self._dlss_output_buffer = wp.zeros((self.height, self.width), dtype=wp.vec4, device="cuda")
            self._dlss_output_surface = self._dlss_color_out_tex.cuda_surface

            res = self._optix.DlssRRResource
            denoiser.setResource(res.RESOURCE_COLOR_IN, self._dlss_color_in_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_COLOR_OUT, self._dlss_output_surface)
            denoiser.setResource(res.RESOURCE_NORMALROUGHNESS, self._dlss_normal_roughness_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_MOTIONVECTOR, self._dlss_motion_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_LINEARDEPTH, self._dlss_depth_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_DIFFUSE_ALBEDO, self._dlss_diffuse_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_SPECULAR_ALBEDO, self._dlss_specular_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_SPECULAR_HITDISTANCE, self._dlss_spec_hit_dist_tex.cuda_texture)

            self._dlss_context = context
            self._dlss_denoiser = denoiser
            self._dlss_enabled = True
            self._dlss_reset_history = True
            self._dlss_status_reported = False
            logger.info(
                "DLSS Ray Reconstruction enabled (render=%dx%d, output=%dx%d).",
                self._render_width,
                self._render_height,
                self.width,
                self.height,
            )
        except Exception as exc:
            self._dlss_init_error = str(exc)
            logger.warning("Failed to initialize DLSS RR: %s", exc)
            self._destroy_dlss_rr()

    def _copy_linear_to_dlss_textures(self):
        if not self._dlss_enabled:
            return
        copies = (
            (self._color_buffer, self._dlss_color_in_tex),
            (self._normal_roughness_buffer, self._dlss_normal_roughness_tex),
            (self._motion_buffer, self._dlss_motion_tex),
            (self._depth_buffer, self._dlss_depth_tex),
            (self._diffuse_buffer, self._dlss_diffuse_tex),
            (self._specular_buffer, self._dlss_specular_tex),
            (self._spec_hit_dist_buffer, self._dlss_spec_hit_dist_tex),
        )
        for src_buffer, dst_tex in copies:
            dst_tex.copy_from_array(src_buffer)

    def _copy_dlss_output_to_color(self):
        if not self._dlss_enabled:
            return
        if self._dlss_output_buffer is None:
            return
        self._dlss_color_out_tex.copy_to_array(self._dlss_output_buffer)

    def _run_dlss_rr(self, reset: bool):
        if not self._dlss_enabled or self._dlss_denoiser is None:
            return False
        try:
            # Match upstream matrix packing in the Vulkan DLSS-RR sample:
            # output in column-major memory order (m11,m21,m31,m41,...).
            view_m = self.camera.get_view_matrix().astype(np.float32)
            proj_m = self.camera.get_projection_matrix().astype(np.float32)
            view = view_m.T.reshape(-1).tolist()
            proj = proj_m.T.reshape(-1).tolist()
            self._dlss_denoiser.denoise(
                int(self._render_width),
                int(self._render_height),
                float(-self._last_jitter[0]),
                float(-self._last_jitter[1]),
                view,
                proj,
                bool(reset or self._dlss_reset_history),
                int(0),
                int(0),
                float(1.0),
                float(1.0),
            )
            self._dlss_reset_history = False
            return True
        except Exception as exc:
            logger.warning("DLSS denoise failed; disabling DLSS RR: %s", exc)
            self._destroy_dlss_rr()
            return False

    def _create_buffers(self):
        """Create output buffers."""
        # HDR color buffer
        self._color_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._accum_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")

        # G-buffer outputs
        self._normal_roughness_buffer = wp.zeros(
            (self._render_height, self._render_width), dtype=wp.vec4, device="cuda"
        )
        self._motion_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec2, device="cuda")
        self._depth_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.float32, device="cuda")
        self._diffuse_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._specular_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._spec_hit_dist_buffer = wp.zeros(
            (self._render_height, self._render_width), dtype=wp.float32, device="cuda"
        )

    def _update_instance_transform_buffers(self):
        """Upload current instance transforms for motion vectors.

        The *previous* transforms buffer is populated by
        :meth:`_snapshot_instance_transforms` which copies the current
        buffer to the previous buffer **after** each rendered frame.
        This guarantees that ``prev`` always holds exactly the transforms
        that were used for the last rendered frame.
        """
        if self._scene is None or self._scene.instance_count == 0:
            self._instance_transforms_buffer = None
            self._prev_instance_transforms_buffer = None
            self._instance_transform_count = 0
            return

        instances = getattr(self._scene, "_instances", None)
        if not instances:
            self._instance_transforms_buffer = None
            self._prev_instance_transforms_buffer = None
            self._instance_transform_count = 0
            return

        count = len(instances)

        # (Re)allocate GPU buffers when the instance count changes.
        if count != self._instance_transform_count:
            self._instance_xform_curr_np = np.empty((count, 12), dtype=np.float32)
            self._instance_transforms_buffer = wp.empty(count * 12, dtype=wp.float32, device="cuda")
            self._prev_instance_transforms_buffer = wp.empty(count * 12, dtype=wp.float32, device="cuda")
            self._instance_transform_count = count
            self._prev_instance_transforms_valid = False

        curr = self._instance_xform_curr_np
        for i, inst in enumerate(instances):
            m = np.asarray(inst.transform, dtype=np.float32).reshape(4, 4)
            curr[i, 0:4] = m[0, :]
            curr[i, 4:8] = m[1, :]
            curr[i, 8:12] = m[2, :]

        self._instance_transforms_buffer.assign(curr.reshape(-1))

        # First frame (or after resize): prev == current so motion is zero.
        if not self._prev_instance_transforms_valid:
            wp.copy(self._prev_instance_transforms_buffer, self._instance_transforms_buffer)
            self._prev_instance_transforms_valid = True

    def _snapshot_instance_transforms(self):
        """Copy current instance transforms to the previous-frame buffer on GPU.

        Must be called once per frame **after** the OptiX launch so that the
        next frame sees the correct previous-frame transforms for rigid-body
        motion vectors.
        """
        if (
            self._instance_transforms_buffer is not None
            and self._prev_instance_transforms_buffer is not None
            and self._instance_transforms_buffer.shape == self._prev_instance_transforms_buffer.shape
        ):
            wp.copy(self._prev_instance_transforms_buffer, self._instance_transforms_buffer)

    def _create_pipeline(self):
        """Create the OptiX pipeline."""
        optix = self._optix
        pipeline_kwargs = {
            "usesMotionBlur": False,
            "traversableGraphFlags": int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING),
            "numPayloadValues": 21,
            "numAttributeValues": 2,
            "exceptionFlags": int(optix.EXCEPTION_FLAG_NONE),
            "pipelineLaunchParamsVariableName": "params",
        }
        if optix.version()[1] >= 2:
            pipeline_kwargs["usesPrimitiveTypeFlags"] = int(optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
        pco = optix.PipelineCompileOptions(**pipeline_kwargs)

        mco = optix.ModuleCompileOptions(
            maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )
        module_result = self._ctx.moduleCreate(mco, pco, self._ptx)
        if isinstance(module_result, tuple):
            self._module = module_result[0]
        else:
            self._module = module_result

        # Match C++ SBT layout: 2 ray subtypes (primary + secondary/shadow).
        # Primary (offset 0): closest hit only, no anyhit — used by primary AND bounce rays.
        # Secondary (offset 1): closest hit + anyhit (simple block) — used by shadow/visibility rays.
        self._sbt_manager = SbtKernelManager(optix, self._ctx, self._module, num_ray_subtypes=2)
        self._sbt_manager.set_raygen_kernel(woptix.get_entry_name(pwk.primary_raygen, woptix.OptixKernelType.RAYGEN))
        self._sbt_manager.add_miss_kernels(
            [
                woptix.get_entry_name(pwk.primary_miss, woptix.OptixKernelType.MISS),
                woptix.get_entry_name(pwk.secondary_miss, woptix.OptixKernelType.MISS),
            ]
        )
        self._sbt_manager.register_hit_shader_type(
            HitKernel(
                woptix.get_entry_name(pwk.primary_closest_hit, woptix.OptixKernelType.CLOSEST_HIT),
            ),
            HitKernel(
                woptix.get_entry_name(pwk.secondary_closest_hit, woptix.OptixKernelType.CLOSEST_HIT),
                any_hit=woptix.get_entry_name(pwk.secondary_any_hit, woptix.OptixKernelType.ANY_HIT),
            ),
        )

        plo = optix.PipelineLinkOptions()
        trace_depth = max(2, int(self.max_bounces) + 1)
        plo.maxTraceDepth = trace_depth
        groups = self._sbt_manager.get_all_program_groups()
        self._pipeline = self._ctx.pipelineCreate(
            pco,
            plo,
            groups,
            "",
        )
        # Single-level instancing => traversable graph depth must stay at 2 (TLAS -> GAS).
        self._pipeline.setStackSize(2048, 2048, 2048, 2)

    def _create_sbt(self):
        """Create the Shader Binding Table."""
        sbt_resources = self._sbt_manager.build_sbt(device="cuda")
        self._sbt = sbt_resources.sbt
        self._sbt_keepalive = sbt_resources.keepalive

    @staticmethod
    def _halton(index: int, base: int) -> float:
        f = 1.0
        r = 0.0
        i = max(0, int(index))
        b = max(2, int(base))
        while i > 0:
            f /= float(b)
            r += f * float(i % b)
            i //= b
        return r

    def _compute_camera_basis(self):
        pos = np.asarray(self.camera.position, dtype=np.float32)
        target = np.asarray(self.camera.target, dtype=np.float32)
        world_up = np.asarray(self.camera.up, dtype=np.float32)

        forward = target - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1.0e-8:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward /= forward_norm

        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1.0e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= right_norm

        up = np.cross(right, forward)
        up_norm = np.linalg.norm(up)
        if up_norm < 1.0e-8:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            up /= up_norm

        return pos, forward, right, up

    def _update_launch_params(
        self,
        frame_index_override: int | None = None,
        *,
        update_instance_transforms: bool = True,
        view: np.ndarray | None = None,
        proj: np.ndarray | None = None,
        view_inv: np.ndarray | None = None,
        proj_inv: np.ndarray | None = None,
    ):
        """Update launch parameters for the current frame."""
        if update_instance_transforms:
            self._update_instance_transform_buffers()

        if view is None:
            view = self.camera.get_view_matrix()
        if proj is None:
            proj = self.camera.get_projection_matrix()
        if view_inv is None:
            view_inv = np.linalg.inv(view)
        if proj_inv is None:
            proj_inv = np.linalg.inv(proj)

        if self._launch_params is None:
            self._launch_params = pwk.PathtraceLaunchParams()

        p = self._launch_params
        p.tlas = wp.uint64(self._scene.tlas_handle)
        p.width = wp.uint32(self._render_width)
        p.height = wp.uint32(self._render_height)
        frame_index_value = self.sample_index if frame_index_override is None else int(frame_index_override)
        p.frame_index = wp.uint32(frame_index_value)
        p.max_bounces = wp.uint32(self.max_bounces)
        p.direct_light_samples = wp.uint32(self.direct_light_samples)
        p.output_mode = int(self.OUTPUT_FINAL)

        if self.use_halton_jitter:
            jitter_x = self._halton(frame_index_value, 2) - 0.5
            jitter_y = self._halton(frame_index_value, 3) - 0.5
            self._last_jitter = (float(jitter_x), float(jitter_y))
        else:
            self._last_jitter = (0.0, 0.0)

        pos, forward, right, up = self._compute_camera_basis()
        p.cam_pos = wp.vec3(float(pos[0]), float(pos[1]), float(pos[2]))
        p.cam_forward = wp.vec3(float(forward[0]), float(forward[1]), float(forward[2]))
        p.cam_right = wp.vec3(float(right[0]), float(right[1]), float(right[2]))
        p.cam_up = wp.vec3(float(up[0]), float(up[1]), float(up[2]))
        p.cam_tan_half_fov = float(np.tan(np.deg2rad(self.camera.fov * 0.5)))
        p.cam_aspect = float(self._render_width) / float(max(1, self._render_height))
        p.jitter = wp.vec2(float(self._last_jitter[0]), float(self._last_jitter[1]))
        p.view = tuple(np.asarray(view, dtype=np.float32).reshape(-1))
        p.proj = tuple(np.asarray(proj, dtype=np.float32).reshape(-1))
        p.view_inv = tuple(np.asarray(view_inv, dtype=np.float32).reshape(-1))
        p.proj_inv = tuple(np.asarray(proj_inv, dtype=np.float32).reshape(-1))
        p.prev_mvp = tuple(np.asarray(self._prev_mvp, dtype=np.float32).reshape(-1))
        p.env_intensity = wp.vec3(
            float(self.env_intensity[0]),
            float(self.env_intensity[1]),
            float(self.env_intensity[2]),
        )
        p.env_rotation = float(self.env_rotation)
        flags = 0
        if self._env_map is None:
            flags |= 1
        if self.use_psr:
            flags |= 2
        if self.use_path_regularization:
            flags |= 4
        p.flags = wp.uint32(flags)
        p.override_roughness = float(self.override_roughness)
        p.override_metallic = float(self.override_metallic)
        p.bitangent_flip = float(self.bitangent_flip)
        p.use_procedural_sky = wp.uint32(1 if self._env_map is None else 0)
        p.env_map = None if self._env_map is None else self._env_map._env_map_buffer
        p.env_map_length = wp.uint32(0 if self._env_map is None else self._env_map.width * self._env_map.height * 4)
        p.env_map_width = wp.uint32(0 if self._env_map is None else self._env_map.width)
        p.env_map_height = wp.uint32(0 if self._env_map is None else self._env_map.height)
        p.env_accel = (
            None
            if self._env_map is None or self._env_map._env_accel_buffer is None
            else wp.array(
                ptr=self._env_map._env_accel_buffer.ptr,
                shape=(self._env_map.accel_count,),
                dtype=pwk.EnvAccel,
            )
        )
        p.env_accel_count = wp.uint32(0 if self._env_map is None else self._env_map.accel_count)

        sky = pwk.PhysicalSkyParams()
        sky.rgb_unit_conversion = wp.vec3(*self.sky_rgb_unit_conversion)
        sky.multiplier = float(self.sky_multiplier)
        sky.haze = float(self.sky_haze)
        sky.redblueshift = float(self.sky_redblueshift)
        sky.saturation = float(self.sky_saturation)
        sky.horizon_height = float(self.sky_horizon_height)
        sky.ground_color = wp.vec3(*self.sky_ground_color)
        sky.horizon_blur = float(self.sky_horizon_blur)
        sky.night_color = wp.vec3(*self.sky_night_color)
        sky.sun_disk_intensity = float(self.sky_sun_disk_intensity)
        sd = self.sky_sun_direction
        sd_len = (sd[0] ** 2 + sd[1] ** 2 + sd[2] ** 2) ** 0.5
        if sd_len > 1.0e-8:
            sd = (sd[0] / sd_len, sd[1] / sd_len, sd[2] / sd_len)
        sky.sun_direction = wp.vec3(*sd)
        sky.sun_disk_scale = float(self.sky_sun_disk_scale)
        sky.sun_glow_intensity = float(self.sky_sun_glow_intensity)
        sky.y_is_up = int(self.sky_y_is_up)
        p.sky = sky

        p.render_primitives = (
            None
            if self._scene._render_primitives is None or self._scene.mesh_count == 0
            else wp.array(
                ptr=self._scene._render_primitives.ptr,
                shape=(self._scene.mesh_count,),
                dtype=pwk.RenderPrimitive,
            )
        )
        p.render_prim_count = wp.uint32(self._scene.mesh_count)
        p.instance_render_prim_ids = self._scene._instance_render_prim_ids
        p.instance_material_ids = self._scene._instance_material_ids
        p.instance_count = wp.uint32(self._scene.instance_count)
        p.instance_transforms = (
            None
            if self._instance_transforms_buffer is None
            else wp.array(
                ptr=self._instance_transforms_buffer.ptr,
                shape=(self._scene.instance_count,),
                dtype=pwk.TransformMatrix3x4,
            )
        )
        p.prev_instance_transforms = (
            None
            if self._prev_instance_transforms_buffer is None
            else wp.array(
                ptr=self._prev_instance_transforms_buffer.ptr,
                shape=(self._scene.instance_count,),
                dtype=pwk.TransformMatrix3x4,
            )
        )
        p.compact_materials = (
            None
            if self._scene._compact_materials is None or self._scene.materials.count == 0
            else wp.array(
                ptr=self._scene._compact_materials.ptr,
                shape=(self._scene.materials.count,),
                dtype=pwk.CompactMaterial,
            )
        )
        p.packed_indices = self._scene._packed_indices
        p.packed_normals = self._scene._packed_normals
        p.packed_tangents = self._scene._packed_tangents
        p.packed_texcoords0 = self._scene._packed_texcoords0
        p.packed_texcoords1 = self._scene._packed_texcoords1
        p.packed_prev_positions = self._scene._packed_prev_positions
        p.packed_material_ids = self._scene._packed_material_ids
        p.material_count = wp.uint32(self._scene.materials.count)
        p.texture_descs = (
            None
            if self._scene._texture_descs is None or self._scene.texture_count == 0
            else wp.array(
                ptr=self._scene._texture_descs.ptr,
                shape=(self._scene.texture_count,),
                dtype=pwk.TextureDesc,
            )
        )
        p.texture_data = self._scene._texture_data
        p.texture_count = wp.uint32(self._scene.texture_count)
        p.texture_data_length = wp.uint32(
            0 if self._scene._texture_data is None else self._scene._texture_data.shape[0]
        )

        p.color_output = self._color_buffer
        p.normal_roughness_output = self._normal_roughness_buffer
        p.motion_output = self._motion_buffer
        p.depth_output = self._depth_buffer
        p.diffuse_output = self._diffuse_buffer
        p.specular_output = self._specular_buffer
        p.spec_hit_dist_output = self._spec_hit_dist_buffer

        woptix.write_launch_params(self._launch_params_buffer, p)

    def _update_temporal_state(
        self, current_view: np.ndarray, current_proj: np.ndarray, use_external_accum: bool
    ) -> bool:
        """Update accumulation state and return whether temporal history resets."""
        if self._dlss_enabled:
            return False

        reset_temporal = (
            self.output_mode != self._last_output_mode
            or (not np.allclose(current_view, self._last_accum_view))
            or (not np.allclose(current_proj, self._last_accum_proj))
        )

        if use_external_accum:
            if reset_temporal:
                wp.launch(
                    _reset_accum_buffer,
                    dim=(self._render_width, self._render_height),
                    inputs=[self._accum_buffer],
                    device="cuda",
                )
                self.frame_index = 0
            return bool(reset_temporal)

        # No persistent external accumulation for this mode.
        wp.launch(
            _reset_accum_buffer,
            dim=(self._render_width, self._render_height),
            inputs=[self._accum_buffer],
            device="cuda",
        )
        return bool(reset_temporal)

    def _launch_samples(
        self,
        samples_this_frame: int,
        use_external_accum: bool,
        *,
        view: np.ndarray,
        proj: np.ndarray,
        view_inv: np.ndarray,
        proj_inv: np.ndarray,
    ):
        """Launch OptiX path tracing and optional external accumulation kernels."""
        for s in range(samples_this_frame):
            launch_frame_index = self.sample_index + s
            self._update_launch_params(
                frame_index_override=launch_frame_index,
                update_instance_transforms=False,
                view=view,
                proj=proj,
                view_inv=view_inv,
                proj_inv=proj_inv,
            )

            woptix.launch(
                self._optix,
                self._pipeline,
                self._sbt,
                self._render_width,
                self._render_height,
                self._launch_params_buffer,
                stream=0,
            )

            if not self._dlss_enabled:
                accum_sample_index = int(self.frame_index if use_external_accum else s)
                wp.launch(
                    _accumulate_sample,
                    dim=(self._render_width, self._render_height),
                    inputs=[self._color_buffer, self._accum_buffer, accum_sample_index],
                    device="cuda",
                )

                if use_external_accum:
                    self.frame_index += 1

    def _process_debug_output(self):
        self._tonemapper.resize(self.width, self.height)
        self._tonemapper.process_debug(
            self.output_mode,
            self._color_buffer,
            self._depth_buffer,
            self._motion_buffer,
            self._normal_roughness_buffer,
            self._diffuse_buffer,
            self._specular_buffer,
            self._spec_hit_dist_buffer,
            self._render_width,
            self._render_height,
        )

    def _process_final_output(self, source_buffer, *, resize_to_render: bool):
        if resize_to_render:
            self._tonemapper.resize(self._render_width, self._render_height)
        else:
            self._tonemapper.resize(self.width, self.height)
        self._tonemapper.process(source_buffer)

    def _process_output(self, source_buffer, *, resize_final_to_render: bool):
        if self.output_mode == self.OUTPUT_FINAL:
            self._process_final_output(source_buffer, resize_to_render=resize_final_to_render)
            return
        self._process_debug_output()

    def render(self):
        """Render a frame."""
        if self._pipeline is None:
            logger.error("Pipeline not built. Call build() first.")
            return

        if not self._dlss_status_reported:
            if self.enable_dlss_rr and self._dlss_enabled:
                logger.warning("DLSS RR active.")
            elif self.enable_dlss_rr and not self._dlss_enabled:
                reason = self._dlss_init_error if self._dlss_init_error else "unknown initialization failure"
                logger.warning("DLSS RR requested but inactive: %s", reason)
            else:
                logger.warning("DLSS RR disabled by configuration.")
            self._dlss_status_reported = True

        current_view = self.camera.get_view_matrix().copy()
        current_proj = self.camera.get_projection_matrix().copy()
        current_view_inv = np.linalg.inv(current_view)
        current_proj_inv = np.linalg.inv(current_proj)
        use_external_accum = self.accumulate_samples and not self._dlss_enabled
        samples_this_frame = 1 if self._dlss_enabled else self.samples_per_frame
        reset_temporal = self._update_temporal_state(current_view, current_proj, use_external_accum)
        self._update_instance_transform_buffers()
        self._launch_samples(
            samples_this_frame,
            use_external_accum,
            view=current_view,
            proj=current_proj,
            view_inv=current_view_inv,
            proj_inv=current_proj_inv,
        )

        # Ensure OptiX work is complete before publishing current transforms as "previous".
        wp.synchronize_device("cuda")

        # Snapshot current instance transforms -> previous for next frame's
        # rigid-body motion vectors.  This is a GPU-side copy so it is cheap.
        self._snapshot_instance_transforms()

        # Keep previous matrices for next frame's motion-vector calculation.
        self._prev_view = current_view.copy()
        self._prev_proj = current_proj.copy()
        self._prev_mvp = (current_view @ current_proj).astype(np.float32)
        self._last_accum_view = current_view.copy()
        self._last_accum_proj = current_proj.copy()
        self._last_output_mode = self.output_mode

        if self._dlss_enabled:
            # Single sync: ensure OptiX launch + Warp kernel writes are visible
            # before copying into DLSS texture resources and running DLSS.
            wp.synchronize_device("cuda")
            self._copy_linear_to_dlss_textures()
            if self._run_dlss_rr(reset_temporal):
                # Single sync: ensure DLSS writes are complete before reading output.
                wp.synchronize_device("cuda")
                self._copy_dlss_output_to_color()
                if self._dlss_output_buffer is not None:
                    self._process_output(self._dlss_output_buffer, resize_final_to_render=False)
                else:
                    self._process_output(self._color_buffer, resize_final_to_render=False)
            else:
                self._process_output(self._color_buffer, resize_final_to_render=True)
        else:
            self._process_output(self._accum_buffer, resize_final_to_render=True)
        self.sample_index += samples_this_frame

    def get_output(self) -> np.ndarray:
        """Get the current output as a numpy array."""
        wp.synchronize_device("cuda")
        return self._tonemapper.get_numpy()

    def resize(self, width: int, height: int):
        """Resize the render buffers."""
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self.camera.set_aspect_ratio(width, height)
            self._sync_prev_camera_matrices_to_current()
            self._set_render_resolution(width, height)
            self._tonemapper.resize(width, height)
            self._init_dlss_rr()
            self.frame_index = 0

    def __del__(self):
        self._destroy_dlss_rr()


def main():
    """Run the path tracing viewer."""
    logger.info("%s", "=" * 60)
    logger.info("OptiX Path Tracing Viewer")
    logger.info("%s", "=" * 60)

    viewer = PathTracingViewer(width=800, height=600)

    if not viewer.build():
        logger.error("Failed to build viewer.")
        return 1

    # Render a few frames
    logger.info("Rendering frames.")
    for i in range(10):
        viewer.render()
        logger.info("Frame %d", i + 1)

    # Get final output
    output = viewer.get_output()
    logger.info("Output shape: %s", output.shape)
    logger.info("Output range: [%.3f, %.3f]", float(output.min()), float(output.max()))

    # Save to file if possible
    if Image is not None:
        img_data = (output[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        img.save("pathtracing_output.png")
        logger.info("Saved output to pathtracing_output.png")
    else:
        logger.info("Pillow not installed; skipping image save.")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
