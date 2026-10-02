# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline quality scoring using the same OptiX sampler as rendering."""

import numpy as np
import warp as wp
import warp_optix as wo

from .device import NeuralTextureView, neural_texture_sample_texel
from .runtime import upload_asset


@wp.struct
class _Params:
    texture: NeuralTextureView
    target: wp.array2d(dtype=wp.float32)
    indices: wp.array(dtype=wp.int32)
    errors: wp.array(dtype=wp.float32)
    stride: int
    channels: int


@wo.optix_kernel(wo.OptixKernelType.RAYGEN)
def _score_texels(params: _Params):
    i = int(wp.optix_get_launch_index()[0])
    pixel = params.indices[i]
    value = neural_texture_sample_texel(
        params.texture, pixel % params.texture.width, pixel // params.texture.width
    )
    for channel in range(params.channels):
        difference = wp.float32(value[channel]) - params.target[pixel, channel]
        params.errors[channel * params.stride + i] = difference * difference


@wo.optix_kernel(wo.OptixKernelType.MISS)
def _miss(params: _Params):
    _ = params.channels


@wp.kernel(enable_backward=False, module="unique")
def _sample_indices(indices: wp.array(dtype=wp.int32), pixels: int):
    i = wp.tid()
    # One deterministic random pixel per disjoint interval; no duplicate samples.
    begin = int(wp.int64(i) * wp.int64(pixels) // wp.int64(indices.shape[0]))
    end = int(wp.int64(i + 1) * wp.int64(pixels) // wp.int64(indices.shape[0]))
    state = wp.rand_init(0, i)
    indices[i] = wp.randi(state, begin, end)


@wp.kernel(enable_backward=False, module="unique")
def _sum_errors(
    errors: wp.array(dtype=wp.float32), stride: int, sums: wp.array(dtype=wp.float32)
):
    block, channel = wp.tid()
    values = wp.tile_load(
        errors, shape=(256,), offset=(channel * stride + block * 256,)
    )
    wp.tile_atomic_add(sums, wp.tile_sum(values), offset=(channel,))


class _QualityEvaluator:
    """Reuse one source upload and scoring pipeline across compression candidates."""

    def __init__(self, image: np.ndarray, *, sampled: bool, device: str):
        self.device = device
        height, width, self.channels = image.shape
        self.count = min(height * width, 65536) if sampled else height * width
        self.stride = ((self.count + 255) // 256) * 256
        self.context = None
        try:
            self._initialize(image, height * width)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.context is not None:
            # OptiX destroys child pipelines, groups and modules too, including
            # objects created during an incomplete initialization.
            with wp.ScopedDevice(self.device):
                try:
                    wp.synchronize_device(self.device)
                finally:
                    self.context.destroy()
                    self.context = None

    def _initialize(self, image, pixels):
        device = self.device
        with wp.ScopedDevice(device):
            self.optix = wo.require_optix()
            cuda_device = wp.get_device(device)
            context = cuda_device.context
            handle = context.value if hasattr(context, "value") else int(context)
            self.context, self.callback = wo.create_context(
                self.optix, handle, log_level=1
            )
            flags = self.context.getProperty(self.optix.DEVICE_PROPERTY_COOP_VEC)
            if not flags & int(self.optix.DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD):
                raise RuntimeError(
                    "the selected OptiX device does not support cooperative vectors"
                )
            self.target = wp.array(
                image.reshape(-1, self.channels), dtype=wp.float32, device=device
            )
            self.indices = wp.empty(self.count, dtype=wp.int32, device=device)
            wp.launch(
                _sample_indices,
                dim=self.count,
                inputs=[self.indices, pixels],
                device=device,
            )
            self.errors = wp.zeros(
                self.stride * self.channels, dtype=wp.float32, device=device
            )
            self.sums = wp.zeros(self.channels, dtype=wp.float32, device=device)
            ptx = wo.compile_warp_module_to_ptx(
                wp.get_module(__name__),
                "",
                "neural_texture_quality",
                __file__,
                device=device,
            )
            self.pipeline, self.sbt, self.resources = wo.create_pipeline_and_sbt(
                self.optix,
                self.context,
                ptx,
                _score_texels,
                _miss,
                None,
                num_payload_values=0,
                num_attribute_values=0,
                device=device,
            )
            self.params_buffer = wo.create_launch_params_buffer(_Params, device)

    def mse(self, asset):
        with wp.ScopedDevice(self.device):
            runtime = upload_asset(asset, self.context, self.optix, device=self.device)
            params = _Params()
            params.texture = runtime.device_view()
            params.target = self.target
            params.indices = self.indices
            params.errors = self.errors
            params.stride = self.stride
            params.channels = self.channels
            wo.write_launch_params(self.params_buffer, params)
            self.sums.zero_()
            stream = wp.get_stream(self.device)
            wo.launch(
                self.optix,
                self.pipeline,
                self.sbt,
                self.count,
                1,
                self.params_buffer,
                stream.cuda_stream,
            )
            wp.launch_tiled(
                _sum_errors,
                dim=(self.stride // 256, self.channels),
                inputs=[self.errors, self.stride],
                outputs=[self.sums],
                block_dim=128,
                device=self.device,
            )
            # Synchronizes before the temporary runtime allocations go out of scope.
            return self.sums.numpy().astype(np.float64) / self.count
