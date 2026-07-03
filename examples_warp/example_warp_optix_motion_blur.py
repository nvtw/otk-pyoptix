# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render vertex motion blur by tracing several shutter times per pixel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import warp_optix as woptix


@wp.struct
class LaunchParams:
    image: wp.array(dtype=wp.uint32)
    width: wp.uint32
    height: wp.uint32
    samples: wp.uint32
    traversable: wp.uint64


@wp.struct
class Payload:
    red: wp.uint32
    green: wp.uint32
    blue: wp.uint32


@wp.func
def _to_u8(value: float) -> wp.uint32:
    return wp.uint32(wp.clamp(value * 255.0, 0.0, 255.0))


@wp.func
def _store_color(color: wp.vec3):
    wp.optix_set_payload_0(_to_u8(color[0]))
    wp.optix_set_payload_1(_to_u8(color[1]))
    wp.optix_set_payload_2(_to_u8(color[2]))


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def raygen(params: LaunchParams):
    index = wp.optix_get_launch_index()
    x = int(index[0])
    y = int(index[1])
    width = int(params.width)
    height = int(params.height)
    samples = int(params.samples)
    sx = (2.0 * (float(x) + 0.5) / float(width) - 1.0) * float(width) / float(height)
    sy = 2.0 * (float(y) + 0.5) / float(height) - 1.0
    origin = wp.vec3(0.0, 0.0, 4.0)
    direction = wp.normalize(wp.vec3(sx, sy, -3.0))

    red = float(0.0)
    green = float(0.0)
    blue = float(0.0)
    for sample in range(samples):
        payload = Payload()
        payload.red = wp.uint32(0)
        payload.green = wp.uint32(0)
        payload.blue = wp.uint32(0)
        ray_time = (float(sample) + 0.5) / float(samples)
        wp.optix_trace(
            params.traversable,
            origin,
            direction,
            0.001,
            1.0e16,
            ray_time,
            wp.uint32(255),
            wp.uint32(0),
            wp.uint32(0),
            wp.uint32(1),
            wp.uint32(0),
            payload,
        )
        red += float(payload.red)
        green += float(payload.green)
        blue += float(payload.blue)

    scale = 1.0 / float(samples)
    red_u8 = wp.uint32(red * scale)
    green_u8 = wp.uint32(green * scale)
    blue_u8 = wp.uint32(blue * scale)
    params.image[y * width + x] = (
        wp.uint32(255) << wp.uint32(24)
        | blue_u8 << wp.uint32(16)
        | green_u8 << wp.uint32(8)
        | red_u8
    )


@woptix.optix_kernel(woptix.OptixKernelType.MISS)
def miss(params: LaunchParams):
    direction = wp.normalize(wp.optix_get_world_ray_direction())
    sky = 0.5 * (direction[1] + 1.0)
    _store_color(wp.vec3(0.025, 0.035, 0.06) * (1.0 - sky) + wp.vec3(0.10, 0.16, 0.25) * sky)


@woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
def closest_hit(params: LaunchParams):
    color = wp.vec3(0.88, 0.90, 0.96)
    if wp.optix_get_primitive_index() == wp.uint32(0):
        ray_time = wp.optix_get_ray_time()
        color = wp.vec3(0.05, 0.80, 1.0) * (1.0 - ray_time) + wp.vec3(1.0, 0.35, 0.04) * ray_time
    _store_color(color)


def _save_bmp(path: Path, pixels: np.ndarray, width: int, height: int) -> None:
    import struct  # noqa: PLC0415

    rgba = pixels.reshape(height, width)
    bgr = np.stack(((rgba >> 16) & 0xFF, (rgba >> 8) & 0xFF, rgba & 0xFF), axis=-1).astype(np.uint8)
    row_stride = (width * 3 + 3) & ~3
    pixel_data_size = row_stride * height
    padding = b"\x00" * (row_stride - width * 3)
    with path.open("wb") as output:
        output.write(struct.pack("<2sIHHI", b"BM", 54 + pixel_data_size, 0, 0, 54))
        output.write(struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0))
        for row in bgr:
            output.write(row.tobytes())
            output.write(padding)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=64, help="Temporal samples per pixel")
    parser.add_argument("--output", type=Path, default=Path("motion_blur.bmp"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    optix = woptix.require_optix()
    wp.init()
    with wp.ScopedDevice(args.device):
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise RuntimeError("This example requires a CUDA device")

        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__), "", "motion_blur", __file__, device=args.device
        )
        cuda_context = device.context.value if hasattr(device.context, "value") else int(device.context)
        context, logger = woptix.create_context(optix, int(cuda_context))

        moving = np.array([[-0.38, 0.12, 0.0], [0.38, 0.12, 0.0], [0.0, 0.92, 0.0]], dtype=np.float32)
        static = np.array([[-0.38, -1.0, 0.0], [0.38, -1.0, 0.0], [0.0, -0.20, 0.0]], dtype=np.float32)
        vertex_keys = np.stack(
            (
                np.concatenate((moving + np.array([-1.15, 0.0, 0.0], dtype=np.float32), static)),
                np.concatenate((moving + np.array([1.15, 0.0, 0.0], dtype=np.float32), static)),
            )
        )
        gas, gas_buffers = woptix.create_triangle_gas(
            optix,
            context,
            vertex_keys,
            np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32),
            args.device,
            motion_time_range=(0.0, 1.0),
        )
        pipeline, sbt, pipeline_buffers = woptix.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            raygen,
            miss,
            closest_hit,
            num_payload_values=3,
            num_attribute_values=2,
            device=args.device,
            uses_motion_blur=True,
        )

        image = wp.empty(args.width * args.height, dtype=wp.uint32, device=args.device)
        params = LaunchParams()
        params.image = image
        params.width = wp.uint32(args.width)
        params.height = wp.uint32(args.height)
        params.samples = wp.uint32(args.samples)
        params.traversable = wp.uint64(gas)
        params_buffer = woptix.create_launch_params_buffer(LaunchParams, args.device)
        woptix.write_launch_params(params_buffer, params)
        woptix.launch(optix, pipeline, sbt, args.width, args.height, params_buffer)
        wp.synchronize_device(args.device)

        pixels = image.numpy()
        _save_bmp(args.output, pixels, args.width, args.height)
        _keepalive = (gas_buffers, pipeline_buffers)
        print(f"Wrote {args.output} (checksum {int(np.bitwise_xor.reduce(pixels)):#010x})")
        print("Top: moving triangle. Bottom: static reference.")
        print(f"OptiX log messages: {logger.num_messages}")


if __name__ == "__main__":
    main()
