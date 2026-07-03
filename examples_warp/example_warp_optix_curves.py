# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a varying-width native OptiX round-linear curve with Warp shaders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import warp_optix as woptix


@wp.struct
class LaunchParams:
    image: wp.array(dtype=wp.uint32)
    curve_vertices: wp.array2d(dtype=wp.float32)
    curve_indices: wp.array(dtype=wp.uint32)
    width: wp.uint32
    height: wp.uint32
    num_segments: wp.uint32
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
    sx = (2.0 * (float(x) + 0.5) / float(width) - 1.0) * float(width) / float(height)
    sy = 2.0 * (float(y) + 0.5) / float(height) - 1.0

    payload = Payload()
    payload.red = wp.uint32(0)
    payload.green = wp.uint32(0)
    payload.blue = wp.uint32(0)
    wp.optix_trace(
        params.traversable,
        wp.vec3(0.0, 0.0, 4.0),
        wp.normalize(wp.vec3(sx, sy, -3.0)),
        0.001,
        1.0e16,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )
    params.image[y * width + x] = (
        wp.uint32(255) << wp.uint32(24)
        | payload.blue << wp.uint32(16)
        | payload.green << wp.uint32(8)
        | payload.red
    )


@woptix.optix_kernel(woptix.OptixKernelType.MISS)
def miss(params: LaunchParams):
    direction = wp.normalize(wp.optix_get_world_ray_direction())
    sky = 0.5 * (direction[1] + 1.0)
    _store_color(wp.vec3(0.94, 0.955, 0.98) * (1.0 - sky) + wp.vec3(0.995, 0.997, 1.0) * sky)


@woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
def curve_closest_hit(params: LaunchParams):
    primitive = int(wp.optix_get_primitive_index())
    vertex = int(params.curve_indices[primitive])
    u = wp.optix_get_curve_parameter()
    p0 = wp.vec3(
        params.curve_vertices[vertex, 0],
        params.curve_vertices[vertex, 1],
        params.curve_vertices[vertex, 2],
    )
    p1 = wp.vec3(
        params.curve_vertices[vertex + 1, 0],
        params.curve_vertices[vertex + 1, 1],
        params.curve_vertices[vertex + 1, 2],
    )
    center = p0 * (1.0 - u) + p1 * u
    world_center = wp.optix_transform_point_from_object_to_world_space(center)
    hit = wp.optix_get_world_ray_origin() + wp.optix_get_ray_tmax() * wp.optix_get_world_ray_direction()
    normal = wp.normalize(hit - world_center)

    along_curve = (float(primitive) + u) / float(params.num_segments)
    albedo = wp.vec3(0.15, 0.75, 0.95) * (1.0 - along_curve) + wp.vec3(0.95, 0.20, 0.55) * along_curve
    light = wp.normalize(wp.vec3(-0.4, 0.8, 0.6))
    intensity = 0.18 + 0.82 * wp.max(wp.dot(normal, light), 0.0)
    _store_color(albedo * intensity)


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
    parser.add_argument("--output", type=Path, default=Path("curves.bmp"))
    args = parser.parse_args()

    optix = woptix.require_optix()
    wp.init()
    with wp.ScopedDevice(args.device):
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise RuntimeError("This example requires a CUDA device")

        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__), "", "curves", __file__, device=args.device
        )
        cuda_context = device.context.value if hasattr(device.context, "value") else int(device.context)
        context, logger = woptix.create_context(optix, int(cuda_context))

        x = np.linspace(-1.65, 1.65, 49, dtype=np.float32)
        vertices = np.stack((x, 0.62 * np.sin(2.4 * x), 0.16 * np.cos(1.8 * x)), axis=1)
        widths = (0.075 + 0.035 * (0.5 + 0.5 * np.sin(4.0 * x))).astype(np.float32)
        segment_indices = np.arange(x.size - 1, dtype=np.uint32)
        curve_type = optix.PRIMITIVE_TYPE_ROUND_LINEAR
        gas, gas_buffers = woptix.create_curve_gas(
            optix,
            context,
            vertices,
            widths,
            segment_indices,
            args.device,
            curve_type=curve_type,
        )

        pipeline, sbt, pipeline_buffers = woptix.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            raygen,
            miss,
            None,
            num_payload_values=3,
            num_attribute_values=1,
            device=args.device,
            hit_groups=[
                woptix.HitKernel(
                    closest_hit=curve_closest_hit,
                    builtin_intersection_type=curve_type,
                )
            ],
        )

        image = wp.empty(args.width * args.height, dtype=wp.uint32, device=args.device)
        params = LaunchParams()
        params.image = image
        params.curve_vertices = gas_buffers["d_vertex_buffers"][0]
        params.curve_indices = gas_buffers["d_indices"]
        params.width = wp.uint32(args.width)
        params.height = wp.uint32(args.height)
        params.num_segments = wp.uint32(segment_indices.size)
        params.traversable = wp.uint64(gas)
        params_buffer = woptix.create_launch_params_buffer(LaunchParams, args.device)
        woptix.write_launch_params(params_buffer, params)
        woptix.launch(optix, pipeline, sbt, args.width, args.height, params_buffer)
        wp.synchronize_device(args.device)

        pixels = image.numpy()
        _save_bmp(args.output, pixels, args.width, args.height)
        _keepalive = (gas_buffers, pipeline_buffers)
        print(f"Wrote {args.output} (checksum {int(np.bitwise_xor.reduce(pixels)):#010x})")
        print(f"OptiX log messages: {logger.num_messages}")


if __name__ == "__main__":
    main()
