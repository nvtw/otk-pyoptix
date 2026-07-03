# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a triangle and a procedural sphere through one OptiX IAS."""

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


@wp.func
def _shade(normal: wp.vec3, albedo: wp.vec3) -> wp.vec3:
    light = wp.normalize(wp.vec3(-0.4, 0.8, 0.6))
    intensity = 0.18 + 0.82 * wp.max(wp.dot(wp.normalize(normal), light), 0.0)
    return albedo * intensity


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def raygen(params: LaunchParams):
    index = wp.optix_get_launch_index()
    x = int(index[0])
    y = int(index[1])
    width = int(params.width)
    height = int(params.height)

    sx = (2.0 * (float(x) + 0.5) / float(width) - 1.0) * float(width) / float(height)
    sy = 2.0 * (float(y) + 0.5) / float(height) - 1.0
    origin = wp.vec3(0.0, 0.0, 3.0)
    direction = wp.normalize(wp.vec3(sx, sy, -2.2))

    payload = Payload()
    payload.red = wp.uint32(0)
    payload.green = wp.uint32(0)
    payload.blue = wp.uint32(0)
    wp.optix_trace(
        params.traversable,
        origin,
        direction,
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

    rgba = (
        wp.uint32(255) << wp.uint32(24)
        | payload.blue << wp.uint32(16)
        | payload.green << wp.uint32(8)
        | payload.red
    )
    params.image[y * width + x] = rgba


@woptix.optix_kernel(woptix.OptixKernelType.MISS)
def miss(params: LaunchParams):
    direction = wp.normalize(wp.optix_get_world_ray_direction())
    sky = 0.5 * (direction[1] + 1.0)
    _store_color(wp.vec3(0.03, 0.04, 0.07) * (1.0 - sky) + wp.vec3(0.12, 0.18, 0.28) * sky)


@woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
def triangle_closest_hit(params: LaunchParams):
    barycentrics = wp.optix_get_triangle_barycentrics()
    albedo = wp.vec3(0.85, 0.25 + 0.4 * barycentrics[0], 0.12 + 0.3 * barycentrics[1])
    vertices = wp.optix_get_triangle_vertex_data()
    v0 = wp.vec3(vertices[0, 0], vertices[0, 1], vertices[0, 2])
    v1 = wp.vec3(vertices[1, 0], vertices[1, 1], vertices[1, 2])
    v2 = wp.vec3(vertices[2, 0], vertices[2, 1], vertices[2, 2])
    object_normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))
    normal = wp.optix_transform_normal_from_object_to_world_space(object_normal)
    _store_color(_shade(normal, albedo))


@woptix.optix_kernel(woptix.OptixKernelType.INTERSECTION)
def sphere_intersection(params: LaunchParams):
    origin = wp.optix_get_object_ray_origin()
    direction = wp.optix_get_object_ray_direction()
    radius = 0.65
    a = wp.dot(direction, direction)
    half_b = wp.dot(origin, direction)
    c = wp.dot(origin, origin) - radius * radius
    discriminant = half_b * half_b - a * c
    if discriminant < 0.0:
        return

    root = wp.sqrt(discriminant)
    hit_t = (-half_b - root) / a
    if hit_t < wp.optix_get_ray_tmin():
        hit_t = (-half_b + root) / a
    if hit_t < wp.optix_get_ray_tmin() or hit_t > wp.optix_get_ray_tmax():
        return

    normal = (origin + hit_t * direction) / radius
    wp.optix_report_intersection(
        hit_t,
        wp.uint32(0),
        wp.float_to_uint32(normal[0]),
        wp.float_to_uint32(normal[1]),
        wp.float_to_uint32(normal[2]),
    )


@woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
def sphere_closest_hit(params: LaunchParams):
    object_normal = wp.vec3(
        wp.uint32_to_float(wp.optix_get_attribute_0()),
        wp.uint32_to_float(wp.optix_get_attribute_1()),
        wp.uint32_to_float(wp.optix_get_attribute_2()),
    )
    world_normal = wp.optix_transform_normal_from_object_to_world_space(object_normal)
    _store_color(_shade(world_normal, wp.vec3(0.12, 0.45, 0.95)))


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
    parser.add_argument("--output", type=Path, default=Path("mixed_geometry.bmp"))
    args = parser.parse_args()

    optix = woptix.require_optix()
    wp.init()
    with wp.ScopedDevice(args.device):
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise RuntimeError("This example requires a CUDA device")

        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__), "", "mixed_geometry", __file__, device=args.device
        )
        cuda_context = device.context.value if hasattr(device.context, "value") else int(device.context)
        context, logger = woptix.create_context(optix, int(cuda_context))

        vertices = np.array([[-1.45, -0.8, 0.0], [-0.2, -0.8, 0.0], [-0.82, 0.8, 0.0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        triangle_gas, triangle_buffers = woptix.create_triangle_gas(
            optix, context, vertices, indices, args.device
        )
        sphere_aabb = np.array([[-0.65, -0.65, -0.65, 0.65, 0.65, 0.65]], dtype=np.float32)
        sphere_gas, sphere_buffers = woptix.create_custom_primitive_gas(
            optix, context, sphere_aabb, args.device
        )

        pipeline, sbt, pipeline_buffers = woptix.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            raygen,
            miss,
            None,
            num_payload_values=3,
            num_attribute_values=3,
            device=args.device,
            hit_groups=[
                woptix.HitKernel(closest_hit=triangle_closest_hit),
                woptix.HitKernel(closest_hit=sphere_closest_hit, intersection=sphere_intersection),
            ],
            traversable_graph_flags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        )
        sbt_manager = pipeline_buffers["sbt_manager"]
        triangle_hit_group, sphere_hit_group = pipeline_buffers["hit_group_handles"]

        identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        sphere_transform = [1.0, 0.0, 0.0, 0.75, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        instances = [
            optix.Instance(
                identity,
                0,
                sbt_manager.get_sbt_offset(triangle_hit_group),
                255,
                optix.INSTANCE_FLAG_NONE,
                triangle_gas,
            ),
            optix.Instance(
                sphere_transform,
                1,
                sbt_manager.get_sbt_offset(sphere_hit_group),
                255,
                optix.INSTANCE_FLAG_NONE,
                sphere_gas,
            ),
        ]
        ias, ias_buffers = woptix.create_instance_acceleration_structure(
            optix, context, instances, args.device
        )

        image = wp.empty(args.width * args.height, dtype=wp.uint32, device=args.device)
        params = LaunchParams()
        params.image = image
        params.width = wp.uint32(args.width)
        params.height = wp.uint32(args.height)
        params.traversable = wp.uint64(ias)
        params_buffer = woptix.create_launch_params_buffer(LaunchParams, args.device)
        woptix.write_launch_params(params_buffer, params)
        woptix.launch(optix, pipeline, sbt, args.width, args.height, params_buffer)
        wp.synchronize_device(args.device)

        pixels = image.numpy()
        _save_bmp(args.output, pixels, args.width, args.height)
        _keepalive = (triangle_buffers, sphere_buffers, ias_buffers, pipeline_buffers)
        print(f"Wrote {args.output} (checksum {int(np.bitwise_xor.reduce(pixels)):#010x})")
        print(f"OptiX log messages: {logger.num_messages}")


if __name__ == "__main__":
    main()
