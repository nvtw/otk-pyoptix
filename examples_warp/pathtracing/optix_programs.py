# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 1:1 translation of optix_programs.h (OptiX entry points and motion vector helpers).
# The actual kernel entry points (primary_raygen, primary_miss, etc.) remain in
# pathtracing_warp_kernels.py because they reference PathtraceLaunchParams and the
# full set of scene accessors.  This file provides the motion vector computation
# functions that are independent of the launch params struct.

from __future__ import annotations

import warp as wp

from .func_common import Mat16f, mul_cm_4x4


@wp.struct
class TransformMatrix3x4:
    row0: wp.vec4
    row1: wp.vec4
    row2: wp.vec4


@wp.func
def transform_point(m: TransformMatrix3x4, p: wp.vec3) -> wp.vec3:
    return wp.vec3(
        m.row0[0] * p[0] + m.row0[1] * p[1] + m.row0[2] * p[2] + m.row0[3],
        m.row1[0] * p[0] + m.row1[1] * p[1] + m.row1[2] * p[2] + m.row1[3],
        m.row2[0] * p[0] + m.row2[1] * p[1] + m.row2[2] * p[2] + m.row2[3],
    )


@wp.func
def transform_vector(m: TransformMatrix3x4, v: wp.vec3) -> wp.vec3:
    return wp.vec3(
        m.row0[0] * v[0] + m.row0[1] * v[1] + m.row0[2] * v[2],
        m.row1[0] * v[0] + m.row1[1] * v[1] + m.row1[2] * v[2],
        m.row2[0] * v[0] + m.row2[1] * v[1] + m.row2[2] * v[2],
    )


@wp.func
def inverse_transform_point(m: TransformMatrix3x4, world_pos: wp.vec3) -> wp.vec3:
    a00 = m.row0[0]
    a01 = m.row0[1]
    a02 = m.row0[2]
    a10 = m.row1[0]
    a11 = m.row1[1]
    a12 = m.row1[2]
    a20 = m.row2[0]
    a21 = m.row2[1]
    a22 = m.row2[2]
    det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
    if wp.abs(det) < 1.0e-12:
        return world_pos
    inv = 1.0 / det
    i00 = (a11 * a22 - a12 * a21) * inv
    i01 = (a02 * a21 - a01 * a22) * inv
    i02 = (a01 * a12 - a02 * a11) * inv
    i10 = (a12 * a20 - a10 * a22) * inv
    i11 = (a00 * a22 - a02 * a20) * inv
    i12 = (a02 * a10 - a00 * a12) * inv
    i20 = (a10 * a21 - a11 * a20) * inv
    i21 = (a01 * a20 - a00 * a21) * inv
    i22 = (a00 * a11 - a01 * a10) * inv
    d = wp.vec3(world_pos[0] - m.row0[3], world_pos[1] - m.row1[3], world_pos[2] - m.row2[3])
    return wp.vec3(
        i00 * d[0] + i01 * d[1] + i02 * d[2],
        i10 * d[0] + i11 * d[1] + i12 * d[2],
        i20 * d[0] + i21 * d[1] + i22 * d[2],
    )


@wp.func
def transforms_equal(a: TransformMatrix3x4, b: TransformMatrix3x4) -> wp.bool:
    return (
        a.row0[0] == b.row0[0]
        and a.row0[1] == b.row0[1]
        and a.row0[2] == b.row0[2]
        and a.row0[3] == b.row0[3]
        and a.row1[0] == b.row1[0]
        and a.row1[1] == b.row1[1]
        and a.row1[2] == b.row1[2]
        and a.row1[3] == b.row1[3]
        and a.row2[0] == b.row2[0]
        and a.row2[1] == b.row2[1]
        and a.row2[2] == b.row2[2]
        and a.row2[3] == b.row2[3]
    )


# -----------------------------------------------------------------------------
# Motion vector computation (matches C++ compute_camera_motion_vector exactly)
# -----------------------------------------------------------------------------


@wp.func
def compute_camera_motion_vector(
    pixel_center: wp.vec2,
    motion_origin: wp.vec4,
    prev_mvp: Mat16f,
    dim_x: wp.uint32,
    dim_y: wp.uint32,
) -> wp.vec2:
    """Project motionOrigin through prevMVP and compute pixel-space displacement.

    motionOrigin.w=1 for world positions, w=0 for directions (sky).
    """
    old = mul_cm_4x4(prev_mvp, motion_origin)
    w = old[3]
    if wp.abs(w) < 1.0e-8:
        w = wp.where(w >= 0.0, 1.0e-8, -1.0e-8)
    inv_w = 1.0 / w
    ox = (old[0] * inv_w * 0.5 + 0.5) * wp.float32(dim_x)
    oy = (old[1] * inv_w * 0.5 + 0.5) * wp.float32(dim_y)
    return wp.vec2(ox - pixel_center[0], oy - pixel_center[1])


@wp.func
def compute_object_motion_vector(
    pixel_center: wp.vec2,
    world_pos: wp.vec3,
    instance_id: wp.int32,
    prev_mvp: Mat16f,
    dim_x: wp.uint32,
    dim_y: wp.uint32,
    instance_transforms_addr: wp.uint64,
    prev_instance_transforms_addr: wp.uint64,
    instance_count: wp.uint32,
) -> wp.vec2:
    if instance_id < 0 or instance_transforms_addr == wp.uint64(0) or prev_instance_transforms_addr == wp.uint64(0):
        return compute_camera_motion_vector(
            pixel_center, wp.vec4(world_pos[0], world_pos[1], world_pos[2], 1.0), prev_mvp, dim_x, dim_y
        )

    iid = wp.uint32(instance_id)
    if iid >= instance_count:
        return compute_camera_motion_vector(
            pixel_center, wp.vec4(world_pos[0], world_pos[1], world_pos[2], 1.0), prev_mvp, dim_x, dim_y
        )

    curr = wp.array(ptr=instance_transforms_addr, shape=(int(instance_count),), dtype=TransformMatrix3x4)
    prev = wp.array(ptr=prev_instance_transforms_addr, shape=(int(instance_count),), dtype=TransformMatrix3x4)
    curr_t = curr[int(iid)]
    prev_t = prev[int(iid)]

    prev_world = world_pos
    if not transforms_equal(curr_t, prev_t):
        prev_world = transform_point(prev_t, inverse_transform_point(curr_t, world_pos))

    old = mul_cm_4x4(prev_mvp, wp.vec4(prev_world[0], prev_world[1], prev_world[2], 1.0))
    w = old[3]
    if wp.abs(w) < 1.0e-8:
        w = wp.where(w >= 0.0, 1.0e-8, -1.0e-8)
    inv_w = 1.0 / w
    ox = (old[0] * inv_w * 0.5 + 0.5) * wp.float32(dim_x)
    oy = (old[1] * inv_w * 0.5 + 0.5) * wp.float32(dim_y)
    return wp.vec2(ox - pixel_center[0], oy - pixel_center[1])


@wp.func
def compute_deformable_motion_vector(
    pixel_center: wp.vec2,
    prev_local_pos: wp.vec3,
    instance_id: wp.int32,
    prev_mvp: Mat16f,
    dim_x: wp.uint32,
    dim_y: wp.uint32,
    prev_instance_transforms_addr: wp.uint64,
    instance_count: wp.uint32,
) -> wp.vec2:
    if instance_id < 0 or prev_instance_transforms_addr == wp.uint64(0):
        return wp.vec2(0.0, 0.0)

    iid = wp.uint32(instance_id)
    if iid >= instance_count:
        return wp.vec2(0.0, 0.0)

    prev = wp.array(ptr=prev_instance_transforms_addr, shape=(int(instance_count),), dtype=TransformMatrix3x4)
    prev_t = prev[int(iid)]
    prev_world = transform_point(prev_t, prev_local_pos)

    old = mul_cm_4x4(prev_mvp, wp.vec4(prev_world[0], prev_world[1], prev_world[2], 1.0))
    w = old[3]
    if wp.abs(w) < 1.0e-8:
        w = wp.where(w >= 0.0, 1.0e-8, -1.0e-8)
    inv_w = 1.0 / w
    ox = (old[0] * inv_w * 0.5 + 0.5) * wp.float32(dim_x)
    oy = (old[1] * inv_w * 0.5 + 0.5) * wp.float32(dim_y)
    return wp.vec2(ox - pixel_center[0], oy - pixel_center[1])
