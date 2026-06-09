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

"""Tiny Warp OptiX ray tracer with optional zero-copy OpenGL live viewer.

Kernels are defined with the new Warp OptiX kernel annotations:
  - @woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
  - @woptix.optix_kernel(woptix.OptixKernelType.MISS)
  - @woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

import warp as wp
import warp_optix as woptix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@wp.func
def pack_rgba8(color: wp.vec3) -> wp.uint32:
    r = wp.uint32(wp.max(0.0, wp.min(255.0, color[0] * 255.0)))
    g = wp.uint32(wp.max(0.0, wp.min(255.0, color[1] * 255.0)))
    b = wp.uint32(wp.max(0.0, wp.min(255.0, color[2] * 255.0)))
    a = wp.uint32(255)
    return (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r


@wp.struct
class TinyLaunchParams:
    image: wp.array(dtype=wp.uint32)
    accum: wp.array(dtype=wp.vec3)
    width: wp.uint32
    height: wp.uint32
    trav_handle: wp.uint64
    frame_index: wp.uint32
    seed: wp.uint32
    cam_pos: wp.vec3
    cam_forward: wp.vec3
    cam_right: wp.vec3
    cam_up: wp.vec3
    tan_half_fov: float
    aspect: float


@wp.struct
class TracePayload:
    hit: wp.uint32
    r_u8: wp.uint32
    g_u8: wp.uint32
    b_u8: wp.uint32
    rand0: wp.uint32
    rand1: wp.uint32


@wp.func
def _to_u8(v: float) -> wp.uint32:
    return wp.uint32(wp.max(0.0, wp.min(255.0, v * 255.0)))


@wp.func
def _hash_u32(x: wp.uint32) -> wp.uint32:
    x = (x ^ wp.uint32(61)) ^ (x >> wp.uint32(16))
    x *= wp.uint32(9)
    x = x ^ (x >> wp.uint32(4))
    x *= wp.uint32(0x27D4EB2D)
    x = x ^ (x >> wp.uint32(15))
    return x


@wp.func
def _u32_to_unit_float(x: wp.uint32) -> float:
    return float(x) * (1.0 / 4294967295.0)


@wp.func
def _cosine_sample_hemisphere(n: wp.vec3, u1: float, u2: float) -> wp.vec3:
    r = wp.sqrt(wp.max(0.0, u1))
    phi = 6.28318530718 * u2
    x = r * wp.cos(phi)
    y = r * wp.sin(phi)
    z = wp.sqrt(wp.max(0.0, 1.0 - u1))

    up = wp.vec3(0.0, 1.0, 0.0) if wp.abs(n[1]) < 0.999 else wp.vec3(1.0, 0.0, 0.0)
    t = wp.normalize(wp.cross(up, n))
    b = wp.cross(n, t)
    return wp.normalize(t * x + b * y + n * z)


@wp.func
def _room_radiance(p: wp.vec3, d: wp.vec3) -> wp.vec3:
    eps = 1.0e-4
    t_best = 1.0e20
    c = wp.vec3(0.0, 0.0, 0.0)

    if wp.abs(d[0]) > 1.0e-6:
        txl = (-1.0 - p[0]) / d[0]
        if txl > eps and txl < t_best:
            q = p + d * txl
            if q[1] >= -1.0 and q[1] <= 1.0 and q[2] >= -1.0 and q[2] <= 1.0:
                t_best = txl
                c = wp.vec3(0.78, 0.18, 0.14)
        txr = (1.0 - p[0]) / d[0]
        if txr > eps and txr < t_best:
            q = p + d * txr
            if q[1] >= -1.0 and q[1] <= 1.0 and q[2] >= -1.0 and q[2] <= 1.0:
                t_best = txr
                c = wp.vec3(0.16, 0.64, 0.20)

    if wp.abs(d[1]) > 1.0e-6:
        tyf = (-1.0 - p[1]) / d[1]
        if tyf > eps and tyf < t_best:
            q = p + d * tyf
            if q[0] >= -1.0 and q[0] <= 1.0 and q[2] >= -1.0 and q[2] <= 1.0:
                t_best = tyf
                c = wp.vec3(0.74, 0.74, 0.74) * 0.8
        tyc = (1.0 - p[1]) / d[1]
        if tyc > eps and tyc < t_best:
            q = p + d * tyc
            if q[0] >= -1.0 and q[0] <= 1.0 and q[2] >= -1.0 and q[2] <= 1.0:
                t_best = tyc
                if q[0] >= -0.25 and q[0] <= 0.25 and q[2] >= -0.65 and q[2] <= -0.2:
                    c = wp.vec3(3.5, 3.3, 2.9)
                else:
                    c = wp.vec3(0.74, 0.74, 0.74) * 0.7

    if wp.abs(d[2]) > 1.0e-6:
        tzb = (-1.0 - p[2]) / d[2]
        if tzb > eps and tzb < t_best:
            q = p + d * tzb
            if q[0] >= -1.0 and q[0] <= 1.0 and q[1] >= -1.0 and q[1] <= 1.0:
                t_best = tzb
                c = wp.vec3(0.72, 0.72, 0.72) * 0.75

    if t_best < 1.0e19:
        return c
    return wp.vec3(0.03, 0.04, 0.05)


@wp.func
def _segment_hits_aabb(p0: wp.vec3, p1: wp.vec3, bmin: wp.vec3, bmax: wp.vec3) -> bool:
    d = p1 - p0
    tmin = 0.0
    tmax = 1.0
    eps = 1.0e-6

    for axis in range(3):
        p = p0[axis]
        v = d[axis]
        lo = bmin[axis]
        hi = bmax[axis]
        if wp.abs(v) < eps:
            if p < lo or p > hi:
                return False
        else:
            inv_v = 1.0 / v
            t1 = (lo - p) * inv_v
            t2 = (hi - p) * inv_v
            if t1 > t2:
                tmp = t1
                t1 = t2
                t2 = tmp
            tmin = wp.max(tmin, t1)
            tmax = wp.min(tmax, t2)
            if tmax < tmin:
                return False
    return tmax > 0.0 and tmin < 1.0


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def tiny_raygen(params: TinyLaunchParams):
    launch_idx = wp.optix_get_launch_index()
    ix = int(launch_idx[0])
    iy = int(launch_idx[1])

    width = int(params.width)
    height = int(params.height)

    if ix >= width or iy >= height:
        return

    ndc_x = 2.0 * ((float(ix) + 0.5) / float(width)) - 1.0
    ndc_y = 2.0 * ((float(iy) + 0.5) / float(height)) - 1.0
    px = ndc_x * float(params.aspect) * float(params.tan_half_fov)
    py = ndc_y * float(params.tan_half_fov)

    ray_origin = params.cam_pos
    ray_direction = wp.normalize(params.cam_forward + px * params.cam_right + py * params.cam_up)

    payload = TracePayload()
    payload.hit = wp.uint32(0)
    payload.r_u8 = wp.uint32(0)
    payload.g_u8 = wp.uint32(0)
    payload.b_u8 = wp.uint32(0)
    pixel_idx = iy * width + ix
    rng_base = wp.uint32(pixel_idx) ^ (params.frame_index * wp.uint32(747796405)) ^ params.seed
    payload.rand0 = _hash_u32(rng_base ^ wp.uint32(2891336453))
    payload.rand1 = _hash_u32(rng_base ^ wp.uint32(1181783497))

    wp.optix_trace(
        params.trav_handle,
        ray_origin,
        ray_direction,
        0.001,
        1.0e16,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(5),
        wp.uint32(0),
        payload,
    )

    hit_color = wp.vec3(float(payload.r_u8) / 255.0, float(payload.g_u8) / 255.0, float(payload.b_u8) / 255.0)
    sky_t = 0.5 * (ray_direction[1] + 1.0)
    sky_color = wp.vec3(0.02, 0.025, 0.03) * (1.0 - sky_t) + wp.vec3(0.08, 0.11, 0.16) * sky_t
    color = hit_color if payload.hit == wp.uint32(1) else sky_color

    prev = params.accum[pixel_idx]
    fi = float(params.frame_index)
    accum = color if fi <= 0.0 else (prev * fi + color) / (fi + 1.0)
    params.accum[pixel_idx] = accum
    display = wp.vec3(
        wp.pow(wp.max(0.0, wp.min(1.0, accum[0])), 1.0 / 2.2),
        wp.pow(wp.max(0.0, wp.min(1.0, accum[1])), 1.0 / 2.2),
        wp.pow(wp.max(0.0, wp.min(1.0, accum[2])), 1.0 / 2.2),
    )
    params.image[pixel_idx] = pack_rgba8(display)


@woptix.optix_kernel(woptix.OptixKernelType.MISS)
def tiny_miss(params: TinyLaunchParams):
    wp.optix_set_payload_0(wp.uint32(0))
    wp.optix_set_payload_1(wp.uint32(0))
    wp.optix_set_payload_2(wp.uint32(0))
    wp.optix_set_payload_3(wp.uint32(0))


@woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
def tiny_closest_hit(params: TinyLaunchParams):
    pid = int(wp.optix_get_primitive_index())

    n = wp.vec3(0.0, 0.0, 1.0)
    albedo = wp.vec3(0.75, 0.75, 0.75)
    emissive = wp.vec3(0.0, 0.0, 0.0)

    # Room quads are emitted in fixed order in _build_cornell_box_mesh().
    if pid < 2:
        n = wp.vec3(0.0, 1.0, 0.0)  # floor
        albedo = wp.vec3(0.74, 0.74, 0.74)
    elif pid < 4:
        n = wp.vec3(0.0, -1.0, 0.0)  # ceiling
        albedo = wp.vec3(0.74, 0.74, 0.74)
    elif pid < 6:
        n = wp.vec3(0.0, 0.0, 1.0)  # back wall
        albedo = wp.vec3(0.72, 0.72, 0.72)
    elif pid < 8:
        n = wp.vec3(1.0, 0.0, 0.0)  # left wall (red)
        albedo = wp.vec3(0.78, 0.18, 0.14)
    elif pid < 10:
        n = wp.vec3(-1.0, 0.0, 0.0)  # right wall (green)
        albedo = wp.vec3(0.16, 0.64, 0.2)
    elif pid < 12:
        # Small emissive ceiling panel.
        n = wp.vec3(0.0, -1.0, 0.0)
        emissive = wp.vec3(1.0, 0.95, 0.8)
    elif pid < 24:
        face = (pid - 12) // 2  # short box
        if face == 0:
            n = wp.vec3(1.0, 0.0, 0.0)
        elif face == 1:
            n = wp.vec3(-1.0, 0.0, 0.0)
        elif face == 2:
            n = wp.vec3(0.0, 0.0, 1.0)
        elif face == 3:
            n = wp.vec3(0.0, 0.0, -1.0)
        elif face == 4:
            n = wp.vec3(0.0, 1.0, 0.0)
        else:
            n = wp.vec3(0.0, -1.0, 0.0)
        albedo = wp.vec3(0.72, 0.72, 0.72)
    elif pid < 36:
        face = (pid - 24) // 2  # tall box
        if face == 0:
            n = wp.vec3(1.0, 0.0, 0.0)
        elif face == 1:
            n = wp.vec3(-1.0, 0.0, 0.0)
        elif face == 2:
            n = wp.vec3(0.0, 0.0, 1.0)
        elif face == 3:
            n = wp.vec3(0.0, 0.0, -1.0)
        elif face == 4:
            n = wp.vec3(0.0, 1.0, 0.0)
        else:
            n = wp.vec3(0.0, -1.0, 0.0)
        albedo = wp.vec3(0.68, 0.68, 0.68)

    ro = wp.optix_get_world_ray_origin()
    rd = wp.normalize(wp.optix_get_world_ray_direction())
    t_hit = wp.optix_get_ray_tmax()
    p = ro + rd * t_hit

    light_pos = wp.vec3(0.0, 0.99, -0.42)
    p_shadow = p + n * 0.002
    to_light = light_pos - p_shadow
    dist = wp.length(to_light)
    dist2 = wp.max(dist * dist, 1.0e-4)
    l = to_light / wp.max(dist, 1.0e-4)

    occluded_short = _segment_hits_aabb(p_shadow, light_pos, wp.vec3(-0.65, -1.0, -0.35), wp.vec3(-0.15, -0.25, 0.25))
    occluded_tall = _segment_hits_aabb(p_shadow, light_pos, wp.vec3(0.2, -1.0, -0.8), wp.vec3(0.65, 0.45, -0.25))
    visibility = 0.0 if (occluded_short or occluded_tall) else 1.0

    ndotl = wp.max(wp.dot(n, l), 0.0)
    key = visibility * ndotl * 8.0 / (1.0 + 6.0 * dist2)

    fill_dir = wp.normalize(wp.vec3(-0.25, 0.9, 0.35))
    fill = wp.max(wp.dot(n, fill_dir), 0.0) * 0.08
    ambient = 0.08

    u1 = _u32_to_unit_float(wp.optix_get_payload_4())
    u2 = _u32_to_unit_float(wp.optix_get_payload_5())
    bounce_dir = _cosine_sample_hemisphere(n, u1, u2)
    bounce_p = p + n * 0.002
    bounce_radiance = _room_radiance(bounce_p, bounce_dir)
    indirect = wp.cw_mul(albedo, bounce_radiance) * 0.35

    lit = albedo * wp.min(ambient + key + fill, 1.35) + indirect + emissive
    color = wp.vec3(wp.min(lit[0], 1.0), wp.min(lit[1], 1.0), wp.min(lit[2], 1.0))

    wp.optix_set_payload_0(wp.uint32(1))
    wp.optix_set_payload_1(_to_u8(color[0]))
    wp.optix_set_payload_2(_to_u8(color[1]))
    wp.optix_set_payload_3(_to_u8(color[2]))


def _build_optix_ptx_from_warp(device: str = "cuda") -> tuple[bytes, str, str, str]:
    module = wp.get_module(__name__)
    ptx = woptix.compile_warp_module_to_ptx(
        module=module,
        launch_preamble="",
        module_tag="tiny_rt",
        script_dir=SCRIPT_DIR,
        device=device,
    )

    raygen_name = woptix.get_entry_name(tiny_raygen, expected_kernel_type=woptix.OptixKernelType.RAYGEN)
    miss_name = woptix.get_entry_name(tiny_miss, expected_kernel_type=woptix.OptixKernelType.MISS)
    hit_name = woptix.get_entry_name(tiny_closest_hit, expected_kernel_type=woptix.OptixKernelType.CLOSEST_HIT)
    return ptx, raygen_name, miss_name, hit_name


def _save_bmp(path: str, pixels_rgba_u32: np.ndarray, width: int, height: int):
    import struct  # noqa: PLC0415

    rgba = pixels_rgba_u32.reshape(height, width)
    bgr = np.empty((height, width, 3), dtype=np.uint8)
    bgr[..., 0] = ((rgba >> 16) & 0xFF).astype(np.uint8)
    bgr[..., 1] = ((rgba >> 8) & 0xFF).astype(np.uint8)
    bgr[..., 2] = (rgba & 0xFF).astype(np.uint8)

    row_stride = (width * 3 + 3) & ~3
    pixel_data_size = row_stride * height
    # Positive height means bottom-to-top row order, which matches our
    # kernel layout (iy=0 is the bottom of the image) -- no flip needed.
    with open(path, "wb") as f:
        f.write(struct.pack("<2sIHHI", b"BM", 54 + pixel_data_size, 0, 0, 54))
        f.write(struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0))
        pad = b"\x00" * (row_stride - width * 3)
        for y in range(height):
            f.write(bgr[y].tobytes())
            if pad:
                f.write(pad)


def _add_quad(vertices: list[tuple[float, float, float]], indices: list[tuple[int, int, int]], v0, v1, v2, v3):
    base = len(vertices)
    vertices.extend((v0, v1, v2, v3))
    indices.append((base, base + 1, base + 2))
    indices.append((base, base + 2, base + 3))


def _add_box(
    vertices: list[tuple[float, float, float]],
    indices: list[tuple[int, int, int]],
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
):
    p000 = (x0, y0, z0)
    p001 = (x0, y0, z1)
    p010 = (x0, y1, z0)
    p011 = (x0, y1, z1)
    p100 = (x1, y0, z0)
    p101 = (x1, y0, z1)
    p110 = (x1, y1, z0)
    p111 = (x1, y1, z1)

    # Keep quad order stable so primitive id -> normal mapping stays tiny and deterministic.
    _add_quad(vertices, indices, p100, p110, p111, p101)  # +X
    _add_quad(vertices, indices, p001, p011, p010, p000)  # -X
    _add_quad(vertices, indices, p101, p111, p011, p001)  # +Z
    _add_quad(vertices, indices, p000, p010, p110, p100)  # -Z
    _add_quad(vertices, indices, p010, p011, p111, p110)  # +Y
    _add_quad(vertices, indices, p001, p000, p100, p101)  # -Y


def _build_cornell_box_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    indices: list[tuple[int, int, int]] = []

    # Cornell-style room: floor, ceiling, back wall, left wall, right wall.
    _add_quad(vertices, indices, (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (-1.0, -1.0, 1.0))
    _add_quad(vertices, indices, (-1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0))
    _add_quad(vertices, indices, (-1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (1.0, 1.0, -1.0), (1.0, -1.0, -1.0))
    _add_quad(vertices, indices, (-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0))
    _add_quad(vertices, indices, (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0))

    # Ceiling light panel.
    _add_quad(vertices, indices, (-0.25, 0.99, -0.65), (0.25, 0.99, -0.65), (0.25, 0.99, -0.2), (-0.25, 0.99, -0.2))

    # Two interior blocks.
    _add_box(vertices, indices, -0.65, -1.0, -0.35, -0.15, -0.25, 0.25)
    _add_box(vertices, indices, 0.2, -1.0, -0.8, 0.65, 0.45, -0.25)

    verts_np = np.asarray(vertices, dtype=np.float32)
    inds_np = np.asarray(indices, dtype=np.uint32)
    return verts_np, inds_np


def _set_camera_params(
    params: TinyLaunchParams,
    position: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    width: int,
    height: int,
):
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    cos_pitch = math.cos(pitch_rad)
    forward = np.array(
        [math.sin(yaw_rad) * cos_pitch, math.sin(pitch_rad), math.cos(yaw_rad) * cos_pitch],
        dtype=np.float32,
    )
    forward /= np.linalg.norm(forward) + 1.0e-8
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1.0e-8
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1.0e-8

    params.cam_pos = wp.vec3(float(position[0]), float(position[1]), float(position[2]))
    params.cam_forward = wp.vec3(float(forward[0]), float(forward[1]), float(forward[2]))
    params.cam_right = wp.vec3(float(right[0]), float(right[1]), float(right[2]))
    params.cam_up = wp.vec3(float(up[0]), float(up[1]), float(up[2]))
    params.tan_half_fov = float(math.tan(math.radians(fov_deg) * 0.5))
    params.aspect = float(width) / float(height)


class FreeCameraController:
    def __init__(self, viewer: woptix.GLInteropViewer, params: TinyLaunchParams, width: int, height: int):
        self.window = viewer.window
        self.pyglet = viewer.pyglet
        self.params = params
        self.width = int(width)
        self.height = int(height)
        self.position = np.array([0.0, 0.05, 2.5], dtype=np.float32)
        self.yaw = 180.0
        self.pitch = -4.0
        self.fov = 50.0
        self._keys_down: set[int] = set()
        self._cam_speed = 1.6
        self._look_sensitivity = 0.12
        self._dirty = True
        self.window.push_handlers(self)
        self._write(mark_dirty=False)

    def _forward_right(self) -> tuple[np.ndarray, np.ndarray]:
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        cos_pitch = math.cos(pitch_rad)
        forward = np.array(
            [math.sin(yaw_rad) * cos_pitch, math.sin(pitch_rad), math.cos(yaw_rad) * cos_pitch], dtype=np.float32
        )
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1.0e-8
        return forward, right

    def _write(self, mark_dirty: bool = True):
        _set_camera_params(self.params, self.position, self.yaw, self.pitch, self.fov, self.width, self.height)
        if mark_dirty:
            self._dirty = True

    def resize(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self._write()

    def update(self, dt: float):
        try:
            key = self.pyglet.window.key
        except Exception:
            return

        forward, right = self._forward_right()
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        move = np.zeros(3, dtype=np.float32)
        if key.W in self._keys_down or key.UP in self._keys_down:
            move += forward
        if key.S in self._keys_down or key.DOWN in self._keys_down:
            move -= forward
        if key.A in self._keys_down or key.LEFT in self._keys_down:
            move -= right
        if key.D in self._keys_down or key.RIGHT in self._keys_down:
            move += right
        if key.Q in self._keys_down:
            move -= world_up
        if key.E in self._keys_down:
            move += world_up

        move_norm = float(np.linalg.norm(move))
        if move_norm > 1.0e-6:
            speed = self._cam_speed
            if key.LSHIFT in self._keys_down or key.RSHIFT in self._keys_down:
                speed *= 3.5
            self.position += (move / move_norm) * speed * max(0.0, min(dt, 0.1))
            self._write()

    def consume_changed(self) -> bool:
        was_dirty = self._dirty
        self._dirty = False
        return was_dirty

    def on_key_press(self, symbol, _modifiers):
        self._keys_down.add(symbol)

    def on_key_release(self, symbol, _modifiers):
        self._keys_down.discard(symbol)

    def on_mouse_drag(self, _x, _y, dx, dy, buttons, _modifiers):
        try:
            mouse = self.pyglet.window.mouse
        except Exception:
            return
        if buttons & mouse.LEFT:
            self.yaw -= float(dx) * self._look_sensitivity
            self.pitch += float(dy) * self._look_sensitivity
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self._write()

    def on_mouse_scroll(self, _x, _y, _sx, sy):
        self.fov = max(20.0, min(90.0, self.fov - float(sy) * 2.0))
        self._write()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", type=str, default="tiny_optix_raytracer.bmp")
    parser.add_argument("--live", action="store_true", help="Run OpenGL interop viewer (zero-copy PBO path)")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames in live mode (0=forever)")
    args = parser.parse_args()

    optix = woptix.require_optix()
    wp.init()

    with wp.ScopedDevice(args.device):
        if not wp.get_device().is_cuda:
            raise RuntimeError("This sample requires a CUDA device.")

        ptx, raygen_name, miss_name, hit_name = _build_optix_ptx_from_warp(device=args.device)
        wp_device = wp.get_device(args.device)
        cu_context = wp_device.context.value if hasattr(wp_device.context, "value") else int(wp_device.context)
        ctx, logger = woptix.create_context(optix, int(cu_context))
        vertices, indices = _build_cornell_box_mesh()
        gas_handle, gas_keepalive = woptix.create_triangle_gas(optix, ctx, vertices, indices, args.device)
        pipeline, sbt, pipe_keepalive = woptix.create_pipeline_and_sbt(
            optix=optix,
            ctx=ctx,
            ptx=ptx,
            raygen_entry=tiny_raygen,
            miss_entry=tiny_miss,
            closest_hit_entry=tiny_closest_hit,
            num_payload_values=6,
            num_attribute_values=2,
            device=args.device,
        )

        params = TinyLaunchParams()
        params.width = wp.uint32(args.width)
        params.height = wp.uint32(args.height)
        params.trav_handle = wp.uint64(gas_handle)
        params.frame_index = wp.uint32(0)
        params.seed = wp.uint32(1337)
        _set_camera_params(
            params=params,
            position=np.array([0.0, 0.05, 2.5], dtype=np.float32),
            yaw_deg=180.0,
            pitch_deg=-4.0,
            fov_deg=50.0,
            width=args.width,
            height=args.height,
        )
        params_buffer = woptix.create_launch_params_buffer(TinyLaunchParams, device=args.device)
        accum = wp.zeros(args.width * args.height, dtype=wp.vec3, device=args.device)
        params.accum = accum

        if args.live:
            render_width = int(args.width)
            render_height = int(args.height)
            sample_count = 0
            controller: FreeCameraController | None = None

            def _on_resize(width: int, height: int):
                nonlocal render_width, render_height, accum, sample_count, controller
                render_width = int(width)
                render_height = int(height)
                params.width = wp.uint32(render_width)
                params.height = wp.uint32(render_height)
                accum = wp.zeros(render_width * render_height, dtype=wp.vec3, device=args.device)
                params.accum = accum
                if controller is not None:
                    controller.resize(render_width, render_height)
                sample_count = 0

            viewer = woptix.GLInteropViewer(
                width=args.width,
                height=args.height,
                device=args.device,
                title="Warp OptiX Tiny RT",
                fps=args.fps,
                on_resize=_on_resize,
            )
            controller = FreeCameraController(viewer, params, args.width, args.height)
            last_elapsed = 0.0

            def _render(mapped_image: wp.array, _frame_idx: int, elapsed_sec: float):
                nonlocal last_elapsed, sample_count
                dt = elapsed_sec - last_elapsed
                last_elapsed = elapsed_sec
                controller.update(dt)
                if controller.consume_changed():
                    sample_count = 0
                params.frame_index = wp.uint32(sample_count)
                params.image = mapped_image
                woptix.write_launch_params(params_buffer, params)
                woptix.launch(optix, pipeline, sbt, render_width, render_height, params_buffer)
                sample_count += 1

            viewer.run(_render, max_frames=args.max_frames)
            checksum = 0
            print("Controls: WASD move, Q/E down/up, Shift faster, Left-drag look, Mouse wheel zoom.")
        else:
            image = wp.empty(args.width * args.height, dtype=wp.uint32, device=args.device)
            params.image = image
            params.frame_index = wp.uint32(0)
            woptix.write_launch_params(params_buffer, params)
            woptix.launch(optix, pipeline, sbt, args.width, args.height, params_buffer)
            wp.synchronize_device(args.device)
            pixels = image.numpy()
            checksum = int(np.bitwise_xor.reduce(pixels))
            _save_bmp(args.output, pixels, args.width, args.height)

        print(f"OptiX log messages: {logger.num_messages}")
        print(f"Raygen entry: {raygen_name}")
        print(f"Miss entry:   {miss_name}")
        print(f"Hit entry:    {hit_name}")
        if args.live:
            print("Live viewer finished.")
        else:
            print(f"Image checksum: {checksum}")
            print(f"Wrote: {os.path.abspath(args.output)}")

        _keepalive = {
            "gas": gas_keepalive,
            "pipeline": pipe_keepalive,
            "params": params_buffer,
        }
        _ = _keepalive


if __name__ == "__main__":
    main()
