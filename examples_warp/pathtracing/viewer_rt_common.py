# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 1:1 translation of viewer_rt_common.h (common helpers for pathtracing viewer).
# Functions that depend on launch params (eval_environment, sample_environment_importance,
# sky_params_from_launch) remain in pathtracing_warp_kernels.py because they access
# PathtraceLaunchParams fields directly.  This file contains the param-free helpers.

from __future__ import annotations

import warp as wp

from .func_common import M_1_OVER_PI, Mat16f


@wp.func
def load_mat4_from_array(m: Mat16f) -> Mat16f:
    """Identity passthrough -- in Warp, matrices are already stored as Mat16f.

    In C++ this copies 16 floats from a pointer into a float4x4 struct.
    Here the data is already in the right format.
    """
    return m


@wp.func
def get_spherical_uv_csharp(v: wp.vec3) -> wp.vec2:
    """Spherical UV mapping matching the reference EnvMap.getSphericalUv.

    Y-up convention: Y is vertical axis.
    """
    gamma = wp.asin(-v[1])
    theta = wp.atan2(v[2], v[0])
    return wp.vec2(theta * M_1_OVER_PI * 0.5 + 0.5, gamma * M_1_OVER_PI + 0.5)


@wp.func
def rotate_environment_dir(d: wp.vec3, angle: wp.float32, y_is_up: wp.int32) -> wp.vec3:
    """Rotate a direction for environment map lookup."""
    s = wp.sin(angle)
    c = wp.cos(angle)
    if y_is_up != 0:
        return wp.vec3(c * d[0] + s * d[2], d[1], -s * d[0] + c * d[2])
    return wp.vec3(c * d[0] - s * d[1], s * d[0] + c * d[1], d[2])
