# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 1:1 translation of dlss_helper.h (DLSS helper functions).

from __future__ import annotations

import warp as wp

from .func_common import saturate_v3

DLSS_INF_DISTANCE = float(65504.0)
FLT_MIN_DLSS = 1.0e-15


@wp.func
def positive_rcp(x: wp.float32) -> wp.float32:
    return 1.0 / wp.max(x, FLT_MIN_DLSS)


@wp.func
def environment_term_rtg(rf0: wp.vec3, n_dot_v: wp.float32, alpha_roughness: wp.float32) -> wp.vec3:
    """Ray Tracing Gems, Chapter 32, Equation 4.

    Approximation assuming GGX VNDF and Schlick's approximation.
    """
    xx = 1.0
    xy = n_dot_v
    xz = n_dot_v * n_dot_v
    xw = n_dot_v * xz

    yx = 1.0
    yy = alpha_roughness
    yz = alpha_roughness * alpha_roughness
    yw = alpha_roughness * yz

    # M1 * X.xy  (GLSL column-major: col0=(0.99044, -1.28514), col1=(1.29678, -0.755907))
    m1x_x = 0.99044 * xx + 1.29678 * xy
    m1x_y = -1.28514 * xx + (-0.755907) * xy

    # M2 * X.xyw  (col0=(1.0, 2.92338, 59.4188), col1=(20.3225, -27.0302, 222.592), col2=(121.563, 626.13, 316.627))
    m2x_x = 1.0 * xx + 20.3225 * xy + 121.563 * xw
    m2x_y = 2.92338 * xx + (-27.0302) * xy + 626.13 * xw
    m2x_z = 59.4188 * xx + 222.592 * xy + 316.627 * xw

    # M3 * X.xy  (col0=(0.0365463, 3.32707), col1=(9.0632, -9.04756))
    m3x_x = 0.0365463 * xx + 9.0632 * xy
    m3x_y = 3.32707 * xx + (-9.04756) * xy

    # M4 * X.xzw  (col0=(1.0, 3.59685, -1.36772), col1=(9.04401, -16.3174, 9.22949), col2=(5.56589, 19.7886, -20.2123))
    m4x_x = 1.0 * xx + 9.04401 * xz + 5.56589 * xw
    m4x_y = 3.59685 * xx + (-16.3174) * xz + 19.7886 * xw
    m4x_z = -1.36772 * xx + 9.22949 * xz + (-20.2123) * xw

    dot_m1_y = m1x_x * yx + m1x_y * yy
    dot_m2_y = m2x_x * yx + m2x_y * yy + m2x_z * yw
    dot_m3_y = m3x_x * yx + m3x_y * yy
    dot_m4_y = m4x_x * yx + m4x_y * yy + m4x_z * yw

    bias = dot_m1_y * positive_rcp(dot_m2_y)
    scale = dot_m3_y * positive_rcp(dot_m4_y)

    return saturate_v3(rf0 * scale + wp.vec3(bias, bias, bias))
