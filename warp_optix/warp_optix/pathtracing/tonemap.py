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

"""Tonemapping for path tracing viewer.

Matches the Vulkan tonemap shader behavior and defaults from the reference sample.
"""

import warp as wp

# Tonemap methods (match Vulkan reference tonemap shader behavior)
TONEMAP_FILMIC = 0
TONEMAP_UNCHARTED2 = 1
TONEMAP_CLIP = 2
TONEMAP_ACES = 3
TONEMAP_AGX = 4
TONEMAP_KHRONOS_PBR = 5
TONEMAP_ACES_SRGB = 6

# Debug output modes (match pathtracing_viewer.py enums)
OUTPUT_FINAL = 0
OUTPUT_RADIANCE = 1
OUTPUT_DEPTH = 2
OUTPUT_MOTION = 3
OUTPUT_NORMAL = 4
OUTPUT_ROUGHNESS = 5
OUTPUT_DIFFUSE = 6
OUTPUT_SPECULAR = 7
OUTPUT_SPEC_HITDIST = 8


@wp.func
def to_srgb(color: wp.vec3) -> wp.vec3:
    """Linear->sRGB transfer function."""
    r = wp.where(
        color[0] <= 0.0031308,
        color[0] * 12.92,
        wp.pow(wp.max(color[0], 0.0), 1.0 / 2.4) * 1.055 - 0.055,
    )
    g = wp.where(
        color[1] <= 0.0031308,
        color[1] * 12.92,
        wp.pow(wp.max(color[1], 0.0), 1.0 / 2.4) * 1.055 - 0.055,
    )
    b = wp.where(
        color[2] <= 0.0031308,
        color[2] * 12.92,
        wp.pow(wp.max(color[2], 0.0), 1.0 / 2.4) * 1.055 - 0.055,
    )
    return wp.vec3(r, g, b)


@wp.func
def tonemap_filmic_hejl(color: wp.vec3) -> wp.vec3:
    """Filmic (Jim Hejl / Richard Burgess-Dawson)."""
    t = wp.max(color - wp.vec3(0.004, 0.004, 0.004), wp.vec3(0.0, 0.0, 0.0))
    num = wp.vec3(
        t[0] * (t[0] * 6.2 + 0.5),
        t[1] * (t[1] * 6.2 + 0.5),
        t[2] * (t[2] * 6.2 + 0.5),
    )
    den = wp.vec3(
        t[0] * (t[0] * 6.2 + 1.7) + 0.06,
        t[1] * (t[1] * 6.2 + 1.7) + 0.06,
        t[2] * (t[2] * 6.2 + 1.7) + 0.06,
    )
    return wp.vec3(num[0] / den[0], num[1] / den[1], num[2] / den[2])


@wp.func
def tonemap_uncharted2_impl(x: float) -> float:
    """Uncharted 2 tonemap partial function."""
    A = 0.15  # Shoulder strength
    B = 0.50  # Linear strength
    C = 0.10  # Linear angle
    D = 0.20  # Toe strength
    E = 0.02  # Toe numerator
    F = 0.30  # Toe denominator
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


@wp.func
def tonemap_uncharted2(color: wp.vec3) -> wp.vec3:
    """Uncharted 2 tonemap (with display gamma in-method)."""
    exposure_bias = 2.0
    W = 11.2

    r = tonemap_uncharted2_impl(color[0] * exposure_bias)
    g = tonemap_uncharted2_impl(color[1] * exposure_bias)
    b = tonemap_uncharted2_impl(color[2] * exposure_bias)

    white_scale = 1.0 / tonemap_uncharted2_impl(W)
    c = wp.vec3(r * white_scale, g * white_scale, b * white_scale)
    inv_gamma = 1.0 / 2.2
    return wp.vec3(
        wp.pow(wp.max(c[0], 0.0), inv_gamma),
        wp.pow(wp.max(c[1], 0.0), inv_gamma),
        wp.pow(wp.max(c[2], 0.0), inv_gamma),
    )


@wp.func
def tonemap_aces(color: wp.vec3) -> wp.vec3:
    """ACES approximation (Stephen Hill), output converted to sRGB."""
    c = wp.vec3(
        0.59719 * color[0] + 0.07600 * color[1] + 0.02840 * color[2],
        0.35458 * color[0] + 0.90834 * color[1] + 0.13383 * color[2],
        0.04823 * color[0] + 0.01566 * color[1] + 0.83777 * color[2],
    )
    a = wp.vec3(
        c[0] * (c[0] + 0.0245786) - 0.000090537,
        c[1] * (c[1] + 0.0245786) - 0.000090537,
        c[2] * (c[2] + 0.0245786) - 0.000090537,
    )
    b = wp.vec3(
        c[0] * (0.983729 * c[0] + 0.4329510) + 0.238081,
        c[1] * (0.983729 * c[1] + 0.4329510) + 0.238081,
        c[2] * (0.983729 * c[2] + 0.4329510) + 0.238081,
    )
    c = wp.vec3(a[0] / b[0], a[1] / b[1], a[2] / b[2])
    c = wp.vec3(
        1.60475 * c[0] + -0.10208 * c[1] + -0.00327 * c[2],
        -0.53108 * c[0] + 1.10813 * c[1] + -0.07276 * c[2],
        -0.07367 * c[0] + -0.00605 * c[1] + 1.07602 * c[2],
    )
    return to_srgb(c)


@wp.func
def tonemap_agx(color: wp.vec3) -> wp.vec3:
    """AgX tonemapper (returns linear display-referred color, matches shader)."""
    c = wp.vec3(
        0.842479062253094 * color[0]
        + 0.0423282422610123 * color[1]
        + 0.0423756549057051 * color[2],
        0.0784335999999992 * color[0]
        + 0.878468636469772 * color[1]
        + 0.0784336 * color[2],
        0.0792237451477643 * color[0]
        + 0.0791661274605434 * color[1]
        + 0.879142973793104 * color[2],
    )
    min_ev = -12.47393
    max_ev = 4.026069
    inv_ln2 = 1.0 / wp.log(2.0)
    c = wp.vec3(
        wp.clamp(wp.log(wp.max(c[0], 1.0e-10)) * inv_ln2, min_ev, max_ev),
        wp.clamp(wp.log(wp.max(c[1], 1.0e-10)) * inv_ln2, min_ev, max_ev),
        wp.clamp(wp.log(wp.max(c[2], 1.0e-10)) * inv_ln2, min_ev, max_ev),
    )
    c = (c - wp.vec3(min_ev, min_ev, min_ev)) / (max_ev - min_ev)
    v = c * 15.5 + wp.vec3(-40.14, -40.14, -40.14)
    v = wp.vec3(c[0] * v[0] + 31.96, c[1] * v[1] + 31.96, c[2] * v[2] + 31.96)
    v = wp.vec3(c[0] * v[0] - 6.868, c[1] * v[1] - 6.868, c[2] * v[2] - 6.868)
    v = wp.vec3(c[0] * v[0] + 0.4298, c[1] * v[1] + 0.4298, c[2] * v[2] + 0.4298)
    v = wp.vec3(c[0] * v[0] + 0.1191, c[1] * v[1] + 0.1191, c[2] * v[2] + 0.1191)
    v = wp.vec3(c[0] * v[0] - 0.0023, c[1] * v[1] - 0.0023, c[2] * v[2] - 0.0023)
    return wp.vec3(
        1.19687900512017 * v[0]
        + -0.0528968517574562 * v[1]
        + -0.0529716355144438 * v[2],
        -0.0980208811401368 * v[0]
        + 1.15190312990417 * v[1]
        + -0.0980434501171241 * v[2],
        -0.0990297440797205 * v[0]
        + -0.0989611768448433 * v[1]
        + 1.15107367264116 * v[2],
    )


@wp.func
def tonemap_khronos_pbr(color: wp.vec3) -> wp.vec3:
    """Khronos PBR neutral tonemapper."""
    start_compression = 0.8 - 0.04
    desaturation = 0.15
    x = wp.min(color[0], wp.min(color[1], color[2]))
    peak = wp.max(color[0], wp.max(color[1], color[2]))
    offset = wp.where(x < 0.08, x * (-6.25 * x + 1.0), 0.04)
    c = color - wp.vec3(offset, offset, offset)
    if peak >= start_compression:
        d = 1.0 - start_compression
        new_peak = 1.0 - d * d / (peak + d - start_compression)
        c = c * (new_peak / peak)
        g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0)
        c = c * (1.0 - g) + wp.vec3(new_peak, new_peak, new_peak) * g
    return to_srgb(c)


@wp.func
def tonemap_aces_srgb(color: wp.vec3) -> wp.vec3:
    """Apply hue-preserving ACES highlight rolloff and encode as sRGB."""
    peak = wp.max(color[0], wp.max(color[1], color[2]))
    mapped_peak = peak * (2.51 * peak + 0.03) / (peak * (2.43 * peak + 0.59) + 0.14)
    scale = wp.clamp(mapped_peak, 0.0, 1.0) / wp.max(peak, 1.0e-6)
    mapped = wp.vec3(color[0] * scale, color[1] * scale, color[2] * scale)
    return to_srgb(mapped)


@wp.kernel
def _reset_auto_exposure_stats(stats: wp.array(dtype=wp.float32)):
    stats[0] = 0.0
    stats[1] = 0.0


@wp.kernel
def _meter_auto_exposure(
    hdr_input: wp.array2d(dtype=wp.vec4),
    stats: wp.array(dtype=wp.float32),
    width: int,
    height: int,
    min_luminance_ev: float,
    max_luminance_ev: float,
):
    """Accumulate a center-weighted log-average luminance on a 4x4 grid."""
    tile_x, tile_y = wp.tid()
    x = tile_x * 4 + 2
    y = tile_y * 4 + 2
    if x >= width or y >= height:
        return
    color = hdr_input[y, x]
    luminance = color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
    min_luminance = wp.pow(2.0, min_luminance_ev)
    max_luminance = wp.pow(2.0, max_luminance_ev)
    luminance = wp.clamp(luminance, min_luminance, max_luminance)
    nx = (wp.float32(x) + 0.5) / wp.float32(width) * 2.0 - 1.0
    ny = (wp.float32(y) + 0.5) / wp.float32(height) * 2.0 - 1.0
    weight = 1.0 / (1.0 + 4.0 * (nx * nx + ny * ny))
    log_luminance = wp.log(luminance) / wp.log(2.0)
    wp.atomic_add(stats, 0, log_luminance * weight)
    wp.atomic_add(stats, 1, weight)


@wp.kernel
def _adapt_auto_exposure(
    stats: wp.array(dtype=wp.float32),
    exposure_ev: wp.array(dtype=wp.float32),
    target_luminance: float,
    min_exposure_ev: float,
    max_exposure_ev: float,
    brighten_speed: float,
    darken_speed: float,
    delta_time: float,
    initialize: int,
):
    """Temporally adapt exposure in EV without a CPU readback."""
    average_log_luminance = stats[0] / wp.max(stats[1], 1.0e-6)
    target_ev = wp.clamp(
        wp.log(wp.max(target_luminance, 1.0e-6)) / wp.log(2.0) - average_log_luminance,
        min_exposure_ev,
        max_exposure_ev,
    )
    if initialize != 0:
        exposure_ev[0] = target_ev
        return
    current_ev = exposure_ev[0]
    speed = brighten_speed if target_ev > current_ev else darken_speed
    alpha = 1.0 - wp.exp(-wp.max(speed, 0.0) * wp.max(delta_time, 0.0))
    exposure_ev[0] = current_ev + (target_ev - current_ev) * alpha


@wp.kernel
def tonemap_kernel(
    hdr_input: wp.array2d(dtype=wp.vec4),
    ldr_output: wp.array2d(dtype=wp.vec4),
    width: int,
    height: int,
    exposure: float,
    auto_exposure_ev: wp.array(dtype=wp.float32),
    auto_exposure_enabled: int,
    method: int,
    is_active: int,
    brightness: float,
    contrast: float,
    saturation: float,
    vignette: float,
):
    """
    Tonemap HDR image to LDR.

    Args:
        hdr_input: Input HDR image (RGBA float)
        ldr_output: Output LDR image (RGBA float, 0-1 range)
        width: Output width
        height: Output height
        exposure: Exposure multiplier
        method: Tonemap method enum
        is_active: Tonemap enable flag
        brightness: Brightness curve control (1.0 neutral)
        contrast: Contrast control (1.0 neutral)
        saturation: Saturation control (1.0 neutral)
        vignette: Vignette strength (0.0 disabled)
    """
    x, y = wp.tid()

    # Read HDR color from vertically mirrored source row.
    # This compensates the camera-space Y flip applied in projection.
    source_y = height - 1 - y
    hdr = hdr_input[source_y, x]
    exposure_multiplier = exposure
    if auto_exposure_enabled == 1:
        exposure_multiplier *= wp.pow(2.0, auto_exposure_ev[0])
    color = wp.vec3(hdr[0], hdr[1], hdr[2]) * exposure_multiplier

    if is_active == 1:
        if method == TONEMAP_FILMIC:
            color = tonemap_filmic_hejl(color)
        elif method == TONEMAP_UNCHARTED2:
            color = tonemap_uncharted2(color)
        elif method == TONEMAP_CLIP:
            color = to_srgb(
                wp.vec3(
                    wp.clamp(color[0], 0.0, 1.0),
                    wp.clamp(color[1], 0.0, 1.0),
                    wp.clamp(color[2], 0.0, 1.0),
                )
            )
        elif method == TONEMAP_ACES:
            color = tonemap_aces(color)
        elif method == TONEMAP_AGX:
            color = tonemap_agx(color)
        elif method == TONEMAP_KHRONOS_PBR:
            color = tonemap_khronos_pbr(color)
        elif method == TONEMAP_ACES_SRGB:
            color = tonemap_aces_srgb(color)
        else:
            color = tonemap_filmic_hejl(color)

        # Contrast + clamp
        color = wp.vec3(
            wp.clamp(0.5 + (color[0] - 0.5) * contrast, 0.0, 1.0),
            wp.clamp(0.5 + (color[1] - 0.5) * contrast, 0.0, 1.0),
            wp.clamp(0.5 + (color[2] - 0.5) * contrast, 0.0, 1.0),
        )

        # Brightness curve
        inv_brightness = 1.0 / brightness
        color = wp.vec3(
            wp.pow(color[0], inv_brightness),
            wp.pow(color[1], inv_brightness),
            wp.pow(color[2], inv_brightness),
        )

        # Saturation
        luma = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
        color = wp.vec3(
            luma + (color[0] - luma) * saturation,
            luma + (color[1] - luma) * saturation,
            luma + (color[2] - luma) * saturation,
        )

        # Vignette
        w = float(width)
        h = float(height)
        uvx = (float(x) + 0.5) / w
        uvy = (float(y) + 0.5) / h
        dx = (uvx - 0.5) * 2.0
        dy = (uvy - 0.5) * 2.0
        vig = 1.0 - (dx * dx + dy * dy) * vignette
        vig = wp.clamp(vig, 0.0, 1.0)
        color = color * vig

    # Write output
    ldr_output[y, x] = wp.vec4(color[0], color[1], color[2], 1.0)


@wp.kernel
def debug_visualize_kernel(
    color_hdr: wp.array2d(dtype=wp.vec4),
    depth: wp.array2d(dtype=wp.float32),
    motion: wp.array2d(dtype=wp.vec2),
    normal_roughness: wp.array2d(dtype=wp.vec4),
    diffuse: wp.array2d(dtype=wp.vec4),
    specular: wp.array2d(dtype=wp.vec4),
    spec_hit_dist: wp.array2d(dtype=wp.float32),
    ldr_output: wp.array2d(dtype=wp.vec4),
    output_width: int,
    output_height: int,
    src_width: int,
    src_height: int,
    mode: int,
    max_depth: float,
):
    x, y = wp.tid()

    # Map output pixel to source buffer pixel (render vs display resolution).
    sx = int(float(x) * float(src_width) / float(output_width))
    sy = int(float(y) * float(src_height) / float(output_height))
    if sx < 0:
        sx = 0
    if sy < 0:
        sy = 0
    if sx >= src_width:
        sx = src_width - 1
    if sy >= src_height:
        sy = src_height - 1

    # Keep output orientation consistent with tonemap path.
    source_y = src_height - 1 - sy

    out = wp.vec3(1.0, 0.0, 1.0)  # Magenta for invalid mode.
    if mode == OUTPUT_RADIANCE:
        hdr = color_hdr[source_y, sx]
        c = wp.vec3(hdr[0], hdr[1], hdr[2])
        out = wp.vec3(c[0] / (c[0] + 1.0), c[1] / (c[1] + 1.0), c[2] / (c[2] + 1.0))
    elif mode == OUTPUT_DEPTH:
        depth_sample = depth[source_y, sx]
        n = wp.clamp(depth_sample / max_depth, 0.0, 1.0)
        out = wp.vec3(n, n, n)
    elif mode == OUTPUT_MOTION:
        m = motion[source_y, sx]
        mx = wp.clamp(m[0] * 10.0 + 0.5, 0.0, 1.0)
        my = wp.clamp(m[1] * 10.0 + 0.5, 0.0, 1.0)
        out = wp.vec3(mx, my, 0.0)
    elif mode == OUTPUT_NORMAL:
        nr = normal_roughness[source_y, sx]
        out = wp.vec3(nr[0] * 0.5 + 0.5, nr[1] * 0.5 + 0.5, nr[2] * 0.5 + 0.5)
    elif mode == OUTPUT_ROUGHNESS:
        r = normal_roughness[source_y, sx][3]
        out = wp.vec3(r, r, r)
    elif mode == OUTPUT_DIFFUSE:
        diffuse_sample = diffuse[source_y, sx]
        out = wp.vec3(diffuse_sample[0], diffuse_sample[1], diffuse_sample[2])
    elif mode == OUTPUT_SPECULAR:
        s = specular[source_y, sx]
        out = wp.vec3(s[0], s[1], s[2])
    elif mode == OUTPUT_SPEC_HITDIST:
        h = spec_hit_dist[source_y, sx]
        n = wp.clamp(h / max_depth, 0.0, 1.0)
        out = wp.vec3(n, n, n)

    ldr_output[y, x] = wp.vec4(out[0], out[1], out[2], 1.0)


class Tonemapper:
    """
    HDR to LDR tonemapping processor.

    Matches Vulkan reference tonemap behavior.
    """

    def __init__(self, width: int, height: int):
        """
        Create a tonemapper.

        Args:
            width: Image width
            height: Image height
        """
        self.width = width
        self.height = height
        self.exposure = 1.0
        self.method = TONEMAP_ACES_SRGB
        self.is_active = True
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.vignette = 0.0
        self.auto_exposure = False
        self.auto_exposure_target = 0.18
        self.auto_exposure_min_ev = 0.0
        self.auto_exposure_max_ev = 6.0
        self.auto_exposure_brighten_speed = 1.5
        self.auto_exposure_darken_speed = 3.0
        self.auto_exposure_min_luminance_ev = -12.0
        self.auto_exposure_max_luminance_ev = 16.0
        self._auto_exposure_stats = wp.zeros(2, dtype=wp.float32, device="cuda")
        self._auto_exposure_ev = wp.zeros(1, dtype=wp.float32, device="cuda")
        self._auto_exposure_initialized = False

        # Output buffer
        self._ldr_output = wp.zeros((height, width), dtype=wp.vec4, device="cuda")

    @property
    def output(self) -> wp.array:
        """Get the LDR output buffer."""
        return self._ldr_output

    def resize(self, width: int, height: int):
        """Resize the output buffer."""
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self._ldr_output = wp.zeros((height, width), dtype=wp.vec4, device="cuda")

    def configure_auto_exposure(
        self,
        enabled: bool,
        *,
        target_luminance: float | None = None,
        min_ev: float | None = None,
        max_ev: float | None = None,
        brighten_speed: float | None = None,
        darken_speed: float | None = None,
    ) -> None:
        """Configure temporal automatic exposure."""
        was_enabled = self.auto_exposure
        self.auto_exposure = bool(enabled)
        if self.auto_exposure and not was_enabled:
            self._auto_exposure_initialized = False
        if target_luminance is not None:
            self.auto_exposure_target = max(float(target_luminance), 1.0e-6)
        if min_ev is not None:
            self.auto_exposure_min_ev = float(min_ev)
        if max_ev is not None:
            self.auto_exposure_max_ev = float(max_ev)
        if self.auto_exposure_min_ev > self.auto_exposure_max_ev:
            raise ValueError("min_ev must not exceed max_ev")
        if brighten_speed is not None:
            self.auto_exposure_brighten_speed = max(float(brighten_speed), 0.0)
        if darken_speed is not None:
            self.auto_exposure_darken_speed = max(float(darken_speed), 0.0)

    def process(self, hdr_input: wp.array, delta_time: float = 1.0 / 60.0):
        """
        Apply tonemapping to HDR input.

        Args:
            hdr_input: HDR input image (wp.array2d of vec4)
            delta_time: Elapsed display time used for eye adaptation.
        """
        if self.auto_exposure:
            wp.launch(
                _reset_auto_exposure_stats,
                dim=1,
                inputs=[self._auto_exposure_stats],
                device="cuda",
            )
            wp.launch(
                _meter_auto_exposure,
                dim=((self.width + 3) // 4, (self.height + 3) // 4),
                inputs=[
                    hdr_input,
                    self._auto_exposure_stats,
                    int(self.width),
                    int(self.height),
                    self.auto_exposure_min_luminance_ev,
                    self.auto_exposure_max_luminance_ev,
                ],
                device="cuda",
            )
            wp.launch(
                _adapt_auto_exposure,
                dim=1,
                inputs=[
                    self._auto_exposure_stats,
                    self._auto_exposure_ev,
                    self.auto_exposure_target,
                    self.auto_exposure_min_ev,
                    self.auto_exposure_max_ev,
                    self.auto_exposure_brighten_speed,
                    self.auto_exposure_darken_speed,
                    float(delta_time),
                    0 if self._auto_exposure_initialized else 1,
                ],
                device="cuda",
            )
            self._auto_exposure_initialized = True
        wp.launch(
            tonemap_kernel,
            dim=(self.width, self.height),
            inputs=[
                hdr_input,
                self._ldr_output,
                int(self.width),
                int(self.height),
                self.exposure,
                self._auto_exposure_ev,
                1 if self.auto_exposure else 0,
                int(self.method),
                1 if self.is_active else 0,
                self.brightness,
                self.contrast,
                self.saturation,
                self.vignette,
            ],
            device="cuda",
        )

    def process_debug(
        self,
        mode: int,
        color_hdr: wp.array,
        depth: wp.array,
        motion: wp.array,
        normal_roughness: wp.array,
        diffuse: wp.array,
        specular: wp.array,
        spec_hit_dist: wp.array,
        src_width: int,
        src_height: int,
        max_depth: float = 100.0,
    ):
        """Debug visualization from DLSS input buffers."""
        wp.launch(
            debug_visualize_kernel,
            dim=(self.width, self.height),
            inputs=[
                color_hdr,
                depth,
                motion,
                normal_roughness,
                diffuse,
                specular,
                spec_hit_dist,
                self._ldr_output,
                int(self.width),
                int(self.height),
                int(src_width),
                int(src_height),
                int(mode),
                float(max_depth),
            ],
            device="cuda",
        )

    def get_numpy(self):
        """Get the LDR output as a numpy array."""
        return self._ldr_output.numpy()
