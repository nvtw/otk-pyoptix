# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive PBR path-traced hair ball using native OptiX curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp
import warp_optix as woptix
from example_warp_optix_basic_pathtracing import (
    FreeCameraController,
    _pack_display_rgba8,
)
from warp_optix.pathtracing import PathTracerAPI


_RAINBOW_SRGB_U8 = np.array(
    [
        (229, 57, 53),
        (244, 81, 30),
        (251, 140, 0),
        (255, 179, 0),
        (253, 216, 53),
        (192, 202, 51),
        (124, 179, 66),
        (0, 166, 90),
        (0, 191, 165),
        (0, 172, 193),
        (30, 136, 229),
        (94, 53, 177),
    ],
    dtype=np.float32,
)


def _srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32) / 255.0
    return np.where(
        colors <= 0.04045,
        colors / 12.92,
        ((colors + 0.055) / 1.055) ** 2.4,
    )


def rainbow_segment_slots(points: np.ndarray, segments: int) -> np.ndarray:
    """Assign one palette slot per segment using gently rippled height bands."""
    roots = points[:: segments + 1]
    directions = roots / np.linalg.norm(roots, axis=1, keepdims=True)
    height = 0.5 * (directions[:, 1] + 1.0)
    azimuth = np.arctan2(directions[:, 2], directions[:, 0])
    position = np.clip(height + 0.045 * np.sin(3.0 * azimuth + 0.5), 0.0, 1.0)
    strand_slots = np.minimum(
        (position * len(_RAINBOW_SRGB_U8)).astype(np.uint32),
        len(_RAINBOW_SRGB_U8) - 1,
    )
    return np.repeat(strand_slots, segments)


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--hair-count", type=int, default=4000)
    parser.add_argument(
        "--segments", type=int, default=14, help="Linear segments per hair."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ball-radius", type=float, default=0.8)
    parser.add_argument("--hair-length", type=float, default=0.48)
    parser.add_argument("--curl-radius", type=float, default=0.065)
    parser.add_argument("--curl-turns", type=float, default=1.65)
    parser.add_argument("--root-radius", type=float, default=0.012)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--fps", type=int, default=0, help="Presentation rate cap; 0 is unlimited."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Auto-exit after N frames; 0 runs forever.",
    )
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--title", default="Warp OptiX Path-traced Hair Ball")
    parser.add_argument("--camera-speed", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=0.32)
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--saturation", type=float, default=1.1)
    parser.add_argument("--no-dlss-rr", action="store_true")
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--no-set", action="store_true")
    return parser.parse_args()


def generate_hair_ball(
    hair_count: int,
    segments: int,
    ball_radius: float,
    hair_length: float,
    curl_radius: float,
    curl_turns: float,
    root_radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack disjoint tapered helical hairs into one curve geometry."""
    if hair_count < 1:
        raise ValueError("hair_count must be positive")
    if segments < 2:
        raise ValueError("segments must be at least 2")
    if min(ball_radius, hair_length, root_radius) <= 0.0 or curl_radius < 0.0:
        raise ValueError("ball_radius, hair_length, and root_radius must be positive")

    rng = np.random.default_rng(seed)
    root_id = np.arange(hair_count, dtype=np.float64)
    y = 1.0 - 2.0 * (root_id + 0.5) / hair_count
    azimuth = root_id * (np.pi * (3.0 - np.sqrt(5.0)))
    radial = np.column_stack(
        (
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.cos(azimuth),
            y,
            np.sqrt(np.maximum(0.0, 1.0 - y * y)) * np.sin(azimuth),
        )
    )
    radial += rng.normal(scale=0.012, size=radial.shape)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)

    reference = np.zeros_like(radial)
    reference[:, 1] = 1.0
    reference[np.abs(radial[:, 1]) > 0.9] = (1.0, 0.0, 0.0)
    tangent = np.cross(reference, radial)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(radial, tangent)

    phase = rng.uniform(0.0, 2.0 * np.pi, (hair_count, 1))
    length = hair_length * rng.uniform(0.78, 1.18, (hair_count, 1))
    turns = curl_turns * rng.uniform(0.72, 1.28, (hair_count, 1))
    curl = curl_radius * rng.uniform(0.65, 1.35, (hair_count, 1))
    lean_angle = rng.uniform(0.0, 2.0 * np.pi, (hair_count, 1))
    lean = (
        np.cos(lean_angle)[..., None] * tangent[:, None, :]
        + np.sin(lean_angle)[..., None] * bitangent[:, None, :]
    )

    u = np.linspace(0.0, 1.0, segments + 1, dtype=np.float64)[None, :]
    angle = phase + 2.0 * np.pi * turns * u
    envelope = np.sin(0.5 * np.pi * u) ** 1.25
    helix = (
        np.cos(angle)[..., None] * tangent[:, None, :]
        + np.sin(angle)[..., None] * bitangent[:, None, :]
    )
    center = (ball_radius + length * u)[..., None] * radial[:, None, :]
    points = (
        center
        + (curl * envelope)[..., None] * helix
        + (0.13 * length * u * u)[..., None] * lean
    )

    width_scale = rng.uniform(0.8, 1.2, (hair_count, 1))
    radii = root_radius * width_scale * (0.08 + 0.92 * (1.0 - u) ** 0.75)
    points_per_hair = segments + 1
    segment_indices = (
        np.arange(hair_count, dtype=np.uint32)[:, None] * points_per_hair
        + np.arange(segments, dtype=np.uint32)[None, :]
    ).reshape(-1)
    return (
        points.astype(np.float32).reshape(-1, 3),
        radii.astype(np.float32).reshape(-1),
        segment_indices,
    )


def main():
    args = _parse_args()
    if args.width < 1 or args.height < 1:
        raise ValueError("width and height must be positive")
    points, radii, segment_indices = generate_hair_ball(
        args.hair_count,
        args.segments,
        args.ball_radius,
        args.hair_length,
        args.curl_radius,
        args.curl_turns,
        args.root_radius,
        args.seed,
    )

    wp.init()
    api = PathTracerAPI(
        width=args.width,
        height=args.height,
        enable_dlss_rr=not args.no_dlss_rr,
        enable_set=not args.no_set,
        enable_cuda_graphs=not args.no_cuda_graphs,
    )
    if not api.initialize():
        raise RuntimeError("Failed to initialize pathtracing API.")
    api.tonemap_exposure = args.exposure
    api.tonemap_contrast = args.contrast
    api.tonemap_saturation = args.saturation

    core_mat = api.create_pbr_material((0.025, 0.03, 0.04), 0.72, 0.0)
    palette_materials = np.array(
        [
            api.create_pbr_material(
                color,
                roughness=0.82,
                metallic=0.0,
                ior=1.0,
                specular=0.0,
                clearcoat=0.0,
                emissive=color * 0.45,
            )
            for color in _srgb_to_linear(_RAINBOW_SRGB_U8)
        ],
        dtype=np.uint32,
    )
    segment_material_ids = palette_materials[
        rainbow_segment_slots(points, args.segments)
    ]
    ground_mat = api.create_pbr_material((0.18, 0.2, 0.22), 0.78, 0.0)
    api.add_sphere((0.0, 0.0, 0.0), args.ball_radius * 1.005, 64, core_mat)
    api.add_box((-3.5, -1.42, -3.5), (3.5, -1.38, 3.5), ground_mat)
    curve_id = api.create_curve(
        points,
        radii,
        segment_indices,
        material_id=int(palette_materials[0]),
        material_ids=segment_material_ids,
    )
    api.create_instance(curve_id)
    api.build_scene()
    api.set_use_procedural_sky(True)
    api.set_sky_parameters(
        (-0.55, 0.78, 0.30),
        multiplier=1.25,
        haze=0.18,
        saturation=0.9,
        ground_color=(0.12, 0.13, 0.15),
        sun_disk_intensity=1.2,
        sun_glow_intensity=0.8,
    )

    render_width, render_height = args.width, args.height
    last_elapsed = 0.0

    def _on_resize(width: int, height: int):
        nonlocal render_width, render_height, last_elapsed
        render_width, render_height = int(width), int(height)
        api.resize(render_width, render_height)
        last_elapsed = 0.0

    viewer = woptix.GLInteropViewer(
        width=args.width,
        height=args.height,
        device="cuda",
        title=args.title,
        fps=args.fps,
        on_resize=_on_resize,
        vsync=args.fps > 0,
    )
    controller = FreeCameraController(
        viewer, api, (0.0, 0.18, 3.55), 180.0, -3.0, 42.0, args.camera_speed
    )

    def _render(mapped_image: wp.array, _frame_idx: int, elapsed_sec: float):
        nonlocal last_elapsed
        controller.update(elapsed_sec - last_elapsed)
        last_elapsed = elapsed_sec
        api.render_frame()
        wp.launch(
            _pack_display_rgba8,
            dim=(render_width, render_height),
            inputs=[
                api.viewer.tonemapped_output,
                mapped_image,
                render_width,
                render_height,
            ],
            device="cuda",
        )

    print(
        f"[optix] hair ball: {args.hair_count:,} strands, {len(segment_indices):,} curve segments"
    )
    print("[optix] controls: left-drag look, WASD move, Q/E down/up, wheel zoom")
    viewer.run(_render, max_frames=args.max_frames)
    if args.screenshot is not None:
        from PIL import Image  # noqa: PLC0415

        screenshot = args.screenshot.expanduser().resolve()
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        frame = np.clip(api.get_frame(), 0.0, 1.0)
        Image.fromarray(
            (frame[..., :3] * 255.0 + 0.5).astype(np.uint8), mode="RGB"
        ).save(screenshot)
        print(f"[optix] saved screenshot: {screenshot}")


if __name__ == "__main__":
    main()
