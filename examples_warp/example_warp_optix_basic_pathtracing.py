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

"""Basic OptiX pathtracing example (glTF-only scene path).

This reproduces the glTF scene flow used by Newton's basic OptiX example,
without any Newton dependency.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import warp as wp
import warp_optix as woptix
from warp_optix.pathtracing import PathTracerAPI


@wp.kernel
def _pack_display_rgba8(
    src: wp.array2d(dtype=wp.vec4),
    dst: wp.array(dtype=wp.uint32),
    width: int,
    height: int,
):
    x, y = wp.tid()
    if x >= width or y >= height:
        return

    c = src[y, x]
    r = wp.uint32(wp.clamp(c[0] * 255.0, 0.0, 255.0))
    g = wp.uint32(wp.clamp(c[1] * 255.0, 0.0, 255.0))
    b = wp.uint32(wp.clamp(c[2] * 255.0, 0.0, 255.0))
    a = wp.uint32(255)
    dst[y * width + x] = (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--scene-gltf",
        type=str,
        default=None,
        help="Optional path to glTF scene file. If omitted, tries known ABeautifulGame locations.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames (0 = run forever).")
    parser.add_argument("--title", type=str, default="Warp OptiX Basic Pathtracing")
    parser.add_argument("--camera-speed", type=float, default=0.5, help="Camera movement speed in scene units/second.")
    parser.add_argument("--no-dlss-rr", action="store_true", help="Disable DLSS Ray Reconstruction.")
    parser.add_argument("--no-set", action="store_true", help="Disable Shader Execution Reordering.")
    return parser.parse_args()


def _resolve_scene_gltf(scene_gltf_arg: str | None) -> Path:
    if scene_gltf_arg:
        scene_gltf = Path(scene_gltf_arg).expanduser().resolve()
        if scene_gltf.exists():
            return scene_gltf
        raise FileNotFoundError(f"--scene-gltf does not exist: {scene_gltf}")

    candidates = [
        Path(r"C:\git\downloaded_resources\ABeautifulGame\glTF\ABeautifulGame.gltf"),
        Path(r"C:\git\single-file-vulkan-pathtracing\assets\gltf\ABeautifulGame\ABeautifulGame.gltf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find ABeautifulGame.gltf. Pass --scene-gltf explicitly or place it in one of:\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}"
    )


class FreeCameraController:
    """Newton-style interactive camera controls for the basic pathtracing example."""

    def __init__(
        self, viewer, api: PathTracerAPI, position, yaw: float, pitch: float, fov: float, movement_speed: float
    ):
        self.window = viewer.window
        self.pyglet = viewer.pyglet
        self.api = api
        self.position = np.array(position, dtype=np.float32)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov = float(fov)
        self._keys_down: set[int] = set()
        self._cam_speed = max(0.0, float(movement_speed))
        self._look_sensitivity = 0.1

        # Push initial camera state.
        self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)
        self.window.push_handlers(self)

    def _forward_right(self) -> tuple[np.ndarray, np.ndarray]:
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        cos_pitch = math.cos(pitch_rad)

        forward = np.array(
            [
                math.sin(yaw_rad) * cos_pitch,
                math.sin(pitch_rad),
                math.cos(yaw_rad) * cos_pitch,
            ],
            dtype=np.float32,
        )
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm > 1.0e-6:
            right /= right_norm
        return forward, right

    def update(self, dt: float):
        try:
            key = self.pyglet.window.key
        except Exception:
            # Fallback to no-op if backend doesn't expose key symbols.
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
                speed *= 4.0
            self.position += (move / move_norm) * speed * max(0.0, min(dt, 0.1))
            self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)

    def on_key_press(self, symbol, _modifiers):
        try:
            key = self.pyglet.window.key
        except Exception:
            key = None

        if key is not None:
            mode_by_key = {
                key._1: 0,  # final
                key._2: 1,  # radiance
                key._3: 2,  # depth
                key._4: 3,  # motion
                key._5: 4,  # normal
                key._6: 5,  # roughness
                key._7: 6,  # diffuse
                key._8: 7,  # specular
                key._9: 8,  # spec hit distance
            }
            if symbol in mode_by_key:
                self.api.set_debug_buffer_mode(mode_by_key[symbol])

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
            self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y):
        self.fov = max(15.0, min(90.0, self.fov - float(scroll_y) * 2.0))
        self.api.set_camera_angles(self.position, self.yaw, self.pitch, self.fov)


def main():
    args = _parse_args()
    scene_gltf = _resolve_scene_gltf(args.scene_gltf)

    wp.init()

    api = PathTracerAPI(
        width=args.width,
        height=args.height,
        enable_dlss_rr=not args.no_dlss_rr,
        enable_set=not args.no_set,
    )
    if not api.initialize():
        raise RuntimeError("Failed to initialize pathtracing API.")

    if not api.load_scene_from_gltf(str(scene_gltf), build_scene=True):
        raise RuntimeError(f"Failed to load glTF scene: {scene_gltf}")

    # Camera preset mirrored from Newton basic OptiX pathtracing example.
    cam_pos = (-0.803, 0.340, 0.327)
    cam_yaw = 115.2
    cam_pitch = -21.8
    cam_fov = 45.0

    render_width = int(args.width)
    render_height = int(args.height)
    last_elapsed = 0.0

    def _on_resize(width: int, height: int):
        nonlocal render_width, render_height, last_elapsed
        render_width = int(width)
        render_height = int(height)
        api.resize(render_width, render_height)
        last_elapsed = 0.0

    viewer = woptix.GLInteropViewer(
        width=args.width,
        height=args.height,
        device="cuda",
        title=args.title,
        fps=args.fps,
        on_resize=_on_resize,
    )

    # Attach Newton-style free camera controls to the viewer window.
    controller = FreeCameraController(viewer, api, cam_pos, cam_yaw, cam_pitch, cam_fov, args.camera_speed)

    def _render(mapped_image: wp.array, _frame_idx: int, elapsed_sec: float):
        nonlocal last_elapsed
        dt = elapsed_sec - last_elapsed
        last_elapsed = elapsed_sec
        controller.update(dt)
        api.render_frame()
        wp.launch(
            _pack_display_rgba8,
            dim=(render_width, render_height),
            inputs=[api.viewer.tonemapped_output, mapped_image, render_width, render_height],
            device="cuda",
        )

    print(f"[optix] loaded glTF scene: {scene_gltf}")
    viewer.run(_render, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
