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

"""Minimal OpenGL viewer with CUDA zero-copy PBO interop."""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable

import warp as wp


class OptixGLInteropViewer:
    @staticmethod
    def _create_texture_2d_compat(pyglet_module, gl_module, width: int, height: int):
        try:
            # pyglet 2.x style (legacy rectangle arg still accepted on some versions)
            return pyglet_module.image.Texture.create(
                width=width, height=height, rectangle=False
            )
        except TypeError:
            try:
                # pyglet compatibility path used by the working viewer implementation
                return pyglet_module.image.Texture.create(
                    width=width, height=height, target=gl_module.GL_TEXTURE_2D
                )
            except TypeError:
                # fallback for older pyglet variants
                return pyglet_module.image.Texture.create(width=width, height=height)

    def __init__(
        self,
        width: int,
        height: int,
        device: str,
        title: str = "Warp OptiX Tiny Raytracer",
        fps: int = 60,
        on_resize: Callable[[int, int], None] | None = None,
        on_draw_overlay: Callable[[], None] | None = None,
        vsync: bool = True,
        fallback_to_copy: bool = True,
    ):
        import pyglet
        from pyglet import gl

        self.width = width
        self.height = height
        self.device = device
        self.pyglet = pyglet
        self.gl = gl
        self.frame_index = 0
        self.start_time = time.perf_counter()
        self.max_frames = 0
        self.closed = False
        self._render_callback: Callable[[wp.array, int, float], None] | None = None
        self._on_resize_callback = on_resize
        self._on_draw_overlay = on_draw_overlay
        self._fallback_to_copy = bool(fallback_to_copy)

        self.window = pyglet.window.Window(
            width=width, height=height, caption=title, vsync=bool(vsync), resizable=True
        )
        self.window.push_handlers(
            on_draw=self._on_draw, on_close=self._on_close, on_resize=self._on_resize
        )
        self._recreate_gl_resources()
        pyglet.clock.schedule_interval(self._update, 1.0 / float(max(1, fps)))

    def _recreate_gl_resources(self):
        # Recreate texture + PBO + CUDA interop for current resolution.
        gl = self.gl

        self.cuda_gl = None
        if hasattr(self, "pbo") and self.pbo is not None:
            gl.glDeleteBuffers(1, self.pbo)

        self.texture = self._create_texture_2d_compat(
            self.pyglet, gl, self.width, self.height
        )
        self.texture.min_filter = gl.GL_NEAREST
        self.texture.mag_filter = gl.GL_NEAREST
        self.sprite = self.pyglet.sprite.Sprite(self.texture, x=0, y=0)

        self.pbo = gl.GLuint()
        gl.glGenBuffers(1, self.pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBufferData(
            gl.GL_PIXEL_UNPACK_BUFFER,
            self.width * self.height * 4,
            None,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self.cuda_gl = wp.RegisteredGLBuffer(
            int(self.pbo.value),
            device=self.device,
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            fallback_to_copy=self._fallback_to_copy,
        )
        gl.glViewport(0, 0, self.width, self.height)

    def run(
        self,
        render_callback: Callable[[wp.array, int, float], None],
        max_frames: int = 0,
    ):
        self._render_callback = render_callback
        self.max_frames = max_frames
        self.pyglet.app.run()

    def render_once(
        self, render_callback: Callable[[wp.array, int, float], None] | None = None
    ):
        """Render and present one frame without entering pyglet's event loop."""
        if self.closed:
            return
        if render_callback is not None:
            self._render_callback = render_callback
        self.window.dispatch_events()
        if self.closed:
            return
        self._render_frame()

    def _update(self, _dt):
        self._render_frame()

    def _render_frame(self):
        if self._render_callback is None:
            return
        with wp.ScopedDevice(self.device):
            mapped = self.cuda_gl.map(
                dtype=wp.uint32, shape=(self.width * self.height,)
            )
            elapsed = time.perf_counter() - self.start_time
            self._render_callback(mapped, self.frame_index, elapsed)
            wp.synchronize_device(self.device)
            self.cuda_gl.unmap()

        # Drive presentation explicitly each update so rendering does not depend
        # on backend-specific on_draw invalidation behavior.
        self.window.switch_to()
        self.window.dispatch_event("on_draw")
        self.window.flip()

        self.frame_index += 1
        if self.max_frames > 0 and self.frame_index >= self.max_frames:
            self.pyglet.app.exit()

    def _on_draw(self):
        gl = self.gl
        self.window.clear()
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBindTexture(self.texture.target, self.texture.id)
        gl.glTexSubImage2D(
            self.texture.target,
            0,
            0,
            0,
            self.width,
            self.height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindTexture(self.texture.target, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.sprite.draw()
        if self._on_draw_overlay is not None:
            self._on_draw_overlay()

    def _on_close(self):
        if self.closed:
            return
        self.closed = True
        if self.cuda_gl is not None:
            self.cuda_gl = None
        if self.pbo is not None:
            self.gl.glDeleteBuffers(1, self.pbo)
            self.pbo = None
        self.pyglet.app.exit()

    def close(self):
        """Release presentation resources and close the window."""
        if self.closed:
            return
        self._on_close()
        self.window.close()

    def is_running(self) -> bool:
        """Return whether the presentation window remains open."""
        return not self.closed and not self.window.has_exit

    def _on_resize(self, width: int, height: int):
        if width <= 0 or height <= 0:
            return

        self.width = int(width)
        self.height = int(height)
        self._recreate_gl_resources()

        if self._on_resize_callback is not None:
            self._on_resize_callback(self.width, self.height)
