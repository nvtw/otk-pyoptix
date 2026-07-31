# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal latest-development Newton simulation using the OptiX viewer backend."""

from __future__ import annotations

import argparse

import newton
import warp as wp
from newton.viewer import ViewerBase
from warp_optix.pathtracing import PathTracingViewerBackend


class ViewerOptix(PathTracingViewerBackend, ViewerBase):
    """Thin Newton integration; the renderer package has no Newton dependency."""


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-dlss-rr", action="store_true")
    parser.add_argument("--no-imgui", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    wp.init()

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=wp.quat_identity())
    )
    builder.add_shape_box(
        body,
        hx=0.6,
        hy=0.6,
        hz=0.6,
        color=wp.vec3(0.2, 0.45, 0.85),
    )
    builder.add_ground_plane()
    model = builder.finalize(device="cuda")

    solver = newton.solvers.SolverXPBD(model, iterations=4)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    state_0 = model.state()
    state_1 = model.state()

    viewer = ViewerOptix(
        width=args.width,
        height=args.height,
        max_instances=10000,
        num_frames=args.max_frames,
        enable_dlss_rr=not args.no_dlss_rr,
        enable_imgui=not args.no_imgui,
    )
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(6.0, -6.0, 4.0), -15.0, 135.0)

    frame_dt = 1.0 / 60.0
    sim_time = 0.0
    try:
        while viewer.is_running():
            if not viewer.is_paused():
                state_0.clear_forces()
                viewer.apply_forces(state_0)
                collision_pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, frame_dt)
                state_0, state_1 = state_1, state_0
                sim_time += frame_dt

            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
