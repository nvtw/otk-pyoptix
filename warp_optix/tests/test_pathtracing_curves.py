# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from warp_optix.pathtracing import Curve, PathTracerAPI, Scene


def _points():
    return np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
        dtype=np.float32,
    )


def test_curve_defaults_to_one_linear_segment_per_consecutive_pair():
    curve = Curve(_points(), radii=np.array((0.1, 0.2, 0.3)))

    np.testing.assert_array_equal(curve.indices, (0, 1))
    np.testing.assert_allclose(curve.radii, (0.1, 0.2, 0.3))
    assert curve.primitive_type == "round_linear"
    assert curve.sbt_offset == 2


def test_curve_accepts_scalar_radius_and_disjoint_segment_starts():
    points = np.vstack((_points(), _points() + (4.0, 0.0, 0.0)))
    curve = Curve(points, radii=0.05, segment_indices=np.array((0, 1, 3, 4)))

    np.testing.assert_array_equal(curve.indices, (0, 1, 3, 4))
    np.testing.assert_allclose(curve.radii, 0.05)


@pytest.mark.parametrize(
    ("radii", "segments", "message"),
    [
        ((0.1, 0.2), None, "shape"),
        ((0.1, 0.0, 0.1), None, "positive"),
        (0.1, (2,), "two consecutive"),
        (0.1, (), "non-empty"),
    ],
)
def test_curve_rejects_invalid_radii_and_segments(radii, segments, message):
    with pytest.raises(ValueError, match=message):
        Curve(_points(), radii=radii, segment_indices=segments)


def test_pathtracer_api_creates_reusable_curve_geometry():
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)

    curve_id = api.create_curve(
        _points(),
        radii=(0.04, 0.06, 0.08),
        segment_indices=(0, 1),
    )
    instance_id = api.create_instance(curve_id)

    assert curve_id == 0
    assert instance_id == 0
    assert scene.curve_count == 1
    assert scene.mesh_count == 0
    assert scene.geometry_count == 1
    assert scene._instances[instance_id].mesh_index == curve_id
    assert scene._meshes[curve_id].material_id == 0
    assert scene.materials.count == 1
