# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from warp_optix.pathtracing import Curve, Mesh, PathTracerAPI, Scene


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


def test_cubic_bezier_curve_uses_native_topology_and_sbt_slot():
    points = np.zeros((7, 3), dtype=np.float32)
    points[:, 0] = np.arange(7)
    curve = Curve(points, radii=np.linspace(0.1, 0.04, 7), basis="cubic_bezier")

    np.testing.assert_array_equal(curve.indices, (0, 3))
    assert curve.basis == "cubic_bezier"
    assert curve.primitive_type == "round_cubic_bezier"
    assert curve.control_point_count == 4
    assert curve.sbt_offset == 4
    assert curve.geometry_type == 2


@pytest.mark.parametrize(
    ("points", "indices", "message"),
    [
        (np.zeros((6, 3), dtype=np.float32), None, "3\\*N \\+ 1"),
        (np.zeros((7, 3), dtype=np.float32), (4,), "four consecutive"),
    ],
)
def test_cubic_bezier_curve_rejects_invalid_topology(points, indices, message):
    with pytest.raises(ValueError, match=message):
        Curve(points, radii=0.1, segment_indices=indices, basis="cubic_bezier")


def test_pathtracer_api_creates_cubic_bezier_curve():
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    curve_id = api.create_curve(
        np.zeros((4, 3), dtype=np.float32), radii=0.1, basis="cubic_bezier"
    )

    assert scene._meshes[curve_id].basis == "cubic_bezier"


def test_curve_accepts_scalar_radius_and_disjoint_segment_starts():
    points = np.vstack((_points(), _points() + (4.0, 0.0, 0.0)))
    curve = Curve(points, radii=0.05, segment_indices=np.array((0, 1, 3, 4)))

    np.testing.assert_array_equal(curve.indices, (0, 1, 3, 4))
    np.testing.assert_allclose(curve.radii, 0.05)


def test_curve_and_mesh_accept_one_material_id_per_primitive():
    curve = Curve(_points(), radii=0.05, material_id=7, material_ids=(2, 4))
    mesh = Mesh(
        vertices=np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
            dtype=np.float32,
        ),
        indices=np.array(((0, 1, 2), (1, 3, 2)), dtype=np.uint32),
        material_id=7,
        material_ids=(3, 5),
    )

    np.testing.assert_array_equal(curve.material_ids, (2, 4))
    np.testing.assert_array_equal(mesh.material_ids, (3, 5))


@pytest.mark.parametrize("material_ids", [(1,), (1, 2, 3), (0.0, 1.0), (-1, 0)])
def test_curve_rejects_invalid_per_segment_material_ids(material_ids):
    with pytest.raises(ValueError, match="material_ids"):
        Curve(_points(), radii=0.05, material_ids=material_ids)


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


def test_pathtracer_api_pbr_material_accepts_emissive_term():
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)

    material_id = api.create_pbr_material(
        (0.2, 0.3, 0.4), roughness=0.8, metallic=0.0, emissive=(0.1, 0.2, 0.3)
    )

    material = scene.materials._materials[material_id]
    np.testing.assert_allclose(material["pbrBaseColorFactor"][:3], (0.2, 0.3, 0.4))
    np.testing.assert_allclose(material["emissiveFactor"], (0.1, 0.2, 0.3))


def test_pathtracer_api_creates_reusable_curve_geometry():
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    scene.materials.add_diffuse((0.2, 0.3, 0.4))
    scene.materials.add_diffuse((0.8, 0.7, 0.6))

    curve_id = api.create_curve(
        _points(),
        radii=(0.04, 0.06, 0.08),
        segment_indices=(0, 1),
        material_ids=(0, 1),
    )
    instance_id = api.create_instance(curve_id)

    assert curve_id == 0
    assert instance_id == 0
    assert scene.curve_count == 1
    assert scene.mesh_count == 0
    assert scene.geometry_count == 1
    assert scene._instances[instance_id].mesh_index == curve_id
    assert scene._meshes[curve_id].material_id == 0
    np.testing.assert_array_equal(scene._meshes[curve_id].material_ids, (0, 1))
    assert scene.materials.count == 2


def test_dynamic_curve_accepts_zero_radius_but_static_curve_does_not():
    with pytest.raises(ValueError, match="positive"):
        Curve(_points(), radii=(0.1, 0.0, 0.1))

    curve = Curve(_points(), radii=(0.1, 0.0, 0.1), dynamic=True)

    assert curve.dynamic
    np.testing.assert_allclose(curve.radii, (0.1, 0.0, 0.1))


def test_pathtracer_api_creates_one_dynamic_curve_instance_for_arrow_batch():
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    scene.materials.add_diffuse((0.2, 0.3, 0.4))
    scene.materials.add_diffuse((0.8, 0.7, 0.6))

    batch = api.create_arrow_batch(
        capacity=4,
        small_radius=0.02,
        large_radius=0.06,
        tip_length_ratio=0.25,
        material_ids=(0, 1, 0, 1),
    )
    curve = scene._meshes[batch.geometry_id]

    assert batch.capacity == 4
    assert scene.geometry_count == 1
    assert scene.instance_count == 1
    assert scene._instances[batch.instance_id].mesh_index == batch.geometry_id
    assert curve.dynamic
    np.testing.assert_array_equal(curve.indices, (0, 2, 4, 6, 8, 10, 12, 14))
    np.testing.assert_array_equal(curve.material_ids, (0, 0, 1, 1, 0, 0, 1, 1))
    np.testing.assert_allclose(curve.radii, 0.0)


def test_host_arrow_batch_update_builds_shaft_tip_and_clears_inactive_slots():
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    scene.materials.add_diffuse((0.2, 0.3, 0.4))
    scene.materials.add_diffuse((0.8, 0.7, 0.6))
    batch = api.create_arrow_batch(3, 0.02, 0.06, 0.25)

    api.update_arrow_batch(
        batch,
        starts=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ends=((0.0, 0.0, 2.0), (2.0, 0.0, 0.0)),
        material_ids=(1, 0),
    )
    curve = scene._meshes[batch.geometry_id]
    points = curve.vertices.reshape(3, 4, 3)
    widths = curve.radii.reshape(3, 4)

    np.testing.assert_allclose(
        points[0],
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.5), (0.0, 0.0, 1.5), (0.0, 0.0, 2.0)),
    )
    np.testing.assert_allclose(
        points[1],
        ((1.0, 0.0, 0.0), (1.75, 0.0, 0.0), (1.75, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    np.testing.assert_allclose(widths[:2], ((0.02, 0.02, 0.06, 0.0),) * 2)
    np.testing.assert_allclose(points[2], 0.0)
    np.testing.assert_allclose(widths[2], 0.0)
    np.testing.assert_array_equal(curve.material_ids[:4], (1, 1, 0, 0))
    assert batch.active_count == 2

    api.update_arrow_batch(
        batch,
        starts=((3.0, 0.0, 0.0),),
        ends=((3.0, 1.0, 0.0),),
    )
    np.testing.assert_allclose(curve.vertices[4:], 0.0)
    np.testing.assert_allclose(curve.radii[4:], 0.0)
    assert batch.active_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": 0, "small_radius": 0.1, "large_radius": 0.2}, "capacity"),
        ({"capacity": 1, "small_radius": 0.0, "large_radius": 0.2}, "positive"),
        ({"capacity": 1, "small_radius": 0.2, "large_radius": 0.1}, "at least"),
        (
            {
                "capacity": 1,
                "small_radius": 0.1,
                "large_radius": 0.2,
                "tip_length_ratio": 1.0,
            },
            "between",
        ),
    ],
)
def test_arrow_batch_rejects_invalid_shape_parameters(kwargs, message):
    scene = Scene(None)
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    with pytest.raises(ValueError, match=message):
        api.create_arrow_batch(**kwargs)
