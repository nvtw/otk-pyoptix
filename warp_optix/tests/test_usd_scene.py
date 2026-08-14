# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from warp_optix.pathtracing.scene import Scene

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402


def _write_transform_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    body = UsdGeom.Xform.Define(stage, "/body")
    body.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))
    mesh = UsdGeom.Mesh.Define(stage, "/body/mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()


def test_usd_scene_retains_paths_and_local_geometry(tmp_path):
    path = tmp_path / "transform.usda"
    _write_transform_stage(path)
    scene = Scene(None)

    assert scene.load_from_usd(path, apply_stage_units=False)
    usd_scene = scene.usd_scene
    body = usd_scene.require_transform("/body")
    mesh = usd_scene.require_transform("/body/mesh")

    assert usd_scene.get_transform("/missing") is None
    assert usd_scene.transform_count == 2
    assert usd_scene.source_path == str(path.resolve())
    assert usd_scene.get_prim("/body").IsValid()
    np.testing.assert_allclose(scene._meshes[0].vertices[0], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(usd_scene.get_world_transform(mesh)[:3, 3], (2.0, 0.0, 0.0))
    assert usd_scene.instance_ids(body) == ()
    assert usd_scene.instance_ids(mesh) == (0,)

    usd_scene.set_visible(False)
    assert not scene._instance_visibility_cache[0]
    usd_scene.set_visible(True)
    assert scene._instance_visibility_cache[0]

    updated = np.eye(4, dtype=np.float32)
    updated[:3, 3] = (4.0, 5.0, 6.0)
    usd_scene.update_local_transform(body, updated)
    np.testing.assert_allclose(usd_scene.get_world_transform(mesh)[:3, 3], (4.0, 5.0, 6.0))
    np.testing.assert_allclose(scene._instance_transform_cache[0, :3, 3], (4.0, 5.0, 6.0))

    scene.clear()
    with pytest.raises(RuntimeError, match="invalidated"):
        usd_scene.get_world_transform(mesh)


def test_reflected_face_varying_mesh_keeps_world_shading_frame_aligned(tmp_path):
    path = tmp_path / "reflected.usda"
    stage = Usd.Stage.CreateNew(str(path))
    reflected = UsdGeom.Xform.Define(stage, "/reflected")
    reflected.AddScaleOp().Set(Gf.Vec3f(-1.0, 1.0, 1.0))
    mesh = UsdGeom.Mesh.Define(stage, "/reflected/mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(0.0, 1.0)])
    stage.GetRootLayer().Save()

    scene = Scene(None)
    assert scene.load_from_usd(path, apply_stage_units=False, convert_up_axis=False)
    loaded = scene._meshes[0]
    world = scene._instance_transform_cache[0]
    world_positions = (
        np.concatenate((loaded.vertices, np.ones((len(loaded.vertices), 1), dtype=np.float32)), axis=1)
        @ world.T
    )[:, :3]
    tri = loaded.indices[0]
    geometric = np.cross(
        world_positions[tri[1]] - world_positions[tri[0]],
        world_positions[tri[2]] - world_positions[tri[0]],
    )
    normal_matrix = np.linalg.inv(world[:3, :3]).T
    shading = loaded.normals[tri[0]] @ normal_matrix.T

    assert np.dot(geometric, shading) > 0.0
    np.testing.assert_array_equal(tri, (0, 1, 2))


def test_authored_normals_are_preserved_in_robust_two_sided_mode(tmp_path):
    path = tmp_path / "authored_normals.usda"
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, -1.0)] * 3)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(0.0, 1.0)])
    stage.GetRootLayer().Save()

    scene = Scene(None)
    assert scene.load_from_usd(path, apply_stage_units=False, convert_up_axis=False)
    np.testing.assert_allclose(scene._meshes[0].normals, ((0.0, 0.0, -1.0),) * 3)
    np.testing.assert_array_equal(scene._meshes[0].indices[0], (0, 1, 2))
    assert scene._instances[0].double_sided

    strict_scene = Scene(None)
    assert strict_scene.load_from_usd(
        path, apply_stage_units=False, convert_up_axis=False, strict_sidedness=True
    )
    assert not strict_scene._instances[0].double_sided


def test_left_handed_winding_and_double_sided_are_converted_for_optix(tmp_path):
    path = tmp_path / "left_handed.usda"
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    mesh.CreateDoubleSidedAttr(True)
    stage.GetRootLayer().Save()

    scene = Scene(None)
    assert scene.load_from_usd(path, apply_stage_units=False, convert_up_axis=False)
    np.testing.assert_array_equal(scene._meshes[0].indices[0], (0, 2, 1))
    np.testing.assert_allclose(scene._meshes[0].normals, ((0.0, 0.0, -1.0),) * 3)
    assert scene._instances[0].double_sided
