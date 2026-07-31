# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
from warp_optix.pathtracing import PathTracingViewerBackend


class _FakeScene:
    def __init__(self):
        self._meshes = []
        self._instances = []
        self.uploaded_material_ids = None

    def set_instance_material_ids_host(self, material_ids):
        self.uploaded_material_ids = np.asarray(material_ids).copy()


class _FakePathTracerAPI:
    def __init__(self):
        self.scene = _FakeScene()
        self.viewer = SimpleNamespace(tonemapped_output=None)
        self.materials = []
        self.sky = None
        self.camera = None
        self.build_count = 0
        self.refit_count = 0
        self.render_count = 0

    def initialize(self):
        return True

    def set_use_procedural_sky(self, enabled):
        self.procedural_sky = enabled

    def set_sky_parameters(self, **kwargs):
        self.sky = kwargs

    def set_camera_look_at(self, position, target, up, fov):
        self.camera = (np.asarray(position), np.asarray(target), tuple(up), fov)

    def create_pbr_material(self, color, roughness, metallic):
        self.materials.append((tuple(color), roughness, metallic))
        return len(self.materials) - 1

    def create_mesh(self, positions, indices, normals, uvs, material_id):
        self.scene._meshes.append(SimpleNamespace(material_id=material_id))
        return len(self.scene._meshes) - 1

    def create_instance(self, mesh_id):
        self.scene._instances.append(
            SimpleNamespace(
                mesh_index=mesh_id,
                material_id=None,
                visible=True,
                transform=np.eye(4, dtype=np.float32),
            )
        )
        return len(self.scene._instances) - 1

    def set_instance_visible(self, instance_id, visible):
        self.scene._instances[instance_id].visible = visible

    def set_instance_transform_matrix(self, instance_id, matrix):
        self.scene._instances[instance_id].transform = np.asarray(matrix).copy()

    def set_instance_material(self, instance_id, material_id):
        self.scene._instances[instance_id].material_id = material_id

    def build_scene(self):
        self.build_count += 1

    def rebuild_tlas(self):
        self.refit_count += 1

    def render_frame(self):
        self.render_count += 1

    def clear_scene(self):
        self.scene = _FakeScene()

    def close(self):
        pass


def _triangle():
    return (
        np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32),
        np.array(((0, 1, 2),), dtype=np.uint32),
    )


def test_hybrid_sky_and_srgb_material_mapping():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)
    points, indices = _triangle()

    viewer.log_mesh(
        "triangle",
        points,
        indices,
        color=(0.8, 0.8, 0.8),
        roughness=0.25,
        metallic=0.75,
    )

    assert api.procedural_sky is True
    assert api.sky["sun_direction"] == (-0.3, 0.7, 0.5)
    assert api.sky["multiplier"] == 1.5
    assert api.sky["haze"] == 0.03
    assert api.sky["ground_color"] == (0.7, 0.7, 0.75)
    assert api.sky["horizon_blur"] == 0.3
    assert api.sky["sun_glow_intensity"] == 0.8
    np.testing.assert_allclose(
        api.materials[0][0], (0.60382736, 0.60382736, 0.60382736), rtol=1.0e-6
    )
    assert api.materials[0][1:] == (0.25, 0.75)


def test_logged_instances_apply_up_axis_materials_and_frame_lifecycle():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(
        device="cpu", headless=True, num_frames=1, up_axis="Z", api=api
    )
    points, indices = _triangle()
    viewer.log_mesh("triangle", points, indices)
    xforms = np.array(((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),), dtype=np.float32)
    scales = np.array(((2.0, 3.0, 4.0),), dtype=np.float32)
    colors = np.array(((1.0, 0.0, 0.0),), dtype=np.float32)
    materials = np.array(((0.2, 0.6, 0.0, 0.0),), dtype=np.float32)

    viewer.log_instances("batch", "triangle", xforms, scales, colors, materials)
    instance = api.scene._instances[0]
    np.testing.assert_allclose(instance.transform[:3, 3], (1.0, 3.0, -2.0))
    np.testing.assert_allclose(
        instance.transform[:3, :3], ((2, 0, 0), (0, 0, 4), (0, -3, 0))
    )
    assert instance.visible is True
    assert api.materials[instance.material_id][1:] == (0.2, 0.6)

    viewer.end_frame()

    assert api.build_count == 1
    assert api.render_count == 1
    assert not viewer.is_running()


def test_instance_visibility_and_cached_material_updates():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)
    points, indices = _triangle()
    viewer.log_mesh("triangle", points, indices)
    xforms = np.array(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    viewer.log_instances("batch", "triangle", xforms, None, None, None)
    viewer.end_frame()
    viewer.log_instances("batch", "triangle", xforms[:1], None, None, None)
    viewer.end_frame()

    assert api.build_count == 1
    assert api.refit_count == 1
    assert api.scene._instances[0].visible is True
    assert api.scene._instances[1].visible is False
    assert api.scene.uploaded_material_ids is not None
