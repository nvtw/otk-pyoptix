# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
from warp_optix.pathtracing import (
    PathTracerAPI,
    PathTracingRenderer,
    PathTracingViewerBackend,
)
from warp_optix.pathtracing import pathtracing_viewer as viewer_module
from warp_optix.pathtracing import viewer as recording_viewer_module
from warp_optix.pathtracing.scene import Scene


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
        self.debug_mode = 0
        self.dlss_enabled = True
        self.tonemap_exposure = 1.0
        self.tonemap_contrast = 1.0
        self.tonemap_saturation = 1.0
        self.auto_exposure_enabled = False
        self.auto_exposure_config = None
        self.usd_scene = None
        self.usd_load = None

    def initialize(self):
        return True

    def set_use_procedural_sky(self, enabled):
        self.procedural_sky = enabled

    def set_sky_parameters(self, **kwargs):
        self.sky = kwargs

    def set_camera_look_at(self, position, target, up, fov):
        self.camera = (np.asarray(position), np.asarray(target), tuple(up), fov)

    def create_pbr_material(
        self,
        color,
        roughness,
        metallic,
        ior=1.5,
        specular=1.0,
        clearcoat=0.0,
        clearcoat_roughness=0.1,
        u_subdiv=0.0,
        v_subdiv=0.0,
        base_color_scale=0.75,
    ):
        self.materials.append(
            (
                tuple(color),
                roughness,
                metallic,
                ior,
                specular,
                clearcoat,
                clearcoat_roughness,
                u_subdiv,
                v_subdiv,
                base_color_scale,
            )
        )
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

    def set_instance_transform_matrices(self, instance_ids, matrices):
        for instance_id, matrix in zip(instance_ids, matrices, strict=True):
            self.scene._instances[instance_id].transform = np.asarray(matrix).copy()

    def set_instances_visible(self, instance_ids, visible):
        for instance_id in instance_ids:
            self.scene._instances[instance_id].visible = bool(visible)

    def set_instance_material(self, instance_id, material_id):
        self.scene._instances[instance_id].material_id = material_id

    def build_scene(self):
        self.build_count += 1

    def rebuild_tlas(self):
        self.refit_count += 1

    def render_frame(self):
        self.render_count += 1

    def get_frame_uint8(self):
        return np.full((2, 3, 4), 127, dtype=np.uint8)

    def set_debug_buffer_mode(self, mode):
        self.debug_mode = int(mode)

    def set_environment_color(self, color):
        self.environment_color = tuple(color)

    def configure_auto_exposure(self, enabled, **kwargs):
        self.auto_exposure_enabled = bool(enabled)
        self.auto_exposure_config = kwargs

    def clear_scene(self):
        self.scene = _FakeScene()

    def load_scene_from_usd(self, path, **kwargs):
        self.usd_load = (path, kwargs)
        self.usd_scene = object()
        return True

    def close(self):
        pass


def _triangle():
    return (
        np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32),
        np.array(((0, 1, 2),), dtype=np.uint32),
    )


def test_scene_batches_instance_transforms_and_visibility():
    scene = Scene(None)
    instance_ids = [scene.add_instance(0) for _ in range(3)]
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    transforms[:, 0, 3] = (1.0, 2.0, 3.0)

    scene.set_instance_transforms_batch(instance_ids, transforms)
    scene.set_instances_visible_batch(instance_ids[1:], False)

    np.testing.assert_array_equal(scene._instance_transform_cache[:3], transforms)
    np.testing.assert_array_equal(
        scene._instance_visibility_cache[:3], (True, False, False)
    )


def test_reference_sky_and_srgb_color_mapping():
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
    assert api.sky is None
    np.testing.assert_allclose(
        api.materials[0][0], (0.60382736, 0.60382736, 0.60382736), rtol=1.0e-6
    )
    assert api.materials[0][1:3] == (0.25, 0.75)
    assert api.materials[0][3:] == (1.5, 1.0, 0.0, 0.1, 0.0, 0.0, 0.75)

    viewer.set_sky_parameters(
        sun_direction=(0.0, 1.0, 0.0),
        ground_color=(0.5, 0.25, 0.75),
        night_color=(0.2, 0.4, 0.6),
    )
    viewer.set_environment_color((0.1, 0.5, 0.9))
    viewer.tonemap_saturation = 1.25
    viewer.tonemap_contrast = 1.1
    viewer.tonemap_exposure = 0.75
    assert viewer.tonemap_exposure == 0.75
    assert viewer.tonemap_saturation == 1.25
    assert viewer.tonemap_contrast == 1.1

    viewer.configure_auto_exposure(True, min_ev=-5.0, max_ev=7.0)
    assert viewer.auto_exposure_enabled
    assert api.auto_exposure_config == {
        "target_luminance": None,
        "min_ev": -5.0,
        "max_ev": 7.0,
        "brighten_speed": None,
        "darken_speed": None,
    }
    np.testing.assert_allclose(
        api.sky["ground_color"], (0.21404114, 0.05087609, 0.52252155), rtol=1.0e-6
    )
    np.testing.assert_allclose(
        api.sky["night_color"], (0.03310477, 0.13286832, 0.31854678), rtol=1.0e-6
    )
    np.testing.assert_allclose(
        api.environment_color, (0.01002283, 0.21404114, 0.78741229), rtol=1.0e-6
    )


def test_sky_parameters_normalize_sun_direction():
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace()
    api.initialize = lambda: True

    api.set_sky_parameters((0.0, 2.0, 1.0), grayscale=True)

    assert api._viewer.sky_grayscale is True
    np.testing.assert_allclose(
        api._viewer.sky_sun_direction, (0.0, 0.8944272, 0.4472136), rtol=1.0e-6
    )
    with pytest.raises(ValueError, match="nonzero"):
        api.set_sky_parameters((0.0, 0.0, 0.0))


def test_load_scene_from_usd_forwards_options_and_builds():
    calls = []
    scene = SimpleNamespace(
        load_from_usd=lambda *args, **kwargs: calls.append((args, kwargs)) or True
    )
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    api.initialize = lambda: True
    api.build_scene = lambda: calls.append(("build", {}))

    assert api.load_scene_from_usd(
        "asset.usd",
        clear_existing=False,
        apply_stage_units=False,
        convert_up_axis=False,
    )
    assert calls == [
        (
            ("asset.usd",),
            {
                "root_transform": None,
                "clear_existing": False,
                "apply_stage_units": False,
                "convert_up_axis": False,
                "max_texture_size": None,
                "max_texture_memory_bytes": None,
                "strict_sidedness": False,
                "enable_emissive_materials": True,
                "load_usd_lights": False,
                "usd_light_radius": None,
            },
        ),
        ("build", {}),
    ]


def test_resize_releases_old_dlss_resources_before_native_reallocation(monkeypatch):
    viewer = PathTracingRenderer.__new__(PathTracingRenderer)
    viewer.width = 800
    viewer.height = 600
    viewer._render_stream = object()
    viewer.frame_index = 9
    events = []
    viewer.camera = SimpleNamespace(
        set_aspect_ratio=lambda width, height: events.append(("aspect", width, height))
    )
    viewer._tonemapper = SimpleNamespace(
        resize=lambda width, height: events.append(("tonemap", width, height))
    )
    viewer._sync_prev_camera_matrices_to_current = lambda: events.append(("camera",))
    viewer._destroy_dlss_rr = lambda **kwargs: events.append(("destroy", kwargs))
    viewer._init_dlss_rr = lambda: events.append(("init",))
    monkeypatch.setattr(
        viewer_module.wp, "synchronize_stream", lambda stream: events.append(("sync",))
    )

    viewer.resize(3840, 2160)

    assert viewer.width == 3840
    assert viewer.height == 2160
    assert events[:2] == [("sync",), ("destroy", {"restore_resolution": False})]
    assert events[-2:] == [("init",), ("tonemap", 3840, 2160)]


def test_load_scene_from_usd_can_enable_composed_environment():
    calls = []
    scene = SimpleNamespace(
        load_from_usd=lambda *args, **kwargs: True,
        usd_environment_path="environment.hdr",
    )
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(_scene=scene)
    api.initialize = lambda: True
    api.build_scene = lambda: calls.append(("build", {}))
    api.set_environment_hdr = lambda path, scaling=1.0: calls.append((path, scaling))

    assert api.load_scene_from_usd(
        "asset.usd", load_usd_environment=True, usd_environment_scale=0.25
    )
    assert calls == [("environment.hdr", 0.25), ("build", {})]


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
    materials = np.array(((0.2, 0.6, 12.5, 7.25),), dtype=np.float32)

    viewer.log_instances("batch", "triangle", xforms, scales, colors, materials)
    instance = api.scene._instances[0]
    np.testing.assert_allclose(instance.transform[:3, 3], (1.0, 3.0, -2.0))
    np.testing.assert_allclose(
        instance.transform[:3, :3], ((2, 0, 0), (0, 0, 4), (0, -3, 0))
    )
    assert instance.visible is True
    assert api.materials[instance.material_id][1:3] == (0.2, 0.6)
    assert api.materials[instance.material_id][3:] == (
        1.5,
        1.0,
        0.0,
        0.1,
        12.5,
        7.25,
        0.75,
    )

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
    viewer.update_instance_transforms("batch", xforms[:1])
    viewer.end_frame()

    assert api.build_count == 1
    assert api.refit_count == 1
    assert api.scene._instances[0].visible is True
    assert api.scene._instances[1].visible is False
    assert api.scene.uploaded_material_ids is not None


class _FakePicking:
    def __init__(self, model, pick_stiffness, pick_damping):
        self.model = model
        self.settings = (pick_stiffness, pick_damping)
        self.active = False
        self.applied = []
        self.picked = None
        self.updated = None

    def is_picking(self):
        return self.active

    def _apply_picking_force(self, state):
        self.applied.append(state)

    def pick(self, state, origin, direction):
        self.active = True
        self.picked = (state, tuple(origin), tuple(direction))

    def update(self, origin, direction):
        self.updated = (tuple(origin), tuple(direction))

    def release(self):
        self.active = False


class _FakeVideoWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(np.asarray(frame).copy())

    def close(self):
        self.closed = True


def test_optional_picking_uses_physics_camera_ray_and_applies_forces():
    api = _FakePathTracerAPI()
    created = []

    def factory(*args, **kwargs):
        picking = _FakePicking(*args, **kwargs)
        created.append(picking)
        return picking

    viewer = PathTracingViewerBackend(
        device="cpu", headless=True, api=api, up_axis="Z", picking_factory=factory
    )
    model = SimpleNamespace(up_axis=2)
    viewer.set_model(model)
    picking = created[0]
    assert picking.settings == (10000.0, 1000.0)

    state = object()
    origin, direction = viewer._get_ray_from_mouse(viewer.width / 2, viewer.height / 2)
    np.testing.assert_allclose(origin, viewer._camera_position)
    np.testing.assert_allclose(direction, viewer._physics_camera_front(), atol=1.0e-6)
    picking.pick(state, origin, direction)
    viewer.log_state(state)
    viewer.apply_forces(state)
    assert picking.applied == [state]


def test_recording_debug_and_bridge_transform_compatibility(tmp_path):
    api = _FakePathTracerAPI()
    writer = _FakeVideoWriter()
    writer_options = {}

    def writer_factory(*_args, **kwargs):
        writer_options.update(kwargs)
        return writer

    viewer = PathTracingViewerBackend(
        device="cpu",
        headless=True,
        api=api,
        recording_writer_factory=writer_factory,
    )

    viewer.set_debug_buffer_mode(7)
    assert api.debug_mode == 7
    key = SimpleNamespace(
        SPACE=10,
        ESCAPE=11,
        R=12,
        T=13,
        _0=20,
        _1=21,
        _2=22,
        _3=23,
        _4=24,
        _5=25,
        _6=26,
        _7=27,
        _8=28,
        BACKSPACE=29,
    )
    viewer._presenter = SimpleNamespace(
        pyglet=SimpleNamespace(window=SimpleNamespace(key=key))
    )
    viewer.on_key_press(key._1, None)
    assert api.debug_mode == 2
    viewer.on_key_press(key._1, None)
    assert api.debug_mode == 0
    viewer._presenter = None
    path = viewer.start_recording(tmp_path / "capture.mp4", frame_skip=2)
    assert path.endswith("capture.mp4")
    viewer.end_frame()
    viewer.end_frame()
    viewer.end_frame()
    viewer.stop_recording()

    assert len(writer.frames) == 2
    assert writer.frames[0].shape == (2, 3, 3)
    assert writer_options["output_params"][:2] == ["-vf", "vflip"]
    assert writer.closed
    assert viewer.get_instance_transform_gl_buffer() == 0
    assert viewer.get_instance_transform_capacity() == 10000
    assert not viewer.is_gpu_transform_available()


def test_recording_defaults_to_system_videos_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(recording_viewer_module, "_system_videos_dir", lambda: tmp_path)
    writer = _FakeVideoWriter()
    viewer = PathTracingViewerBackend(
        device="cpu",
        headless=True,
        api=_FakePathTracerAPI(),
        recording_writer_factory=lambda *_args, **_kwargs: writer,
    )

    path = viewer.start_recording()
    viewer.stop_recording()

    assert path.startswith(str(tmp_path / "NewtonRecordings"))
    assert path.endswith(".mp4")
    assert writer.closed


def test_viewer_loads_usd_through_public_backend_api():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, api=api)
    viewer._mesh_ids["old"] = 1

    loaded = viewer.load_scene_from_usd(
        "scene.usd",
        max_texture_size=2048,
        load_usd_environment=True,
    )

    assert loaded
    assert api.usd_load == (
        "scene.usd",
        {
            "clear_existing": True,
            "apply_stage_units": True,
            "convert_up_axis": True,
            "max_texture_size": 2048,
            "max_texture_memory_bytes": None,
            "strict_sidedness": False,
            "enable_emissive_materials": True,
            "load_usd_lights": False,
            "usd_light_radius": None,
            "load_usd_environment": True,
            "usd_environment_scale": 1.0,
        },
    )
    assert viewer.usd_scene is api.usd_scene
    assert viewer._mesh_ids == {}


def test_renderer_space_look_at_preserves_level_z_up_camera():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(device="cpu", headless=True, up_axis="Z", api=api)
    viewer._ensure_initialized()

    position = np.array((1.0, 2.0, 3.0), dtype=np.float32)
    target = np.array((4.0, 6.0, 3.0), dtype=np.float32)
    viewer.set_camera_look_at(position, target, fov=52.0, renderer_space=True)

    camera_position, camera_target, camera_up, camera_fov = api.camera
    np.testing.assert_allclose(camera_position, position)
    np.testing.assert_allclose(
        camera_target - camera_position,
        (target - position) / np.linalg.norm(target - position),
        atol=1.0e-6,
    )
    assert camera_up == (0.0, 1.0, 0.0)
    assert camera_fov == 52.0


def test_instance_capacity_is_enforced():
    api = _FakePathTracerAPI()
    viewer = PathTracingViewerBackend(
        device="cpu", headless=True, api=api, max_instances=1
    )
    points, indices = _triangle()
    viewer.log_mesh("triangle", points, indices)
    xforms = np.array(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    with pytest.raises(RuntimeError, match="instance capacity"):
        viewer.log_instances("batch", "triangle", xforms, None, None, None)


def test_tlas_refit_preserves_temporal_sequence():
    class _Scene:
        def __init__(self):
            self.refit_count = 0

        def rebuild_tlas(self):
            self.refit_count += 1

    scene = _Scene()
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(
        _scene=scene,
        sample_index=17,
        frame_index=9,
    )

    api.rebuild_tlas()

    assert scene.refit_count == 1
    assert api._viewer.sample_index == 17
    assert api._viewer.frame_index == 9


def test_scene_rebuild_resets_temporal_history():
    class _Scene:
        def build(self, optix):
            self.optix = optix

    scene = _Scene()
    api = PathTracerAPI.__new__(PathTracerAPI)
    api._viewer = SimpleNamespace(
        _scene=scene,
        _optix=object(),
        _dlss_reset_history=False,
        _prev_instance_transforms_valid=True,
        sample_index=17,
        frame_index=9,
        _create_sbt=lambda: None,
        _sync_prev_camera_matrices_to_current=lambda: None,
    )

    api.build_scene()

    assert scene.optix is api._viewer._optix
    assert api._viewer.sample_index == 0
    assert api._viewer.frame_index == 0
    assert api._viewer._dlss_reset_history is True
    assert api._viewer._prev_instance_transforms_valid is False

    api._viewer._dlss_reset_history = False
    api.reset_temporal_history()
    assert api._viewer._dlss_reset_history is True


def test_quality_modes_and_runtime_ray_budgets():
    assert (
        PathTracingRenderer._normalize_dlss_quality("ultra-performance")
        == "ultra_performance"
    )
    assert PathTracingRenderer._normalize_dlss_quality("native") == "native"
    with pytest.raises(ValueError, match="dlss_quality"):
        PathTracingRenderer._normalize_dlss_quality("cinematic")

    renderer = PathTracingRenderer.__new__(PathTracingRenderer)
    renderer._pipeline_max_bounces = 6
    renderer.max_bounces = 4
    renderer.direct_light_samples = 1
    renderer.russian_roulette_start_bounce = 3
    renderer.samples_per_frame = 1

    renderer.set_ray_budget(
        max_bounces=2,
        direct_light_samples=3,
        russian_roulette_start_bounce=5,
        samples_per_frame=4,
    )

    assert renderer.max_bounces == 2
    assert renderer.direct_light_samples == 3
    assert renderer.russian_roulette_start_bounce == 5
    assert renderer.samples_per_frame == 4
    with pytest.raises(ValueError, match="compiled limit"):
        renderer.set_ray_budget(max_bounces=7)
    with pytest.raises(ValueError, match="at least 1"):
        renderer.set_ray_budget(direct_light_samples=0)
    with pytest.raises(ValueError, match="at least 1"):
        renderer.set_ray_budget(russian_roulette_start_bounce=0)
