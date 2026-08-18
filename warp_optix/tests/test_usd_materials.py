# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from warp_optix.pathtracing import usd_loader
from warp_optix.pathtracing.usd_loader import (
    _ambient_light_from_custom_layer_data,
    _decode_packed_orm,
    _decode_udim,
    _mdl_texture_inputs,
    _fit_textures_to_budget,
    _normalize_mesh_uvs_for_udim_atlas,
    _rebase_missing_texture,
)


def test_texture_budget_proportionally_downscales_large_atlas():
    textures = [
        np.zeros((16, 16, 4), dtype=np.uint8),
        np.zeros((16, 16, 4), dtype=np.uint8),
    ]

    resized = _fit_textures_to_budget(textures, max_bytes=512)

    assert sum(texture.nbytes for texture in resized) <= 512
    assert resized[0].shape == resized[1].shape
    assert resized[0].shape[0] < textures[0].shape[0]
    assert all(texture.flags.c_contiguous for texture in resized)
    assert _fit_textures_to_budget(textures, max_bytes=4096) is textures


def test_texture_budget_uses_available_gpu_memory(monkeypatch):
    """Preserve source textures when live free VRAM provides enough headroom."""
    import warp as wp

    class DeviceStub:
        free_memory = 4096

    textures = [np.zeros((16, 16, 4), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(wp, "get_device", lambda _alias: DeviceStub())
    monkeypatch.setattr(usd_loader, "_MIN_TEXTURE_MEMORY_BUDGET_BYTES", 256)
    monkeypatch.setattr(usd_loader, "_TEXTURE_VRAM_RESERVE_BYTES", 128)

    assert _fit_textures_to_budget(textures) is textures


def test_texture_budget_retains_headroom_on_small_gpu(monkeypatch):
    """Downscale textures when they would consume reserved GPU headroom."""
    import warp as wp

    class DeviceStub:
        free_memory = 1024

    textures = [np.zeros((16, 16, 4), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(wp, "get_device", lambda _alias: DeviceStub())
    monkeypatch.setattr(usd_loader, "_MIN_TEXTURE_MEMORY_BUDGET_BYTES", 256)
    monkeypatch.setattr(usd_loader, "_TEXTURE_VRAM_RESERVE_BYTES", 128)

    resized = _fit_textures_to_budget(textures)

    assert sum(texture.nbytes for texture in resized) <= 512


def test_missing_texture_rebases_to_stage_local_texture_tree(tmp_path):
    stage_path = tmp_path / "variant" / "scene.usd"
    stage_path.parent.mkdir()
    stage_path.touch()
    texture = stage_path.parent / "textures" / "kit" / "albedo.1001.png"
    texture.parent.mkdir(parents=True)
    texture.touch()

    broken = tmp_path / "textures" / "kit" / "albedo.<UDIM>.png"

    assert _rebase_missing_texture(stage_path, broken) == texture.with_name(
        "albedo.<UDIM>.png"
    )


def test_omniverse_ambient_light_metadata_is_preserved():
    custom_data = {
        "renderSettings": {
            "rtx:sceneDb:ambientLightColor": (0.68, 0.68, 0.95),
            "rtx:sceneDb:ambientLightIntensity": 1.5,
        }
    }

    np.testing.assert_allclose(
        _ambient_light_from_custom_layer_data(custom_data),
        (1.02, 1.02, 1.425),
    )


class _Input:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def GetBaseName(self):
        return self._name

    def Get(self):
        return self._value


class _Shader:
    def __init__(self, inputs, shader_id="", source_asset="", sub_identifier=""):
        self._inputs = [_Input(name, value) for name, value in inputs.items()]
        self._shader_id = shader_id
        self._source_asset = source_asset
        self._sub_identifier = sub_identifier

    def GetInputs(self):
        return self._inputs

    def GetInput(self, _name):
        return None

    def GetIdAttr(self):
        return _Input("id", self._shader_id)

    def GetSourceAsset(self, context):
        if context != "mdl" or not self._source_asset:
            return None
        return type("Asset", (), {"resolvedPath": "", "path": self._source_asset})()

    def GetSourceAssetSubIdentifier(self, context):
        return self._sub_identifier if context == "mdl" else ""


def test_collected_mdl_source_exposes_pbr_textures(monkeypatch, tmp_path):
    texture_paths = {}
    for name in ("base.png", "normal.png", "mask.png"):
        texture_path = tmp_path / name
        texture_path.touch()
        texture_paths[name] = texture_path
    mdl_path = tmp_path / "material.mdl"
    mdl_path.write_text(
        """
        export material Example(
            uniform texture_2d Num1_BaseColor = texture_2d("base.png"),
            uniform texture_2d Num2_Normal = texture_2d("normal.png"),
            uniform texture_2d Num3_Mask = texture_2d("mask.png"))
        = material();
        """
    )
    shader = _Shader({}, source_asset=str(mdl_path))
    inputs = _mdl_texture_inputs(shader)

    assert inputs == {
        "diffuse_texture": texture_paths["base.png"],
        "normalmap_texture": texture_paths["normal.png"],
        "ORM_texture": texture_paths["mask.png"],
    }

    requests = []
    monkeypatch.setattr(
        usd_loader, "_surface_shader", lambda material, UsdShade: shader
    )
    material = type("Material", (), {"GetInputs": lambda self: []})()
    result = usd_loader._material_to_pbr(
        material,
        object(),
        lambda path, srgb: requests.append((path, srgb)) or len(requests) - 1,
    )

    assert result["base_color_texture"]["index"] == 0
    assert result["normal_texture"]["index"] == 2
    assert result["metallic_roughness_texture"]["index"] == 1
    assert requests == [
        (texture_paths["base.png"], True),
        (texture_paths["mask.png"], False),
        (texture_paths["normal.png"], False),
    ]


def test_collected_mdl_source_infers_common_texture_parameter_names(tmp_path):
    texture_names = (
        "albedo.png",
        "normal.png",
        "roughness.png",
        "metalness.png",
        "emission.png",
        "opacity.png",
    )
    for name in texture_names:
        (tmp_path / name).touch()
    mdl_path = tmp_path / "material.mdl"
    mdl_path.write_text(
        """
        export material Example(
            uniform texture_2d AlbedoMap = texture_2d("albedo.png"),



            uniform texture_2d SurfaceNormalTexture = texture_2d("normal.png"),
            uniform texture_2d SurfaceRoughnessMap = texture_2d("roughness.png"),
            uniform texture_2d MetalnessTexture = texture_2d("metalness.png"),
            uniform texture_2d EmissionMap = texture_2d("emission.png"),
            uniform texture_2d OpacityMap = texture_2d("opacity.png"))
        = material();
        """
    )

    assert _mdl_texture_inputs(_Shader({}, source_asset=str(mdl_path))) == {
        "diffuse_texture": tmp_path / "albedo.png",
        "normalmap_texture": tmp_path / "normal.png",
        "reflectionroughness_texture": tmp_path / "roughness.png",
        "metallic_texture": tmp_path / "metalness.png",
        "emissive_texture": tmp_path / "emission.png",
    }


def test_authored_mdl_inputs_infer_common_texture_parameter_names(
    monkeypatch, tmp_path
):
    albedo = tmp_path / "albedo.png"
    albedo.touch()
    shader = _Shader({"Base_Color_Map": str(albedo)}, shader_id="CustomMdl")
    monkeypatch.setattr(
        usd_loader, "_surface_shader", lambda material, UsdShade: shader
    )
    requests = []

    result = usd_loader._material_to_pbr(
        None, object(), lambda path, srgb: requests.append((path, srgb)) or 0
    )

    assert result["base_color_texture"]["index"] == 0
    assert requests == [(albedo, True)]


def test_separate_roughness_map_is_packed_with_metallic_constant(tmp_path):
    roughness_path = tmp_path / "roughness.png"
    Image.fromarray(
        np.asarray([[[64, 64, 64, 255]]], dtype=np.uint8), mode="RGBA"
    ).save(roughness_path)
    spec = (
        "packed_orm",
        Path(roughness_path),
        None,
        0.1,
        1.0,
        0.8,
        1.0,
    )

    packed = _decode_packed_orm(spec)

    assert packed.shape == (1, 1, 4)
    assert packed.dtype == np.uint8
    np.testing.assert_allclose(packed[0, 0, 0], 255)
    np.testing.assert_allclose(
        packed[0, 0, 1], round((0.1 * 0.2 + (64.0 / 255.0) * 0.8) * 255.0)
    )
    np.testing.assert_allclose(packed[0, 0, 2], 255)


def test_omnipbr_separate_maps_select_packed_gltf_workflow(monkeypatch, tmp_path):
    roughness_path = tmp_path / "roughness.dds"
    roughness_path.touch()
    asset = type("Asset", (), {"resolvedPath": str(roughness_path), "path": ""})()
    shader = _Shader(
        {
            "reflectionroughness_texture": asset,
            "reflection_roughness_constant": 0.084,
            "reflection_roughness_texture_influence": 0.806,
            "metallic_constant": 1.0,
        }
    )
    monkeypatch.setattr(
        usd_loader, "_surface_shader", lambda material, UsdShade: shader
    )
    requests = []

    material = type("Material", (), {"GetInputs": lambda self: []})()
    result = usd_loader._material_to_pbr(
        material,
        object(),
        lambda path, srgb: -1,
        lambda *args: requests.append(args) or 7,
    )

    assert result["roughness"] == 1.0
    assert result["metallic"] == 1.0
    assert result["metallic_roughness_texture"]["index"] == 7
    assert requests == [(roughness_path, None, 0.084, 1.0, 0.806, 1.0)]


def test_fresnel_emissive_mdl_inputs_are_preserved(monkeypatch):
    shader = _Shader(
        {
            "albedo_color": (0.001, 0.012, 0.0),
            "emissive_color_normal": (0.302, 1.0, 0.0),
            "emissive_intensity": 7.0,
            "reflection_roughness": 0.05,
        }
    )
    monkeypatch.setattr(
        usd_loader, "_surface_shader", lambda material, UsdShade: shader
    )
    material = type("Material", (), {"GetInputs": lambda self: []})()

    result = usd_loader._material_to_pbr(material, object(), lambda path, srgb: -1)

    np.testing.assert_allclose(result["base_color"][:3], (0.001, 0.012, 0.0))
    np.testing.assert_allclose(result["emissive_factor"], (2.114, 7.0, 0.0))
    assert result["roughness"] == 0.05


def test_omniglass_source_asset_selects_transmission_without_authored_inputs(
    monkeypatch,
):
    shader = _Shader({}, source_asset="OmniGlass.mdl", sub_identifier="OmniGlass")
    monkeypatch.setattr(
        usd_loader, "_surface_shader", lambda material, UsdShade: shader
    )
    material = type("Material", (), {"GetInputs": lambda self: []})()

    result = usd_loader._material_to_pbr(material, object(), lambda path, srgb: -1)

    assert result["transmission"] == 1.0
    assert result["roughness"] == 0.0
    assert result["thickness"] == 1.0


def test_omnipbr_honors_disabled_orm_and_surface_controls(monkeypatch, tmp_path):
    orm_path = tmp_path / "orm.png"
    orm_path.touch()
    asset = type("Asset", (), {"resolvedPath": str(orm_path), "path": ""})()
    shader = _Shader(
        {
            "ORM_texture": asset,
            "enable_ORM_texture": False,
            "specular_level": 0.4,
            "albedo_add": 0.2,
            "albedo_desaturation": 0.3,
            "texture_scale": (2.0, 3.0),
            "texture_translate": (0.1, 0.25),
            "texture_rotate": 90.0,
        },
        source_asset="OmniPBR.mdl",
        sub_identifier="OmniPBR",
    )
    monkeypatch.setattr(
        usd_loader, "_surface_shader", lambda material, UsdShade: shader
    )
    material = type("Material", (), {"GetInputs": lambda self: []})()

    result = usd_loader._material_to_pbr(material, object(), lambda path, srgb: 3)

    assert result["metallic_roughness_texture"]["index"] == -1
    assert result["occlusion_texture"]["index"] == -1
    assert result["specular"] == 0.4
    assert result["base_color_add"] == 0.2
    assert result["base_color_desaturation"] == 0.3
    assert result["base_color_texture"]["transform"]["scale"] == (2.0, 3.0)
    assert result["base_color_texture"]["transform"]["offset"] == (0.1, 0.25)
    assert result["base_color_texture"]["transform"]["rotation"] == pytest.approx(
        np.pi / 2.0
    )


def test_udim_tiles_decode_to_horizontal_atlas(tmp_path):
    paths = []
    for tile, color in (
        (1001, (255, 0, 0, 255)),
        (1002, (0, 255, 0, 255)),
        (1003, (0, 0, 255, 255)),
    ):
        tile_path = tmp_path / f"color.{tile}.png"
        Image.fromarray(np.full((2, 2, 4), color, dtype=np.uint8), mode="RGBA").save(
            tile_path
        )
        paths.append((tile, tile_path))

    atlas = _decode_udim(tuple(paths))

    assert atlas.shape == (2, 6, 4)
    assert atlas.dtype == np.uint8
    np.testing.assert_array_equal(
        atlas[0, (0, 2, 4), :3], np.eye(3, dtype=np.uint8) * 255
    )


def test_udim_size_limit_applies_per_tile_not_to_completed_atlas(tmp_path):
    paths = []
    for tile in (1001, 1002, 1003):
        tile_path = tmp_path / f"tile.{tile}.png"
        Image.new("RGBA", (8, 8), (tile - 1000, 0, 0, 255)).save(tile_path)
        paths.append((tile, tile_path))

    atlas = _decode_udim(tuple(paths), max_size=4)

    assert atlas.shape == (4, 12, 4)


def test_local_uv_mesh_maps_to_first_cell_of_udim_atlas():
    texcoords = np.asarray(((0.0, 0.25), (0.5, 0.75), (1.0, 1.0)), dtype=np.float32)

    adjusted = _normalize_mesh_uvs_for_udim_atlas(texcoords, 3, 1)

    np.testing.assert_allclose(adjusted, texcoords / (3.0, 1.0))


def test_multitile_uv_mesh_uses_material_atlas_dimensions():
    texcoords = np.asarray(((0.0, 0.0), (1.5, 0.5), (2.9, 1.0)), dtype=np.float32)

    adjusted = _normalize_mesh_uvs_for_udim_atlas(texcoords, 4, 1)

    np.testing.assert_allclose(adjusted, texcoords / (4.0, 1.0))
