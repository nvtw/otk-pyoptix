# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
from PIL import Image

from warp_optix.pathtracing import usd_loader
from warp_optix.pathtracing.usd_loader import (
    _ambient_light_from_custom_layer_data,
    _normalize_mesh_uvs_for_udim_atlas,
    _decode_packed_orm,
    _decode_udim,
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
    def __init__(self, inputs):
        self._inputs = [_Input(name, value) for name, value in inputs.items()]

    def GetInputs(self):
        return self._inputs

    def GetInput(self, _name):
        return None

    def GetIdAttr(self):
        return _Input("id", "")


def test_separate_roughness_map_is_packed_with_metallic_constant(tmp_path):
    roughness_path = tmp_path / "roughness.png"
    Image.fromarray(np.asarray([[[64, 64, 64, 255]]], dtype=np.uint8), mode="RGBA").save(
        roughness_path
    )
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
    np.testing.assert_allclose(packed[0, 0, 0], 1.0)
    np.testing.assert_allclose(packed[0, 0, 1], 0.1 * 0.2 + (64.0 / 255.0) * 0.8)
    np.testing.assert_allclose(packed[0, 0, 2], 1.0)


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
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
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
    monkeypatch.setattr(usd_loader, "_surface_shader", lambda material, UsdShade: shader)
    material = type("Material", (), {"GetInputs": lambda self: []})()

    result = usd_loader._material_to_pbr(material, object(), lambda path, srgb: -1)

    np.testing.assert_allclose(result["base_color"][:3], (0.001, 0.012, 0.0))
    np.testing.assert_allclose(result["emissive_factor"], (2.114, 7.0, 0.0))
    assert result["roughness"] == 0.05


def test_udim_tiles_decode_to_horizontal_atlas(tmp_path):
    paths = []
    for tile, color in (
        (1001, (255, 0, 0, 255)),
        (1002, (0, 255, 0, 255)),
        (1003, (0, 0, 255, 255)),
    ):
        tile_path = tmp_path / f"color.{tile}.png"
        Image.fromarray(np.full((2, 2, 4), color, dtype=np.uint8), mode="RGBA").save(tile_path)
        paths.append((tile, tile_path))

    atlas = _decode_udim(tuple(paths))

    assert atlas.shape == (2, 6, 4)
    np.testing.assert_allclose(atlas[0, (0, 2, 4), :3], np.eye(3), atol=1.0e-6)


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
