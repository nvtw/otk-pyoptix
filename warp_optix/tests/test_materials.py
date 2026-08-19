# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from warp_optix.pathtracing.materials import MaterialManager


def test_gltf_material_configures_checker_overlay():
    manager = MaterialManager()

    material_id = manager.add_gltf_material(
        u_subdiv=12.5,
        v_subdiv=7.25,
        base_color_scale=0.4,
    )
    material = manager.get_material_entries()[material_id]

    assert material["uSubdiv"] == pytest.approx(12.5)
    assert material["vSubdiv"] == pytest.approx(7.25)
    assert material["baseColorScale"] == pytest.approx(0.4)


def test_checker_overlay_defaults_to_disabled():
    manager = MaterialManager()

    material_id = manager.add_gltf_material()
    material = manager.get_material_entries()[material_id]

    assert material["uSubdiv"] == pytest.approx(0.0)
    assert material["vSubdiv"] == pytest.approx(0.0)
    assert material["baseColorScale"] == pytest.approx(0.75)


def test_gltf_material_preserves_extended_surface_controls():
    manager = MaterialManager()

    material_id = manager.add_gltf_material(
        alpha_mode="MASK",
        alpha_cutoff=0.35,
        occlusion_texture={"index": 4, "texCoord": 1},
        occlusion_strength=0.7,
        specular=0.45,
        specular_color=(0.8, 0.7, 0.6),
        clearcoat=0.85,
        clearcoat_roughness=0.12,
        thickness=2.0,
        base_color_add=0.1,
        base_color_desaturation=0.2,
    )
    material = manager.get_material_entries()[material_id]

    assert material["alphaMode"] == 1
    assert material["alphaCutoff"] == pytest.approx(0.35)
    assert material["occlusionTexture"]["index"] == 4
    assert material["occlusionTexture"]["texCoord"] == 1
    assert material["occlusionStrength"] == pytest.approx(0.7)
    assert material["specularFactor"] == pytest.approx(0.45)
    assert material["specularColorFactor"] == pytest.approx((0.8, 0.7, 0.6))
    assert material["clearcoatFactor"] == pytest.approx(0.85)
    assert material["clearcoatRoughness"] == pytest.approx(0.12)
    assert material["thicknessFactor"] == pytest.approx(2.0)
    assert material["pbrDiffuseFactor"][0] == pytest.approx(0.1)
    assert material["pbrDiffuseFactor"][1] == pytest.approx(0.2)


def test_gltf_material_preserves_optional_opacity_fresnel():
    """Preserve optional angular opacity without changing material defaults."""
    manager = MaterialManager()

    default_id = manager.add_gltf_material()
    fresnel_id = manager.add_gltf_material(
        opacity_fresnel=(0.05, 0.2, 2.0), transmission_color=(1.0, 1.0, 1.0)
    )
    default = manager.get_material_entries()[default_id]
    fresnel = manager.get_material_entries()[fresnel_id]

    assert default["opacityFresnelLow"] == pytest.approx(-1.0)
    assert fresnel["opacityFresnelLow"] == pytest.approx(0.05)
    assert fresnel["opacityFresnelHigh"] == pytest.approx(0.2)
    assert fresnel["opacityFresnelFalloff"] == pytest.approx(2.0)

    assert default["transmissionColorFactor"] == pytest.approx((-1.0, -1.0, -1.0))
    assert fresnel["transmissionColorFactor"] == pytest.approx((1.0, 1.0, 1.0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"u_subdiv": -1.0}, "u_subdiv and v_subdiv"),
        ({"v_subdiv": -1.0}, "u_subdiv and v_subdiv"),
        ({"base_color_scale": -0.1}, "base_color_scale"),
        ({"base_color_scale": 1.1}, "base_color_scale"),
    ],
)
def test_checker_overlay_rejects_out_of_range_values(kwargs, message):
    manager = MaterialManager()

    with pytest.raises(ValueError, match=message):
        manager.add_gltf_material(**kwargs)


def test_normal_texture_preserves_per_axis_scale():
    """Preserve independent tangent-axis normal-map scales."""
    manager = MaterialManager()

    material_id = manager.add_gltf_material(
        normal_texture={"index": 0, "scale": (0.75, -0.75)}
    )

    np.testing.assert_array_equal(
        manager.get_material_entries()[material_id]["normalTextureScale"],
        (0.75, -0.75),
    )
