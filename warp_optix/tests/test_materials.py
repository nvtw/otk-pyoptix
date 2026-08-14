# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    assert material["baseColorScale"] == pytest.approx(0.9)


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
