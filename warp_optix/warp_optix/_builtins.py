"""OptiX builtin registrations for Warp.

This module was extracted from ``warp/_src/builtins.py``; importing it
registers every OptiX-related builtin through Warp's public extension API.
``warp_optix._addon`` triggers the import.

Source extracted on migration from warp branch ``dev/tw/add_minimal_optix_supprt``.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from warp_optix._compat import get_add_builtin

add_builtin = get_add_builtin(wp)

bool = wp.bool  # noqa: A001
float = wp.float32  # noqa: A001
uint32 = wp.uint32
uint64 = wp.uint64
int32 = wp.int32
vec2 = wp.vec2
vec3 = wp.vec3
vec3ui = wp.vec3ui
mat33 = wp.mat33


def register_addon_builtins() -> None:
    """Register all OptiX-specific builtins with warp.

    Called from ``warp_optix._addon`` at import time. The body below was lifted
    verbatim from ``warp/_src/builtins.py``; only the surrounding ``def`` wrapper
    is new so the registrations don't run as a side-effect of merely importing
    this module's symbols.
    """
    add_builtin(
        "float_to_uint32",
        input_types={"x": float},
        value_type=uint32,
        doc="Reinterpret the bits of a float as a uint32 (bit-cast, no conversion).",
    )

    add_builtin(
        "uint32_to_float",
        input_types={"u": uint32},
        value_type=float,
        doc="Reinterpret the bits of a uint32 as a float (bit-cast, no conversion).",
    )

    add_builtin(
        "optix_get_launch_index",
        input_types={},
        value_type=vec3ui,
    )

    add_builtin(
        "optix_get_launch_dimensions",
        input_types={},
        value_type=vec3ui,
    )

    add_builtin(
        "optix_get_world_ray_origin",
        input_types={},
        value_type=vec3,
    )

    add_builtin(
        "optix_get_world_ray_direction",
        input_types={},
        value_type=vec3,
    )

    add_builtin(
        "optix_get_object_ray_origin",
        input_types={},
        value_type=vec3,
    )

    add_builtin(
        "optix_get_object_ray_direction",
        input_types={},
        value_type=vec3,
    )

    add_builtin(
        "optix_get_ray_tmin",
        input_types={},
        value_type=float,
    )

    add_builtin(
        "optix_get_ray_time",
        input_types={},
        value_type=float,
    )

    for _name in ("optix_get_ray_flags", "optix_get_ray_visibility_mask"):
        add_builtin(
            _name,
            input_types={},
            value_type=uint32,
        )

    add_builtin(
        "optix_get_ray_tmax",
        input_types={},
        value_type=float,
    )

    add_builtin(
        "optix_get_triangle_barycentrics",
        input_types={},
        value_type=vec2,
    )

    add_builtin(
        "optix_get_triangle_vertex_data",
        input_types={},
        value_type=mat33,
        doc="Return the current triangle's three object-space vertices as rows of a mat33.",
    )

    add_builtin(
        "optix_get_curve_parameter",
        input_types={},
        value_type=float,
    )

    add_builtin(
        "optix_get_primitive_index",
        input_types={},
        value_type=uint32,
    )

    add_builtin(
        "optix_get_instance_id",
        input_types={},
        value_type=uint32,
    )

    for _name in (
        "optix_get_instance_index",
        "optix_get_sbt_gas_index",
        "optix_get_primitive_type",
    ):
        add_builtin(
            _name,
            input_types={},
            value_type=uint32,
        )

    add_builtin(
        "optix_get_gas_traversable_handle",
        input_types={},
        value_type=uint64,
    )

    for _name in ("optix_is_front_face_hit", "optix_is_back_face_hit"):
        add_builtin(
            _name,
            input_types={},
            value_type=bool,
        )

    add_builtin(
        "optix_get_hit_kind",
        input_types={},
        value_type=uint32,
    )

    for _attribute_i in range(8):
        add_builtin(
            f"optix_get_attribute_{_attribute_i}",
            input_types={},
            value_type=uint32,
        )

    add_builtin(
        "optix_transform_normal_from_object_to_world_space",
        input_types={"normal": vec3},
        value_type=vec3,
    )

    add_builtin(
        "optix_transform_point_from_object_to_world_space",
        input_types={"point": vec3},
        value_type=vec3,
    )

    add_builtin(
        "optix_transform_vector_from_object_to_world_space",
        input_types={"vector": vec3},
        value_type=vec3,
    )

    for _name, _argument in (
        ("optix_transform_point_from_world_to_object_space", "point"),
        ("optix_transform_vector_from_world_to_object_space", "vector"),
        ("optix_transform_normal_from_world_to_object_space", "normal"),
    ):
        add_builtin(
            _name,
            input_types={_argument: vec3},
            value_type=vec3,
        )

    add_builtin(
        "optix_terminate_ray",
        input_types={},
        value_type=None,
    )

    add_builtin(
        "optix_ignore_intersection",
        input_types={},
        value_type=None,
    )

    add_builtin(
        "optix_direct_call",
        input_types={"sbt_index": uint32},
        value_type=None,
    )

    add_builtin(
        "optix_continuation_call",
        input_types={"sbt_index": uint32},
        value_type=None,
    )

    add_builtin(
        "optix_get_exception_code",
        input_types={},
        value_type=int32,
    )

    for _detail_i in range(8):
        add_builtin(
            f"optix_get_exception_detail_{_detail_i}",
            input_types={},
            value_type=uint32,
        )

    for _num_details in range(9):
        _input_types = {"exception_code": int32}
        _input_types.update({f"detail_{i}": uint32 for i in range(_num_details)})
        add_builtin(
            "optix_throw_exception",
            input_types=_input_types,
            value_type=None,
        )

    # OptiX accepts zero to eight 32-bit attributes when an intersection is
    # reported. Register each arity explicitly so Warp can type-check calls and
    # emit a normal overload call into the variadic C++ wrapper.
    for _num_attributes in range(9):
        _input_types = {"hit_t": float, "hit_kind": uint32}
        _input_types.update({f"attribute_{i}": uint32 for i in range(_num_attributes)})
        add_builtin(
            "optix_report_intersection",
            input_types=_input_types,
            value_type=bool,
            doc="Report a custom-primitive intersection with up to eight 32-bit attributes.",
        )

    add_builtin(
        "optix_trace",
        input_types={
            "traversable": uint64,
            "ray_origin": vec3,
            "ray_direction": vec3,
            "tmin": float,
            "tmax": float,
            "ray_time": float,
            "visibility_mask": uint32,
            "ray_flags": uint32,
            "sbt_offset": uint32,
            "sbt_stride": uint32,
            "miss_sbt_index": uint32,
            "payload": Any,
        },
        value_type=None,
    )

    add_builtin(
        "optix_traverse",
        input_types={
            "traversable": uint64,
            "ray_origin": vec3,
            "ray_direction": vec3,
            "tmin": float,
            "tmax": float,
            "ray_time": float,
            "visibility_mask": uint32,
            "ray_flags": uint32,
            "sbt_offset": uint32,
            "sbt_stride": uint32,
            "miss_sbt_index": uint32,
            "payload": Any,
        },
        value_type=None,
    )

    add_builtin(
        "optix_reorder",
        input_types={
            "coherence_hint": uint32,
            "num_coherence_hint_bits_from_lsb": uint32,
        },
        value_type=None,
    )

    add_builtin(
        "optix_reorder",
        input_types={},
        value_type=None,
    )

    add_builtin(
        "optix_hit_object_is_hit",
        input_types={},
        value_type=bool,
    )

    add_builtin(
        "optix_hit_object_get_primitive_index",
        input_types={},
        value_type=uint32,
    )

    add_builtin(
        "optix_hit_object_get_instance_id",
        input_types={},
        value_type=uint32,
    )

    add_builtin(
        "optix_invoke",
        input_types={"payload": Any},
        value_type=None,
    )

    add_builtin(
        "optix_load_payload",
        input_types={"payload": Any},
        value_type=None,
    )

    add_builtin(
        "optix_store_payload",
        input_types={"payload": Any},
        value_type=None,
    )

    for _payload_i in range(32):
        add_builtin(
            f"optix_get_payload_{_payload_i}",
            input_types={},
            value_type=uint32,
        )
        add_builtin(
            f"optix_set_payload_{_payload_i}",
            input_types={"value": uint32},
            value_type=None,
        )
