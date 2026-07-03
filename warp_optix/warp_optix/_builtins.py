"""OptiX builtin registrations for warp.

This module was extracted from ``warp/_src/builtins.py``; importing it
calls ``warp.add_builtin(...)`` for every OptiX-related builtin so that the rest
of warp can codegen against them. ``warp_optix._addon`` triggers the import.

Source extracted on migration from warp branch ``dev/tw/add_minimal_optix_supprt``.
"""

from __future__ import annotations

from typing import Any, Mapping

import warp as wp

add_builtin = wp.add_builtin

bool = wp.bool  # noqa: A001
float = wp.float32  # noqa: A001
uint32 = wp.uint32
uint64 = wp.uint64
vec2 = wp.vec2
vec3 = wp.vec3
vec3ui = wp.vec3ui


def register_addon_builtins() -> None:
    """Register all OptiX-specific builtins with warp.

    Called from ``warp_optix._addon`` at import time. The body below was lifted
    verbatim from ``warp/_src/builtins.py``; only the surrounding ``def`` wrapper
    is new so the registrations don't run as a side-effect of merely importing
    this module's symbols.
    """
    def optix_trace_payload_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Any]):
        func_args = (
            args["traversable"],
            args["ray_origin"],
            args["ray_direction"],
            args["tmin"],
            args["tmax"],
            args["ray_time"],
            args["visibility_mask"],
            args["ray_flags"],
            args["sbt_offset"],
            args["sbt_stride"],
            args["miss_sbt_index"],
            args["payload"],
        )
        return (func_args, ())

    def optix_load_payload_dispatch_func(input_types: Mapping[str, type], return_type: Any, args: Mapping[str, Any]):
        return ((args["payload"],), ())

    add_builtin(
        "float_to_uint32",
        input_types={"x": float},
        value_type=uint32,
        group="Utility",
        doc="Reinterpret the bits of a float as a uint32 (bit-cast, no conversion).",
        is_differentiable=False,
    )

    add_builtin(
        "uint32_to_float",
        input_types={"u": uint32},
        value_type=float,
        group="Utility",
        doc="Reinterpret the bits of a uint32 as a float (bit-cast, no conversion).",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_launch_index",
        input_types={},
        value_type=vec3ui,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_launch_dimensions",
        input_types={},
        value_type=vec3ui,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_world_ray_origin",
        input_types={},
        value_type=vec3,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_world_ray_direction",
        input_types={},
        value_type=vec3,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_object_ray_origin",
        input_types={},
        value_type=vec3,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_object_ray_direction",
        input_types={},
        value_type=vec3,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_ray_tmin",
        input_types={},
        value_type=float,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_ray_tmax",
        input_types={},
        value_type=float,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_triangle_barycentrics",
        input_types={},
        value_type=vec2,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_primitive_index",
        input_types={},
        value_type=uint32,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_instance_id",
        input_types={},
        value_type=uint32,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_get_hit_kind",
        input_types={},
        value_type=uint32,
        group="Utility",
        is_differentiable=False,
    )

    for _attribute_i in range(8):
        add_builtin(
            f"optix_get_attribute_{_attribute_i}",
            input_types={},
            value_type=uint32,
            group="Utility",
            is_differentiable=False,
        )

    add_builtin(
        "optix_transform_normal_from_object_to_world_space",
        input_types={"normal": vec3},
        value_type=vec3,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_transform_vector_from_object_to_world_space",
        input_types={"vector": vec3},
        value_type=vec3,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_terminate_ray",
        input_types={},
        value_type=None,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_ignore_intersection",
        input_types={},
        value_type=None,
        group="Utility",
        is_differentiable=False,
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
            group="Utility",
            doc="Report a custom-primitive intersection with up to eight 32-bit attributes.",
            is_differentiable=False,
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
        dispatch_func=optix_trace_payload_dispatch_func,
        export=False,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_load_payload",
        input_types={"payload": Any},
        value_type=None,
        dispatch_func=optix_load_payload_dispatch_func,
        export=False,
        group="Utility",
        is_differentiable=False,
    )

    add_builtin(
        "optix_store_payload",
        input_types={"payload": Any},
        value_type=None,
        export=False,
        group="Utility",
        is_differentiable=False,
    )

    for _payload_i in range(32):
        add_builtin(
            f"optix_get_payload_{_payload_i}",
            input_types={},
            value_type=uint32,
            group="Utility",
            is_differentiable=False,
        )
        add_builtin(
            f"optix_set_payload_{_payload_i}",
            input_types={"value": uint32},
            value_type=None,
            group="Utility",
            is_differentiable=False,
        )
