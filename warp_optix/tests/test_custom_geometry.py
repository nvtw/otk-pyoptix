# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import warp as wp  # noqa: E402
from warp_optix._runtime import runtime  # noqa: E402
from warp_optix._runtime import sbt as sbt_module  # noqa: E402
from warp_optix._runtime.hit_kernels import HitKernel  # noqa: E402
from warp_optix._runtime.sbt import SbtKernelManager  # noqa: E402


class _DeviceArray:
    _next_ptr = 1000

    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.ptr = _DeviceArray._next_ptr
        _DeviceArray._next_ptr += 1000


class _FakeWarp:
    Kernel = wp.Kernel
    float32 = np.float32
    uint32 = np.uint32
    uint8 = np.uint8

    @staticmethod
    def array(values, dtype, device, copy=True):
        del device, copy
        return _DeviceArray(np.asarray(values, dtype=dtype))

    @staticmethod
    def empty(size, dtype, device):
        del device
        return _DeviceArray(np.empty(size, dtype=dtype))

    @staticmethod
    def synchronize_device(device):
        del device


def test_custom_primitive_builtins_register_all_attribute_arities():
    calls = []
    fake_warp = ModuleType("warp")
    fake_warp.build = SimpleNamespace(add_builtin=lambda name, **kwargs: calls.append((name, kwargs)))
    fake_warp.bool = bool
    fake_warp.float32 = np.float32
    fake_warp.uint32 = np.uint32
    fake_warp.uint64 = np.uint64
    fake_warp.int32 = np.int32
    fake_warp.vec2 = type("vec2", (), {})
    fake_warp.vec3 = type("vec3", (), {})
    fake_warp.vec3ui = type("vec3ui", (), {})
    fake_warp.mat33 = type("mat33", (), {})

    module_name = "warp_optix_builtins_under_test"
    builtins_path = Path(__file__).parents[1] / "warp_optix" / "_builtins.py"
    spec = importlib.util.spec_from_file_location(module_name, builtins_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_warp = sys.modules.get("warp")
    sys.modules["warp"] = fake_warp
    try:
        spec.loader.exec_module(module)
        module.register_addon_builtins()
    finally:
        if previous_warp is None:
            sys.modules.pop("warp", None)
        else:
            sys.modules["warp"] = previous_warp

    reports = [kwargs for name, kwargs in calls if name == "optix_report_intersection"]
    assert [len(call["input_types"]) for call in reports] == list(range(2, 11))
    registered_names = {name for name, _ in calls}
    assert "optix_get_object_ray_origin" in registered_names
    assert "optix_get_object_ray_direction" in registered_names
    assert "optix_get_ray_tmin" in registered_names
    assert "optix_get_hit_kind" in registered_names
    assert "optix_get_triangle_vertex_data" in registered_names
    assert "optix_get_curve_parameter" in registered_names
    assert "optix_direct_call" in registered_names
    assert "optix_continuation_call" in registered_names
    assert "optix_get_exception_code" in registered_names
    assert all(f"optix_get_exception_detail_{i}" in registered_names for i in range(8))
    exception_overloads = [kwargs for name, kwargs in calls if name == "optix_throw_exception"]
    assert [len(call["input_types"]) for call in exception_overloads] == list(range(1, 10))
    assert all(f"optix_get_attribute_{i}" in registered_names for i in range(8))
    assert {
        "optix_get_ray_time",
        "optix_get_ray_flags",
        "optix_get_ray_visibility_mask",
        "optix_get_instance_index",
        "optix_get_sbt_gas_index",
        "optix_get_primitive_type",
        "optix_get_gas_traversable_handle",
        "optix_is_front_face_hit",
        "optix_is_back_face_hit",
        "optix_transform_point_from_object_to_world_space",
        "optix_transform_point_from_world_to_object_space",
        "optix_transform_vector_from_world_to_object_space",
        "optix_transform_normal_from_world_to_object_space",
    } <= registered_names


class _Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Optix:
    BUILD_FLAG_NONE = 0
    BUILD_OPERATION_BUILD = 1
    GEOMETRY_FLAG_NONE = 0
    TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1
    TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING = 2
    EXCEPTION_FLAG_NONE = 0
    PRIMITIVE_TYPE_FLAGS_TRIANGLE = 1
    PRIMITIVE_TYPE_FLAGS_CUSTOM = 4
    COMPILE_DEFAULT_MAX_REGISTER_COUNT = 0
    COMPILE_OPTIMIZATION_DEFAULT = 0
    COMPILE_DEBUG_LEVEL_DEFAULT = 0
    SBT_RECORD_HEADER_SIZE = 32
    SBT_RECORD_ALIGNMENT = 16

    AccelBuildOptions = _Struct
    BuildInputCustomPrimitiveArray = _Struct
    BuildInputInstanceArray = _Struct
    PipelineCompileOptions = _Struct
    ModuleCompileOptions = _Struct
    ProgramGroupDesc = _Struct
    ProgramGroupOptions = _Struct
    PipelineLinkOptions = _Struct
    ShaderBindingTable = _Struct

    @staticmethod
    def version():
        return (9, 0, 0)

    @staticmethod
    def sbtRecordPackHeader(group, record):
        del group, record

    @staticmethod
    def getDeviceRepresentation(instances):
        return bytes(80 * len(instances))


class _Context:
    def __init__(self):
        self.build_input = None
        self.pipeline_options = None
        self.program_descs = []
        self.pipeline = None

    def accelComputeMemoryUsage(self, options, build_inputs):
        del options
        self.build_input = build_inputs[0]
        return SimpleNamespace(tempSizeInBytes=64, outputSizeInBytes=128)

    def accelBuild(self, stream, options, inputs, temp, temp_size, output, output_size, emitted):
        del stream, options, inputs, temp, temp_size, output, output_size, emitted
        return 42

    def moduleCreate(self, module_options, pipeline_options, ptx):
        del module_options, ptx
        self.pipeline_options = pipeline_options
        return object(), ""

    def programGroupCreate(self, descs, options=None):
        del options
        self.program_descs.extend(descs)
        return [object()], ""

    def pipelineCreate(self, pipeline_options, link_options, groups, log):
        del pipeline_options, link_options, groups, log
        self.pipeline = SimpleNamespace(stack_size=None)
        self.pipeline.setStackSize = lambda *args: setattr(self.pipeline, "stack_size", args)
        return self.pipeline


def test_create_custom_primitive_gas_with_sbt_offsets(monkeypatch):
    monkeypatch.setattr(runtime, "wp", _FakeWarp)
    optix = _Optix()
    ctx = _Context()
    aabbs = np.array([[[-1, -2, -3], [1, 2, 3]], [[4, 5, 6], [7, 8, 9]]], dtype=np.float64)

    handle, keepalive = runtime.create_custom_primitive_gas(
        optix,
        ctx,
        aabbs,
        "cuda",
        geometry_flags=[3, 5],
        sbt_index_offsets=np.array([0, 1]),
        num_sbt_records=2,
        primitive_index_offset=9,
    )

    assert handle == 42
    assert keepalive["d_aabbs"].shape == (2, 6)
    assert keepalive["d_sbt_indices"].shape == (2,)
    assert ctx.build_input.numPrimitives == 2
    assert ctx.build_input.strideInBytes == 24
    assert ctx.build_input.flags == [3, 5]
    assert ctx.build_input.numSbtRecords == 2
    assert ctx.build_input.primitiveIndexOffset == 9


@pytest.mark.parametrize(
    "aabbs, message",
    [
        (np.zeros((2, 5)), "shape"),
        (np.zeros((0, 6)), "at least one"),
        (np.array([[0, 0, 0, np.inf, 1, 1]]), "finite"),
        (np.array([[1, 0, 0, -1, 1, 1]]), "minimum"),
    ],
)
def test_create_custom_primitive_gas_validates_aabbs(monkeypatch, aabbs, message):
    monkeypatch.setattr(runtime, "wp", _FakeWarp)
    with pytest.raises(ValueError, match=message):
        runtime.create_custom_primitive_gas(_Optix(), _Context(), aabbs, "cuda")


def test_custom_pipeline_infers_primitive_flag_and_wires_intersection(monkeypatch):
    monkeypatch.setattr(runtime, "wp", _FakeWarp)
    monkeypatch.setattr(sbt_module, "wp", _FakeWarp)
    optix = _Optix()
    ctx = _Context()

    runtime.create_pipeline_and_sbt(
        optix,
        ctx,
        b"ptx",
        "__raygen__rg",
        "__miss__ms",
        "__closesthit__ch",
        2,
        3,
        "cuda",
        intersection_entry="__intersection__is",
        any_hit_entry="__anyhit__ah",
    )

    assert ctx.pipeline_options.usesPrimitiveTypeFlags == optix.PRIMITIVE_TYPE_FLAGS_CUSTOM
    hit_desc = ctx.program_descs[2]
    assert hit_desc.hitgroupEntryFunctionNameCH == "__closesthit__ch"
    assert hit_desc.hitgroupEntryFunctionNameAH == "__anyhit__ah"
    assert hit_desc.hitgroupEntryFunctionNameIS == "__intersection__is"


def test_triangle_pipeline_default_is_preserved(monkeypatch):
    monkeypatch.setattr(runtime, "wp", _FakeWarp)
    monkeypatch.setattr(sbt_module, "wp", _FakeWarp)
    optix = _Optix()
    ctx = _Context()

    runtime.create_pipeline_and_sbt(
        optix,
        ctx,
        b"ptx",
        "__raygen__rg",
        "__miss__ms",
        "__closesthit__ch",
        2,
        2,
        "cuda",
    )

    assert ctx.pipeline_options.usesPrimitiveTypeFlags == optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE
    assert ctx.program_descs[2].hitgroupEntryFunctionNameCH == "__closesthit__ch"


def test_mixed_pipeline_uses_contiguous_hit_records(monkeypatch):
    monkeypatch.setattr(runtime, "wp", _FakeWarp)
    monkeypatch.setattr(sbt_module, "wp", _FakeWarp)
    optix = _Optix()
    ctx = _Context()

    _, sbt, resources = runtime.create_pipeline_and_sbt(
        optix,
        ctx,
        b"ptx",
        "__raygen__rg",
        "__miss__ms",
        None,
        3,
        3,
        "cuda",
        hit_groups=[
            HitKernel(closest_hit="__closesthit__triangle"),
            HitKernel(closest_hit="__closesthit__sphere", intersection="__intersection__sphere"),
        ],
        traversable_graph_flags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
    )

    expected_flags = optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE | optix.PRIMITIVE_TYPE_FLAGS_CUSTOM
    assert ctx.pipeline_options.usesPrimitiveTypeFlags == expected_flags
    assert sbt.hitgroupRecordCount == 2
    assert resources["sbt"]["d_hg"].shape == (2 * optix.SBT_RECORD_HEADER_SIZE,)
    manager = resources["sbt_manager"]
    assert [manager.get_sbt_offset(handle) for handle in resources["hit_group_handles"]] == [0, 1]
    assert ctx.pipeline.stack_size[-1] == 2


def test_create_instance_acceleration_structure(monkeypatch):
    monkeypatch.setattr(runtime, "wp", _FakeWarp)
    ctx = _Context()

    handle, keepalive = runtime.create_instance_acceleration_structure(
        _Optix(), ctx, [object(), object()], "cuda"
    )

    assert handle == 42
    assert keepalive["d_instances"].shape == (160,)
    assert ctx.build_input.numInstances == 2


def test_sbt_manager_standardizes_multiple_geometry_and_ray_types(monkeypatch):
    monkeypatch.setattr(sbt_module, "wp", _FakeWarp)
    manager = SbtKernelManager(_Optix(), _Context(), object(), num_ray_subtypes=2)
    manager.set_raygen_kernel("__raygen__rg")
    manager.add_miss_kernels(["__miss__primary", "__miss__shadow"])

    triangle = manager.register_hit_shader_type(
        HitKernel(closest_hit="__closesthit__triangle"),
        HitKernel(any_hit="__anyhit__triangle_shadow"),
    )
    custom = manager.register_hit_shader_type(
        HitKernel(closest_hit="__closesthit__custom", intersection="__intersection__custom"),
        HitKernel(any_hit="__anyhit__custom_shadow", intersection="__intersection__custom"),
    )
    resources = manager.build_sbt("cuda")

    assert manager.get_sbt_offset(triangle) == 0
    assert manager.get_sbt_offset(custom) == 2
    assert resources.sbt.missRecordCount == 2
    assert resources.sbt.hitgroupRecordCount == 4
    assert resources.keepalive["d_hg"].shape == (4 * _Optix.SBT_RECORD_HEADER_SIZE,)
