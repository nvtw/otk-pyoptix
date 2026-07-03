# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import warp as wp  # noqa: E402
import warp_optix as woptix  # noqa: E402


@wp.struct
class QueryParams:
    output: wp.array(dtype=wp.uint32)
    traversable: wp.uint64
    direct_callable_index: wp.uint32
    continuation_callable_index: wp.uint32


@wp.struct
class QueryPayload:
    value: wp.uint32


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def query_raygen(params: QueryParams):
    payload = QueryPayload()
    payload.value = wp.uint32(0)
    wp.optix_trace(
        params.traversable,
        wp.vec3(2.0, 0.0, 3.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.001,
        100.0,
        0.25,
        wp.uint32(0x5A),
        wp.uint32(1),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )


@woptix.optix_kernel(woptix.OptixKernelType.MISS)
def query_miss(params: QueryParams):
    params.output[0] = wp.uint32(0xFFFFFFFF)


@woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
def query_closest_hit(params: QueryParams):
    params.output[0] = wp.optix_get_instance_id()
    params.output[1] = wp.optix_get_instance_index()
    params.output[2] = wp.optix_get_sbt_gas_index()
    params.output[3] = wp.optix_get_primitive_type()
    params.output[4] = wp.optix_get_ray_flags()
    params.output[5] = wp.optix_get_ray_visibility_mask()
    params.output[6] = wp.float_to_uint32(wp.optix_get_ray_time())
    params.output[7] = wp.uint32(1) if wp.optix_is_front_face_hit() else wp.uint32(0)
    params.output[8] = wp.uint32(1) if wp.optix_is_back_face_hit() else wp.uint32(0)

    gas = wp.optix_get_gas_traversable_handle()
    params.output[9] = wp.uint32(gas & wp.uint64(0xFFFFFFFF))
    params.output[10] = wp.uint32(gas >> wp.uint64(32))

    error = wp.uint32(0)
    object_origin = wp.vec3(0.0, 0.0, 0.0)
    world_origin = wp.vec3(2.0, 0.0, 0.0)
    if wp.length(wp.optix_transform_point_from_object_to_world_space(object_origin) - world_origin) > 1.0e-5:
        error |= wp.uint32(1)
    if wp.length(wp.optix_transform_point_from_world_to_object_space(world_origin) - object_origin) > 1.0e-5:
        error |= wp.uint32(2)

    vector = wp.vec3(1.0, 2.0, 3.0)
    world_vector = wp.optix_transform_vector_from_object_to_world_space(vector)
    if wp.length(wp.optix_transform_vector_from_world_to_object_space(world_vector) - vector) > 1.0e-5:
        error |= wp.uint32(4)

    normal = wp.normalize(wp.vec3(1.0, 2.0, 3.0))
    world_normal = wp.optix_transform_normal_from_object_to_world_space(normal)
    if wp.length(wp.optix_transform_normal_from_world_to_object_space(world_normal) - normal) > 1.0e-5:
        error |= wp.uint32(8)

    vertices = wp.optix_get_triangle_vertex_data()
    if wp.length(wp.vec3(vertices[0, 0], vertices[0, 1], vertices[0, 2]) - wp.vec3(-1.0, -1.0, 0.0)) > 1.0e-5:
        error |= wp.uint32(16)
    if wp.length(wp.vec3(vertices[1, 0], vertices[1, 1], vertices[1, 2]) - wp.vec3(1.0, -1.0, 0.0)) > 1.0e-5:
        error |= wp.uint32(32)
    if wp.length(wp.vec3(vertices[2, 0], vertices[2, 1], vertices[2, 2]) - wp.vec3(0.0, 1.0, 0.0)) > 1.0e-5:
        error |= wp.uint32(64)
    params.output[11] = error


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def motion_raygen(params: QueryParams):
    payload = QueryPayload()
    payload.value = wp.uint32(0)
    wp.optix_trace(
        params.traversable,
        wp.vec3(0.0, 0.0, 3.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.001,
        100.0,
        0.25,
        wp.uint32(255),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )


@woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
def motion_closest_hit(params: QueryParams):
    params.output[0] = wp.float_to_uint32(wp.optix_get_ray_time())


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def curve_raygen(params: QueryParams):
    payload = QueryPayload()
    payload.value = wp.uint32(0)
    wp.optix_trace(
        params.traversable,
        wp.vec3(0.0, 0.0, 3.0),
        wp.vec3(0.0, 0.0, -1.0),
        0.001,
        100.0,
        0.0,
        wp.uint32(255),
        wp.uint32(0),
        wp.uint32(0),
        wp.uint32(1),
        wp.uint32(0),
        payload,
    )


@woptix.optix_kernel(woptix.OptixKernelType.CLOSEST_HIT)
def curve_closest_hit(params: QueryParams):
    params.output[0] = wp.uint32(1)
    params.output[1] = wp.optix_get_primitive_type()
    params.output[2] = wp.float_to_uint32(wp.optix_get_curve_parameter())


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def callable_raygen(params: QueryParams):
    wp.optix_direct_call(params.direct_callable_index)
    wp.optix_continuation_call(params.continuation_callable_index)


@woptix.optix_kernel(woptix.OptixKernelType.DIRECT_CALLABLE)
def direct_callable(params: QueryParams):
    params.output[0] = wp.uint32(42)


@woptix.optix_kernel(woptix.OptixKernelType.CONTINUATION_CALLABLE)
def continuation_callable(params: QueryParams):
    params.output[1] = wp.uint32(84)


@woptix.optix_kernel(woptix.OptixKernelType.RAYGEN)
def exception_raygen(params: QueryParams):
    wp.optix_throw_exception(wp.int32(1234), wp.uint32(77), wp.uint32(88))


@woptix.optix_kernel(woptix.OptixKernelType.EXCEPTION)
def exception_program(params: QueryParams):
    params.output[0] = wp.uint32(wp.optix_get_exception_code())
    params.output[1] = wp.optix_get_exception_detail_0()
    params.output[2] = wp.optix_get_exception_detail_1()


def test_common_device_queries_on_gpu(tmp_path, monkeypatch):
    try:
        optix = woptix.require_optix()
        wp.init()
    except Exception as error:
        pytest.skip(f"OptiX/Warp unavailable: {error}")
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")

    device_name = "cuda:0"
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp_cache"))
    with wp.ScopedDevice(device_name):
        device = wp.get_device(device_name)
        cuda_context = device.context.value if hasattr(device.context, "value") else int(device.context)
        context, _ = woptix.create_context(optix, int(cuda_context), log_level=1)
        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__), "", "test_device_queries", __file__, device=device_name
        )

        vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        gas, gas_buffers = woptix.create_triangle_gas(
            optix, context, vertices, indices, device_name, compact=True
        )
        assert gas_buffers.build_flags & int(optix.BUILD_FLAG_ALLOW_COMPACTION)
        assert gas_buffers.output_size_in_bytes <= gas_buffers.uncompacted_size_in_bytes
        pipeline, sbt, pipeline_buffers = woptix.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            query_raygen,
            query_miss,
            query_closest_hit,
            num_payload_values=1,
            num_attribute_values=2,
            device=device_name,
            traversable_graph_flags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        )
        hit_handle = pipeline_buffers["hit_group_handles"][0]
        hit_offset = pipeline_buffers["sbt_manager"].get_sbt_offset(hit_handle)
        transform = [1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        instance = optix.Instance(transform, 77, hit_offset, 0xFF, optix.INSTANCE_FLAG_NONE, gas)
        ias, ias_buffers = woptix.create_instance_acceleration_structure(
            optix, context, [instance], device_name
        )

        output = wp.zeros(12, dtype=wp.uint32, device=device_name)
        params = QueryParams()
        params.output = output
        params.traversable = wp.uint64(ias)
        params_buffer = woptix.create_launch_params_buffer(QueryParams, device_name)
        woptix.write_launch_params(params_buffer, params)
        woptix.launch(optix, pipeline, sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        result = output.numpy()

        assert result[0] == 77
        assert result[1] == 0
        assert result[2] == 0
        assert result[3] == int(optix.PRIMITIVE_TYPE_TRIANGLE)
        assert result[4] == 1
        assert result[5] == 0x5A
        # Pipelines without motion blur report time zero; stage 5 exercises a
        # non-zero value with motion enabled.
        assert result[6] == np.float32(0.0).view(np.uint32)
        assert tuple(result[7:9]) == (1, 0)
        assert int(result[9]) | (int(result[10]) << 32) == gas
        assert result[11] == 0

        dynamic_flags = int(optix.BUILD_FLAG_ALLOW_UPDATE | optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS)
        dynamic_gas, dynamic_buffers = woptix.create_triangle_gas(
            optix, context, vertices, indices, device_name, build_flags=dynamic_flags
        )
        dynamic_instance = optix.Instance(transform, 77, hit_offset, 0xFF, optix.INSTANCE_FLAG_NONE, dynamic_gas)
        dynamic_ias, dynamic_ias_buffers = woptix.create_instance_acceleration_structure(
            optix, context, [dynamic_instance], device_name
        )
        params.traversable = wp.uint64(dynamic_ias)
        woptix.write_launch_params(params_buffer, params)

        moved_vertices = vertices + np.array([10.0, 0.0, 0.0], dtype=np.float32)
        dynamic_buffers["d_vertices"].assign(moved_vertices)
        assert woptix.refit_acceleration_structure(optix, context, dynamic_buffers) == dynamic_gas
        output.zero_()
        woptix.launch(optix, pipeline, sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        assert output.numpy()[0] == np.uint32(0xFFFFFFFF)

        _keepalive = (
            gas_buffers,
            pipeline_buffers,
            ias_buffers,
            dynamic_buffers,
            dynamic_ias_buffers,
            params_buffer,
        )


def test_callables_and_exception_programs_on_gpu(tmp_path, monkeypatch):
    try:
        optix = woptix.require_optix()
        wp.init()
    except Exception as error:
        pytest.skip(f"OptiX/Warp unavailable: {error}")
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")

    device_name = "cuda:0"
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp_cache"))
    with wp.ScopedDevice(device_name):
        device = wp.get_device(device_name)
        cuda_context = device.context.value if hasattr(device.context, "value") else int(device.context)
        context, _ = woptix.create_context(optix, int(cuda_context), log_level=1)
        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__), "", "test_callables_exceptions", __file__, device=device_name
        )

        callable_pipeline, callable_sbt, callable_resources = woptix.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            callable_raygen,
            query_miss,
            None,
            num_payload_values=1,
            num_attribute_values=0,
            device=device_name,
            direct_callable_entries=[direct_callable],
            continuation_callable_entries=[continuation_callable],
        )
        manager = callable_resources["sbt_manager"]
        direct_handle = callable_resources["direct_callable_handles"][0]
        continuation_handle = callable_resources["continuation_callable_handles"][0]

        output = wp.zeros(12, dtype=wp.uint32, device=device_name)
        params = QueryParams()
        params.output = output
        params.direct_callable_index = wp.uint32(manager.get_callable_index(direct_handle))
        params.continuation_callable_index = wp.uint32(manager.get_callable_index(continuation_handle))
        params_buffer = woptix.create_launch_params_buffer(QueryParams, device_name)
        woptix.write_launch_params(params_buffer, params)
        woptix.launch(optix, callable_pipeline, callable_sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        assert tuple(output.numpy()[:2]) == (42, 84)

        exception_pipeline, exception_sbt, exception_resources = woptix.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            exception_raygen,
            query_miss,
            None,
            num_payload_values=1,
            num_attribute_values=0,
            device=device_name,
            exception_entry=exception_program,
        )
        output.zero_()
        woptix.launch(optix, exception_pipeline, exception_sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        assert tuple(output.numpy()[:3]) == (1234, 77, 88)

        _keepalive = (callable_resources, exception_resources, params_buffer)


def test_motion_blur_and_round_curves_on_gpu(tmp_path, monkeypatch):
    try:
        optix = woptix.require_optix()
        wp.init()
    except Exception as error:
        pytest.skip(f"OptiX/Warp unavailable: {error}")
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")

    device_name = "cuda:0"
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp_cache"))
    with wp.ScopedDevice(device_name):
        device = wp.get_device(device_name)
        cuda_context = device.context.value if hasattr(device.context, "value") else int(device.context)
        context, _ = woptix.create_context(optix, int(cuda_context), log_level=1)
        ptx = woptix.compile_warp_module_to_ptx(
            wp.get_module(__name__), "", "test_motion_curves", __file__, device=device_name
        )

        vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        vertex_keys = np.stack((vertices, vertices + np.array([0.0, 0.0, -0.5], dtype=np.float32)))
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        motion_gas, motion_buffers = woptix.create_triangle_gas(
            optix, context, vertex_keys, indices, device_name, motion_time_range=(0.0, 1.0)
        )
        motion_pipeline, motion_sbt, motion_pipeline_buffers = woptix.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            motion_raygen,
            query_miss,
            motion_closest_hit,
            num_payload_values=1,
            num_attribute_values=2,
            device=device_name,
            uses_motion_blur=True,
        )

        output = wp.zeros(12, dtype=wp.uint32, device=device_name)
        params = QueryParams()
        params.output = output
        params.traversable = wp.uint64(motion_gas)
        params_buffer = woptix.create_launch_params_buffer(QueryParams, device_name)
        woptix.write_launch_params(params_buffer, params)
        woptix.launch(optix, motion_pipeline, motion_sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        assert output.numpy()[0] == np.float32(0.25).view(np.uint32)

        curve_vertices = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        curve_widths = np.array([0.25, 0.25], dtype=np.float32)
        curve_gas, curve_buffers = woptix.create_curve_gas(
            optix,
            context,
            curve_vertices,
            curve_widths,
            np.array([0], dtype=np.uint32),
            device_name,
        )
        curve_pipeline, curve_sbt, curve_pipeline_buffers = woptix.create_pipeline_and_sbt(
            optix,
            context,
            ptx,
            curve_raygen,
            query_miss,
            None,
            num_payload_values=1,
            num_attribute_values=1,
            device=device_name,
            hit_groups=[
                woptix.HitKernel(
                    closest_hit=curve_closest_hit,
                    builtin_intersection_type=optix.PRIMITIVE_TYPE_ROUND_LINEAR,
                )
            ],
        )
        output.zero_()
        params.traversable = wp.uint64(curve_gas)
        woptix.write_launch_params(params_buffer, params)
        woptix.launch(optix, curve_pipeline, curve_sbt, 1, 1, params_buffer)
        wp.synchronize_device(device_name)
        result = output.numpy()
        assert result[0] == 1
        assert result[1] == int(optix.PRIMITIVE_TYPE_ROUND_LINEAR)
        curve_u = np.array(result[2], dtype=np.uint32).view(np.float32).item()
        assert 0.0 <= curve_u <= 1.0

        _keepalive = (
            motion_buffers,
            motion_pipeline_buffers,
            curve_buffers,
            curve_pipeline_buffers,
            params_buffer,
        )
