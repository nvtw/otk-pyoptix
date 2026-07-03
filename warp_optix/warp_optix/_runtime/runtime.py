# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal reusable runtime helpers for Warp + otk-pyoptix integration."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np

import warp as wp

from warp_optix._addon import get_module_build_options


def require_optix():
    try:
        import optix  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError("Install otk-pyoptix first: https://github.com/NVIDIA/otk-pyoptix") from e
    return optix


def _optix_version_at_least(optix, major: int, minor: int) -> bool:
    version = tuple(optix.version())
    return version >= (major, minor)


class OptixLogger:
    def __init__(self):
        self.num_messages = 0

    def __call__(self, level, tag, message):
        print(f"[{level:>2}][{tag:>12}]: {message}")
        self.num_messages += 1


def create_optix_context(optix, cuda_context, log_level: int = 4):
    logger = OptixLogger()
    options = optix.DeviceContextOptions(logCallbackFunction=logger, logCallbackLevel=log_level)
    if _optix_version_at_least(optix, 7, 2):
        options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL
    return optix.deviceContextCreate(cuda_context, options), logger


@dataclass
class LaunchParamsBuffer:
    struct_type: type
    struct_ctype: type[ctypes.Structure]
    bytes: wp.array
    nbytes: int
    device: str


def _is_warp_struct_type(value) -> bool:
    return hasattr(value, "ctype") and hasattr(value, "instance_type")


def _get_struct_type_info(params_struct_type):
    if _is_warp_struct_type(params_struct_type):
        wp_struct = params_struct_type
    else:
        wp_struct = getattr(params_struct_type, "_cls", None)
        if wp_struct is None and hasattr(params_struct_type, "__class__"):
            wp_struct = getattr(params_struct_type.__class__, "_cls", None)

    if not _is_warp_struct_type(wp_struct):
        raise TypeError(
            "params_struct_type must be a @wp.struct definition (example: create_launch_params_buffer(MyLaunchParams, ...))"
        )
    return wp_struct, wp_struct.ctype


def create_launch_params_buffer(params_struct_type: type, device: str = "cuda") -> LaunchParamsBuffer:
    wp_struct, struct_ctype = _get_struct_type_info(params_struct_type)
    nbytes = ctypes.sizeof(struct_ctype)
    return LaunchParamsBuffer(
        struct_type=wp_struct.instance_type,
        struct_ctype=struct_ctype,
        bytes=wp.empty(nbytes, dtype=wp.uint8, device=device),
        nbytes=nbytes,
        device=device,
    )


def write_launch_params(buffer: LaunchParamsBuffer, params_struct_instance) -> None:
    if not isinstance(buffer, LaunchParamsBuffer):
        raise TypeError("buffer must be a LaunchParamsBuffer created by create_launch_params_buffer()")
    if not hasattr(params_struct_instance, "_ctype"):
        raise TypeError("params_struct_instance must be a Warp struct instance")
    if not isinstance(params_struct_instance, buffer.struct_type):
        raise TypeError(
            f"params struct type mismatch: expected {buffer.struct_type.__name__}, got {type(params_struct_instance).__name__}"
        )

    host_bytes = np.frombuffer(bytes(params_struct_instance._ctype), dtype=np.uint8)
    if host_bytes.size != buffer.nbytes:
        raise RuntimeError(
            f"params struct size mismatch: expected {buffer.nbytes} bytes, got {host_bytes.size} bytes from struct ctype"
        )
    wp.copy(buffer.bytes, wp.array(host_bytes, dtype=wp.uint8, device="cpu", copy=False))


def launch(optix, pipeline, sbt, width: int, height: int, params_buffer: LaunchParamsBuffer, stream: int = 0) -> None:
    if not isinstance(params_buffer, LaunchParamsBuffer):
        raise TypeError("params_buffer must be a LaunchParamsBuffer")
    optix.launch(pipeline, stream, params_buffer.bytes.ptr, params_buffer.nbytes, sbt, width, height, 1)


def get_optix_entry_name(kernel_or_entry, expected_kernel_type=None) -> str:
    if isinstance(kernel_or_entry, str):
        return kernel_or_entry

    if not isinstance(kernel_or_entry, wp.Kernel):
        raise TypeError("expected a Warp kernel object or explicit entry name string")

    kernel_type = kernel_or_entry.options.get("optix_kernel_type")
    if kernel_type is None:
        raise TypeError(
            f"kernel '{kernel_or_entry.key}' is not an OptiX kernel "
            "(decorate it with @warp_optix.optix_kernel(...) instead of @wp.kernel)"
        )
    if expected_kernel_type is not None and kernel_type is not expected_kernel_type:
        raise TypeError(f"kernel '{kernel_or_entry.key}' has type {kernel_type}, expected {expected_kernel_type}")

    return f"{kernel_type.value}{kernel_or_entry.get_mangled_name()}"


def _prepend_device_preamble(build_options: wp.ModuleBuildOptions, preamble: str) -> wp.ModuleBuildOptions:
    if not preamble:
        return build_options

    if not preamble.endswith("\n"):
        preamble += "\n"

    return wp.ModuleBuildOptions(
        extra_include_dirs=build_options.extra_include_dirs,
        extra_cuda_include_dirs=build_options.extra_cuda_include_dirs,
        extra_cpu_include_dirs=build_options.extra_cpu_include_dirs,
        extra_device_preamble=preamble + build_options.extra_device_preamble,
        extra_cpu_preamble=build_options.extra_cpu_preamble,
    )


def compile_warp_module_to_ptx(
    module: wp.Module,
    launch_preamble: str,
    module_tag: str,
    script_dir: str,
    device: str = "cuda",
) -> bytes:
    if not hasattr(wp, "compile_module_to_ptx"):
        raise RuntimeError("warp_optix requires a Warp build with wp.compile_module_to_ptx()")

    del script_dir  # Preserved for backward-compatible call sites.

    old_build_options = module.options.get("extra_build_options")
    build_options = get_module_build_options(wp, old_build_options)
    module.options["extra_build_options"] = _prepend_device_preamble(build_options, launch_preamble)
    try:
        wp.get_device(device)
        module_dir = os.path.join(wp.config.kernel_cache_dir, "optix", module_tag)
        return wp.compile_module_to_ptx(module, device=device, module_dir=module_dir)
    finally:
        module.options["extra_build_options"] = old_build_options


def create_triangle_gas(optix, ctx, vertices: np.ndarray, indices: np.ndarray, device: str):
    d_vertices = wp.array(vertices, dtype=wp.float32, device=device)
    d_indices = wp.array(indices, dtype=wp.uint32, device=device)

    options = optix.AccelBuildOptions(
        buildFlags=int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
        operation=optix.BUILD_OPERATION_BUILD,
    )
    tri = optix.BuildInputTriangleArray()
    tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
    tri.numVertices = vertices.shape[0]
    tri.vertexStrideInBytes = 12
    tri.vertexBuffers = [d_vertices.ptr]
    tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
    tri.numIndexTriplets = indices.shape[0]
    tri.indexStrideInBytes = 12
    tri.indexBuffer = d_indices.ptr
    tri.flags = [optix.GEOMETRY_FLAG_NONE]
    tri.numSbtRecords = 1

    sizes = ctx.accelComputeMemoryUsage([options], [tri])
    d_temp = wp.empty(sizes.tempSizeInBytes, dtype=wp.uint8, device=device)
    d_gas = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device=device)
    handle = ctx.accelBuild(
        0,
        [options],
        [tri],
        d_temp.ptr,
        sizes.tempSizeInBytes,
        d_gas.ptr,
        sizes.outputSizeInBytes,
        [],
    )
    wp.synchronize_device(device)
    keepalive = {"d_vertices": d_vertices, "d_indices": d_indices, "d_temp": d_temp, "d_gas": d_gas}
    return int(handle), keepalive


def create_custom_primitive_gas(
    optix,
    ctx,
    aabbs: np.ndarray,
    device: str,
    *,
    geometry_flags=None,
    build_flags=None,
    sbt_index_offsets: np.ndarray | None = None,
    num_sbt_records: int = 1,
    primitive_index_offset: int = 0,
):
    """Build a GAS for custom primitives from object-space AABBs.

    ``aabbs`` must contain one ``(min_x, min_y, min_z, max_x, max_y,
    max_z)`` record per primitive. It may be shaped ``(N, 6)`` or
    ``(N, 2, 3)``. Optional SBT index offsets select a hit-group record per
    primitive, following the normal OptiX custom-primitive build semantics.
    """
    host_aabbs = np.asarray(aabbs, dtype=np.float32)
    if host_aabbs.ndim == 3 and host_aabbs.shape[1:] == (2, 3):
        host_aabbs = host_aabbs.reshape((-1, 6))
    if host_aabbs.ndim != 2 or host_aabbs.shape[1] != 6:
        raise ValueError("aabbs must have shape (N, 6) or (N, 2, 3)")
    if host_aabbs.shape[0] == 0:
        raise ValueError("aabbs must contain at least one primitive")
    if not np.all(np.isfinite(host_aabbs)):
        raise ValueError("aabbs must contain only finite values")
    if np.any(host_aabbs[:, :3] > host_aabbs[:, 3:]):
        raise ValueError("each AABB minimum must be less than or equal to its maximum")
    host_aabbs = np.ascontiguousarray(host_aabbs)

    num_sbt_records = int(num_sbt_records)
    if num_sbt_records < 1:
        raise ValueError("num_sbt_records must be at least 1")

    if geometry_flags is None:
        flags = [int(optix.GEOMETRY_FLAG_NONE)] * num_sbt_records
    elif isinstance(geometry_flags, (int, np.integer)):
        flags = [int(geometry_flags)] * num_sbt_records
    else:
        flags = [int(flag) for flag in geometry_flags]
        if len(flags) != num_sbt_records:
            raise ValueError("geometry_flags must contain one flag per SBT record")

    d_aabbs = wp.array(host_aabbs, dtype=wp.float32, device=device)
    d_sbt_indices = None
    if sbt_index_offsets is not None:
        host_sbt_indices = np.asarray(sbt_index_offsets, dtype=np.uint32)
        if host_sbt_indices.shape != (host_aabbs.shape[0],):
            raise ValueError("sbt_index_offsets must have shape (N,), matching the AABB count")
        if np.any(host_sbt_indices >= num_sbt_records):
            raise ValueError("sbt_index_offsets values must be less than num_sbt_records")
        d_sbt_indices = wp.array(np.ascontiguousarray(host_sbt_indices), dtype=wp.uint32, device=device)

    options = optix.AccelBuildOptions(
        buildFlags=int(optix.BUILD_FLAG_NONE if build_flags is None else build_flags),
        operation=optix.BUILD_OPERATION_BUILD,
    )
    custom = optix.BuildInputCustomPrimitiveArray(
        aabbBuffers=[d_aabbs.ptr],
        numPrimitives=host_aabbs.shape[0],
        strideInBytes=6 * np.dtype(np.float32).itemsize,
        flags=flags,
        numSbtRecords=num_sbt_records,
        sbtIndexOffsetBuffer=0 if d_sbt_indices is None else d_sbt_indices.ptr,
        sbtIndexOffsetSizeInBytes=0 if d_sbt_indices is None else np.dtype(np.uint32).itemsize,
        sbtIndexOffsetStrideInBytes=0 if d_sbt_indices is None else np.dtype(np.uint32).itemsize,
        primitiveIndexOffset=int(primitive_index_offset),
    )

    sizes = ctx.accelComputeMemoryUsage([options], [custom])
    d_temp = wp.empty(sizes.tempSizeInBytes, dtype=wp.uint8, device=device)
    d_gas = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device=device)
    handle = ctx.accelBuild(
        0,
        [options],
        [custom],
        d_temp.ptr,
        sizes.tempSizeInBytes,
        d_gas.ptr,
        sizes.outputSizeInBytes,
        [],
    )
    wp.synchronize_device(device)
    keepalive = {
        "d_aabbs": d_aabbs,
        "d_sbt_indices": d_sbt_indices,
        "d_temp": d_temp,
        "d_gas": d_gas,
    }
    return int(handle), keepalive


def create_instance_acceleration_structure(optix, ctx, instances, device: str, *, build_flags=None):
    """Build an IAS from a non-empty sequence of ``optix.Instance`` objects."""
    instances = list(instances)
    if not instances:
        raise ValueError("instances must contain at least one optix.Instance")

    host_bytes = np.frombuffer(optix.getDeviceRepresentation(instances), dtype=np.uint8)
    d_instances = wp.array(host_bytes, dtype=wp.uint8, device=device)
    instance_input = optix.BuildInputInstanceArray(instances=d_instances.ptr, numInstances=len(instances))
    options = optix.AccelBuildOptions(
        buildFlags=int(optix.BUILD_FLAG_NONE if build_flags is None else build_flags),
        operation=optix.BUILD_OPERATION_BUILD,
    )
    sizes = ctx.accelComputeMemoryUsage([options], [instance_input])
    d_temp = wp.empty(sizes.tempSizeInBytes, dtype=wp.uint8, device=device)
    d_ias = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device=device)
    handle = ctx.accelBuild(
        0,
        [options],
        [instance_input],
        d_temp.ptr,
        sizes.tempSizeInBytes,
        d_ias.ptr,
        sizes.outputSizeInBytes,
        [],
    )
    wp.synchronize_device(device)
    return int(handle), {"d_instances": d_instances, "d_temp": d_temp, "d_ias": d_ias}


def _set_pipeline_stack_size(optix, pipeline, program_groups, max_trace_depth: int, max_traversable_depth: int):
    if hasattr(optix, "util") and hasattr(optix, "StackSizes"):
        stack_sizes = optix.StackSizes()
        for group in program_groups:
            if _optix_version_at_least(optix, 7, 7):
                optix.util.accumulateStackSizes(group, stack_sizes, pipeline)
            else:
                optix.util.accumulateStackSizes(group, stack_sizes)
        direct_traversal, direct_state, continuation = optix.util.computeStackSizes(
            stack_sizes, max_trace_depth, 0, 0
        )
        pipeline.setStackSize(
            direct_traversal, direct_state, continuation, max_traversable_depth
        )
    else:
        pipeline.setStackSize(2 * 1024, 2 * 1024, 2 * 1024, max_traversable_depth)


def create_pipeline_and_sbt(
    optix,
    ctx,
    ptx: bytes,
    raygen_entry: wp.Kernel | str,
    miss_entry: wp.Kernel | str,
    closest_hit_entry: wp.Kernel | str | None,
    num_payload_values: int,
    num_attribute_values: int,
    device: str,
    intersection_entry: wp.Kernel | str | None = None,
    any_hit_entry: wp.Kernel | str | None = None,
    primitive_type_flags: int | None = None,
    hit_groups=None,
    traversable_graph_flags: int | None = None,
    max_traversable_depth: int | None = None,
):
    from warp_optix._codegen import OptixKernelType  # noqa: PLC0415
    from warp_optix._runtime.hit_kernels import HitKernel  # noqa: PLC0415
    from warp_optix._runtime.sbt import SbtKernelManager  # noqa: PLC0415

    raygen_name = get_optix_entry_name(raygen_entry, expected_kernel_type=OptixKernelType.RAYGEN)
    miss_name = get_optix_entry_name(miss_entry, expected_kernel_type=OptixKernelType.MISS)

    if hit_groups is not None:
        if any(entry is not None for entry in (closest_hit_entry, intersection_entry, any_hit_entry)):
            raise ValueError("hit_groups cannot be combined with the individual hit entry arguments")
        requested_hit_groups = list(hit_groups)
        if not all(isinstance(group, HitKernel) for group in requested_hit_groups):
            raise TypeError("hit_groups must contain HitKernel objects")
    else:
        requested_hit_groups = []
        if any(entry is not None for entry in (closest_hit_entry, intersection_entry, any_hit_entry)):
            requested_hit_groups.append(HitKernel(closest_hit_entry, any_hit_entry, intersection_entry))

    resolved_hit_groups = []
    for group in requested_hit_groups:
        closest_hit_name = (
            get_optix_entry_name(group.closest_hit, expected_kernel_type=OptixKernelType.CLOSEST_HIT)
            if group.closest_hit is not None
            else None
        )
        any_hit_name = (
            get_optix_entry_name(group.any_hit, expected_kernel_type=OptixKernelType.ANY_HIT)
            if group.any_hit is not None
            else None
        )
        intersection_name = (
            get_optix_entry_name(group.intersection, expected_kernel_type=OptixKernelType.INTERSECTION)
            if group.intersection is not None
            else None
        )
        if not (closest_hit_name or any_hit_name or intersection_name):
            raise ValueError("each hit group must define at least one program")
        resolved_hit_groups.append((closest_hit_name, any_hit_name, intersection_name))

    if traversable_graph_flags is None:
        traversable_graph_flags = optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
    if max_traversable_depth is None:
        max_traversable_depth = (
            1 if int(traversable_graph_flags) == int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS) else 2
        )

    kwargs = {
        "usesMotionBlur": False,
        "traversableGraphFlags": int(traversable_graph_flags),
        "numPayloadValues": num_payload_values,
        "numAttributeValues": num_attribute_values,
        "exceptionFlags": int(optix.EXCEPTION_FLAG_NONE),
        "pipelineLaunchParamsVariableName": "params",
    }
    if _optix_version_at_least(optix, 7, 2):
        if primitive_type_flags is None:
            primitive_type_flags = 0
            for _, _, intersection_name in resolved_hit_groups:
                primitive_type_flags |= int(
                    optix.PRIMITIVE_TYPE_FLAGS_CUSTOM
                    if intersection_name is not None
                    else optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE
                )
            if not resolved_hit_groups:
                primitive_type_flags = optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE
        kwargs["usesPrimitiveTypeFlags"] = int(primitive_type_flags)
    pipeline_options = optix.PipelineCompileOptions(**kwargs)
    module_options = optix.ModuleCompileOptions(
        maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
    )
    module, log = ctx.moduleCreate(module_options, pipeline_options, ptx)
    if log:
        print(f"Module create log:\n{log}")

    sbt_manager = SbtKernelManager(optix, ctx, module)
    sbt_manager.set_raygen_kernel(raygen_name)
    sbt_manager.add_miss_kernels([miss_name])
    hit_group_handles = []
    for closest_hit_name, any_hit_name, intersection_name in resolved_hit_groups:
        hit_group_handles.append(
            sbt_manager.register_hit_shader_type(
                HitKernel(closest_hit_name, any_hit_name, intersection_name)
            )
        )

    groups = sbt_manager.get_all_program_groups()

    link_options = optix.PipelineLinkOptions()
    link_options.maxTraceDepth = 1
    pipeline = ctx.pipelineCreate(pipeline_options, link_options, groups, "")
    _set_pipeline_stack_size(optix, pipeline, groups, link_options.maxTraceDepth, int(max_traversable_depth))

    sbt_resources = sbt_manager.build_sbt(device=device)
    keepalive = {
        "module": module,
        "program_groups": groups,
        "sbt": sbt_resources.keepalive,
        "sbt_manager": sbt_manager,
        "hit_group_handles": hit_group_handles,
    }
    return pipeline, sbt_resources.sbt, keepalive
