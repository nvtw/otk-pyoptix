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
import hashlib
import os
from dataclasses import dataclass

import numpy as np

import warp as wp
import warp._src.build as wp_build
import warp._src.types
from warp._src.thirdparty import appdirs

from warp_optix._addon import get_module_build_options


def require_optix():
    try:
        import optix  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError("Install otk-pyoptix first: https://github.com/NVIDIA/otk-pyoptix") from e
    return optix


class OptixLogger:
    def __init__(self):
        self.num_messages = 0

    def __call__(self, level, tag, message):
        print(f"[{level:>2}][{tag:>12}]: {message}")
        self.num_messages += 1


def create_optix_context(optix, cuda_context, log_level: int = 4):
    logger = OptixLogger()
    options = optix.DeviceContextOptions(logCallbackFunction=logger, logCallbackLevel=log_level)
    if optix.version()[1] >= 2:
        options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL
    return optix.deviceContextCreate(cuda_context, options), logger


def _round_up(v: int, alignment: int) -> int:
    return v if (v % alignment) == 0 else v + alignment - (v % alignment)


def aligned_record_dtype(optix) -> np.dtype:
    header_format = f"{optix.SBT_RECORD_HEADER_SIZE}B"
    itemsize = _round_up(optix.SBT_RECORD_HEADER_SIZE, optix.SBT_RECORD_ALIGNMENT)
    return np.dtype({"names": ["header"], "formats": [header_format], "itemsize": itemsize, "align": True})


def to_device_bytes(host_records: np.ndarray, device: str) -> wp.array:
    host_bytes = np.ascontiguousarray(host_records).view(np.uint8).reshape(-1)
    return wp.array(host_bytes, dtype=wp.uint8, device=device)


@dataclass
class LaunchParamsBuffer:
    struct_type: type
    struct_ctype: type[ctypes.Structure]
    bytes: wp.array
    nbytes: int
    device: str


def _get_struct_type_info(params_struct_type):
    if isinstance(params_struct_type, wp._src.codegen.Struct):
        wp_struct = params_struct_type
    else:
        wp_struct = getattr(params_struct_type, "_cls", None)
        if wp_struct is None and hasattr(params_struct_type, "__class__"):
            wp_struct = getattr(params_struct_type.__class__, "_cls", None)

    if not isinstance(wp_struct, wp._src.codegen.Struct):
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
    if not warp._src.types.is_struct(params_struct_instance):
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


def _default_kernel_cache_root() -> str:
    cache_root_dir = appdirs.user_cache_dir(appname="warp", appauthor="NVIDIA", version=wp.config.version)

    if os.name == "nt" and os.path.isabs(cache_root_dir) and not cache_root_dir.startswith("\\\\?\\"):
        # Match build.init_kernel_cache() long-path handling.
        if cache_root_dir.startswith("\\\\"):
            cache_root_dir = "\\\\?\\UNC\\" + cache_root_dir.removeprefix("\\\\")
        else:
            cache_root_dir = "\\\\?\\" + cache_root_dir

    return cache_root_dir


def compile_warp_module_to_ptx(
    module: wp.Module,
    launch_preamble: str,
    module_tag: str,
    script_dir: str,
    device: str = "cuda",
) -> bytes:
    device_obj = wp.get_device(device)
    if not device_obj.is_cuda:
        raise RuntimeError(f"PTX can only be generated for CUDA devices, got '{device_obj}'")

    old_build_options = module.options.get("extra_build_options")
    module.options["extra_build_options"] = get_module_build_options(wp, old_build_options)
    try:
        options = module.resolve_options(wp.config)
        hasher = wp._src.context.ModuleHasher(module._get_live_kernels(), options)
        builder = wp._src.context.ModuleBuilder(module, options=options, hasher=hasher)
        warp_cuda = builder.codegen("cuda")
    finally:
        module.options["extra_build_options"] = old_build_options

    cuda_source = launch_preamble + "\n" + warp_cuda

    digest = hashlib.sha256(cuda_source.encode("utf-8")).hexdigest()[:16]
    module_dir = os.path.join(wp.config.kernel_cache_dir, f"wp_optix_{module_tag}_{digest}")
    try:
        os.makedirs(module_dir, exist_ok=True)
    except PermissionError:
        fallback_cache_root = os.path.join(_default_kernel_cache_root(), "optix")
        module_dir = os.path.join(fallback_cache_root, f"wp_optix_{module_tag}_{digest}")
        os.makedirs(module_dir, exist_ok=True)

    cu_path = os.path.join(module_dir, f"wp_optix_{module_tag}_{digest}.cu")
    ptx_path = os.path.join(module_dir, f"wp_optix_{module_tag}_{digest}.ptx")
    if not os.path.exists(ptx_path) or not wp.config.cache_kernels:
        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(cuda_source)

        old_use_pch = wp.config.use_precompiled_headers
        try:
            wp.config.use_precompiled_headers = False
            wp_build.build_cuda(
                cu_path,
                arch=device_obj.arch,
                output_path=ptx_path,
                pch_dir=module_dir,
                extra_include_dirs=options["extra_cuda_include_dirs"],
            )
        finally:
            wp.config.use_precompiled_headers = old_use_pch

    with open(ptx_path, "rb") as f:
        return f.read()


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
):
    from warp_optix._codegen import OptixKernelType  # noqa: PLC0415

    raygen_name = get_optix_entry_name(raygen_entry, expected_kernel_type=OptixKernelType.RAYGEN)
    miss_name = get_optix_entry_name(miss_entry, expected_kernel_type=OptixKernelType.MISS)
    closest_hit_name = (
        get_optix_entry_name(closest_hit_entry, expected_kernel_type=OptixKernelType.CLOSEST_HIT)
        if closest_hit_entry is not None
        else None
    )

    kwargs = {
        "usesMotionBlur": False,
        "traversableGraphFlags": int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
        "numPayloadValues": num_payload_values,
        "numAttributeValues": num_attribute_values,
        "exceptionFlags": int(optix.EXCEPTION_FLAG_NONE),
        "pipelineLaunchParamsVariableName": "params",
    }
    if optix.version()[1] >= 2:
        kwargs["usesPrimitiveTypeFlags"] = int(optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
    pipeline_options = optix.PipelineCompileOptions(**kwargs)
    module_options = optix.ModuleCompileOptions(
        maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
    )
    module, log = ctx.moduleCreate(module_options, pipeline_options, ptx)
    if log:
        print(f"Module create log:\n{log}")

    rg_desc = optix.ProgramGroupDesc()
    rg_desc.raygenModule = module
    rg_desc.raygenEntryFunctionName = raygen_name

    ms_desc = optix.ProgramGroupDesc()
    ms_desc.missModule = module
    ms_desc.missEntryFunctionName = miss_name

    groups_desc = [rg_desc, ms_desc]
    if closest_hit_name is not None:
        hg_desc = optix.ProgramGroupDesc()
        hg_desc.hitgroupModuleCH = module
        hg_desc.hitgroupEntryFunctionNameCH = closest_hit_name
        groups_desc.append(hg_desc)

    groups = []
    if optix.version()[1] >= 4:
        pg_options = optix.ProgramGroupOptions()
        for desc in groups_desc:
            groups.append(ctx.programGroupCreate([desc], pg_options)[0][0])
    else:
        for desc in groups_desc:
            groups.append(ctx.programGroupCreate([desc])[0][0])

    link_options = optix.PipelineLinkOptions()
    link_options.maxTraceDepth = 1
    pipeline = ctx.pipelineCreate(pipeline_options, link_options, groups, "")
    pipeline.setStackSize(2 * 1024, 2 * 1024, 2 * 1024, 1)

    record_dtype = aligned_record_dtype(optix)
    h_records = [np.zeros(1, dtype=record_dtype) for _ in groups]
    for host_record, group in zip(h_records, groups):
        optix.sbtRecordPackHeader(group, host_record)

    d_records = [to_device_bytes(h, device=device) for h in h_records]
    sbt = optix.ShaderBindingTable()
    sbt.raygenRecord = d_records[0].ptr
    sbt.missRecordBase = d_records[1].ptr
    sbt.missRecordStrideInBytes = h_records[1].dtype.itemsize
    sbt.missRecordCount = 1
    if len(d_records) > 2:
        sbt.hitgroupRecordBase = d_records[2].ptr
        sbt.hitgroupRecordStrideInBytes = h_records[2].dtype.itemsize
        sbt.hitgroupRecordCount = 1
    else:
        sbt.hitgroupRecordBase = 0
        sbt.hitgroupRecordStrideInBytes = 0
        sbt.hitgroupRecordCount = 0

    keepalive = {"module": module, "program_groups": groups, "records": d_records}
    return pipeline, sbt, keepalive
