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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import warp as wp
from warp_optix._runtime.hit_kernels import HitKernel, HitKernelManager


def _round_up(value: int, alignment: int) -> int:
    return value if (value % alignment) == 0 else value + alignment - (value % alignment)


def _aligned_record_dtype(optix, payload_dtype: np.dtype | None = None) -> np.dtype:
    header_format = f"{optix.SBT_RECORD_HEADER_SIZE}B"
    payload_size = 0 if payload_dtype is None else payload_dtype.itemsize
    itemsize = _round_up(optix.SBT_RECORD_HEADER_SIZE + payload_size, optix.SBT_RECORD_ALIGNMENT)

    if payload_dtype is None:
        names = ["header"]
        formats = [header_format]
    else:
        names = ["header", "data"]
        formats = [header_format, payload_dtype]

    return np.dtype({"names": names, "formats": formats, "itemsize": itemsize, "align": True})


def _pack_headers(optix, program_groups, record_dtype: np.dtype) -> np.ndarray:
    host_records = np.zeros(len(program_groups), dtype=record_dtype)
    for i, pg in enumerate(program_groups):
        optix.sbtRecordPackHeader(pg, host_records[i : i + 1])
    return host_records


def _to_device_bytes(host_records: np.ndarray, device: str) -> wp.array:
    host_bytes = np.ascontiguousarray(host_records).view(np.uint8).reshape(-1)
    return wp.array(host_bytes, dtype=wp.uint8, device=device)


@dataclass
class SbtResources:
    sbt: object
    keepalive: dict


class SbtKernelManager:
    """Builds raygen/miss/hit program groups and a Shader Binding Table."""

    def __init__(self, optix, ctx, module, num_ray_subtypes: int = 1) -> None:
        self.optix = optix
        self.ctx = ctx
        self.module = module
        self.raygen_group = None
        self.miss_groups = []
        self.hit_kernels = HitKernelManager(optix, ctx, module, num_ray_subtypes)

    @staticmethod
    def _entry_name(kernel_or_name, kernel_type):
        from warp_optix._runtime.runtime import get_optix_entry_name  # noqa: PLC0415

        return get_optix_entry_name(kernel_or_name, expected_kernel_type=kernel_type)

    def set_raygen_kernel(self, kernel_or_name) -> None:
        from warp_optix._codegen import OptixKernelType  # noqa: PLC0415

        kernel_name = self._entry_name(kernel_or_name, OptixKernelType.RAYGEN)
        desc = self.optix.ProgramGroupDesc()
        desc.raygenModule = self.module
        desc.raygenEntryFunctionName = kernel_name
        if tuple(self.optix.version()) >= (7, 4):
            self.raygen_group = self.ctx.programGroupCreate([desc], self.optix.ProgramGroupOptions())[0][0]
        else:
            self.raygen_group = self.ctx.programGroupCreate([desc])[0][0]

    def add_miss_kernels(self, kernels) -> None:
        from warp_optix._codegen import OptixKernelType  # noqa: PLC0415

        for kernel in kernels:
            name = self._entry_name(kernel, OptixKernelType.MISS)
            desc = self.optix.ProgramGroupDesc()
            desc.missModule = self.module
            desc.missEntryFunctionName = name
            if tuple(self.optix.version()) >= (7, 4):
                pg = self.ctx.programGroupCreate([desc], self.optix.ProgramGroupOptions())[0][0]
            else:
                pg = self.ctx.programGroupCreate([desc])[0][0]
            self.miss_groups.append(pg)

    def register_hit_shader_type(self, *kernels: str | HitKernel):
        from warp_optix._codegen import OptixKernelType  # noqa: PLC0415

        resolved = []
        for kernel in kernels:
            if not isinstance(kernel, HitKernel):
                kernel = HitKernel(closest_hit=kernel)
            resolved.append(
                HitKernel(
                    closest_hit=(
                        self._entry_name(kernel.closest_hit, OptixKernelType.CLOSEST_HIT)
                        if kernel.closest_hit is not None
                        else None
                    ),
                    any_hit=(
                        self._entry_name(kernel.any_hit, OptixKernelType.ANY_HIT)
                        if kernel.any_hit is not None
                        else None
                    ),
                    intersection=(
                        self._entry_name(kernel.intersection, OptixKernelType.INTERSECTION)
                        if kernel.intersection is not None
                        else None
                    ),
                    builtin_intersection_type=kernel.builtin_intersection_type,
                    intersection_module=kernel.intersection_module,
                )
            )
        return self.hit_kernels.register_hit_shader_type(*resolved)

    def get_sbt_offset(self, handle) -> int:
        """Resolve an opaque hit-group handle to its SBT record offset."""
        return self.hit_kernels.get_sbt_offset(handle)

    def get_all_program_groups(self):
        groups = []
        if self.raygen_group is not None:
            groups.append(self.raygen_group)
        groups.extend(self.miss_groups)
        groups.extend(self.hit_kernels.get_list())
        return groups

    def build_sbt(self, device: str = "cuda") -> SbtResources:
        if self.raygen_group is None:
            raise RuntimeError("Raygen kernel not set")
        if not self.miss_groups:
            raise RuntimeError("At least one miss kernel is required")

        record_dtype = _aligned_record_dtype(self.optix)
        host_raygen = _pack_headers(self.optix, [self.raygen_group], record_dtype)
        host_miss = _pack_headers(self.optix, self.miss_groups, record_dtype)
        hit_groups = self.hit_kernels.get_list()
        host_hit = (
            _pack_headers(self.optix, hit_groups, record_dtype) if hit_groups else np.zeros(0, dtype=record_dtype)
        )

        device_raygen = _to_device_bytes(host_raygen, device=device)
        device_miss = _to_device_bytes(host_miss, device=device)
        device_hit = _to_device_bytes(host_hit, device=device) if len(host_hit) > 0 else None

        sbt = self.optix.ShaderBindingTable()
        sbt.raygenRecord = device_raygen.ptr
        sbt.missRecordBase = device_miss.ptr
        sbt.missRecordStrideInBytes = host_miss.dtype.itemsize
        sbt.missRecordCount = len(host_miss)

        if device_hit is not None:
            sbt.hitgroupRecordBase = device_hit.ptr
            sbt.hitgroupRecordStrideInBytes = host_hit.dtype.itemsize
            sbt.hitgroupRecordCount = len(host_hit)
        else:
            sbt.hitgroupRecordBase = 0
            sbt.hitgroupRecordStrideInBytes = 0
            sbt.hitgroupRecordCount = 0

        keepalive = {"d_rg": device_raygen, "d_ms": device_miss, "d_hg": device_hit}
        return SbtResources(sbt=sbt, keepalive=keepalive)
