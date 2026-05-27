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


@dataclass(frozen=True)
class _Handle:
    value: int


class _HandleBuffer:
    def __init__(self) -> None:
        self._data: dict[int, int] = {}
        self._next = 0

    @property
    def count(self) -> int:
        return len(self._data)

    def add(self, value: int) -> _Handle:
        handle = _Handle(self._next)
        self._next += 1
        self._data[handle.value] = value
        return handle

    def try_get_value(self, handle: _Handle) -> tuple[bool, int | None]:
        if handle.value in self._data:
            return True, self._data[handle.value]
        return False, None


@dataclass
class HitKernel:
    closest_hit: str
    any_hit: str | None = None
    intersection: str | None = None


class HitKernelManager:
    """Utility for registering hit shader program groups and SBT offsets."""

    def __init__(self, optix, ctx, module, num_ray_subtypes: int) -> None:
        self.optix = optix
        self.ctx = ctx
        self.module = module
        self.num_ray_types_per_intersection_type = int(num_ray_subtypes)
        self._handle_to_offset = _HandleBuffer()
        self._hit_shaders = []

    @property
    def count(self) -> int:
        return self._handle_to_offset.count

    def register_hit_shader_type(self, *kernel_names: str | HitKernel):
        kernels: list[HitKernel] = []
        for kernel in kernel_names:
            if isinstance(kernel, HitKernel):
                kernels.append(kernel)
            else:
                kernels.append(HitKernel(str(kernel)))

        if len(kernels) != self.num_ray_types_per_intersection_type:
            raise ValueError(f"Expected {self.num_ray_types_per_intersection_type} kernels, got {len(kernels)}")

        index = len(self._hit_shaders)
        handle = self._handle_to_offset.add(index)
        for kernel in kernels:
            desc = self.optix.ProgramGroupDesc()
            desc.hitgroupModuleCH = self.module
            desc.hitgroupEntryFunctionNameCH = kernel.closest_hit
            if kernel.any_hit:
                desc.hitgroupModuleAH = self.module
                desc.hitgroupEntryFunctionNameAH = kernel.any_hit
            if kernel.intersection:
                desc.hitgroupModuleIS = self.module
                desc.hitgroupEntryFunctionNameIS = kernel.intersection

            if self.optix.version()[1] >= 4:
                pg_options = self.optix.ProgramGroupOptions()
                pg = self.ctx.programGroupCreate([desc], pg_options)[0][0]
            else:
                pg = self.ctx.programGroupCreate([desc])[0][0]
            self._hit_shaders.append(pg)

        return handle

    def get_sbt_offset(self, handle) -> int:
        ok, offset = self._handle_to_offset.try_get_value(handle)
        if not ok or offset is None:
            raise KeyError(f"Handle {handle.value} not found")
        return int(offset)

    def get_list(self):
        return list(self._hit_shaders)
