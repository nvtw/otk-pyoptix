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

"""Public OptiX integration helpers for Warp."""

from __future__ import annotations


from warp_optix._addon import register_with_warp as _register_with_warp
_register_with_warp()

from warp_optix._codegen import OptixKernelType as OptixKernelType
from warp_optix._codegen import optix_kernel as optix_kernel
from warp_optix._runtime.gl_interop import OptixGLInteropViewer as GLInteropViewer
from warp_optix._runtime.gl_line_overlay import GLLineOverlay as GLLineOverlay
from warp_optix._runtime.hit_kernels import HitKernel as HitKernel
from warp_optix._runtime.hit_kernels import HitKernelManager as HitKernelManager
from warp_optix._runtime.runtime import LaunchParamsBuffer as LaunchParamsBuffer
from warp_optix._runtime.runtime import AccelResources as AccelResources
from warp_optix._runtime.runtime import compile_warp_module_to_ptx as compile_warp_module_to_ptx
from warp_optix._runtime.runtime import create_custom_primitive_gas as create_custom_primitive_gas
from warp_optix._runtime.runtime import create_curve_gas as create_curve_gas
from warp_optix._runtime.runtime import create_instance_acceleration_structure as create_instance_acceleration_structure
from warp_optix._runtime.runtime import create_launch_params_buffer as create_launch_params_buffer
from warp_optix._runtime.runtime import create_optix_context as create_context
from warp_optix._runtime.runtime import create_pipeline_and_sbt as create_pipeline_and_sbt
from warp_optix._runtime.runtime import create_triangle_gas as create_triangle_gas
from warp_optix._runtime.runtime import get_optix_entry_name as get_entry_name
from warp_optix._runtime.runtime import launch as launch
from warp_optix._runtime.runtime import require_optix as require_optix
from warp_optix._runtime.runtime import refit_acceleration_structure as refit_acceleration_structure
from warp_optix._runtime.runtime import write_launch_params as write_launch_params
from warp_optix._runtime.sbt import SbtKernelManager as SbtKernelManager
from warp_optix._runtime.sbt import SbtResources as SbtResources

__all__ = [
    "GLInteropViewer",
    "GLLineOverlay",
    "AccelResources",
    "HitKernel",
    "HitKernelManager",
    "LaunchParamsBuffer",
    "OptixKernelType",
    "SbtKernelManager",
    "SbtResources",
    "compile_warp_module_to_ptx",
    "create_context",
    "create_custom_primitive_gas",
    "create_curve_gas",
    "create_instance_acceleration_structure",
    "create_launch_params_buffer",
    "create_pipeline_and_sbt",
    "create_triangle_gas",
    "get_entry_name",
    "launch",
    "optix_kernel",
    "require_optix",
    "refit_acceleration_structure",
    "write_launch_params",
]
