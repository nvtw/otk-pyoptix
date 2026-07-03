from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import warp as wp

class AccelResources(dict):
    handle: int
    build_flags: int
    output_size_in_bytes: int
    uncompacted_size_in_bytes: int
    update_temp_size_in_bytes: int
    compacted: bool
    device: str

class OptixKernelType(Enum):
    RAYGEN: str
    MISS: str
    CLOSEST_HIT: str
    ANY_HIT: str
    INTERSECTION: str

def optix_kernel(kind: OptixKernelType, **kernel_kwargs: Any): ...

@dataclass
class HitKernel:
    closest_hit: wp.Kernel | str | None = None
    any_hit: wp.Kernel | str | None = None
    intersection: wp.Kernel | str | None = None

class HitKernelManager:
    @property
    def count(self) -> int: ...
    def register_hit_shader_type(self, *kernel_names: str | HitKernel): ...
    def get_sbt_offset(self, handle) -> int: ...
    def get_list(self): ...

@dataclass
class SbtResources:
    sbt: object
    keepalive: dict

class SbtKernelManager:
    def __init__(self, optix, ctx, module, num_ray_subtypes: int = 1) -> None: ...
    def set_raygen_kernel(self, kernel: wp.Kernel | str) -> None: ...
    def add_miss_kernels(self, kernels: list[wp.Kernel | str]) -> None: ...
    def register_hit_shader_type(self, *kernels: wp.Kernel | str | HitKernel): ...
    def get_sbt_offset(self, handle) -> int: ...
    def get_all_program_groups(self): ...
    def build_sbt(self, device: str = "cuda") -> SbtResources: ...

@dataclass
class LaunchParamsBuffer:
    struct_type: type
    struct_ctype: type
    bytes: wp.array
    nbytes: int
    device: str

class GLInteropViewer:
    def __init__(
        self,
        width: int,
        height: int,
        device: str,
        title: str = "Warp OptiX Tiny Raytracer",
        fps: int = 60,
        resizable: bool = False,
    ): ...
    def run(self, render_callback: Any, max_frames: int = 0): ...

def require_optix(): ...
def create_context(optix, cuda_context, log_level: int = 4): ...
def compile_warp_module_to_ptx(
    module: wp.Module,
    launch_preamble: str,
    module_tag: str,
    script_dir: str,
    device: str = "cuda",
) -> bytes: ...
def create_triangle_gas(
    optix, ctx, vertices, indices, device: str, *, build_flags=None, compact: bool = False
) -> tuple[int, AccelResources]: ...
def create_custom_primitive_gas(
    optix,
    ctx,
    aabbs,
    device: str,
    *,
    geometry_flags=None,
    build_flags=None,
    sbt_index_offsets=None,
    num_sbt_records: int = 1,
    primitive_index_offset: int = 0,
    compact: bool = False,
) -> tuple[int, AccelResources]: ...
def create_instance_acceleration_structure(
    optix, ctx, instances, device: str, *, build_flags=None, compact: bool = False
) -> tuple[int, AccelResources]: ...
def refit_acceleration_structure(optix, ctx, resources: AccelResources, *, stream: int = 0) -> int: ...
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
    hit_groups: list[HitKernel] | None = None,
    traversable_graph_flags: int | None = None,
    max_traversable_depth: int | None = None,
): ...
def get_entry_name(kernel_or_entry, expected_kernel_type: OptixKernelType | None = None) -> str: ...
def create_launch_params_buffer(params_struct_type: type, device: str = "cuda") -> LaunchParamsBuffer: ...
def write_launch_params(buffer: LaunchParamsBuffer, params_struct_instance) -> None: ...
def launch(
    optix, pipeline, sbt, width: int, height: int, params_buffer: LaunchParamsBuffer, stream: int = 0
) -> None: ...
