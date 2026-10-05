# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned Warp-facing OptiX resources."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import warp as wp

from ._backend import get_backend
from ._enums import InstanceFlags, PrimitiveType
from ._runtime import runtime


@dataclass(frozen=True)
class HitKernel:
    closest_hit: wp.Kernel | str | None = None
    any_hit: wp.Kernel | str | None = None
    intersection: wp.Kernel | str | None = None
    builtin_intersection_type: PrimitiveType | None = None


@dataclass(frozen=True)
class Instance:
    """An instance of geometry owned by the same Context as its IAS."""

    acceleration_structure: AccelerationStructure
    transform: Sequence[float] = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    instance_id: int = 0
    sbt_offset: int = 0
    visibility_mask: int = 255
    flags: InstanceFlags = InstanceFlags.NONE


class _TrackedContext:
    """Private adapter recording allocations made by the existing runtime."""

    def __init__(self, owner, handle):
        self.owner = owner
        self.handle = handle
        self.objects = []

    def __getattr__(self, name):
        self.owner._check_open()
        method = getattr(self.handle, name)
        if name not in (
            "moduleCreate",
            "moduleCreateFromPTX",
            "builtinISModuleGet",
            "programGroupCreate",
            "pipelineCreate",
        ):
            return method

        def create(*args, **kwargs):
            self.owner._check_open()
            result = method(*args, **kwargs)
            if name in ("moduleCreate", "moduleCreateFromPTX"):
                objects = [result[0]]
            elif name == "programGroupCreate":
                objects = result[0]
            else:
                objects = [result]
            for obj in objects:
                # builtinISModuleGet can return another wrapper for the same handle.
                if not any(
                    type(obj) is type(old) and obj == old for old in self.objects
                ):
                    self.objects.append(obj)
            return result

        return create

    def release_since(self, index):
        first_error = None
        while len(self.objects) > index:
            obj = self.objects.pop()
            try:
                obj.destroy()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


class Context:
    """Own OptiX resources on a Warp CUDA device until close().

    Use as a context manager. close() synchronizes the device and invalidates
    all child resources. OptiX launches cannot be captured in CUDA graphs.
    """

    def __init__(
        self,
        device: str | wp.Device = "cuda",
        *,
        log_level: int = 4,
        enable_validation: bool = False,
    ):
        self._closed = True
        self._resources = []
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("OptiX requires a Warp CUDA device")
        self._backend = get_backend()
        with wp.ScopedDevice(self.device):
            cuda_context = self.device.context
            cuda_context = getattr(cuda_context, "value", cuda_context)
            handle, self._logger = runtime.create_optix_context(
                self._backend,
                int(cuda_context),
                log_level,
                enable_validation=enable_validation,
            )
        self._closed = False
        self._native_context = _TrackedContext(self, handle)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("The OptiX context is closed")

    @property
    def native_handle(self):
        """Borrowed native context; the caller must not destroy it."""
        self._check_open()
        return self._native_context.handle

    @property
    def log_message_count(self) -> int:
        return self._logger.num_messages

    @property
    def closed(self) -> bool:
        return self._closed

    def _stream(self, stream):
        self._check_open()
        stream = wp.get_stream(self.device) if stream is None else stream
        if not isinstance(stream, wp.Stream):
            raise TypeError("stream must be a Warp Stream or None")
        if stream.device != self.device:
            raise ValueError("Stream and context must use the same CUDA device")
        return stream

    def _check_arrays(self, *values):
        for value in values:
            if isinstance(value, wp.array) and value.device != self.device:
                raise ValueError("Array and context must use the same CUDA device")

    @staticmethod
    def _host(value):
        return value.numpy() if isinstance(value, wp.array) else value

    @contextmanager
    def _transaction(self):
        self._check_open()
        index = len(self._native_context.objects)
        with wp.ScopedDevice(self.device):
            try:
                yield
            except BaseException:
                # Preserve the original construction error if cleanup also fails.
                try:
                    self._native_context.release_since(index)
                except Exception:
                    pass
                raise

    def _geometry(self, builder, *arrays, dependencies=(), **options):
        self._check_arrays(*arrays)
        with self._transaction():
            _, buffers = builder(
                self._backend,
                self._native_context,
                *(self._host(a) for a in arrays),
                device=self.device,
                **options,
            )
            result = AccelerationStructure(self, buffers, dependencies)
            self._resources.append(result)
            return result

    def create_triangle_gas(
        self,
        vertices,
        indices,
        *,
        build_flags=None,
        compact=False,
        motion_time_range=None,
    ) -> AccelerationStructure:
        return self._geometry(
            runtime.create_triangle_gas,
            vertices,
            indices,
            build_flags=build_flags,
            compact=compact,
            motion_time_range=motion_time_range,
        )

    def create_curve_gas(
        self,
        vertices,
        widths,
        segment_indices,
        *,
        curve_type=PrimitiveType.ROUND_LINEAR,
        geometry_flag=None,
        build_flags=None,
        compact=False,
        motion_time_range=None,
    ) -> AccelerationStructure:
        return self._geometry(
            runtime.create_curve_gas,
            vertices,
            widths,
            segment_indices,
            curve_type=self._backend.PrimitiveType(int(curve_type)),
            geometry_flag=geometry_flag,
            build_flags=build_flags,
            compact=compact,
            motion_time_range=motion_time_range,
        )

    def create_custom_primitive_gas(
        self,
        aabbs,
        *,
        geometry_flags=None,
        build_flags=None,
        sbt_index_offsets=None,
        num_sbt_records=1,
        primitive_index_offset=0,
        compact=False,
    ) -> AccelerationStructure:
        self._check_arrays(sbt_index_offsets)
        return self._geometry(
            runtime.create_custom_primitive_gas,
            aabbs,
            geometry_flags=geometry_flags,
            build_flags=build_flags,
            sbt_index_offsets=self._host(sbt_index_offsets),
            num_sbt_records=num_sbt_records,
            primitive_index_offset=primitive_index_offset,
            compact=compact,
        )

    def create_instance_acceleration_structure(
        self, instances: Sequence[Instance], *, build_flags=None, compact=False
    ) -> AccelerationStructure:
        self._check_open()
        native_instances, dependencies = [], []
        for instance in instances:
            if not isinstance(instance, Instance):
                raise TypeError(
                    "instances must contain warp_optix.Instance descriptors"
                )
            geometry = instance.acceleration_structure
            if (
                not isinstance(geometry, AccelerationStructure)
                or geometry._context is not self
            ):
                raise ValueError("Instance geometry must belong to this context")
            transform = np.asarray(instance.transform, dtype=np.float32)
            if transform.size != 12 or not np.isfinite(transform).all():
                raise ValueError(
                    "Instance transform must contain 12 finite values (row-major 3x4)"
                )
            native_instances.append(
                self._backend.Instance(
                    transform.reshape(-1).tolist(),
                    instance.instance_id,
                    instance.sbt_offset,
                    instance.visibility_mask,
                    int(instance.flags),
                    geometry.handle,
                )
            )
            dependencies.append(geometry)
        return self._geometry(
            runtime.create_instance_acceleration_structure,
            native_instances,
            dependencies=dependencies,
            build_flags=build_flags,
            compact=compact,
        )

    def create_pipeline(
        self,
        ptx: bytes,
        raygen_entry: wp.Kernel | str,
        miss_entry: wp.Kernel | str,
        closest_hit_entry: wp.Kernel | str | None = None,
        *,
        num_payload_values: int,
        num_attribute_values: int,
        intersection_entry=None,
        any_hit_entry=None,
        primitive_type_flags=None,
        hit_groups: Sequence[HitKernel] | None = None,
        traversable_graph_flags=None,
        max_traversable_depth=None,
        uses_motion_blur=False,
        exception_entry=None,
        direct_callable_entries=None,
        continuation_callable_entries=None,
        exception_flags=None,
    ) -> Pipeline:
        from ._runtime.hit_kernels import HitKernel as NativeHitKernel

        native_groups = None
        if hit_groups is not None:
            native_groups = []
            for group in hit_groups:
                if not isinstance(group, HitKernel):
                    raise TypeError(
                        "hit_groups must contain warp_optix.HitKernel descriptors"
                    )
                native_groups.append(
                    NativeHitKernel(
                        group.closest_hit,
                        group.any_hit,
                        group.intersection,
                        builtin_intersection_type=None
                        if group.builtin_intersection_type is None
                        else self._backend.PrimitiveType(
                            int(group.builtin_intersection_type)
                        ),
                    )
                )
        with self._transaction():
            handle, sbt, resources = runtime.create_pipeline_and_sbt(
                self._backend,
                self._native_context,
                ptx,
                raygen_entry,
                miss_entry,
                closest_hit_entry,
                num_payload_values,
                num_attribute_values,
                self.device,
                intersection_entry=intersection_entry,
                any_hit_entry=any_hit_entry,
                primitive_type_flags=primitive_type_flags,
                hit_groups=native_groups,
                traversable_graph_flags=traversable_graph_flags,
                max_traversable_depth=max_traversable_depth,
                uses_motion_blur=uses_motion_blur,
                exception_entry=exception_entry,
                direct_callable_entries=direct_callable_entries,
                continuation_callable_entries=continuation_callable_entries,
                exception_flags=exception_flags,
            )
            result = Pipeline(self, handle, sbt, resources)
            self._resources.append(result)
            return result

    def close(self) -> None:
        if self._closed:
            return
        with wp.ScopedDevice(self.device):
            wp.synchronize_device(self.device)
            try:
                for resource in reversed(self._resources):
                    if not isinstance(resource, (Pipeline, AccelerationStructure)):
                        resource._release()
                # OptiX destroys its child API objects here. Keep GPU allocations
                # alive until its resource-use event bookkeeping is torn down.
                self._native_context.handle.destroy()
            finally:
                self._closed = True
                self._native_context.objects.clear()
                for resource in reversed(self._resources):
                    if isinstance(resource, (Pipeline, AccelerationStructure)):
                        resource._release()
                self._resources.clear()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class AccelerationStructure:
    """Geometry buffers retained by their Context; handle is a device traversable."""

    def __init__(self, context, resources, dependencies=()):
        self._context = context
        self._resources = resources
        self._dependencies = tuple(dependencies)

    @property
    def handle(self) -> int:
        self._context._check_open()
        return self._resources.handle

    @property
    def buffers(self) -> Mapping[str, object]:
        self._context._check_open()
        return MappingProxyType(self._resources)

    @property
    def device(self):
        return self._context.device

    def refit(self, stream: wp.Stream | None = None) -> int:
        stream = self._context._stream(stream)
        with wp.ScopedDevice(self.device):
            return runtime.refit_acceleration_structure(
                self._context._backend,
                self._context._native_context,
                self._resources,
                stream=stream.cuda_stream,
            )

    @property
    def build_flags(self) -> int:
        self._context._check_open()
        return self._resources.build_flags

    @property
    def output_size_in_bytes(self) -> int:
        self._context._check_open()
        return self._resources.output_size_in_bytes

    @property
    def uncompacted_size_in_bytes(self) -> int:
        self._context._check_open()
        return self._resources.uncompacted_size_in_bytes

    @property
    def update_temp_size_in_bytes(self) -> int:
        self._context._check_open()
        return self._resources.update_temp_size_in_bytes

    @property
    def compacted(self) -> bool:
        self._context._check_open()
        return self._resources.compacted

    def _release(self):
        self._resources = None
        self._dependencies = ()


class Pipeline:
    """Pipeline, SBT, and program modules retained by their Context."""

    def __init__(self, context, handle, sbt, resources):
        self._context = context
        self._handle = handle
        self._sbt = sbt
        self._resources = resources
        self._params = {}
        manager = resources["sbt_manager"]
        self.hit_group_offsets = tuple(
            manager.get_sbt_offset(h) for h in resources["hit_group_handles"]
        )
        self.direct_callable_indices = tuple(
            manager.get_callable_index(h) for h in resources["direct_callable_handles"]
        )
        self.continuation_callable_indices = tuple(
            manager.get_callable_index(h)
            for h in resources["continuation_callable_handles"]
        )

    @property
    def native_handle(self):
        """Borrowed native pipeline; the caller must not destroy it."""
        self._context._check_open()
        return self._handle

    def launch(
        self,
        params_buffer: runtime.LaunchParamsBuffer,
        *,
        width: int,
        height: int = 1,
        stream: wp.Stream | None = None,
    ) -> None:
        stream = self._context._stream(stream)
        if stream.is_capturing:
            raise RuntimeError(
                "OptiX launch capture is unsupported: graph replay can corrupt GPU state. "
                "Submit the OptiX launch outside capture, between ordinary Warp graphs."
            )
        if not isinstance(params_buffer, runtime.LaunchParamsBuffer):
            raise TypeError("params_buffer must be a LaunchParamsBuffer")
        if wp.get_device(params_buffer.device) != self._context.device:
            raise ValueError(
                "Launch parameters and pipeline must use the same CUDA device"
            )
        if width < 1 or height < 1:
            raise ValueError("Launch dimensions must be positive")
        self._params[id(params_buffer)] = params_buffer
        with wp.ScopedDevice(self._context.device):
            runtime.launch(
                self._context._backend,
                self._handle,
                self._sbt,
                width,
                height,
                params_buffer,
                stream.cuda_stream,
            )

    def _release(self):
        self._params.clear()
        self._resources = None
        self._sbt = None
        self._handle = None
