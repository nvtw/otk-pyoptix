# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

import warp_optix as wo
from warp_optix import native

pytestmark = pytest.mark.gpu

Vec16h = wp.types.vector(length=16, dtype=wp.float16)
N = K = 16


@wp.kernel(module="unique")
def coop_vec_raygen(output: wp.array(dtype=wp.float32)):
    a = wp.vec4h(wp.float16(1.0), wp.float16(2.0), wp.float16(3.0), wp.float16(4.0))
    b = wp.vec4h(wp.float16(2.0), wp.float16(3.0), wp.float16(4.0), wp.float16(5.0))
    summed = wp.vec4h()
    wp.optix_coop_vec_add(a, b, summed)
    for i in range(4):
        output[i] = wp.float32(summed[i])


@wp.kernel(module="unique")
def coop_vec_matmul_raygen(output: wp.array(dtype=wp.float32), matrix: wp.uint64):
    result = Vec16h()
    wp.optix_coop_vec_matmul_fp16(Vec16h(wp.float16(1.0)), matrix, wp.uint32(0), wp.uint32(0), result)
    for i in range(16):
        output[i] = wp.float32(result[i])


@wp.kernel(module="unique")
def coop_vec_fp8_matmul_raygen(output: wp.array(dtype=wp.float32), matrix: wp.uint64):
    result = Vec16h()
    wp.optix_coop_vec_matmul_fp8_e4m3(Vec16h(wp.float16(1.0)), matrix, wp.uint32(0), wp.uint32(0), result)
    for i in range(16):
        output[i] = wp.float32(result[i])


@pytest.fixture
def ctx():
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    context = wo.create_context("cuda")
    if not context.getProperty(native.DEVICE_PROPERTY_COOP_VEC):
        pytest.skip("GPU does not support OptiX cooperative vectors")
    return context


def _identity_matrix(ctx, element_type) -> wp.array:
    """Convert a row-major fp16 identity into the inference-optimal layout of ``element_type``."""

    def description(element_type, layout, size):
        desc = native.CoopVecMatrixDescription()
        desc.N = N
        desc.K = K
        desc.elementType = element_type
        desc.layout = layout
        desc.sizeInBytes = size
        return desc

    row_major = native.COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR
    optimal = native.COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL
    source_size = ctx.coopVecMatrixComputeSize(N, K, native.COOP_VEC_ELEM_TYPE_FLOAT16, row_major)
    packed_size = ctx.coopVecMatrixComputeSize(N, K, element_type, optimal)
    source = wp.array(np.eye(N, dtype=np.float16).reshape(-1), device="cuda")
    packed = wp.empty(packed_size, dtype=wp.uint8, device="cuda")
    ctx.coopVecMatrixConvert(
        wp.get_stream("cuda").cuda_stream,
        [description(native.COOP_VEC_ELEM_TYPE_FLOAT16, row_major, source_size)],
        source.ptr,
        [description(element_type, optimal, packed_size)],
        packed.ptr,
    )
    return packed


def test_coop_vec_elementwise_executes_in_optix(ctx):
    output = wp.zeros(4, dtype=wp.float32, device="cuda")
    wo.launch(wo.create_pipeline(ctx, coop_vec_raygen), dim=(1,), inputs=[output])
    np.testing.assert_array_equal(output.numpy(), [3.0, 5.0, 7.0, 9.0])


@pytest.mark.parametrize(
    ("raygen", "element_type"),
    [
        (coop_vec_matmul_raygen, native.COOP_VEC_ELEM_TYPE_FLOAT16),
        (coop_vec_fp8_matmul_raygen, native.COOP_VEC_ELEM_TYPE_FLOAT8_E4M3),
    ],
)
def test_coop_vec_matmul_executes_in_optix(ctx, raygen, element_type):
    matrix = _identity_matrix(ctx, element_type)
    output = wp.zeros(N, dtype=wp.float32, device="cuda")
    wo.launch(wo.create_pipeline(ctx, raygen), dim=(1,), inputs=[output, matrix.ptr])
    np.testing.assert_array_equal(output.numpy(), np.ones(N, dtype=np.float32))
