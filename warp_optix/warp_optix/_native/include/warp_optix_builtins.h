// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// warp_optix_builtins.h
//
// OptiX device-side wrappers for warp's generated kernels. This header is
// added to the compile include path by warp_optix._addon and is pulled into
// every CUDA translation unit via a registered preamble.
//
// Extracted from warp/native/builtin.h on migration from warp branch
// `dev/tw/add_minimal_optix_supprt`.

#pragma once

// This header is injected through `wp.ModuleBuildOptions.extra_cuda_preamble`
// before warp's own `cuda_module_header`, so we must declare `WP_NO_CRT`
// ourselves before pulling
// in `builtin.h` — otherwise `crt.h` tries to `#include <assert.h>` which is
// unavailable under NVRTC. `cuda_module_header`'s own `#define WP_NO_CRT` and
// `#include "builtin.h"` further down become no-ops thanks to `#pragma once`.
#ifndef WP_NO_CRT
#define WP_NO_CRT
#endif

#define WP_ENABLE_OPTIX 1
#include "optix_device.h"
#include "builtin.h"

namespace wp {

// Bit-cast between float and uint32 (reinterpret the bit pattern, no conversion).
inline CUDA_CALLABLE uint32 float_to_uint32(float x)
{
    uint32 u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

inline CUDA_CALLABLE float uint32_to_float(uint32 u)
{
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

template <typename Payload> union optix_payload_words {
    static_assert(__is_trivially_copyable(Payload), "OptiX payload must be trivially copyable");
    static_assert(
        (alignof(Payload) % alignof(uint32)) == 0, "OptiX payload alignment must be a multiple of 4 bytes"
    );
    static_assert((sizeof(Payload) % sizeof(uint32)) == 0, "OptiX payload size must be a multiple of 4 bytes");

    static constexpr size_t count = sizeof(Payload) / sizeof(uint32);
    static_assert(count > 0, "OptiX payload must contain at least one 32-bit word");
    static_assert(count <= 32, "OptiX payload exceeds OptiX payload register capacity");

    Payload payload;
    uint32 words[count];
};


#if defined(WP_ENABLE_OPTIX)
template <size_t... I> struct wp_index_sequence { };

template <size_t N, size_t... I>
struct wp_make_index_sequence_impl : wp_make_index_sequence_impl<N - 1, N - 1, I...> { };

template <size_t... I> struct wp_make_index_sequence_impl<0, I...> {
    using type = wp_index_sequence<I...>;
};

template <size_t N> using wp_make_index_sequence = typename wp_make_index_sequence_impl<N>::type;

template <size_t N, size_t... I>
inline CUDA_CALLABLE_DEVICE void optix_trace_words(
    OptixTraversableHandle handle,
    const vec3& ray_origin,
    const vec3& ray_direction,
    float tmin,
    float tmax,
    float ray_time,
    uint32 visibility_mask,
    uint32 ray_flags,
    uint32 sbt_offset,
    uint32 sbt_stride,
    uint32 miss_sbt_index,
    uint32 (&words)[N],
    wp_index_sequence<I...>
)
{
    optixTrace(
        handle, make_float3(ray_origin[0], ray_origin[1], ray_origin[2]),
        make_float3(ray_direction[0], ray_direction[1], ray_direction[2]), tmin, tmax, ray_time,
        static_cast<OptixVisibilityMask>(visibility_mask), ray_flags, sbt_offset, sbt_stride, miss_sbt_index,
        words[I]...
    );
}

template <size_t N, size_t... I>
inline CUDA_CALLABLE_DEVICE void optix_traverse_words(
    OptixTraversableHandle handle,
    const vec3& ray_origin,
    const vec3& ray_direction,
    float tmin,
    float tmax,
    float ray_time,
    uint32 visibility_mask,
    uint32 ray_flags,
    uint32 sbt_offset,
    uint32 sbt_stride,
    uint32 miss_sbt_index,
    uint32 (&words)[N],
    wp_index_sequence<I...>
)
{
    optixTraverse(
        handle, make_float3(ray_origin[0], ray_origin[1], ray_origin[2]),
        make_float3(ray_direction[0], ray_direction[1], ray_direction[2]), tmin, tmax, ray_time,
        static_cast<OptixVisibilityMask>(visibility_mask), ray_flags, sbt_offset, sbt_stride, miss_sbt_index,
        words[I]...
    );
}

template <size_t N, size_t... I>
inline CUDA_CALLABLE_DEVICE void optix_invoke_words(uint32 (&words)[N], wp_index_sequence<I...>)
{
    optixInvoke(words[I]...);
}
#endif

inline CUDA_CALLABLE_DEVICE vec3ui optix_get_launch_index()
{
#if defined(WP_ENABLE_OPTIX)
    const uint3 idx = optixGetLaunchIndex();
    return vec3ui(idx.x, idx.y, idx.z);
#else
    return vec3ui(0u, 0u, 0u);
#endif
}

inline CUDA_CALLABLE_DEVICE vec3ui optix_get_launch_dimensions()
{
#if defined(WP_ENABLE_OPTIX)
    const uint3 dim = optixGetLaunchDimensions();
    return vec3ui(dim.x, dim.y, dim.z);
#else
    return vec3ui(0u, 0u, 0u);
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_get_world_ray_origin()
{
#if defined(WP_ENABLE_OPTIX)
    const float3 o = optixGetWorldRayOrigin();
    return vec3(o.x, o.y, o.z);
#else
    return vec3(0.0f, 0.0f, 0.0f);
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_get_world_ray_direction()
{
#if defined(WP_ENABLE_OPTIX)
    const float3 d = optixGetWorldRayDirection();
    return vec3(d.x, d.y, d.z);
#else
    return vec3(0.0f, 0.0f, 0.0f);
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_get_object_ray_origin()
{
#if defined(WP_ENABLE_OPTIX)
    const float3 o = optixGetObjectRayOrigin();
    return vec3(o.x, o.y, o.z);
#else
    return vec3(0.0f, 0.0f, 0.0f);
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_get_object_ray_direction()
{
#if defined(WP_ENABLE_OPTIX)
    const float3 d = optixGetObjectRayDirection();
    return vec3(d.x, d.y, d.z);
#else
    return vec3(0.0f, 0.0f, 0.0f);
#endif
}

inline CUDA_CALLABLE_DEVICE float optix_get_ray_tmin()
{
#if defined(WP_ENABLE_OPTIX)
    return optixGetRayTmin();
#else
    return 0.0f;
#endif
}

inline CUDA_CALLABLE_DEVICE float optix_get_ray_time()
{
#if defined(WP_ENABLE_OPTIX)
    return optixGetRayTime();
#else
    return 0.0f;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_ray_flags()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(optixGetRayFlags());
#else
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_ray_visibility_mask()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(optixGetRayVisibilityMask());
#else
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE float optix_get_ray_tmax()
{
#if defined(WP_ENABLE_OPTIX)
    return optixGetRayTmax();
#else
    return 0.0f;
#endif
}

inline CUDA_CALLABLE_DEVICE vec2 optix_get_triangle_barycentrics()
{
#if defined(WP_ENABLE_OPTIX)
    const float2 bary = optixGetTriangleBarycentrics();
    return vec2(bary.x, bary.y);
#else
    return vec2(0.0f, 0.0f);
#endif
}

inline CUDA_CALLABLE_DEVICE mat33 optix_get_triangle_vertex_data()
{
#if defined(WP_ENABLE_OPTIX)
    float3 vertices[3];
    optixGetTriangleVertexData(vertices);
    return mat33(
        vertices[0].x,
        vertices[0].y,
        vertices[0].z,
        vertices[1].x,
        vertices[1].y,
        vertices[1].z,
        vertices[2].x,
        vertices[2].y,
        vertices[2].z
    );
#else
    return mat33(0.0f);
#endif
}

inline CUDA_CALLABLE_DEVICE bool optix_is_triangle_hit()
{
#if defined(WP_ENABLE_OPTIX)
    return optixIsTriangleHit();
#else
    return false;
#endif
}

inline CUDA_CALLABLE_DEVICE bool optix_is_triangle_front_face_hit()
{
#if defined(WP_ENABLE_OPTIX)
    return optixIsTriangleFrontFaceHit();
#else
    return false;
#endif
}

inline CUDA_CALLABLE_DEVICE bool optix_is_triangle_back_face_hit()
{
#if defined(WP_ENABLE_OPTIX)
    return optixIsTriangleBackFaceHit();
#else
    return false;
#endif
}

template <unsigned Count>
inline CUDA_CALLABLE_DEVICE mat_t<Count, 4, float> optix_curve_vertices_to_matrix(const float4 (&vertices)[Count])
{
    mat_t<Count, 4, float> result;
    for (unsigned i = 0; i < Count; ++i) {
        result.data[i][0] = vertices[i].x;
        result.data[i][1] = vertices[i].y;
        result.data[i][2] = vertices[i].z;
        result.data[i][3] = vertices[i].w;
    }
    return result;
}

#define WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(NAME, OPTIX_NAME, COUNT) \
    inline CUDA_CALLABLE_DEVICE mat_t<COUNT, 4, float> NAME()       \
    {                                                               \
        float4 vertices[COUNT] = {};                                \
        OPTIX_NAME(vertices);                                        \
        return optix_curve_vertices_to_matrix(vertices);             \
    }

#if defined(WP_ENABLE_OPTIX)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_linear_curve_vertex_data, optixGetLinearCurveVertexData, 2)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_quadratic_bspline_vertex_data, optixGetQuadraticBSplineVertexData, 3)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_cubic_bspline_vertex_data, optixGetCubicBSplineVertexData, 4)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_catmull_rom_vertex_data, optixGetCatmullRomVertexData, 4)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_cubic_bezier_vertex_data, optixGetCubicBezierVertexData, 4)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_ribbon_vertex_data, optixGetRibbonVertexData, 3)
#else
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_linear_curve_vertex_data, (void), 2)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_quadratic_bspline_vertex_data, (void), 3)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_cubic_bspline_vertex_data, (void), 4)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_catmull_rom_vertex_data, (void), 4)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_cubic_bezier_vertex_data, (void), 4)
WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY(optix_get_ribbon_vertex_data, (void), 3)
#endif

#undef WP_DEFINE_OPTIX_CURVE_VERTEX_QUERY

inline CUDA_CALLABLE_DEVICE vec2 optix_get_ribbon_parameters()
{
#if defined(WP_ENABLE_OPTIX)
    const float2 parameters = optixGetRibbonParameters();
    return vec2(parameters.x, parameters.y);
#else
    return vec2(0.0f);
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_get_ribbon_normal(const vec2& parameters)
{
#if defined(WP_ENABLE_OPTIX)
    const float3 normal = optixGetRibbonNormal(make_float2(parameters[0], parameters[1]));
    return vec3(normal.x, normal.y, normal.z);
#else
    (void)parameters;
    return vec3(0.0f);
#endif
}

inline CUDA_CALLABLE_DEVICE float optix_get_curve_parameter()
{
#if defined(WP_ENABLE_OPTIX)
    return optixGetCurveParameter();
#else
    return 0.0f;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_primitive_index()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(optixGetPrimitiveIndex());
#else
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_instance_id()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(optixGetInstanceId());
#else
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_instance_index()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(optixGetInstanceIndex());
#else
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_sbt_gas_index()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(optixGetSbtGASIndex());
#else
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_primitive_type()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(optixGetPrimitiveType());
#else
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE uint64 optix_get_gas_traversable_handle()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint64>(optixGetGASTraversableHandle());
#else
    return 0ull;
#endif
}

inline CUDA_CALLABLE_DEVICE bool optix_is_front_face_hit()
{
#if defined(WP_ENABLE_OPTIX)
    return optixIsFrontFaceHit();
#else
    return false;
#endif
}

inline CUDA_CALLABLE_DEVICE bool optix_is_back_face_hit()
{
#if defined(WP_ENABLE_OPTIX)
    return optixIsBackFaceHit();
#else
    return false;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_hit_kind()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(optixGetHitKind());
#else
    return 0u;
#endif
}

#define WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR(I)                  \
    inline CUDA_CALLABLE_DEVICE uint32 optix_get_attribute_##I() \
    {                                                          \
        return static_cast<uint32>(optixGetAttribute_##I());   \
    }

WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR(0)
WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR(1)
WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR(2)
WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR(3)
WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR(4)
WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR(5)
WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR(6)
WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR(7)

#undef WP_DEFINE_OPTIX_ATTRIBUTE_ACCESSOR

inline CUDA_CALLABLE_DEVICE vec3 optix_transform_normal_from_object_to_world_space(const vec3& normal)
{
#if defined(WP_ENABLE_OPTIX)
    const float3 n = optixTransformNormalFromObjectToWorldSpace(make_float3(normal[0], normal[1], normal[2]));
    return vec3(n.x, n.y, n.z);
#else
    return normal;
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_transform_point_from_object_to_world_space(const vec3& point)
{
#if defined(WP_ENABLE_OPTIX)
    const float3 p = optixTransformPointFromObjectToWorldSpace(make_float3(point[0], point[1], point[2]));
    return vec3(p.x, p.y, p.z);
#else
    return point;
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_transform_vector_from_object_to_world_space(const vec3& vector)
{
#if defined(WP_ENABLE_OPTIX)
    const float3 v = optixTransformVectorFromObjectToWorldSpace(make_float3(vector[0], vector[1], vector[2]));
    return vec3(v.x, v.y, v.z);
#else
    return vector;
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_transform_point_from_world_to_object_space(const vec3& point)
{
#if defined(WP_ENABLE_OPTIX)
    const float3 p = optixTransformPointFromWorldToObjectSpace(make_float3(point[0], point[1], point[2]));
    return vec3(p.x, p.y, p.z);
#else
    return point;
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_transform_vector_from_world_to_object_space(const vec3& vector)
{
#if defined(WP_ENABLE_OPTIX)
    const float3 v = optixTransformVectorFromWorldToObjectSpace(make_float3(vector[0], vector[1], vector[2]));
    return vec3(v.x, v.y, v.z);
#else
    return vector;
#endif
}

inline CUDA_CALLABLE_DEVICE vec3 optix_transform_normal_from_world_to_object_space(const vec3& normal)
{
#if defined(WP_ENABLE_OPTIX)
    const float3 n = optixTransformNormalFromWorldToObjectSpace(make_float3(normal[0], normal[1], normal[2]));
    return vec3(n.x, n.y, n.z);
#else
    return normal;
#endif
}

inline CUDA_CALLABLE_DEVICE mat44 optix_get_object_to_world_transform_matrix()
{
    float matrix[12] = {};
#if defined(WP_ENABLE_OPTIX)
    optixGetObjectToWorldTransformMatrix(matrix);
#else
    matrix[0] = matrix[5] = matrix[10] = 1.0f;
#endif
    return mat44(matrix[0], matrix[1], matrix[2], matrix[3],
                 matrix[4], matrix[5], matrix[6], matrix[7],
                 matrix[8], matrix[9], matrix[10], matrix[11],
                 0.0f, 0.0f, 0.0f, 1.0f);
}

inline CUDA_CALLABLE_DEVICE mat44 optix_get_world_to_object_transform_matrix()
{
    float matrix[12] = {};
#if defined(WP_ENABLE_OPTIX)
    optixGetWorldToObjectTransformMatrix(matrix);
#else
    matrix[0] = matrix[5] = matrix[10] = 1.0f;
#endif
    return mat44(matrix[0], matrix[1], matrix[2], matrix[3],
                 matrix[4], matrix[5], matrix[6], matrix[7],
                 matrix[8], matrix[9], matrix[10], matrix[11],
                 0.0f, 0.0f, 0.0f, 1.0f);
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_transform_list_size()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(optixGetTransformListSize());
#else
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE uint64 optix_get_transform_list_handle(uint32 index)
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint64>(optixGetTransformListHandle(index));
#else
    (void)index;
    return 0ull;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_transform_type_from_handle(uint64 handle)
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<uint32>(
        optixGetTransformTypeFromHandle(static_cast<OptixTraversableHandle>(handle)));
#else
    (void)handle;
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE void optix_terminate_ray()
{
#if defined(WP_ENABLE_OPTIX)
    optixTerminateRay();
#endif
}

inline CUDA_CALLABLE_DEVICE void optix_ignore_intersection()
{
#if defined(WP_ENABLE_OPTIX)
    optixIgnoreIntersection();
#endif
}

inline CUDA_CALLABLE_DEVICE void optix_direct_call(uint32 sbt_index)
{
#if defined(WP_ENABLE_OPTIX)
    optixDirectCall<void>(sbt_index);
#else
    (void)sbt_index;
#endif
}

inline CUDA_CALLABLE_DEVICE void optix_continuation_call(uint32 sbt_index)
{
#if defined(WP_ENABLE_OPTIX)
    optixContinuationCall<void>(sbt_index);
#else
    (void)sbt_index;
#endif
}

inline CUDA_CALLABLE_DEVICE int32 optix_get_exception_code()
{
#if defined(WP_ENABLE_OPTIX)
    return static_cast<int32>(optixGetExceptionCode());
#else
    return 0;
#endif
}

#define WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR(I)                  \
    inline CUDA_CALLABLE_DEVICE uint32 optix_get_exception_detail_##I() \
    {                                                                 \
        return static_cast<uint32>(optixGetExceptionDetail_##I());    \
    }

WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR(0)
WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR(1)
WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR(2)
WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR(3)
WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR(4)
WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR(5)
WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR(6)
WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR(7)

#undef WP_DEFINE_OPTIX_EXCEPTION_DETAIL_ACCESSOR

template <typename... Details>
inline CUDA_CALLABLE_DEVICE void optix_throw_exception(int32 exception_code, Details... details)
{
    static_assert(sizeof...(Details) <= 8, "optixThrowException accepts at most eight details");
#if defined(WP_ENABLE_OPTIX)
    optixThrowException(exception_code, static_cast<uint32>(details)...);
#else
    (void)exception_code;
#endif
}

template <typename... Attributes>
inline CUDA_CALLABLE_DEVICE bool optix_report_intersection(float hit_t, uint32 hit_kind, Attributes... attributes)
{
    static_assert(sizeof...(Attributes) <= 8, "optixReportIntersection accepts at most eight attributes");
#if defined(WP_ENABLE_OPTIX)
    return optixReportIntersection(hit_t, hit_kind, static_cast<uint32>(attributes)...);
#else
    (void)hit_t;
    (void)hit_kind;
    return false;
#endif
}

template <typename Payload>
inline CUDA_CALLABLE_DEVICE void optix_trace(
    uint64 traversable,
    const vec3& ray_origin,
    const vec3& ray_direction,
    float tmin,
    float tmax,
    float ray_time,
    uint32 visibility_mask,
    uint32 ray_flags,
    uint32 sbt_offset,
    uint32 sbt_stride,
    uint32 miss_sbt_index,
    Payload& payload
)
{
#if defined(WP_ENABLE_OPTIX)
    using PayloadWords = optix_payload_words<Payload>;
    constexpr size_t kPayloadWords = PayloadWords::count;
    PayloadWords packed = {};
    packed.payload = payload;

    optix_trace_words(
        static_cast<OptixTraversableHandle>(traversable), ray_origin, ray_direction, tmin, tmax, ray_time,
        visibility_mask, ray_flags, sbt_offset, sbt_stride, miss_sbt_index, packed.words,
        wp_make_index_sequence<kPayloadWords> {}
    );

    payload = packed.payload;
#else
    (void)traversable;
    (void)ray_origin;
    (void)ray_direction;
    (void)tmin;
    (void)tmax;
    (void)ray_time;
    (void)visibility_mask;
    (void)ray_flags;
    (void)sbt_offset;
    (void)sbt_stride;
    (void)miss_sbt_index;
    (void)payload;
#endif
}

template <typename Payload>
inline CUDA_CALLABLE_DEVICE void optix_traverse(
    uint64 traversable,
    const vec3& ray_origin,
    const vec3& ray_direction,
    float tmin,
    float tmax,
    float ray_time,
    uint32 visibility_mask,
    uint32 ray_flags,
    uint32 sbt_offset,
    uint32 sbt_stride,
    uint32 miss_sbt_index,
    Payload& payload
)
{
#if defined(WP_ENABLE_OPTIX)
    using PayloadWords = optix_payload_words<Payload>;
    constexpr size_t kPayloadWords = PayloadWords::count;
    PayloadWords packed = {};
    packed.payload = payload;
    optix_traverse_words(
        static_cast<OptixTraversableHandle>(traversable), ray_origin, ray_direction, tmin, tmax, ray_time,
        visibility_mask, ray_flags, sbt_offset, sbt_stride, miss_sbt_index, packed.words,
        wp_make_index_sequence<kPayloadWords> {}
    );
    payload = packed.payload;
#else
    (void)traversable;
    (void)ray_origin;
    (void)ray_direction;
    (void)tmin;
    (void)tmax;
    (void)ray_time;
    (void)visibility_mask;
    (void)ray_flags;
    (void)sbt_offset;
    (void)sbt_stride;
    (void)miss_sbt_index;
    (void)payload;
#endif
}

inline CUDA_CALLABLE_DEVICE void optix_reorder(uint32 coherence_hint, uint32 num_coherence_hint_bits_from_lsb)
{
#if defined(WP_ENABLE_OPTIX)
    optixReorder(coherence_hint, num_coherence_hint_bits_from_lsb);
#else
    (void)coherence_hint;
    (void)num_coherence_hint_bits_from_lsb;
#endif
}

inline CUDA_CALLABLE_DEVICE void optix_reorder()
{
#if defined(WP_ENABLE_OPTIX)
    optixReorder();
#endif
}

inline CUDA_CALLABLE_DEVICE bool optix_hit_object_is_hit()
{
#if defined(WP_ENABLE_OPTIX)
    return optixHitObjectIsHit();
#else
    return false;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_hit_object_get_primitive_index()
{
#if defined(WP_ENABLE_OPTIX)
    return optixHitObjectGetPrimitiveIndex();
#else
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_hit_object_get_instance_id()
{
#if defined(WP_ENABLE_OPTIX)
    return optixHitObjectGetInstanceId();
#else
    return 0u;
#endif
}

template <typename Payload>
inline CUDA_CALLABLE_DEVICE void optix_invoke(Payload& payload)
{
#if defined(WP_ENABLE_OPTIX)
    using PayloadWords = optix_payload_words<Payload>;
    constexpr size_t kPayloadWords = PayloadWords::count;
    PayloadWords packed = {};
    packed.payload = payload;
    optix_invoke_words(packed.words, wp_make_index_sequence<kPayloadWords> {});
    payload = packed.payload;
#else
    (void)payload;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_payload_index(uint32 index);
inline CUDA_CALLABLE_DEVICE void optix_set_payload_index(uint32 index, uint32 value);

template <typename Payload> inline CUDA_CALLABLE_DEVICE Payload optix_load_payload()
{
    using PayloadWords = optix_payload_words<Payload>;
    constexpr size_t kPayloadWords = PayloadWords::count;
    PayloadWords packed = {};
#if defined(WP_ENABLE_OPTIX)
#pragma unroll
    for (uint32 i = 0; i < uint32(kPayloadWords); ++i) {
        packed.words[i] = optix_get_payload_index(i);
    }
#endif
    return packed.payload;
}

template <typename Payload> inline CUDA_CALLABLE_DEVICE void optix_load_payload(Payload& payload)
{
    payload = optix_load_payload<Payload>();
}

template <typename Payload> inline CUDA_CALLABLE_DEVICE void optix_store_payload(const Payload& payload)
{
    using PayloadWords = optix_payload_words<Payload>;
    constexpr size_t kPayloadWords = PayloadWords::count;
    PayloadWords packed = {};
    packed.payload = payload;
#if defined(WP_ENABLE_OPTIX)
#pragma unroll
    for (uint32 i = 0; i < uint32(kPayloadWords); ++i) {
        optix_set_payload_index(i, packed.words[i]);
    }
#else
    (void)payload;
#endif
}

inline CUDA_CALLABLE_DEVICE uint32 optix_get_payload_index(uint32 index)
{
#if defined(WP_ENABLE_OPTIX)
    switch (index) {
    case 0:
        return optixGetPayload_0();
    case 1:
        return optixGetPayload_1();
    case 2:
        return optixGetPayload_2();
    case 3:
        return optixGetPayload_3();
    case 4:
        return optixGetPayload_4();
    case 5:
        return optixGetPayload_5();
    case 6:
        return optixGetPayload_6();
    case 7:
        return optixGetPayload_7();
    case 8:
        return optixGetPayload_8();
    case 9:
        return optixGetPayload_9();
    case 10:
        return optixGetPayload_10();
    case 11:
        return optixGetPayload_11();
    case 12:
        return optixGetPayload_12();
    case 13:
        return optixGetPayload_13();
    case 14:
        return optixGetPayload_14();
    case 15:
        return optixGetPayload_15();
    case 16:
        return optixGetPayload_16();
    case 17:
        return optixGetPayload_17();
    case 18:
        return optixGetPayload_18();
    case 19:
        return optixGetPayload_19();
    case 20:
        return optixGetPayload_20();
    case 21:
        return optixGetPayload_21();
    case 22:
        return optixGetPayload_22();
    case 23:
        return optixGetPayload_23();
    case 24:
        return optixGetPayload_24();
    case 25:
        return optixGetPayload_25();
    case 26:
        return optixGetPayload_26();
    case 27:
        return optixGetPayload_27();
    case 28:
        return optixGetPayload_28();
    case 29:
        return optixGetPayload_29();
    case 30:
        return optixGetPayload_30();
    case 31:
        return optixGetPayload_31();
    default:
        return 0u;
    }
#else
    (void)index;
    return 0u;
#endif
}

inline CUDA_CALLABLE_DEVICE void optix_set_payload_index(uint32 index, uint32 value)
{
#if defined(WP_ENABLE_OPTIX)
    switch (index) {
    case 0:
        optixSetPayload_0(value);
        return;
    case 1:
        optixSetPayload_1(value);
        return;
    case 2:
        optixSetPayload_2(value);
        return;
    case 3:
        optixSetPayload_3(value);
        return;
    case 4:
        optixSetPayload_4(value);
        return;
    case 5:
        optixSetPayload_5(value);
        return;
    case 6:
        optixSetPayload_6(value);
        return;
    case 7:
        optixSetPayload_7(value);
        return;
    case 8:
        optixSetPayload_8(value);
        return;
    case 9:
        optixSetPayload_9(value);
        return;
    case 10:
        optixSetPayload_10(value);
        return;
    case 11:
        optixSetPayload_11(value);
        return;
    case 12:
        optixSetPayload_12(value);
        return;
    case 13:
        optixSetPayload_13(value);
        return;
    case 14:
        optixSetPayload_14(value);
        return;
    case 15:
        optixSetPayload_15(value);
        return;
    case 16:
        optixSetPayload_16(value);
        return;
    case 17:
        optixSetPayload_17(value);
        return;
    case 18:
        optixSetPayload_18(value);
        return;
    case 19:
        optixSetPayload_19(value);
        return;
    case 20:
        optixSetPayload_20(value);
        return;
    case 21:
        optixSetPayload_21(value);
        return;
    case 22:
        optixSetPayload_22(value);
        return;
    case 23:
        optixSetPayload_23(value);
        return;
    case 24:
        optixSetPayload_24(value);
        return;
    case 25:
        optixSetPayload_25(value);
        return;
    case 26:
        optixSetPayload_26(value);
        return;
    case 27:
        optixSetPayload_27(value);
        return;
    case 28:
        optixSetPayload_28(value);
        return;
    case 29:
        optixSetPayload_29(value);
        return;
    case 30:
        optixSetPayload_30(value);
        return;
    case 31:
        optixSetPayload_31(value);
        return;
    default:
        return;
    }
#else
    (void)index;
    (void)value;
#endif
}

#define WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(I)                    \
    inline CUDA_CALLABLE_DEVICE uint32 optix_get_payload_##I() \
    {                                                          \
        return optix_get_payload_index(I);                     \
    }                                                          \
    inline CUDA_CALLABLE_DEVICE void optix_set_payload_##I(uint32 value) \
    {                                                          \
        optix_set_payload_index(I, value);                     \
    }

WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(0)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(1)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(2)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(3)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(4)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(5)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(6)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(7)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(8)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(9)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(10)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(11)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(12)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(13)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(14)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(15)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(16)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(17)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(18)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(19)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(20)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(21)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(22)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(23)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(24)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(25)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(26)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(27)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(28)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(29)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(30)
WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR(31)

#undef WP_DEFINE_OPTIX_PAYLOAD_ACCESSOR

// OptiX cooperative vectors use CUDA's global half type, while Warp uses its
// own ABI-compatible wp::half. This adapter keeps Warp kernels on wp.vector.
template <typename A, typename B> struct optix_coop_is_same { static constexpr bool value = false; };
template <typename A> struct optix_coop_is_same<A, A> { static constexpr bool value = true; };
template <typename T> struct optix_coop_elem_type;
template <> struct optix_coop_elem_type<half> { static constexpr OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16; };
template <> struct optix_coop_elem_type<float> { static constexpr OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32; };
template <> struct optix_coop_elem_type<int8> { static constexpr OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_INT8; };
template <> struct optix_coop_elem_type<uint8> { static constexpr OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_UINT8; };
template <> struct optix_coop_elem_type<int32> { static constexpr OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_INT32; };
template <> struct optix_coop_elem_type<uint32> { static constexpr OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_UINT32; };
template <typename T> struct optix_coop_is_supported_scalar { static constexpr bool value = false; };
template <> struct optix_coop_is_supported_scalar<half> { static constexpr bool value = true; };
template <> struct optix_coop_is_supported_scalar<float> { static constexpr bool value = true; };
template <> struct optix_coop_is_supported_scalar<int8> { static constexpr bool value = true; };
template <> struct optix_coop_is_supported_scalar<uint8> { static constexpr bool value = true; };
template <> struct optix_coop_is_supported_scalar<int32> { static constexpr bool value = true; };
template <> struct optix_coop_is_supported_scalar<uint32> { static constexpr bool value = true; };

template <typename T> struct optix_coop_scalar {
    using type = T;
    inline CUDA_CALLABLE_DEVICE static type to_optix(T value) { return value; }
    inline CUDA_CALLABLE_DEVICE static T from_optix(type value) { return value; }
};
template <> struct optix_coop_scalar<half> {
    using type = ::half;
    inline CUDA_CALLABLE_DEVICE static type to_optix(half value) { return __ushort_as_half(value.u); }
    inline CUDA_CALLABLE_DEVICE static half from_optix(type value)
    {
        half result;
        result.u = __half_as_ushort(value);
        return result;
    }
};

template <typename Vec> struct optix_coop_vec_traits;
template <unsigned Length, typename T> struct optix_coop_vec_traits<vec_t<Length, T>> {
    static_assert(Length > 0, "OptiX cooperative vectors must have at least one component");
    static_assert(optix_coop_is_supported_scalar<T>::value, "unsupported OptiX cooperative-vector scalar type");
    using scalar_type = T;
    using optix_scalar_type = typename optix_coop_scalar<T>::type;
    using optix_type = OptixCoopVec<optix_scalar_type, Length>;
    static constexpr unsigned size = Length;
};

template <typename Vec>
inline CUDA_CALLABLE_DEVICE typename optix_coop_vec_traits<Vec>::optix_type optix_coop_to_native(const Vec& value)
{
    using Traits = optix_coop_vec_traits<Vec>;
    typename Traits::optix_type result;
#pragma unroll
    for (unsigned i = 0; i < Traits::size; ++i)
        result[i] = optix_coop_scalar<typename Traits::scalar_type>::to_optix(value[i]);
    return result;
}

template <typename Vec>
inline CUDA_CALLABLE_DEVICE Vec optix_coop_from_native(const typename optix_coop_vec_traits<Vec>::optix_type& value)
{
    using Traits = optix_coop_vec_traits<Vec>;
    Vec result;
#pragma unroll
    for (unsigned i = 0; i < Traits::size; ++i)
        result[i] = optix_coop_scalar<typename Traits::scalar_type>::from_optix(value[i]);
    return result;
}

#if !defined(WP_OPTIX_PROGRAM)
template <typename T> struct optix_coop_requires_optix_program {
    static_assert(sizeof(T) == 0, "OptiX cooperative-vector builtins require an @optix_kernel program");
};
#define WP_OPTIX_COOP_REQUIRE_PROGRAM(T) (void)sizeof(optix_coop_requires_optix_program<T>)
#else
#define WP_OPTIX_COOP_REQUIRE_PROGRAM(T) ((void)0)
#endif

template <typename Vec> inline CUDA_CALLABLE_DEVICE void optix_coop_vec_load(uint64 address, Vec& output)
{
    WP_OPTIX_COOP_REQUIRE_PROGRAM(Vec);
    using Native = typename optix_coop_vec_traits<Vec>::optix_type;
    output = optix_coop_from_native<Vec>(optixCoopVecLoad<Native>(static_cast<CUdeviceptr>(address)));
}

#define WP_DEFINE_OPTIX_COOP_UNARY(NAME, OPTIX_NAME) \
    template <typename Vec> inline CUDA_CALLABLE_DEVICE void NAME(const Vec& input, Vec& output) \
    { WP_OPTIX_COOP_REQUIRE_PROGRAM(Vec); output = optix_coop_from_native<Vec>(OPTIX_NAME(optix_coop_to_native(input))); }
WP_DEFINE_OPTIX_COOP_UNARY(optix_coop_vec_exp2, optixCoopVecExp2)
WP_DEFINE_OPTIX_COOP_UNARY(optix_coop_vec_log2, optixCoopVecLog2)
WP_DEFINE_OPTIX_COOP_UNARY(optix_coop_vec_tanh, optixCoopVecTanh)
#undef WP_DEFINE_OPTIX_COOP_UNARY

template <typename Vec> inline CUDA_CALLABLE_DEVICE void optix_coop_vec_min_scalar(
    const Vec& a, typename optix_coop_vec_traits<Vec>::scalar_type b, Vec& output)
{
    WP_OPTIX_COOP_REQUIRE_PROGRAM(Vec);
    output = optix_coop_from_native<Vec>(optixCoopVecMin(
        optix_coop_to_native(a), optix_coop_scalar<typename optix_coop_vec_traits<Vec>::scalar_type>::to_optix(b)));
}

template <typename Vec> inline CUDA_CALLABLE_DEVICE void optix_coop_vec_max_scalar(
    const Vec& a, typename optix_coop_vec_traits<Vec>::scalar_type b, Vec& output)
{
    WP_OPTIX_COOP_REQUIRE_PROGRAM(Vec);
    output = optix_coop_from_native<Vec>(optixCoopVecMax(
        optix_coop_to_native(a), optix_coop_scalar<typename optix_coop_vec_traits<Vec>::scalar_type>::to_optix(b)));
}

template <typename VecIn, typename VecOut>
inline CUDA_CALLABLE_DEVICE void optix_coop_vec_cvt(const VecIn& input, VecOut& output)
{
    WP_OPTIX_COOP_REQUIRE_PROGRAM(VecIn);
    static_assert(optix_coop_vec_traits<VecIn>::size == optix_coop_vec_traits<VecOut>::size,
        "optix_coop_vec_cvt input and output must have equal vector lengths");
    using NativeOut = typename optix_coop_vec_traits<VecOut>::optix_type;
    output = optix_coop_from_native<VecOut>(optixCoopVecCvt<NativeOut>(optix_coop_to_native(input)));
}

#define WP_DEFINE_OPTIX_COOP_BINARY(NAME, OPTIX_NAME) \
    template <typename Vec> inline CUDA_CALLABLE_DEVICE void NAME(const Vec& a, const Vec& b, Vec& output) \
    { WP_OPTIX_COOP_REQUIRE_PROGRAM(Vec); output = optix_coop_from_native<Vec>(OPTIX_NAME(optix_coop_to_native(a), optix_coop_to_native(b))); }
WP_DEFINE_OPTIX_COOP_BINARY(optix_coop_vec_min, optixCoopVecMin)
WP_DEFINE_OPTIX_COOP_BINARY(optix_coop_vec_max, optixCoopVecMax)
WP_DEFINE_OPTIX_COOP_BINARY(optix_coop_vec_mul, optixCoopVecMul)
WP_DEFINE_OPTIX_COOP_BINARY(optix_coop_vec_add, optixCoopVecAdd)
WP_DEFINE_OPTIX_COOP_BINARY(optix_coop_vec_sub, optixCoopVecSub)
WP_DEFINE_OPTIX_COOP_BINARY(optix_coop_vec_step, optixCoopVecStep)
#undef WP_DEFINE_OPTIX_COOP_BINARY

template <typename Vec>
inline CUDA_CALLABLE_DEVICE void optix_coop_vec_ffma(const Vec& a, const Vec& b, const Vec& c, Vec& output)
{
    WP_OPTIX_COOP_REQUIRE_PROGRAM(Vec);
    output = optix_coop_from_native<Vec>(optixCoopVecFFMA(
        optix_coop_to_native(a), optix_coop_to_native(b), optix_coop_to_native(c)));
}

template <OptixCoopVecElemType MatrixType, OptixCoopVecElemType InputType, typename VecIn, typename VecOut>
inline CUDA_CALLABLE_DEVICE void optix_coop_vec_matmul_impl(const VecIn& input, uint64 matrix,
    uint32 matrix_offset, uint32 stride, VecOut& output)
{
    WP_OPTIX_COOP_REQUIRE_PROGRAM(VecIn);
    using IT = optix_coop_vec_traits<VecIn>;
    using OT = optix_coop_vec_traits<VecOut>;
    using NI = typename IT::optix_type;
    using NO = typename OT::optix_type;
    output = optix_coop_from_native<VecOut>(optixCoopVecMatMul<NO, NI, InputType,
        OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, false, OT::size, IT::size, MatrixType>(
            optix_coop_to_native(input), static_cast<CUdeviceptr>(matrix), matrix_offset, stride));
}

template <OptixCoopVecElemType MatrixType, OptixCoopVecElemType InputType, typename VecIn, typename VecOut>
inline CUDA_CALLABLE_DEVICE void optix_coop_vec_matmul_bias_impl(const VecIn& input, uint64 matrix,
    uint32 matrix_offset, uint64 bias, uint32 bias_offset, uint32 stride, VecOut& output)
{
    WP_OPTIX_COOP_REQUIRE_PROGRAM(VecIn);
    using IT = optix_coop_vec_traits<VecIn>;
    using OT = optix_coop_vec_traits<VecOut>;
    using NI = typename IT::optix_type;
    using NO = typename OT::optix_type;
    constexpr OptixCoopVecElemType BiasType = optix_coop_elem_type<typename OT::scalar_type>::value;
    output = optix_coop_from_native<VecOut>(optixCoopVecMatMul<NO, NI, InputType,
        OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, false, OT::size, IT::size, MatrixType, BiasType>(
            optix_coop_to_native(input), static_cast<CUdeviceptr>(matrix), matrix_offset,
            static_cast<CUdeviceptr>(bias), bias_offset, stride));
}

#define WP_DEFINE_OPTIX_COOP_MATMUL(SUFFIX, TYPE) \
    template <typename VI, typename VO> inline CUDA_CALLABLE_DEVICE void optix_coop_vec_matmul_##SUFFIX( \
        const VI& input, uint64 matrix, uint32 offset, uint32 stride, VO& output) \
    { optix_coop_vec_matmul_impl<TYPE, TYPE>(input, matrix, offset, stride, output); } \
    template <typename VI, typename VO> inline CUDA_CALLABLE_DEVICE void optix_coop_vec_matmul_bias_##SUFFIX( \
        const VI& input, uint64 matrix, uint32 offset, uint64 bias, uint32 bias_offset, uint32 stride, VO& output) \
    { optix_coop_vec_matmul_bias_impl<TYPE, TYPE>(input, matrix, offset, bias, bias_offset, stride, output); }
WP_DEFINE_OPTIX_COOP_MATMUL(fp16, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16)
WP_DEFINE_OPTIX_COOP_MATMUL(fp8_e4m3, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3)
WP_DEFINE_OPTIX_COOP_MATMUL(fp8_e5m2, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E5M2)
WP_DEFINE_OPTIX_COOP_MATMUL(int8, OPTIX_COOP_VEC_ELEM_TYPE_INT8)
WP_DEFINE_OPTIX_COOP_MATMUL(uint8, OPTIX_COOP_VEC_ELEM_TYPE_UINT8)
#undef WP_DEFINE_OPTIX_COOP_MATMUL

#define WP_DEFINE_OPTIX_COOP_MATRIX_SIZE(SUFFIX, TYPE) \
    template <typename VI, typename VO> inline CUDA_CALLABLE_DEVICE uint32 optix_coop_vec_matrix_size_##SUFFIX( \
        const VI&, const VO&) \
    { return optixCoopVecGetMatrixSize<optix_coop_vec_traits<VO>::size, optix_coop_vec_traits<VI>::size, \
        TYPE, OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL, 0>(); }
WP_DEFINE_OPTIX_COOP_MATRIX_SIZE(fp16, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16)
WP_DEFINE_OPTIX_COOP_MATRIX_SIZE(fp8_e4m3, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3)
WP_DEFINE_OPTIX_COOP_MATRIX_SIZE(fp8_e5m2, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E5M2)
WP_DEFINE_OPTIX_COOP_MATRIX_SIZE(int8, OPTIX_COOP_VEC_ELEM_TYPE_INT8)
WP_DEFINE_OPTIX_COOP_MATRIX_SIZE(uint8, OPTIX_COOP_VEC_ELEM_TYPE_UINT8)
#undef WP_DEFINE_OPTIX_COOP_MATRIX_SIZE

template <typename Vec> inline CUDA_CALLABLE_DEVICE void optix_coop_vec_reduce_sum_accumulate(
    const Vec& input, uint64 output, uint32 offset)
{
    WP_OPTIX_COOP_REQUIRE_PROGRAM(Vec);
    using Scalar = typename optix_coop_vec_traits<Vec>::scalar_type;
    static_assert(optix_coop_is_same<Scalar, half>::value || optix_coop_is_same<Scalar, float>::value,
        "optix_coop_vec_reduce_sum_accumulate requires a float16 or float32 vector");
    optixCoopVecReduceSumAccumulate(optix_coop_to_native(input), static_cast<CUdeviceptr>(output), offset);
}

template <typename VecA, typename VecB> inline CUDA_CALLABLE_DEVICE void optix_coop_vec_outer_product_accumulate(
    const VecA& a, const VecB& b, uint64 output, uint32 offset)
{
    WP_OPTIX_COOP_REQUIRE_PROGRAM(VecA);
    using ScalarA = typename optix_coop_vec_traits<VecA>::scalar_type;
    using ScalarB = typename optix_coop_vec_traits<VecB>::scalar_type;
    static_assert(optix_coop_is_same<ScalarA, half>::value && optix_coop_is_same<ScalarB, half>::value,
        "optix_coop_vec_outer_product_accumulate requires two float16 vectors");
    optixCoopVecOuterProductAccumulate<typename optix_coop_vec_traits<VecA>::optix_type,
        typename optix_coop_vec_traits<VecB>::optix_type, OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL>(
            optix_coop_to_native(a), optix_coop_to_native(b), static_cast<CUdeviceptr>(output), offset, 0);
}

#undef WP_OPTIX_COOP_REQUIRE_PROGRAM

}  // namespace wp
