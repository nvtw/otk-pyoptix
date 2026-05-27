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

// This header is injected by `warp_optix._addon` into every CUDA TU via
// `warp.config.extra_device_preamble`. The preamble fires before warp's own
// `cuda_module_header`, so we must declare `WP_NO_CRT` ourselves before pulling
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

inline CUDA_CALLABLE_DEVICE vec3 optix_transform_normal_from_object_to_world_space(const vec3& normal)
{
#if defined(WP_ENABLE_OPTIX)
    const float3 n = optixTransformNormalFromObjectToWorldSpace(make_float3(normal[0], normal[1], normal[2]));
    return vec3(n.x, n.y, n.z);
#else
    return normal;
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

inline CUDA_CALLABLE_DEVICE void optix_get_object_to_world_transform_matrix(float* matrix)
{
#if defined(WP_ENABLE_OPTIX)
    optixGetObjectToWorldTransformMatrix(matrix);
#endif
}

inline CUDA_CALLABLE_DEVICE void optix_get_world_to_object_transform_matrix(float* matrix)
{
#if defined(WP_ENABLE_OPTIX)
    optixGetWorldToObjectTransformMatrix(matrix);
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
    static_assert(__is_trivially_copyable(Payload), "optix_trace payload must be trivially copyable");
    static_assert(
        (alignof(Payload) % alignof(uint32)) == 0, "optix_trace payload alignment must be a multiple of 4 bytes"
    );
    static_assert((sizeof(Payload) % sizeof(uint32)) == 0, "optix_trace payload size must be a multiple of 4 bytes");

    constexpr size_t kPayloadWords = sizeof(Payload) / sizeof(uint32);
    static_assert(kPayloadWords > 0, "optix_trace payload must contain at least one 32-bit word");
    static_assert(kPayloadWords <= 32, "optix_trace payload exceeds OptiX payload register capacity");

    union PayloadWords {
        Payload payload;
        uint32 words[kPayloadWords];
    };

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

inline CUDA_CALLABLE_DEVICE uint32 optix_get_payload_index(uint32 index);
inline CUDA_CALLABLE_DEVICE void optix_set_payload_index(uint32 index, uint32 value);

template <typename Payload> inline CUDA_CALLABLE_DEVICE Payload optix_load_payload()
{
    static_assert(__is_trivially_copyable(Payload), "optix_load_payload type must be trivially copyable");
    static_assert(
        (alignof(Payload) % alignof(uint32)) == 0, "optix_load_payload type alignment must be a multiple of 4 bytes"
    );
    static_assert(
        (sizeof(Payload) % sizeof(uint32)) == 0, "optix_load_payload type size must be a multiple of 4 bytes"
    );

    constexpr size_t kPayloadWords = sizeof(Payload) / sizeof(uint32);
    static_assert(kPayloadWords > 0, "optix_load_payload type must contain at least one 32-bit word");
    static_assert(kPayloadWords <= 32, "optix_load_payload type exceeds OptiX payload register capacity");

    union PayloadWords {
        Payload payload;
        uint32 words[kPayloadWords];
    };

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
    static_assert(__is_trivially_copyable(Payload), "optix_store_payload type must be trivially copyable");
    static_assert(
        (alignof(Payload) % alignof(uint32)) == 0, "optix_store_payload type alignment must be a multiple of 4 bytes"
    );
    static_assert(
        (sizeof(Payload) % sizeof(uint32)) == 0, "optix_store_payload type size must be a multiple of 4 bytes"
    );

    constexpr size_t kPayloadWords = sizeof(Payload) / sizeof(uint32);
    static_assert(kPayloadWords > 0, "optix_store_payload type must contain at least one 32-bit word");
    static_assert(kPayloadWords <= 32, "optix_store_payload type exceeds OptiX payload register capacity");

    union PayloadWords {
        Payload payload;
        uint32 words[kPayloadWords];
    };

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


}  // namespace wp
