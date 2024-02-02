/*******************************************************************************
 * Copyright 2020-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X4_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_u16x4;
class vec_s32x4;
class vec_f32x4 {
public:
    union {
        __m128 v;
        float raw[4];
    } __attribute__((aligned(16)));

    INLINE vec_f32x4() = default;
    INLINE vec_f32x4(float f) { v = _mm_set1_ps(f); }
    INLINE vec_f32x4(float i0, float i1, float i2, float i3) {
        v = _mm_setr_ps(i0, i1, i2, i3);
    }
    INLINE vec_f32x4(__m128 const &x) { v = x; }

    INLINE operator vec_s32x4() const;

    static INLINE vec_f32x4 load(const float *p) { return _mm_loadu_ps(p); }
    static INLINE vec_f32x4 load_aligned(const float *p) {
        return _mm_load_ps(p);
    }
    static INLINE void store(vec_f32x4 v, float *p) { _mm_storeu_ps(p, v.v); }
    static INLINE void store_aligned(vec_f32x4 v, float *p) {
        _mm_store_ps(p, v.v);
    }

#ifdef __AVX512F__
    static INLINE vec_f32x4 mask_load(const float *p, __mmask8 mask) {
        return _mm_mask_loadu_ps(vec_f32x4(0.f).v, mask, p);
    }
    static INLINE void mask_store(float *p, __mmask8 mask, vec_f32x4 const &a) {
        return _mm_mask_storeu_ps(p, mask, a.v);
    }
#elif __AVX2__
    static INLINE vec_f32x4 mask_load(const float *p, uint32_t mask) {
        const __m128i table(_mm_setr_epi32(1 << 0, 1 << 1, 1 << 2, 1 << 3));
        __m128i vmask(_mm_set1_epi32(mask));
        vmask = _mm_and_si128(vmask, table);
        vmask = _mm_cmpeq_epi32(vmask, table);
        return _mm_maskload_ps(p, vmask);
    }
    static INLINE void mask_store(float *p, uint32_t mask, vec_f32x4 const &a) {
        const __m128i table(_mm_setr_epi32(1 << 0, 1 << 1, 1 << 2, 1 << 3));
        __m128i vmask(_mm_set1_epi32(mask));
        vmask = _mm_and_si128(vmask, table);
        vmask = _mm_cmpeq_epi32(vmask, table);
        return _mm_maskstore_ps(p, vmask, a.v);
    }
#endif
};

INLINE vec_f32x4 operator+(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_add_ps(a.v, b.v);
}

INLINE vec_f32x4 operator-(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_sub_ps(a.v, b.v);
}

INLINE vec_f32x4 operator*(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_mul_ps(a.v, b.v);
}

INLINE vec_f32x4 operator/(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_div_ps(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask8 operator!(vec_f32x4 const &a) {
    return _mm_cmp_ps_mask(a.v, _mm_setzero_ps(), _MM_CMPINT_EQ);
}
INLINE __mmask8 operator==(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask8 operator!=(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask8 operator>(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask8 operator<(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask8 operator>=(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask8 operator<=(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_f32x4 sc_select(
        __mmask8 mask, vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_mask_blend_ps(mask, b.v, a.v);
}
#endif

INLINE vec_f32x4 sc_fmadd(
        vec_f32x4 const &a, vec_f32x4 const &b, vec_f32x4 const &c) {
    return _mm_fmadd_ps(a.v, b.v, c.v);
}

INLINE vec_f32x4 sc_fnmadd(
        vec_f32x4 const &a, vec_f32x4 const &b, vec_f32x4 const &c) {
    return _mm_fnmadd_ps(a.v, b.v, c.v);
}

INLINE vec_f32x4 sc_max(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_max_ps(a.v, b.v);
}
INLINE vec_f32x4 sc_min(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_min_ps(a.v, b.v);
}

INLINE vec_f32x4 sc_round(vec_f32x4 const &a) {
    return _mm_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

INLINE vec_f32x4 sc_ceil(vec_f32x4 const &a) {
    return _mm_ceil_ps(a.v);
}
INLINE vec_f32x4 sc_floor(vec_f32x4 const &a) {
    return _mm_floor_ps(a.v);
}

INLINE float sc_reduce_add(vec_f32x4 const &a) {
    __m128 v4 = _mm_hadd_ps(a.v, a.v);
    v4 = _mm_hadd_ps(v4, v4);
    return _mm_cvtss_f32(v4);
}

INLINE vec_f32x4 sc_sqrt(vec_f32x4 const &a) {
    return _mm_sqrt_ps(a.v);
}
INLINE vec_f32x4 sc_rsqrt(vec_f32x4 const &a) {
    return _mm_rsqrt_ps(a.v);
}

INLINE vec_f32x4 sc_pow(vec_f32x4 const &a, vec_f32x4 const &b) {
    vec_f32x4 c;
    for (int i = 0; i < 4; i++) {
        c.raw[i] = powf(a.raw[i], b.raw[i]);
    }
    return c;
}

#endif
