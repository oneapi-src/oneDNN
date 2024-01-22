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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X16_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X16_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
#ifdef __AVX512F__
class vec_u16x16;
class vec_u8x16;
class vec_s8x16;
class vec_s32x16;
#ifdef __AVX512FP16__
class vec_f16x16;
#endif
class vec_f32x16 {
public:
    union {
        __m512 v;
        float raw[16];
    } __attribute__((aligned(64)));

    INLINE vec_f32x16() = default;
    INLINE vec_f32x16(float f) { v = _mm512_set1_ps(f); }
    INLINE vec_f32x16(float i0, float i1, float i2, float i3, float i4,
            float i5, float i6, float i7, float i8, float i9, float i10,
            float i11, float i12, float i13, float i14, float i15) {
        v = _mm512_setr_ps(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
                i12, i13, i14, i15);
    }
    INLINE vec_f32x16(__m512 const &x) { v = x; }
    INLINE operator vec_u8x16() const;
    INLINE operator vec_s8x16() const;
    INLINE operator vec_s32x16() const;
#ifdef __AVX512FP16__
    INLINE operator vec_f16x16() const;
#endif

    static INLINE vec_f32x16 load(const float *p) { return _mm512_loadu_ps(p); }
    static INLINE vec_f32x16 load_aligned(const float *p) {
        // gcc bug here, we need to cast to double
        return _mm512_load_ps((double *)p);
    }
    static INLINE vec_f32x16 mask_load(const float *p, __mmask16 mask) {
        return _mm512_mask_loadu_ps(vec_f32x16(0.f).v, mask, p);
    }
    static INLINE void store(vec_f32x16 v, float *p) {
        _mm512_storeu_ps(p, v.v);
    }
    static INLINE void store_aligned(vec_f32x16 v, float *p) {
        _mm512_store_ps(p, v.v);
    }
    static INLINE void mask_store(
            float *p, __mmask16 mask, vec_f32x16 const &a) {
        return _mm512_mask_storeu_ps(p, mask, a.v);
    }
};

INLINE vec_f32x16 operator+(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_add_ps(a.v, b.v);
}

INLINE vec_f32x16 operator-(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_sub_ps(a.v, b.v);
}

INLINE vec_f32x16 operator*(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_mul_ps(a.v, b.v);
}

INLINE vec_f32x16 operator/(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_div_ps(a.v, b.v);
}

INLINE __mmask16 operator!(vec_f32x16 const &a) {
    return _mm512_cmp_ps_mask(a.v, _mm512_setzero_ps(), _MM_CMPINT_EQ);
}
INLINE __mmask16 operator==(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask16 operator!=(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask16 operator>(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask16 operator<(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask16 operator>=(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask16 operator<=(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_cmp_ps_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_f32x16 sc_select(
        __mmask16 mask, vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_mask_blend_ps(mask, b.v, a.v);
}

INLINE vec_f32x16 sc_max(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_max_ps(a.v, b.v);
}
INLINE vec_f32x16 sc_min(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_min_ps(a.v, b.v);
}

INLINE vec_f32x16 sc_round(vec_f32x16 const &a) {
    return _mm512_roundscale_ps(a.v, _MM_FROUND_TO_NEAREST_INT);
}

INLINE vec_f32x16 sc_ceil(vec_f32x16 const &a) {
    return _mm512_ceil_ps(a.v);
}
INLINE vec_f32x16 sc_floor(vec_f32x16 const &a) {
    return _mm512_floor_ps(a.v);
}

INLINE vec_f32x16 sc_sqrt(vec_f32x16 const &a) {
    return _mm512_sqrt_ps(a.v);
}

INLINE vec_f32x16 sc_rsqrt(vec_f32x16 const &a) {
    return _mm512_rsqrt14_ps(a.v);
}

INLINE float sc_reduce_add(vec_f32x16 const &a) {
    return _mm512_reduce_add_ps(a.v);
}

INLINE float sc_reduce_mul(vec_f32x16 const &a) {
    return _mm512_reduce_mul_ps(a.v);
}

INLINE float sc_reduce_max(vec_f32x16 const &a) {
    return _mm512_reduce_max_ps(a.v);
}

INLINE float sc_reduce_min(vec_f32x16 const &a) {
    return _mm512_reduce_min_ps(a.v);
}

INLINE vec_f32x16 sc_fmadd(
        vec_f32x16 const &a, vec_f32x16 const &b, vec_f32x16 const &c) {
    return _mm512_fmadd_ps(a.v, b.v, c.v);
}

INLINE vec_f32x16 sc_fnmadd(
        vec_f32x16 const &a, vec_f32x16 const &b, vec_f32x16 const &c) {
    return _mm512_fnmadd_ps(a.v, b.v, c.v);
}

INLINE vec_f32x16 sc_abs(vec_f32x16 const &a) {
    return _mm512_abs_ps(a.v);
}

INLINE vec_f32x16 sc_log(vec_f32x16 const &a) {
    vec_f32x16 b;
    for (int i = 0; i < 16; i++) {
        b.raw[i] = logf(a.raw[i]);
    }
    return b;
}

INLINE vec_f32x16 sc_unpack_low_vec_f32x16_32bits(
        vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_unpacklo_ps(a.v, b.v);
}

INLINE vec_f32x16 sc_unpack_low_vec_f32x16_64bits(
        vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_castpd_ps(
            _mm512_unpacklo_pd(_mm512_castps_pd(a.v), _mm512_castps_pd(b.v)));
}

INLINE vec_f32x16 sc_unpack_high_vec_f32x16_32bits(
        vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_unpackhi_ps(a.v, b.v);
}

INLINE vec_f32x16 sc_unpack_high_vec_f32x16_64bits(
        vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_castpd_ps(
            _mm512_unpackhi_pd(_mm512_castps_pd(a.v), _mm512_castps_pd(b.v)));
}

INLINE vec_f32x16 sc_pow(vec_f32x16 const &a, vec_f32x16 const &b) {
    vec_f32x16 c;
    for (int i = 0; i < 16; i++) {
        c.raw[i] = powf(a.raw[i], b.raw[i]);
    }
    return c;
}
#define PARAM_F32X16(X) X.v
#define sc_shuffle_vec_f32x16_128bits(a, b, imm) \
    _mm512_shuffle_f32x4(PARAM_F32X16(a), PARAM_F32X16(b), imm);
#define sc_shuffle_vec_f32x16_32bits(a, b, imm) \
    _mm512_shuffle_ps(PARAM_F32X16(a), PARAM_F32X16(b), imm);

#endif
#endif
