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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X8_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X8_HPP
#include <immintrin.h>
#include <stdint.h>
#ifndef __AVX512F__
#include <cmath>
#endif
#include "common.hpp"

class vec_u16x8;
class vec_s32x8;
#ifdef __AVX512FP16__
class vec_f16x8;
#endif
class vec_f32x8 {
public:
    union {
        __m256 v;
        float raw[8];
    } __attribute__((aligned(32)));

    INLINE vec_f32x8() = default;
    INLINE vec_f32x8(float f) { v = _mm256_set1_ps(f); }
    INLINE vec_f32x8(float i0, float i1, float i2, float i3, float i4, float i5,
            float i6, float i7) {
        v = _mm256_setr_ps(i0, i1, i2, i3, i4, i5, i6, i7);
    }
    INLINE vec_f32x8(__m256 const &x) { v = x; }
    INLINE operator vec_s32x8() const;
#ifdef __AVX512FP16__
    INLINE operator vec_f16x8() const;
#endif

    static INLINE vec_f32x8 load(const float *p) { return _mm256_loadu_ps(p); }
    static INLINE vec_f32x8 load_aligned(const float *p) {
        return _mm256_load_ps(p);
    }
    static INLINE void store(vec_f32x8 v, float *p) {
        _mm256_storeu_ps(p, v.v);
    }
    static INLINE void store_aligned(vec_f32x8 v, float *p) {
        _mm256_store_ps(p, v.v);
    }

#ifdef __AVX512F__
    static INLINE vec_f32x8 mask_load(const float *p, __mmask8 mask) {
        return _mm256_mask_loadu_ps(vec_f32x8(0.f).v, mask, p);
    }
    static INLINE void mask_store(float *p, __mmask8 mask, vec_f32x8 const &a) {
        return _mm256_mask_storeu_ps(p, mask, a.v);
    }
#elif __AVX2__
    static INLINE vec_f32x8 mask_load(const float *p, uint32_t mask) {
        const __m256i table(_mm256_setr_epi32(1 << 0, 1 << 1, 1 << 2, 1 << 3,
                1 << 4, 1 << 5, 1 << 6, 1 << 7));
        __m256i vmask(_mm256_set1_epi32(mask));
        vmask = _mm256_and_si256(vmask, table);
        vmask = _mm256_cmpeq_epi32(vmask, table);
        return _mm256_maskload_ps(p, vmask);
    }
    static INLINE void mask_store(float *p, uint32_t mask, vec_f32x8 const &a) {
        const __m256i table(_mm256_setr_epi32(1 << 0, 1 << 1, 1 << 2, 1 << 3,
                1 << 4, 1 << 5, 1 << 6, 1 << 7));
        __m256i vmask(_mm256_set1_epi32(mask));
        vmask = _mm256_and_si256(vmask, table);
        vmask = _mm256_cmpeq_epi32(vmask, table);
        return _mm256_maskstore_ps(p, vmask, a.v);
    }
#endif
};

INLINE vec_f32x8 operator+(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_add_ps(a.v, b.v);
}

INLINE vec_f32x8 operator-(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_sub_ps(a.v, b.v);
}

INLINE vec_f32x8 operator*(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_mul_ps(a.v, b.v);
}

INLINE vec_f32x8 operator/(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_div_ps(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask8 operator!(vec_f32x8 const &a) {
    return _mm256_cmp_ps_mask(a.v, _mm256_setzero_ps(), _MM_CMPINT_EQ);
}
INLINE __mmask8 operator==(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_cmp_ps_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask8 operator!=(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_cmp_ps_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask8 operator>(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_cmp_ps_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask8 operator<(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_cmp_ps_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask8 operator>=(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_cmp_ps_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask8 operator<=(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_cmp_ps_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_f32x8 sc_select(
        __mmask8 mask, vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_mask_blend_ps(mask, b.v, a.v);
}
#else
INLINE vec_f32x8 sc_select(
        unsigned char mask, vec_f32x8 const &a, vec_f32x8 const &b) {
    float buf[8];
    for (int i = 0; i < 8; i++) {
        int bit = 1 << i;
        if (mask & bit) {
            buf[i] = ((float *)&a)[i];
        } else {
            buf[i] = ((float *)&b)[i];
        }
    }
    return vec_f32x8::load(buf);
}

INLINE unsigned char operator==(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ);
    return _mm256_movemask_ps(ret);
}
INLINE unsigned char operator!=(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_NEQ_OQ);
    return _mm256_movemask_ps(ret);
}
INLINE unsigned char operator>(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ);
    return _mm256_movemask_ps(ret);
}
INLINE unsigned char operator<(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ);
    return _mm256_movemask_ps(ret);
}
INLINE unsigned char operator>=(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ);
    return _mm256_movemask_ps(ret);
}
INLINE unsigned char operator<=(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ);
    return _mm256_movemask_ps(ret);
}
#endif

INLINE vec_f32x8 sc_fmadd(
        vec_f32x8 const &a, vec_f32x8 const &b, vec_f32x8 const &c) {
    return _mm256_fmadd_ps(a.v, b.v, c.v);
}

INLINE vec_f32x8 sc_fnmadd(
        vec_f32x8 const &a, vec_f32x8 const &b, vec_f32x8 const &c) {
    return _mm256_fnmadd_ps(a.v, b.v, c.v);
}

INLINE vec_f32x8 sc_max(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_max_ps(a.v, b.v);
}
INLINE vec_f32x8 sc_min(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_min_ps(a.v, b.v);
}

INLINE vec_f32x8 sc_round(vec_f32x8 const &a) {
    return _mm256_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

INLINE vec_f32x8 sc_ceil(vec_f32x8 const &a) {
    return _mm256_ceil_ps(a.v);
}
INLINE vec_f32x8 sc_floor(vec_f32x8 const &a) {
    return _mm256_floor_ps(a.v);
}

INLINE vec_f32x8 sc_sqrt(vec_f32x8 const &a) {
    return _mm256_sqrt_ps(a.v);
}
INLINE vec_f32x8 sc_rsqrt(vec_f32x8 const &a) {
    return _mm256_rsqrt_ps(a.v);
}

INLINE vec_f32x8 sc_abs(vec_f32x8 const &a) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.v);
}

INLINE float sc_reduce_add(vec_f32x8 const &a) {
    const __m128 v4 = _mm_add_ps(
            _mm256_extractf128_ps(a.v, 1), _mm256_castps256_ps128(a.v));
    const __m128 v2 = _mm_add_ps(v4, _mm_movehl_ps(v4, v4));
    const __m128 v1 = _mm_add_ss(v2, _mm_shuffle_ps(v2, v2, 0x55));
    return _mm_cvtss_f32(v1);
}

INLINE vec_f32x8 sc_unpack_low_vec_f32x8_64bits(
        vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_castpd_ps(
            _mm256_unpacklo_pd(_mm256_castps_pd(a.v), _mm256_castps_pd(b.v)));
}

INLINE vec_f32x8 sc_unpack_low_vec_f32x8_32bits(
        vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_unpacklo_ps(a.v, b.v);
}

INLINE vec_f32x8 sc_unpack_high_vec_f32x8_32bits(
        vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_unpackhi_ps(a.v, b.v);
}

INLINE vec_f32x8 sc_unpack_high_vec_f32x8_64bits(
        vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_castpd_ps(
            _mm256_unpackhi_pd(_mm256_castps_pd(a.v), _mm256_castps_pd(b.v)));
}

#define PARAM_F32X8(X) X.v
#define sc_permute_vec_f32x8(a, b, imm8) \
    _mm256_permute2f128_ps(PARAM_F32X8(a), PARAM_F32X8(b), imm8);
#define sc_shuffle_vec_f32x8_128bits(a, b, imm8) \
    _mm256_shuffle_f32x4(PARAM_F32X8(a), PARAM_F32X8(b), imm8);
#define sc_shuffle_vec_f32x8_32bits(a, b, imm8) \
    _mm256_shuffle_ps(PARAM_F32X8(a), PARAM_F32X8(b), imm8);

INLINE vec_f32x8 sc_exp(vec_f32x8 const &a) {
    float *flo = (float *)&a;
    float out[8];
    for (int i = 0; i < 8; i++) {
        out[i] = std::exp(flo[i]);
    }
    return vec_f32x8::load(out);
}

INLINE vec_f32x8 sc_log(vec_f32x8 const &a) {
    vec_f32x8 b;
    for (int i = 0; i < 8; i++) {
        b.raw[i] = logf(a.raw[i]);
    }
    return b;
}

INLINE vec_f32x8 sc_pow(vec_f32x8 const &a, vec_f32x8 const &b) {
    vec_f32x8 c;
    for (int i = 0; i < 8; i++) {
        c.raw[i] = powf(a.raw[i], b.raw[i]);
    }
    return c;
}
#endif
