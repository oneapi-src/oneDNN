/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F16X16_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F16X16_HPP
#include <cstdint>
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
#include "runtime/kernel_include/x86simd/vec_u16x16.hpp"
class vec_u16x16;
class vec_f32x16;
class vec_u32x16;
class vec_s32x16;
#ifdef __AVX512FP16__
class vec_f16x16 {
public:
    union {
        __m256i v256;
        __m256h v;
        uint16_t raw[16];
    } __attribute__((aligned(32)));

    INLINE vec_f16x16() = default;
    INLINE vec_f16x16(_Float16 f) { v = _mm256_set1_ph(f); }
    INLINE vec_f16x16(_Float16 i0, _Float16 i1, _Float16 i2, _Float16 i3,
            _Float16 i4, _Float16 i5, _Float16 i6, _Float16 i7, _Float16 i8,
            _Float16 i9, _Float16 i10, _Float16 i11, _Float16 i12, _Float16 i13,
            _Float16 i14, _Float16 i15) {
        v = _mm256_setr_ph(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
                i12, i13, i14, i15);
    }
    INLINE vec_f16x16(__m256h const &x) { v = x; }
    INLINE vec_f16x16(__m256i const &x) { v256 = x; }
    INLINE operator vec_u16x16() const;
    INLINE operator vec_f32x16() const;
    INLINE operator vec_u32x16() const;
    INLINE operator vec_s32x16() const;

    static INLINE vec_f16x16 load(const _Float16 *p) {
        return _mm256_loadu_ph(p);
    }
    static INLINE vec_f16x16 load_aligned(const _Float16 *p) {
        return _mm256_load_ph(p);
    }
    static INLINE void store(vec_f16x16 v, _Float16 *p) {
        _mm256_storeu_ph(p, v.v);
    }
    static INLINE void store_aligned(vec_f16x16 v, _Float16 *p) {
        _mm256_store_ph(p, v.v);
    }

    static INLINE vec_f16x16 mask_load(const _Float16 *p, __mmask16 mask) {
        // avx512-fp16 instructions don't have bigger than 128bit mask
        // load/store, we need to do this conversion.
        __m256i load_256b_data
                = _mm256_mask_loadu_epi16(_mm256_setzero_si256(), mask, p);
        return _mm256_cvtepi16_ph(load_256b_data);
    }
    static INLINE void mask_store(
            _Float16 *p, __mmask16 mask, vec_f16x16 const &a) {
        return _mm256_mask_storeu_epi16(p, mask, _mm256_castph_si256(a.v));
    }
};

INLINE vec_f16x16 operator+(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_add_ph(a.v, b.v);
}

INLINE vec_f16x16 operator-(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_sub_ph(a.v, b.v);
}

INLINE vec_f16x16 operator*(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_mul_ph(a.v, b.v);
}

INLINE vec_f16x16 operator/(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_div_ph(a.v, b.v);
}

INLINE __mmask16 operator!(vec_f16x16 const &a) {
    return _mm256_cmp_ph_mask(a.v, _mm256_setzero_ph(), _MM_CMPINT_EQ);
}
INLINE __mmask16 operator==(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_cmp_ph_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask16 operator!=(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_cmp_ph_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask16 operator>(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_cmp_ph_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask16 operator<(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_cmp_ph_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask16 operator>=(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_cmp_ph_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask16 operator<=(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_cmp_ph_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_f16x16 sc_select(
        __mmask16 mask, vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_mask_blend_ph(mask, b.v, a.v);
}

INLINE vec_f16x16 sc_fmadd(
        vec_f16x16 const &a, vec_f16x16 const &b, vec_f16x16 const &c) {
    return _mm256_fmadd_ph(a.v, b.v, c.v);
}

INLINE vec_f16x16 sc_fnmadd(
        vec_f16x16 const &a, vec_f16x16 const &b, vec_f16x16 const &c) {
    return _mm256_fnmadd_ph(a.v, b.v, c.v);
}

INLINE vec_f16x16 sc_max(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_max_ph(a.v, b.v);
}
INLINE vec_f16x16 sc_min(vec_f16x16 const &a, vec_f16x16 const &b) {
    return _mm256_min_ph(a.v, b.v);
}

INLINE vec_f16x16 sc_round(vec_f16x16 const &a) {
    return _mm256_roundscale_ph(
            a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

INLINE vec_f16x16 sc_ceil(vec_f16x16 const &a) {
    return _mm256_roundscale_ph(a.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}
INLINE vec_f16x16 sc_floor(vec_f16x16 const &a) {
    return _mm256_roundscale_ph(a.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

INLINE _Float16 sc_reduce_add(vec_f16x16 const &a) {
    return _mm256_reduce_add_ph(a.v);
}

INLINE vec_f16x16 sc_sqrt(vec_f16x16 const &a) {
    return _mm256_sqrt_ph(a.v);
}
INLINE vec_f16x16 sc_rsqrt(vec_f16x16 const &a) {
    return _mm256_rsqrt_ph(a.v);
}
INLINE vec_f16x16 sc_abs(vec_f16x16 const &a) {
    return _mm256_abs_ph(a.v);
}

#endif
#endif
