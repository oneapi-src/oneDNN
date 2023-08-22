/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F16X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F16X4_HPP
#include <cstdint>
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
#ifdef __AVX512FP16__
class vec_f16x4 {
public:
    union {
        __m128i v128;
        __m128h v;
        uint16_t raw[4];
    } __attribute__((aligned(16)));

    INLINE vec_f16x4() = default;
    INLINE vec_f16x4(_Float16 f) { v = _mm_set1_ph(f); }
    INLINE vec_f16x4(_Float16 i0, _Float16 i1, _Float16 i2, _Float16 i3) {
        v = _mm_setr_ph(i0, i1, i2, i3, 0, 0, 0, 0);
    }
    INLINE vec_f16x4(__m128h const &x) { v = x; }

    static INLINE vec_f16x4 load(const _Float16 *p) { return _mm_loadu_ph(p); }
    static INLINE vec_f16x4 load_aligned(const _Float16 *p) {
        return _mm_load_ph(p);
    }
    static INLINE void store(vec_f16x4 v, _Float16 *p) {
        _mm_storeu_ph(p, v.v);
    }
    static INLINE void store_aligned(vec_f16x4 v, _Float16 *p) {
        _mm_store_ph(p, v.v);
    }

    static INLINE vec_f16x4 mask_load(const _Float16 *p, __mmask8 mask) {
        return _mm_mask_load_sh(vec_f16x4(0).v, mask, p);
    }
    static INLINE void mask_store(
            _Float16 *p, __mmask8 mask, vec_f16x4 const &a) {
        return _mm_mask_store_sh(p, mask, a.v);
    }
};

INLINE vec_f16x4 operator+(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_add_ph(a.v, b.v);
}

INLINE vec_f16x4 operator-(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_sub_ph(a.v, b.v);
}

INLINE vec_f16x4 operator*(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_mul_ph(a.v, b.v);
}

INLINE vec_f16x4 operator/(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_div_ph(a.v, b.v);
}

INLINE __mmask8 operator!(vec_f16x4 const &a) {
    return _mm_cmp_ph_mask(a.v, _mm_setzero_ph(), _MM_CMPINT_EQ);
}
INLINE __mmask8 operator==(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_cmp_ph_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask8 operator!=(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_cmp_ph_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask8 operator>(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_cmp_ph_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask8 operator<(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_cmp_ph_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask8 operator>=(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_cmp_ph_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask8 operator<=(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_cmp_ph_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_f16x4 sc_select(
        __mmask8 mask, vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_mask_blend_ph(mask, b.v, a.v);
}

INLINE vec_f16x4 sc_fmadd(
        vec_f16x4 const &a, vec_f16x4 const &b, vec_f16x4 const &c) {
    return _mm_fmadd_ph(a.v, b.v, c.v);
}

INLINE vec_f16x4 sc_max(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_max_ph(a.v, b.v);
}
INLINE vec_f16x4 sc_min(vec_f16x4 const &a, vec_f16x4 const &b) {
    return _mm_min_ph(a.v, b.v);
}

INLINE vec_f16x4 sc_round(vec_f16x4 const &a) {
    return _mm_roundscale_ph(
            a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

INLINE vec_f16x4 sc_ceil(vec_f16x4 const &a) {
    return _mm_roundscale_ph(a.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}
INLINE vec_f16x4 sc_floor(vec_f16x4 const &a) {
    return _mm_roundscale_ph(a.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

INLINE _Float16 sc_reduce_add(vec_f16x4 const &a) {
    return _mm_reduce_add_ph(a.v);
}

INLINE vec_f16x4 sc_sqrt(vec_f16x4 const &a) {
    return _mm_sqrt_ph(a.v);
}
INLINE vec_f16x4 sc_rsqrt(vec_f16x4 const &a) {
    return _mm_rsqrt_ph(a.v);
}

#endif
#endif
