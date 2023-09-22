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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F16X32_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F16X32_HPP
#include <cstdint>
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
#include "runtime/kernel_include/x86simd/vec_u16x32.hpp"
#ifdef __AVX512FP16__
class vec_u16x32;
class vec_f16x32 {
public:
    union {
        __m512i v512;
        __m512h v;
        uint16_t raw[32];
    } __attribute__((aligned(64)));

    INLINE vec_f16x32() = default;
    INLINE vec_f16x32(_Float16 f) { v = _mm512_set1_ph(f); }
    INLINE vec_f16x32(__m512h const &x) { v = x; }

    static INLINE vec_f16x32 load(const _Float16 *p) {
        return _mm512_loadu_ph(p);
    }
    static INLINE vec_f16x32 load_aligned(const _Float16 *p) {
        return _mm512_load_ph(p);
    }
    static INLINE void store(vec_f16x32 v, _Float16 *p) {
        return _mm512_storeu_ph(p, v.v);
    }
    static INLINE void store_aligned(vec_f16x32 v, _Float16 *p) {
        _mm512_store_ph(p, v.v);
    }

    static INLINE vec_f16x32 mask_load(const _Float16 *p, __mmask32 mask) {
        // avx512-fp16 instructions don't have bigger than 128bit mask
        // load/store, we need to do this conversion.
        __m512i load_512b_data
                = _mm512_mask_loadu_epi16(vec_u16x32(0).v, mask, p);
        return _mm512_cvtepi16_ph(load_512b_data);
    }
    static INLINE void mask_store(
            _Float16 *p, __mmask32 mask, vec_f16x32 const &a) {
        return _mm512_mask_storeu_epi16(p, mask, _mm512_castph_si512(a.v));
    }
};

INLINE vec_f16x32 operator+(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_add_ph(a.v, b.v);
}

INLINE vec_f16x32 operator-(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_sub_ph(a.v, b.v);
}

INLINE vec_f16x32 operator*(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_mul_ph(a.v, b.v);
}

INLINE vec_f16x32 operator/(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_div_ph(a.v, b.v);
}

INLINE __mmask32 operator!(vec_f16x32 const &a) {
    return _mm512_cmp_ph_mask(a.v, _mm512_setzero_ph(), _MM_CMPINT_EQ);
}
INLINE __mmask32 operator==(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_cmp_ph_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask32 operator!=(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_cmp_ph_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask32 operator>(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_cmp_ph_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask32 operator<(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_cmp_ph_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask32 operator>=(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_cmp_ph_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask32 operator<=(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_cmp_ph_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_f16x32 sc_select(
        __mmask32 mask, vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_mask_blend_ph(mask, b.v, a.v);
}

INLINE vec_f16x32 sc_fmadd(
        vec_f16x32 const &a, vec_f16x32 const &b, vec_f16x32 const &c) {
    return _mm512_fmadd_ph(a.v, b.v, c.v);
}

INLINE vec_f16x32 sc_max(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_max_ph(a.v, b.v);
}
INLINE vec_f16x32 sc_min(vec_f16x32 const &a, vec_f16x32 const &b) {
    return _mm512_min_ph(a.v, b.v);
}

INLINE vec_f16x32 sc_round(vec_f16x32 const &a) {
    return _mm512_roundscale_ph(
            a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

INLINE vec_f16x32 sc_ceil(vec_f16x32 const &a) {
    return _mm512_roundscale_ph(a.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}
INLINE vec_f16x32 sc_floor(vec_f16x32 const &a) {
    return _mm512_roundscale_ph(a.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

INLINE _Float16 sc_reduce_add(vec_f16x32 const &a) {
    return _mm512_reduce_add_ph(a.v);
}

INLINE vec_f16x32 sc_sqrt(vec_f16x32 const &a) {
    return _mm512_sqrt_ph(a.v);
}
INLINE vec_f16x32 sc_rsqrt(vec_f16x32 const &a) {
    return _mm512_rsqrt_ph(a.v);
}

#endif
#endif
