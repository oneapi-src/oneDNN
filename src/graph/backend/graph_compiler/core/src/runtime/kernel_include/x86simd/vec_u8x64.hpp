/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U8X64_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U8X64_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
#ifdef __AVX512F__
class vec_u8x64 {
public:
    union {
        __m512i v;
        uint8_t raw[64];
    } __attribute__((aligned(64)));

    INLINE vec_u8x64() = default;
    INLINE vec_u8x64(uint8_t f) { v = _mm512_set1_epi8(f); }
    INLINE vec_u8x64(__m512i const &x) { v = x; }

    static INLINE vec_u8x64 load(const uint8_t *p) {
        return _mm512_loadu_si512((const __m512i *)p);
    }
    static INLINE vec_u8x64 load_aligned(const int8_t *p) {
        return _mm512_load_si512((const __m512i *)p);
    }
    static INLINE vec_u8x64 mask_load(const uint8_t *p, __mmask64 mask) {
        return _mm512_mask_loadu_epi8(vec_u8x64(0).v, mask, p);
    }
    static INLINE void store(vec_u8x64 v, uint8_t *p) {
        _mm512_storeu_si512((__m512i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u8x64 v, int8_t *p) {
        _mm512_store_si512((__m512i *)p, v.v);
    }
    static INLINE void mask_store(
            uint8_t *p, __mmask64 mask, vec_u8x64 const &a) {
        return _mm512_mask_storeu_epi8(p, mask, a.v);
    }
};

INLINE vec_u8x64 operator+(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_add_epi8(a.v, b.v);
}

INLINE vec_u8x64 operator-(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_sub_epi8(a.v, b.v);
}

INLINE vec_u8x64 operator-(vec_u8x64 const &a) {
    return _mm512_sub_epi8(_mm512_setzero_si512(), a.v);
}

// INLINE vec_u8x64 operator*(vec_u8x64 const &a, vec_u8x64 const &b) {
//     return _mm512_mullo_epu8(a.v, b.v);
// }
// INLINE vec_u8x64 operator/(vec_u8x64 const &a, vec_u8x64 const &b) {
//     return _mm512_div_epu8(a.v, b.v);
// }

INLINE vec_u8x64 operator~(vec_u8x64 const &a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi8(-1));
}
INLINE vec_u8x64 operator&(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_and_si512(a.v, b.v);
}
INLINE vec_u8x64 operator|(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_or_si512(a.v, b.v);
}
INLINE vec_u8x64 operator^(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_xor_si512(a.v, b.v);
}

INLINE __mmask64 operator!(vec_u8x64 const &a) {
    return _mm512_cmp_epu8_mask(a.v, _mm512_setzero_si512(), _MM_CMPINT_EQ);
}
INLINE __mmask64 operator==(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask64 operator!=(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask64 operator>(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask64 operator<(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask64 operator>=(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask64 operator<=(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_u8x64 sc_select(
        __mmask64 mask, vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_mask_blend_epi8(mask, b.v, a.v);
}

INLINE vec_u8x64 sc_max(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_max_epu8(a.v, b.v);
}
INLINE vec_u8x64 sc_min(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_min_epu8(a.v, b.v);
}
#endif
#endif
