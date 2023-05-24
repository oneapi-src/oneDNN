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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U16X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U16X4_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_u32x4;
class vec_f32x4;
class vec_u16x4 {
public:
    union {
        __m128i v128;
        __m64 v;
        uint16_t raw[8];
#ifdef __AVX512BF16__
        __m128bh v16;
#endif
    } __attribute__((aligned(16)));

    INLINE vec_u16x4() = default;
    INLINE vec_u16x4(uint16_t f) { v128 = _mm_set1_epi16(f); }
    INLINE vec_u16x4(uint16_t i0, uint16_t i1, uint16_t i2, uint16_t i3) {
        v128 = _mm_setr_epi16(i0, i1, i2, i3, 0, 0, 0, 0);
    }
    INLINE operator vec_u32x4() const;

    INLINE vec_u16x4(__m64 const &x) { v = x; }
    INLINE vec_u16x4(__m128i const &x) { v128 = x; }

    static INLINE vec_u16x4 load(const uint16_t *p) {
        return *(const __m64 *)p;
    }

    static INLINE void store(vec_u16x4 v, uint16_t *p) { *(__m64 *)p = v.v; }
};

// INLINE vec_u16x4 operator+(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_add_epi16(a.v, b.v);
// }

// INLINE vec_u16x4 operator-(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_sub_epi16(a.v, b.v);
// }
// INLINE vec_u16x4 operator-(vec_u16x4 const &a, uint16_t b) {
//     return _mm_sub_epi16(a.v, _mm_set1_epi16(b));s
// }
// INLINE vec_u16x4 operator-(vec_u16x4 const &a) {
//     return _mm_sub_epi16(_mm_setzero_si128(), a.v);
// }

// INLINE vec_u16x4 operator*(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_mullo_epi16(a.v, b.v);
// }
// INLINE vec_u16x4 operator*(vec_u16x4 const &a, uint16_t b) {
//     return _mm_mullo_epi16(a.v, _mm_set1_epi16(b));
// }

// INLINE vec_u16x4 operator~(vec_u16x4 const &a) {
//     return _mm_xor_si128(a.v, _mm_set1_epi16(-1));
// }
INLINE vec_u16x4 operator&(vec_u16x4 const &a, vec_u16x4 const &b) {
    return _mm_and_si64(a.v, b.v);
}
// INLINE vec_u16x4 operator&&(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_and_si128(a.v, b.v);
// }

// INLINE vec_u16x4 operator|(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_or_si128(a.v, b.v);
// }
// INLINE vec_u16x4 operator||(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_or_si128(a.v, b.v);
// }

// INLINE vec_u16x4 operator^(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_xor_si128(a.v, b.v);
// }

// #ifdef __AVX512F__
// INLINE __mmask8 operator!(vec_u16x4 const &a) {
//     return _mm_cmp_epu16_mask(a.v, _mm_setzero_si128(), _MM_CMPINT_EQ);
// }
// INLINE __mmask8 operator==(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_EQ);
// }
// INLINE __mmask8 operator!=(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NE);
// }
// INLINE __mmask8 operator>(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_GT);
// }
// INLINE __mmask8 operator>(vec_u16x4 const &a, float b) {
//     return _mm_cmp_epu16_mask(a.v, _mm_set1_epi16(b), _MM_CMPINT_GT);
// }
// INLINE __mmask8 operator<(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT);
// }
// INLINE __mmask8 operator<(vec_u16x4 const &a, float &b) {
//     return _mm_cmp_epu16_mask(a.v, _mm_set1_epi16(b), _MM_CMPINT_LT);
// }
// INLINE __mmask8 operator>=(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_GE);
// }
// INLINE __mmask8 operator>=(vec_u16x4 const &a, float b) {
//     return _mm_cmp_epu16_mask(a.v, _mm_set1_epi16(b), _MM_CMPINT_GE);
// }
// INLINE __mmask8 operator<=(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE);
// }
// INLINE __mmask8 operator<=(vec_u16x4 const &a, float b) {
//     return _mm_cmp_epu16_mask(a.v, _mm_set1_epi16(b), _MM_CMPINT_LE);
// }
// vec_u16x4 sc_select(__mmask8 mask, vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_mask_blend_epi16(mask, b.v, a.v);
// }
// #endif

INLINE vec_u16x4 operator<<(vec_u16x4 const &a, vec_u16x4 const &b) {
    return _mm_sll_pi16(a.v, b.v);
}
INLINE vec_u16x4 operator>>(vec_u16x4 const &a, vec_u16x4 const &b) {
    return _mm_srl_pi16(a.v, b.v);
}

// operator /

// This is for signed 16-bit integers comparison
// INLINE vec_u16x4 sc_max(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_max_pi16(a.v, b.v);
// }
// INLINE vec_u16x4 sc_min(vec_u16x4 const &a, vec_u16x4 const &b) {
//     return _mm_min_pi16(a.v, b.v);
// }
#endif
