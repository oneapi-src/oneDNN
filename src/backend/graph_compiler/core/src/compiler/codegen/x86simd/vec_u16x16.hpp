/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_X86SIMD_VEC_U16X16_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_X86SIMD_VEC_U16X16_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_u32x16;
class vec_f32x16;
class vec_u16x16 {
public:
    union {
        __m256i v;
        uint16_t raw[16];
#ifdef __AVX512BF16__
        __m256bh v16;
#endif
    } __attribute__((aligned(32)));

    INLINE vec_u16x16() = default;
    INLINE vec_u16x16(uint16_t f) { v = _mm256_set1_epi16(f); }
    INLINE vec_u16x16(uint16_t i0, uint16_t i1, uint16_t i2, uint16_t i3,
            uint16_t i4, uint16_t i5, uint16_t i6, uint16_t i7, uint16_t i8,
            uint16_t i9, uint16_t i10, uint16_t i11, uint16_t i12, uint16_t i13,
            uint16_t i14, uint16_t i15) {
        v = _mm256_setr_epi16(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
                i12, i13, i14, i15);
    }
    INLINE vec_u16x16(__m256i const &x) { v = x; }
    INLINE operator vec_u32x16() const;

    static INLINE vec_u16x16 load(const uint16_t *p) {
        return _mm256_loadu_si256((const __m256i *)p);
    }
    static INLINE vec_u16x16 load_aligned(const uint16_t *p) {
        return _mm256_load_si256((const __m256i *)p);
    }
    static INLINE void store(vec_u16x16 v, uint16_t *p) {
        _mm256_storeu_si256((__m256i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u16x16 v, uint16_t *p) {
        _mm256_store_si256((__m256i *)p, v.v);
    }
};

INLINE vec_u16x16 operator+(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_adds_epu16(a.v, b.v);
}

INLINE vec_u16x16 operator-(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_subs_epu16(a.v, b.v);
}
INLINE vec_u16x16 operator-(vec_u16x16 const &a) {
    return _mm256_subs_epu16(_mm256_setzero_si256(), a.v);
}

// _mm_mulhi_epu16 was supported, but the high 16 bits of result was return.
// INLINE vec_u16x16 operator*(vec_u16x16 const &a, vec_u16x16 const &b) {
//    return _mm256_mulhi_epu16(a.v, b.v);
// }
// INLINE vec_u16x16 operator/(vec_u16x16 const &a, vec_u16x16 const &b) {
//     return _mm256_div_epu16(a.v, b.v);
// }

INLINE vec_u16x16 operator~(vec_u16x16 const &a) {
    return _mm256_xor_si256(a.v, _mm256_set1_epi16(0xFFFF));
}
INLINE vec_u16x16 operator&(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_and_si256(a.v, b.v);
}
INLINE vec_u16x16 operator|(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_or_si256(a.v, b.v);
}
INLINE vec_u16x16 operator^(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_xor_si256(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask16 operator!(vec_u16x16 const &a) {
    return _mm256_cmp_epi16_mask(a.v, _mm256_setzero_si256(), _MM_CMPINT_EQ);
}
INLINE __mmask16 operator==(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask16 operator!=(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask16 operator>(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask16 operator<(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask16 operator>=(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask16 operator<=(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_u16x16 sc_select(
        __mmask16 mask, vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_mask_blend_epi16(mask, b.v, a.v);
}
#endif

INLINE vec_u16x16 operator<<(vec_u16x16 const &a, const int b) {
    return _mm256_sll_epi16(a.v, _mm_cvtsi32_si128(b));
}
INLINE vec_u16x16 operator>>(vec_u16x16 const &a, const int b) {
    return _mm256_srl_epi16(a.v, _mm_cvtsi32_si128(b));
}

INLINE vec_u16x16 sc_max(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_max_epu16(a.v, b.v);
}
INLINE vec_u16x16 sc_min(vec_u16x16 const &a, vec_u16x16 const &b) {
    return _mm256_min_epu16(a.v, b.v);
}
#endif
