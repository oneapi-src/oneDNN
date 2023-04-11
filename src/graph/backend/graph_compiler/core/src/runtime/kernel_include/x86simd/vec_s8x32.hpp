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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S8X32_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S8X32_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_s8x32 {
public:
    union {
        __m256i v;
        int8_t raw[32];
    } __attribute__((aligned(32)));

    INLINE vec_s8x32() = default;
    INLINE vec_s8x32(int8_t f) { v = _mm256_set1_epi8(f); }

    INLINE vec_s8x32(__m256i const &x) { v = x; }

    static INLINE vec_s8x32 load(const int8_t *p) {
        return _mm256_loadu_si256((const __m256i *)p);
    }
    static INLINE vec_s8x32 load_aligned(const int8_t *p) {
        return _mm256_load_si256((const __m256i *)p);
    }
#ifdef __AVX512F__
    static INLINE vec_s8x32 mask_load(const int8_t *p, __mmask32 mask) {
        return _mm256_mask_loadu_epi8(vec_s8x32(0).v, mask, p);
    }
#endif
    static INLINE void store(vec_s8x32 v, int8_t *p) {
        _mm256_storeu_si256((__m256i *)p, v.v);
    }
    static INLINE void store_aligned(vec_s8x32 v, int8_t *p) {
        _mm256_store_si256((__m256i *)p, v.v);
    }
#ifdef __AVX512F__
    static INLINE void mask_store(
            int8_t *p, __mmask32 mask, vec_s8x32 const &a) {
        return _mm256_mask_storeu_epi8(p, mask, a.v);
    }
#endif
};

INLINE vec_s8x32 operator+(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_add_epi8(a.v, b.v);
}

INLINE vec_s8x32 operator-(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_sub_epi8(a.v, b.v);
}

INLINE vec_s8x32 operator-(vec_s8x32 const &a) {
    return _mm256_sub_epi8(_mm256_setzero_si256(), a.v);
}

// INLINE vec_s8x32 operator*(vec_s8x32 const &a, vec_s8x32 const &b) {
//     return _mm256_mullo_epi8(a.v, b.v);
// }
// INLINE vec_s8x32 operator/(vec_s8x32 const &a, vec_s8x32 const &b) {
//     return _mm256_div_epi8(a.v, b.v);
// }

INLINE vec_s8x32 operator~(vec_s8x32 const &a) {
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(0xFFFFFFFF));
}
INLINE vec_s8x32 operator&(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_and_si256(a.v, b.v);
}
INLINE vec_s8x32 operator|(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_or_si256(a.v, b.v);
}
INLINE vec_s8x32 operator^(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_xor_si256(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask32 operator!(vec_s8x32 const &a) {
    return _mm256_cmp_epi8_mask(a.v, _mm256_setzero_si256(), _MM_CMPINT_EQ);
}
INLINE __mmask32 operator==(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask32 operator!=(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask32 operator>(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask32 operator<(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask32 operator>=(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask32 operator<=(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_s8x32 sc_select(
        __mmask32 mask, vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_mask_blend_epi8(mask, b.v, a.v);
}
#endif
// INLINE vec_s8x32 operator<<(vec_s8x32 const &a, int16_t b) {
//     return _mm256_sll_epi8(a.v, _mm256_cvtsi32_si256(b));
// }
// INLINE vec_s8x32 operator>>(vec_s8x32 const &a, int16_t b) {
//     return _mm256_sra_epi8(a.v, _mm256_cvtsi32_si256(b));
// }

// operator /

INLINE vec_s8x32 sc_max(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_max_epi8(a.v, b.v);
}
INLINE vec_s8x32 sc_min(vec_s8x32 const &a, vec_s8x32 const &b) {
    return _mm256_min_epi8(a.v, b.v);
}
INLINE vec_s8x32 sc_abs(vec_s8x32 const &a) {
    return _mm256_abs_epi8(a.v);
}
#endif
