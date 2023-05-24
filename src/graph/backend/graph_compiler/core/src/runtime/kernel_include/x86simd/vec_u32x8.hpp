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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U32X8_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U32X8_HPP

#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_u16x8;
class vec_u32x8 {
public:
    union {
        __m256i v;
        uint32_t raw[8];
    } __attribute__((aligned(32)));

    INLINE vec_u32x8() = default;
    INLINE vec_u32x8(int32_t f) { v = _mm256_set1_epi32(f); }
    INLINE vec_u32x8(int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4,
            int32_t i5, int32_t i6, int32_t i7) {
        v = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
    }
    INLINE vec_u32x8(__m256i const &x) { v = x; }
    INLINE operator vec_u16x8() const;
    static INLINE vec_u32x8 load(const uint32_t *p) {
        return _mm256_loadu_si256((const __m256i *)p);
    }
    static INLINE vec_u32x8 load_aligned(const int32_t *p) {
        return _mm256_load_si256((const __m256i *)p);
    }
#ifdef __AVX512F__
    static INLINE vec_u32x8 mask_load(const uint32_t *p, __mmask8 mask) {
        return _mm256_mask_loadu_epi32(vec_u32x8(0).v, mask, p);
    }
#endif
    static INLINE void store(vec_u32x8 v, uint32_t *p) {
        _mm256_storeu_si256((__m256i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u32x8 v, uint32_t *p) {
        _mm256_store_si256((__m256i *)p, v.v);
    }
#ifdef __AVX512F__
    static INLINE void mask_store(
            uint32_t *p, __mmask8 mask, vec_u32x8 const &a) {
        return _mm256_mask_storeu_epi32(p, mask, a.v);
    }
#endif
};

INLINE vec_u32x8 operator+(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_add_epi32(a.v, b.v);
}

// INLINE vec_u32x8 operator-(vec_u32x8 const &a, vec_u32x8 const &b) {
//     return _mm256_sub_epi32(a.v, b.v);
// }
// INLINE vec_u32x8 operator-(vec_u32x8 const &a) {
//     return _mm256_sub_epi32(_mm256_setzero_si256(), a.v);
// }

// INLINE vec_u32x8 operator*(vec_u32x8 const &a, vec_u32x8 const &b) {
//     return _mm256_mullo_epi32(a.v, b.v);
// }
// INLINE vec_u32x8 operator/(vec_u32x8 const &a, vec_u32x8 const &b) {
//     return _mm256_div_epu32(a.v, b.v);
// }

INLINE vec_u32x8 operator~(vec_u32x8 const &a) {
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(0xFFFFFFFF));
}
INLINE vec_u32x8 operator&(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_and_si256(a.v, b.v);
}
INLINE vec_u32x8 operator|(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_or_si256(a.v, b.v);
}
INLINE vec_u32x8 operator^(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_xor_si256(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask8 operator!(vec_u32x8 const &a) {
    return _mm256_cmp_epu32_mask(a.v, _mm256_setzero_si256(), _MM_CMPINT_EQ);
}
INLINE __mmask8 operator==(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask8 operator!=(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask8 operator>(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask8 operator<(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask8 operator>=(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask8 operator<=(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_u32x8 sc_select(
        __mmask8 mask, vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_mask_blend_epi32(mask, b.v, a.v);
}
#else
INLINE unsigned char operator>(vec_u32x8 const &a, vec_u32x8 const &b) {
    unsigned char ret = 0;
    for (int i = 0; i < 8; i++) {
        ret <<= 1;
        ret |= ((uint32_t *)&a)[i] > ((uint32_t *)&b)[i];
    }
    return ret;
}
vec_u32x8 sc_select(
        unsigned char mask, vec_u32x8 const &a, vec_u32x8 const &b) {
    uint32_t buf[8];
    for (int i = 0; i < 8; i++) {
        int bit = 1 << (7 - i);
        if (mask & bit) {
            buf[i] = ((uint32_t *)&b)[i];
        } else {
            buf[i] = ((uint32_t *)&a)[i];
        }
    }
    return vec_u32x8::load(buf);
}
#endif

INLINE vec_u32x8 operator<<(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_sllv_epi32(a.v, b.v);
}
INLINE vec_u32x8 operator>>(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_srlv_epi32(a.v, b.v);
}

INLINE vec_u32x8 sc_max(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_max_epu32(a.v, b.v);
}
INLINE vec_u32x8 sc_min(vec_u32x8 const &a, vec_u32x8 const &b) {
    return _mm256_min_epu32(a.v, b.v);
}

INLINE vec_u16x8 sc_reinterpret(vec_u32x8 const &x);
INLINE vec_u32x8 sc_reinterpret(vec_u16x8 const &x);
#endif
