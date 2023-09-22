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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U16X8_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U16X8_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_u32x8;
class vec_f32x8;
class vec_u16x8 {
public:
    union {
        __m128i v;
        uint16_t raw[8];
#ifdef __AVX512BF16__
        __m128bh v16;
#endif
    } __attribute__((aligned(16)));

    INLINE vec_u16x8() = default;
    INLINE vec_u16x8(uint16_t f) { v = _mm_set1_epi16(f); }
    INLINE vec_u16x8(uint16_t i0, uint16_t i1, uint16_t i2, uint16_t i3,
            uint16_t i4, uint16_t i5, uint16_t i6, uint16_t i7) {
        v = _mm_setr_epi16(i0, i1, i2, i3, i4, i5, i6, i7);
    }
    // INLINE vec_u16x8(vec_f32x8 val);
    INLINE vec_u16x8(__m128i const &x) { v = x; }
    INLINE operator vec_u32x8() const;

    static INLINE vec_u16x8 load(const uint16_t *p) {
        return _mm_loadu_si128((const __m128i *)p);
    }
    static INLINE vec_u16x8 load_aligned(const uint16_t *p) {
        return _mm_load_si128((const __m128i *)p);
    }
#ifdef __AVX512F__
    static INLINE vec_u16x8 mask_load(const uint16_t *p, __mmask8 mask) {
        return _mm_mask_loadu_epi16(vec_u16x8(0).v, mask, p);
    }
#endif
    static INLINE void store(vec_u16x8 v, uint16_t *p) {
        _mm_storeu_si128((__m128i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u16x8 v, uint16_t *p) {
        _mm_store_si128((__m128i *)p, v.v);
    }
#ifdef __AVX512F__
    static INLINE void mask_store(
            uint16_t *p, __mmask8 mask, vec_u16x8 const &a) {
        return _mm_mask_storeu_epi16(p, mask, a.v);
    }
#endif
};

INLINE vec_u16x8 operator+(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_add_epi16(a.v, b.v);
}

INLINE vec_u16x8 operator-(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_sub_epi16(a.v, b.v);
}
INLINE vec_u16x8 operator-(vec_u16x8 const &a) {
    return _mm_sub_epi16(_mm_setzero_si128(), a.v);
}
// _mm_mulhi_epu16 was supported, but the high 16 bits of result was return.
// INLINE vec_u16x8 operator*(vec_u16x8 const &a, vec_u16x8 const &b) {
//     return _mm_mulhi_epu16(a.v, b.v);
// }
// INLINE vec_u16x8 operator/(vec_u16x8 const &a, vec_u16x8 const &b) {
//     return _mm_div_epu16(a.v, b.v);
// }

INLINE vec_u16x8 operator~(vec_u16x8 const &a) {
    return _mm_xor_si128(a.v, _mm_set1_epi16(0xFFFF));
}
INLINE vec_u16x8 operator&(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_and_si128(a.v, b.v);
}
INLINE vec_u16x8 operator|(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_or_si128(a.v, b.v);
}
INLINE vec_u16x8 operator^(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_xor_si128(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask8 operator!(vec_u16x8 const &a) {
    return _mm_cmp_epu16_mask(a.v, _mm_setzero_si128(), _MM_CMPINT_EQ);
}
INLINE __mmask8 operator==(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask8 operator!=(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask8 operator>(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask8 operator<(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask8 operator>=(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask8 operator<=(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_u16x8 sc_select(
        __mmask8 mask, vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_mask_blend_epi16(mask, b.v, a.v);
}
#endif

INLINE vec_u16x8 operator<<(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_sll_epi16(a.v, b.v);
}
INLINE vec_u16x8 operator>>(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_srl_epi16(a.v, b.v);
}

INLINE vec_u16x8 sc_max(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_max_epu16(a.v, b.v);
}
INLINE vec_u16x8 sc_min(vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_min_epu16(a.v, b.v);
}
INLINE vec_u16x8 sc_unpack_low_vec_u16x8_16bits(
        vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_unpacklo_epi16(a.v, b.v);
}
INLINE vec_u16x8 sc_unpack_high_vec_u16x8_16bits(
        vec_u16x8 const &a, vec_u16x8 const &b) {
    return _mm_unpackhi_epi16(a.v, b.v);
}
#endif
