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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S8X16_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S8X16_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_f32x16;
class vec_s32x16;
class vec_s8x16 {
public:
    union {
        __m128i v;
        int8_t raw[16];
    } __attribute__((aligned(16)));

    INLINE vec_s8x16() = default;
    INLINE vec_s8x16(int8_t f) { v = _mm_set1_epi8(f); }
    INLINE vec_s8x16(int8_t i0, int8_t i1, int8_t i2, int8_t i3, int8_t i4,
            int8_t i5, int8_t i6, int8_t i7, int8_t i8, int8_t i9, int8_t i10,
            int8_t i11, int8_t i12, int8_t i13, int8_t i14, int8_t i15) {
        v = _mm_setr_epi8(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12,
                i13, i14, i15);
    }
    vec_s8x16(__m128i const &x) { v = x; }
    INLINE operator vec_f32x16() const;
    INLINE operator vec_s32x16() const;
    static INLINE vec_s8x16 load(const int8_t *p) {
        return _mm_loadu_si128((const __m128i *)p);
    }
    static INLINE vec_s8x16 load_aligned(const int8_t *p) {
        return _mm_load_si128((const __m128i *)p);
    }
#ifdef __AVX512F__
    static INLINE vec_s8x16 mask_load(const int8_t *p, __mmask16 mask) {
        return _mm_mask_loadu_epi8(vec_s8x16(0).v, mask, p);
    }
#endif
    static INLINE void store(vec_s8x16 v, int8_t *p) {
        _mm_storeu_si128((__m128i *)p, v.v);
    }
    static INLINE void store_aligned(vec_s8x16 v, int8_t *p) {
        _mm_store_si128((__m128i *)p, v.v);
    }
#ifdef __AVX512F__
    static INLINE void mask_store(
            int8_t *p, __mmask16 mask, vec_s8x16 const &a) {
        return _mm_mask_storeu_epi8(p, mask, a.v);
    }
#endif
};

INLINE vec_s8x16 operator+(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_add_epi8(a.v, b.v);
}

INLINE vec_s8x16 operator-(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_sub_epi8(a.v, b.v);
}

INLINE vec_s8x16 operator-(vec_s8x16 const &a) {
    return _mm_sub_epi8(_mm_setzero_si128(), a.v);
}

// INLINE vec_s8x16 operator*(vec_s8x16 const &a, vec_s8x16 const &b) {
//     return _mm_mullo_epi8(a.v, b.v);
// }
// INLINE vec_s8x16 operator/(vec_s8x16 const &a, vec_s8x16 const &b) {
//     return _mm_div_epi8(a.v, b.v);
// }

INLINE vec_s8x16 operator~(vec_s8x16 const &a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(0xFFFFFFFF));
}
INLINE vec_s8x16 operator&(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_and_si128(a.v, b.v);
}
INLINE vec_s8x16 operator|(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_or_si128(a.v, b.v);
}
INLINE vec_s8x16 operator^(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_xor_si128(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask16 operator!(vec_s8x16 const &a) {
    return _mm_cmp_epi8_mask(a.v, _mm_setzero_si128(), _MM_CMPINT_EQ);
}
INLINE __mmask16 operator==(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask16 operator!=(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask16 operator>(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask16 operator<(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask16 operator>=(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask16 operator<=(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_s8x16 sc_select(
        __mmask16 mask, vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_mask_blend_epi8(mask, b.v, a.v);
}
#endif
// INLINE vec_s8x16 operator<<(vec_s8x16 const &a, int16_t b) {
//     return _mm_sll_epi8(a.v, _mm_cvtsi32_si128(b));
// }
// INLINE vec_s8x16 operator>>(vec_s8x16 const &a, int16_t b) {
//     return _mm_sra_epi8(a.v, _mm_cvtsi32_si128(b));
// }

// operator /

INLINE vec_s8x16 sc_unpack_low_vec_s8x16_8bits(
        vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_unpacklo_epi8(a.v, b.v);
}

INLINE vec_s8x16 sc_unpack_low_vec_s8x16_16bits(
        vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_unpacklo_epi16(a.v, b.v);
}

INLINE vec_s8x16 sc_unpack_low_vec_s8x16_32bits(
        vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_unpacklo_epi32(a.v, b.v);
}

INLINE vec_s8x16 sc_unpack_low_vec_s8x16_64bits(
        vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_unpacklo_epi64(a.v, b.v);
}

INLINE vec_s8x16 sc_unpack_high_vec_s8x16_8bits(
        vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_unpackhi_epi8(a.v, b.v);
}

INLINE vec_s8x16 sc_unpack_high_vec_s8x16_16bits(
        vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_unpackhi_epi16(a.v, b.v);
}

INLINE vec_s8x16 sc_unpack_high_vec_s8x16_32bits(
        vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_unpackhi_epi32(a.v, b.v);
}

INLINE vec_s8x16 sc_unpack_high_vec_s8x16_64bits(
        vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_unpackhi_epi64(a.v, b.v);
}

INLINE vec_s8x16 sc_max(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_max_epi8(a.v, b.v);
}
INLINE vec_s8x16 sc_min(vec_s8x16 const &a, vec_s8x16 const &b) {
    return _mm_min_epi8(a.v, b.v);
}
INLINE vec_s8x16 sc_abs(vec_s8x16 const &a) {
    return _mm_abs_epi8(a.v);
}
#endif
