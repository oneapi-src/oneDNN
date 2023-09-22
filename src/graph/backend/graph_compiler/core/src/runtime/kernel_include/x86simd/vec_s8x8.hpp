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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S8X8_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S8X8_HPP
#include <immintrin.h>
#include <stdint.h>
#include <xmmintrin.h>
#include "common.hpp"
class vec_s8x8 {
public:
    union {
        __m128i v;
        int8_t raw[16];
    } __attribute__((aligned(16)));
    INLINE vec_s8x8() = default;
    INLINE vec_s8x8(int8_t f) { v = _mm_set1_epi8(f); }
    INLINE vec_s8x8(int8_t i0, int8_t i1, int8_t i2, int8_t i3, int8_t i4,
            int8_t i5, int8_t i6, int8_t i7) {
        v = _mm_setr_epi8(
                i0, i1, i2, i3, i4, i5, i6, i7, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    vec_s8x8(__m128i const &x) { v = x; }
    static INLINE vec_s8x8 load(const int8_t *p) {
        return _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, ((int8_t *)p)[7],
                ((int8_t *)p)[6], ((int8_t *)p)[5], ((int8_t *)p)[4],
                ((int8_t *)p)[3], ((int8_t *)p)[2], ((int8_t *)p)[1],
                ((int8_t *)p)[0]);
    }
    static INLINE vec_s8x8 load_aligned(const int8_t *p) {
        return _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, ((int8_t *)p)[7],
                ((int8_t *)p)[6], ((int8_t *)p)[5], ((int8_t *)p)[4],
                ((int8_t *)p)[3], ((int8_t *)p)[2], ((int8_t *)p)[1],
                ((int8_t *)p)[0]);
    }
#ifdef __AVX512F__
    static INLINE vec_s8x8 mask_load(const int8_t *p, __mmask8 mask8) {
        __mmask16 mask16 = 0x00FF; // Initialize the 16-bit mask to higher bits
                // 0's (initially unmasked)

        // Manually copy 8-bit mask to the lower 8 bits of the 16-bit mask
        mask16 &= (mask8 & 0xFF);
        return _mm_mask_loadu_epi8(vec_s8x8(0).v, mask16, p);
    }
#endif
    static INLINE void store(vec_s8x8 v, int8_t *p) {
        _mm_storel_pi((__m64 *)p, _mm_castsi128_ps(v.v));
    }
    // static INLINE void store_aligned(vec_s8x8 v, int8_t *p) {
    //     // warning: no aligned store
    //     _mm_storeu_si64((__m64 *)p, v.v);
    // }
#ifdef __AVX512F__
    static INLINE void mask_store(
            int8_t *p, __mmask8 mask8, vec_s8x8 const &a) {
        __mmask16 mask16 = 0x00FF; // Initialize the 16-bit mask to higher bits
                // 0's (initially unmasked)

        // Manually copy 8-bit mask to the lower 8 bits of the 16-bit mask
        mask16 &= (mask8 & 0xFF);
        return _mm_mask_storeu_epi8(p, mask16, a.v);
    }
#endif
};

INLINE vec_s8x8 operator+(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_add_epi8(a.v, b.v);
}

INLINE vec_s8x8 operator-(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_sub_epi8(a.v, b.v);
}

INLINE vec_s8x8 operator-(vec_s8x8 const &a) {
    return _mm_sub_epi8(_mm_setzero_si128(), a.v);
}

// INLINE vec_s8x8 operator*(vec_s8x8 const &a, vec_s8x8 const &b) {
//     return _mm_mullo_epi8(a.v, b.v);
// }
// INLINE vec_s8x8 operator/(vec_s8x8 const &a, vec_s8x8 const &b) {
//     return _mm_div_epi8(a.v, b.v);
// }

INLINE vec_s8x8 operator~(vec_s8x8 const &a) {
    auto xor_helper = _mm_setr_epi8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0, 0, 0, 0, 0, 0, 0, 0);
    return _mm_xor_si128(a.v, xor_helper);
}
INLINE vec_s8x8 operator&(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_and_si128(a.v, b.v);
}
INLINE vec_s8x8 operator|(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_or_si128(a.v, b.v);
}
INLINE vec_s8x8 operator^(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_xor_si128(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask8 operator!(vec_s8x8 const &a) {
    return _mm_cmp_epi8_mask(a.v, _mm_setzero_si128(), _MM_CMPINT_EQ);
}
INLINE __mmask8 operator==(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask8 operator!=(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask8 operator>(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask8 operator<(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask8 operator>=(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask8 operator<=(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_s8x8 sc_select(
        __mmask8 mask8, vec_s8x8 const &a, vec_s8x8 const &b) {
    __mmask16 mask16 = 0x00FF; // Initialize the 16-bit mask to higher bits
            // 0's (initially unmasked)

    // Manually copy 8-bit mask to the lower 8 bits of the 16-bit mask
    mask16 &= (mask8 & 0xFF);
    return _mm_mask_blend_epi8(mask16, b.v, a.v);
}
#endif
// INLINE vec_s8x8 operator<<(vec_s8x8 const &a, int16_t b) {
//     return _mm_sll_epi8(a.v, _mm_cvtsi32_si128(b));
// }
// INLINE vec_s8x8 operator>>(vec_s8x8 const &a, int16_t b) {
//     return _mm_sra_epi8(a.v, _mm_cvtsi32_si128(b));
// }

// operator /

INLINE vec_s8x8 sc_max(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_max_epi8(a.v, b.v);
}
INLINE vec_s8x8 sc_min(vec_s8x8 const &a, vec_s8x8 const &b) {
    return _mm_min_epi8(a.v, b.v);
}
INLINE vec_s8x8 sc_abs(vec_s8x8 const &a) {
    return _mm_abs_epi8(a.v);
}
#endif
