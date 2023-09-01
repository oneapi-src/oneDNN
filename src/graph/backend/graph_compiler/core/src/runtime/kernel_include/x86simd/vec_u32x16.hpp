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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U32X16_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U32X16_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_u16x16;
#ifdef __AVX512F__
#ifdef __AVX512FP16__
class vec_f16x16;
#endif
class vec_u32x16 {
public:
    union {
        __m512i v;
        uint32_t raw[16];
    } __attribute__((aligned(64)));

    INLINE vec_u32x16() = default;
    INLINE vec_u32x16(uint32_t f) { v = _mm512_set1_epi32(f); }
    INLINE vec_u32x16(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3,
            uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7, uint32_t i8,
            uint32_t i9, uint32_t i10, uint32_t i11, uint32_t i12, uint32_t i13,
            uint32_t i14, uint32_t i15) {
        v = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
                i12, i13, i14, i15);
    }
    INLINE operator vec_u16x16() const;

#ifdef __AVX512FP16__
    INLINE operator vec_f16x16() const;
#endif

    INLINE vec_u32x16(__m512i const &x) { v = x; }

    static INLINE vec_u32x16 load(const uint32_t *p) {
        return _mm512_loadu_si512((const __m512i *)p);
    }
    static INLINE vec_u32x16 load_aligned(const uint32_t *p) {
        return _mm512_load_si512((const __m512i *)p);
    }
    static INLINE vec_u32x16 mask_load(const uint32_t *p, __mmask16 mask) {
        return _mm512_mask_loadu_epi32(vec_u32x16(0).v, mask, p);
    }
    static INLINE void store(vec_u32x16 v, uint32_t *p) {
        _mm512_storeu_si512((__m512i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u32x16 v, uint32_t *p) {
        _mm512_store_si512((__m512i *)p, v.v);
    }
    static INLINE void mask_store(
            uint32_t *p, __mmask16 mask, vec_u32x16 const &a) {
        return _mm512_mask_storeu_epi32(p, mask, a.v);
    }
};

// TODO(xxx): here we use signed integer add operation for u32x16.
// Currently, some of BF16 lowering depends on this func.
INLINE vec_u32x16 operator+(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_add_epi32(a.v, b.v);
}
// INLINE vec_u32x16 operator-(vec_u32x16 const &a, vec_u32x16 const &b) {
//     return _mm512_sub_epi32(a.v, b.v);
// }
// INLINE vec_u32x16 operator-(vec_u32x16 const &a) {
//     return _mm512_sub_epi32(_mm512_setzero_si512(), a.v);
// }
// INLINE vec_u32x16 operator*(vec_u32x16 const &a, vec_u32x16 const &b) {
//     return _mm512_mul_epu32(a.v, b.v);
// }
// INLINE vec_u32x16 operator/(vec_u32x16 const &a, vec_u32x16 const &b) {
//     return _mm512_div_epu32(a.v, b.v);
// }

INLINE vec_u32x16 operator~(vec_u32x16 const &a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(0xFFFFFFFF));
}
INLINE vec_u32x16 operator&(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_and_si512(a.v, b.v);
}
INLINE vec_u32x16 operator|(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_or_si512(a.v, b.v);
}
INLINE vec_u32x16 operator^(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_xor_si512(a.v, b.v);
}

INLINE __mmask16 operator!(vec_u32x16 const &a) {
    return _mm512_cmp_epu32_mask(a.v, _mm512_setzero_si512(), _MM_CMPINT_EQ);
}
INLINE __mmask16 operator==(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask16 operator!=(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask16 operator>(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask16 operator<(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask16 operator>=(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask16 operator<=(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_u32x16 sc_select(
        __mmask16 mask, vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_mask_blend_epi32(mask, b.v, a.v);
}

INLINE vec_u32x16 operator<<(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_sllv_epi32(a.v, b.v);
}
INLINE vec_u32x16 operator>>(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_srlv_epi32(a.v, b.v);
}

INLINE vec_u32x16 sc_max(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_max_epi32(a.v, b.v);
}
INLINE vec_u32x16 sc_min(vec_u32x16 const &a, vec_u32x16 const &b) {
    return _mm512_min_epi32(a.v, b.v);
}
#endif
#endif
