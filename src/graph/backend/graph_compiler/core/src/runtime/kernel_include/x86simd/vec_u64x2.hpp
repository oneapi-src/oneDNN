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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U64X2_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U64X2_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_u64x2 {
public:
    union {
        __m128i v;
        uint64_t raw[2];
    } __attribute__((aligned(16)));

    INLINE vec_u64x2() = default;
    INLINE vec_u64x2(__m128i const &x) { v = x; }
    static INLINE vec_u64x2 load(const uint64_t *p) {
        return _mm_loadu_si128((const __m128i *)p);
    }
    static INLINE vec_u64x2 load_aligned(const uint64_t *p) {
        return _mm_load_si128((const __m128i *)p);
    }

    static INLINE void store(vec_u64x2 v, uint64_t *p) {
        _mm_storeu_si128((__m128i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u64x2 v, uint64_t *p) {
        _mm_store_si128((__m128i *)p, v.v);
    }
};

INLINE vec_u64x2 operator+(vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_add_epi64(a.v, b.v);
}

INLINE vec_u64x2 operator-(vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_sub_epi64(a.v, b.v);
}

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
INLINE vec_u64x2 operator*(vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_mullo_epi64(a.v, b.v);
}
#endif

INLINE vec_u64x2 operator~(vec_u64x2 const &a) {
    return _mm_xor_si128(a.v, _mm_set1_epi8(-1));
}
INLINE vec_u64x2 operator&(vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_and_si128(a.v, b.v);
}
INLINE vec_u64x2 operator|(vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_or_si128(a.v, b.v);
}
INLINE vec_u64x2 operator^(vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_xor_si128(a.v, b.v);
}

#if defined(__AVX512F__) && defined(__AVX512VL__)
INLINE vec_u64x2 sc_max(vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_max_epi64(a.v, b.v);
}
INLINE vec_u64x2 sc_min(vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_min_epi64(a.v, b.v);
}
#endif

INLINE vec_u64x2 sc_unpack_low_vec_u64x2_64bits(
        vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_unpacklo_epi64(a.v, b.v);
}

INLINE vec_u64x2 sc_unpack_high_vec_u64x2_64bits(
        vec_u64x2 const &a, vec_u64x2 const &b) {
    return _mm_unpackhi_epi64(a.v, b.v);
}
#endif
