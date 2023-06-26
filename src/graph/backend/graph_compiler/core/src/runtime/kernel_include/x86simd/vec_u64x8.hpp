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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U64X8_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U64X8_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
#ifdef __AVX512F__
class vec_u64x8 {
public:
    union {
        __m512i v;
        uint64_t raw[8];
    } ALIGNAS(64);

    INLINE vec_u64x8() = default;
    INLINE vec_u64x8(__m512i const &x) { v = x; }
    static INLINE vec_u64x8 load(const uint64_t *p) {
        return _mm512_loadu_si512((const __m512i *)p);
    }
    static INLINE vec_u64x8 load_aligned(const uint64_t *p) {
        return _mm512_load_si512((const __m512i *)p);
    }

    static INLINE void store(vec_u64x8 v, uint64_t *p) {
        _mm512_storeu_si512((__m512i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u64x8 v, uint64_t *p) {
        _mm512_store_si512((__m512i *)p, v.v);
    }
};

INLINE vec_u64x8 operator+(vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_add_epi64(a.v, b.v);
}

INLINE vec_u64x8 operator-(vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_sub_epi64(a.v, b.v);
}

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
INLINE vec_u64x8 operator*(vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_mullo_epi64(a.v, b.v);
}
#endif

INLINE vec_u64x8 operator~(vec_u64x8 const &a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi8(-1));
}
INLINE vec_u64x8 operator&(vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_and_si512(a.v, b.v);
}
INLINE vec_u64x8 operator|(vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_or_si512(a.v, b.v);
}
INLINE vec_u64x8 operator^(vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_xor_si512(a.v, b.v);
}

#if defined(__AVX512VL__)
INLINE vec_u64x8 sc_max(vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_max_epi64(a.v, b.v);
}
INLINE vec_u64x8 sc_min(vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_min_epi64(a.v, b.v);
}
#endif

INLINE vec_u64x8 sc_unpack_low_vecu64x8_64bits(
        vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_unpacklo_epi64(a.v, b.v);
}

INLINE vec_u64x8 sc_unpack_high_vecu64x8_64bits(
        vec_u64x8 const &a, vec_u64x8 const &b) {
    return _mm512_unpackhi_epi64(a.v, b.v);
}
#define PARAM_U64X8(X) X.v
#define sc_shuffle_vec_u64x8_128bits(a, b, imm8) \
    _mm512_castpd_si512( \
            _mm512_shuffle_f64x2(_mm512_castsi512_pd(PARAM_U64X8(a)), \
                    _mm512_castsi512_pd(PARAM_U64X8(b)), imm8));
#endif
#endif
