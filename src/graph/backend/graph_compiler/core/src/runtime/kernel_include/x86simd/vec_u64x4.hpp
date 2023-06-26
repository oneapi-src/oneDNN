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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U64X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U64X4_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_u64x4 {
public:
    union {
        __m256i v;
        uint64_t raw[4];
    } ALIGNAS(32);

    INLINE vec_u64x4() = default;
    INLINE vec_u64x4(__m256i const &x) { v = x; }
    static INLINE vec_u64x4 load(const uint64_t *p) {
        return _mm256_loadu_si256((const __m256i *)p);
    }
    static INLINE vec_u64x4 load_aligned(const uint64_t *p) {
        return _mm256_load_si256((const __m256i *)p);
    }

    static INLINE void store(vec_u64x4 v, uint64_t *p) {
        _mm256_storeu_si256((__m256i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u64x4 v, uint64_t *p) {
        _mm256_store_si256((__m256i *)p, v.v);
    }
};

INLINE vec_u64x4 operator+(vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_add_epi64(a.v, b.v);
}

INLINE vec_u64x4 operator-(vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_sub_epi64(a.v, b.v);
}

#if defined(__AVX512DQ__) && defined(__AVX512VL__)
INLINE vec_u64x4 operator*(vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_mullo_epi64(a.v, b.v);
}
#endif

INLINE vec_u64x4 operator~(vec_u64x4 const &a) {
    return _mm256_xor_si256(a.v, _mm256_set1_epi8(-1));
}
INLINE vec_u64x4 operator&(vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_and_si256(a.v, b.v);
}
INLINE vec_u64x4 operator|(vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_or_si256(a.v, b.v);
}
INLINE vec_u64x4 operator^(vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_xor_si256(a.v, b.v);
}

#if defined(__AVX512F__) && defined(__AVX512VL__)
INLINE vec_u64x4 sc_max(vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_max_epi64(a.v, b.v);
}
INLINE vec_u64x4 sc_min(vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_min_epi64(a.v, b.v);
}
#endif

INLINE vec_u64x4 sc_unpack_low_vec_u64x4_64bits(
        vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_unpacklo_epi64(a.v, b.v);
}

INLINE vec_u64x4 sc_unpack_high_vec_u64x4_64bits(
        vec_u64x4 const &a, vec_u64x4 const &b) {
    return _mm256_unpackhi_epi64(a.v, b.v);
}
#define PARAM_U64X4(X) X.v
#define sc_permute_vec_u64x4(a, b, imm8) \
    _mm256_castpd_si256( \
            _mm256_permute2f128_ps(_mm256_castsi256_pd(PARAM_U64X4(a)), \
                    _mm256_castsi256_pd(PARAM_U64X4(b)), imm8));
#define sc_shuffle_vec_u64x4_128bits(a, b, imm8) \
    _mm256_castpd_si256( \
            _mm256_shuffle_f64x2(_mm256_castsi256_pd(PARAM_U64X4(a)), \
                    _mm256_castsi256_pd(PARAM_U64X4(b)), imm8));
#endif
