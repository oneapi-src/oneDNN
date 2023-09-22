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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VECTOR_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VECTOR_UTILS_HPP

template <>
INLINE vec_s32x4 sc_round_and_cast(const vec_f32x4 &x) {
    return _mm_cvtps_epi32(x.v);
}

INLINE vec_f32x4::operator vec_s32x4() const {
    return _mm_cvttps_epi32(v);
}

INLINE vec_s32x4::operator vec_f32x4() const {
    return _mm_cvtepi32_ps(v);
}

#ifdef __AVX__
template <>
INLINE vec_s32x8 sc_round_and_cast(const vec_f32x8 &x) {
    return _mm256_cvtps_epi32(x.v);
}

INLINE vec_s32x8::operator vec_f32x8() const {
    return _mm256_cvtepi32_ps(v);
}

INLINE vec_f32x8::operator vec_s32x8() const {
    return _mm256_cvttps_epi32(v);
}
#endif

INLINE vec_f32x4 sc_gather(float const *a, vec_s32x4 const &b) {
#ifdef __AVX512F__
    __m128 dummy = _mm_set1_ps(0.f);
    return _mm_mmask_i32gather_ps(dummy, 0xff, b.v, a, 4);
#else
#ifdef __AVX2__
    return _mm_i32gather_ps(a, b.v, 4);
#else
    return vec_f32x4(a[b.raw[0]], a[b.raw[1]], a[b.raw[2]], a[b.raw[3]]);
#endif
#endif
}

#ifdef __AVX2__
INLINE vec_f32x8 sc_gather(float const *a, vec_s32x8 const &b) {
#ifdef __AVX512F__
    __m256 dummy = _mm256_set1_ps(0.f);
    union {
        int v;
        float v2;
    } mask;
    mask.v = 0xffffffff;
    return _mm256_mask_i32gather_ps(dummy, a, b.v, vec_f32x8(mask.v2).v, 4);
#else
    return _mm256_i32gather_ps(a, b.v, 4);
#endif
}
#define sc_extract_vec_s8x8(a, imm) _mm_extract_epi8(a.v, imm);
#define sc_extract_vec_u8x8(a, imm) _mm_extract_epi8(a.v, imm);
#define sc_extract_vec_s8x16(a, imm) _mm_extract_epi8(a.v, imm);
#define sc_extract_vec_u8x16(a, imm) _mm_extract_epi8(a.v, imm);
#define sc_extract_vec_u16x8(a, imm) _mm_extract_epi16(a.v, imm);
#define sc_extract_vec_s32x8(a, imm) _mm256_extract_epi32(a.v, imm);
#define sc_insert_vec_s8x16(a, imm) _mm_insert_epi8(a.v, b.v, imm);
#define sc_insert_vec_u8x16(a, imm) _mm_insert_epi8(a.v, b.v, imm);
#define sc_insert_vec_u16x8(a, imm) _mm_insert_epi16(a.v, b.v, imm);
#define sc_insert_vec_s32x8(a, imm) _mm256_insert_epi32(a.v, b.v, imm);
#define sc_permutexvar_vec_u8x32_64bits(idx, a) \
    _mm256_permute4x64_epi64(a.v, idx);

#ifndef __AVX512F__
#define sc_extract_vec_s8x32(a, imm) _mm256_extract_epi8(a.v, imm);
#define sc_extract_vec_u8x32(a, imm) _mm256_extract_epi8(a.v, imm);
#define sc_extract_vec_u16x16(a, imm) _mm256_extract_epi16(a.v, imm);
#define sc_insert_vec_s8x32(a, imm) _mm256_insert_epi8(a.v, b.v, imm);
#define sc_insert_vec_u8x32(a, imm) _mm256_insert_epi8(a.v, b.v, imm);
#define sc_insert_vec_u16x16(a, imm) _mm256_insert_epi16(a.v, b.v, imm);
#define sc_permutexvar_vec_u8x64_64bits(idx, a) \
    _mm512_permutexvar_epi64(a.v, idx);

#endif
#endif

#ifdef __AVX512F__
INLINE vec_f32x16 sc_gather(float const *a, vec_s32x16 const &b) {
    return _mm512_i32gather_ps(b.v, a, 4);
}
INLINE vec_u8x16::operator vec_s32x16() const {
    return _mm512_cvtepu8_epi32(v);
}
INLINE vec_s32x16::operator vec_u8x16() const {
    return _mm512_cvtepi32_epi8(v);
}
INLINE vec_u8x16::operator vec_f32x16() const {
    return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v));
}
INLINE vec_f32x16::operator vec_u8x16() const {
    return _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(v));
}
INLINE vec_s8x16::operator vec_s32x16() const {
    return _mm512_cvtepi8_epi32(v);
}
INLINE vec_s32x16::operator vec_s8x16() const {
    return _mm512_cvtepi32_epi8(v);
}
INLINE vec_f32x16::operator vec_s32x16() const {
    return _mm512_cvttps_epi32(v);
}
INLINE vec_s32x16::operator vec_f32x16() const {
    return _mm512_cvtepi32_ps(v);
}
INLINE vec_s8x16::operator vec_f32x16() const {
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v));
}
INLINE vec_f32x16::operator vec_s8x16() const {
    return _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(v));
}
INLINE vec_u32x16::operator vec_u16x16() const {
    return _mm512_cvtepi32_epi16(v);
}
INLINE vec_u16x16::operator vec_u32x16() const {
    return _mm512_cvtepu16_epi32(v);
}
INLINE vec_u16x16::operator vec_s32x16() const {
    return _mm512_cvtepu16_epi32(v);
}
INLINE vec_u32x4::operator vec_u16x4() const {
    return _mm_cvtepi32_epi16(v);
}
INLINE vec_u16x4::operator vec_u32x4() const {
    return _mm_cvtepu16_epi32(v128);
}
INLINE vec_u32x8::operator vec_u16x8() const {
    return _mm256_cvtepi32_epi16(v);
}
INLINE vec_u16x8::operator vec_u32x8() const {
    return _mm256_cvtepu16_epi32(v);
}
#ifdef __AVX512FP16__
INLINE vec_f16x16::operator vec_u16x16() const {
    return _mm256_cvtph_epu16(v);
}
INLINE vec_f16x16::operator vec_f32x16() const {
    return _mm512_cvtxph_ps(v);
}
INLINE vec_f16x8::operator vec_f32x8() const {
    return _mm256_cvtxph_ps(v);
}
INLINE vec_f16x16::operator vec_u32x16() const {
    return _mm512_cvtph_epu32(v);
}
INLINE vec_f16x16::operator vec_s32x16() const {
    return _mm512_cvtph_epi32(v);
}
INLINE vec_f32x16::operator vec_f16x16() const {
    return _mm512_cvtxps_ph(v);
}
INLINE vec_f32x8::operator vec_f16x8() const {
    return _mm256_cvtxps_ph(v);
}
INLINE vec_u16x16::operator vec_f16x16() const {
    return _mm256_cvtepu16_ph(v);
}
INLINE vec_s32x16::operator vec_f16x16() const {
    return _mm512_cvtepi32_ph(v);
}
INLINE vec_u32x16::operator vec_f16x16() const {
    return _mm512_cvtepu32_ph(v);
}
#endif

INLINE vec_u16x32::vec_u16x32(vec_u16x8 const &x) {
    v = _mm512_broadcast_i32x4(x.v);
}
INLINE vec_u16x32::vec_u16x32(vec_u16x8 const &x, int mask) {
    v = _mm512_maskz_broadcast_i32x4(mask, x.v);
}
INLINE vec_u16x32::vec_u16x32(
        vec_u16x8 const &x, int mask, vec_u16x32 const &src) {
    v = _mm512_mask_broadcast_i32x4(src.v, mask, x.v);
}
#ifdef __AVX512DQ__
#define sc_insert_vec_u16x32(a, b, imm) _mm512_inserti32x8(a.v, b.v, imm);
#define sc_insert_vec_s8x64(a, b, imm) _mm512_inserti32x8(a.v, b.v, imm);
#define sc_insert_vec_u8x64(a, b, imm) _mm512_inserti32x8(a.v, b.v, imm);
#define sc_extract_vec_u16x32(a, imm) _mm512_extracti32x8_epi32(a.v, imm);
#define sc_extract_vec_s8x64(a, imm) _mm512_extracti32x8_epi32(a.v, imm);
#define sc_extract_vec_u8x64(a, imm) _mm512_extracti32x8_epi32(a.v, imm);
#endif
#ifdef __AVX512VL__
#define sc_insert_vec_s8x32(a, b, imm) _mm256_inserti32x4(a.v, b.v, imm);
#define sc_insert_vec_u8x32(a, b, imm) _mm256_inserti32x4(a.v, b.v, imm);
#define sc_extract_vec_s8x32(a, imm) _mm256_extracti32x4_epi32(a.v, imm);
#define sc_extract_vec_u8x32(a, imm) _mm256_extracti32x4_epi32(a.v, imm);

#endif
#ifdef __AVX512VBMI__
INLINE vec_u8x64 sc_permutexvar(vec_u8x64 const &a, vec_u8x64 const &b) {
    return _mm512_permutexvar_epi8(a.v, b.v);
}
INLINE vec_s8x64 sc_permutexvar(vec_s8x64 const &a, vec_s8x64 const &b) {
    return _mm512_permutexvar_epi8(a.v, b.v);
}
INLINE vec_u16x32 sc_permutexvar(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_permutexvar_epi16(a.v, b.v);
}
#endif
template <>
INLINE vec_s32x16 sc_round_and_cast(const vec_f32x16 &x) {
    return _mm512_cvtps_epi32(x.v);
}

// saturated cast
template <>
INLINE vec_s8x16 sc_saturated_cast(const vec_s32x16 &x) {
    return _mm512_cvtsepi32_epi8(x.v);
}

template <>
INLINE vec_u8x16 sc_saturated_cast(const vec_s32x16 &x) {
    return _mm512_cvtusepi32_epi8(x.v);
}

#ifdef __AVX512BF16__
INLINE vec_u16x4 tobf16(vec_f32x4 const &x) {
    vec_u16x4 r;
    r.v16 = _mm_cvtneps_pbh(x.v);
    return r;
}
INLINE vec_u16x8 tobf16(vec_f32x8 const &x) {
    vec_u16x8 r;
    r.v16 = _mm256_cvtneps_pbh(x.v);
    return r;
}
INLINE vec_u16x16 tobf16(vec_f32x16 const &x) {
    vec_u16x16 r;
    r.v16 = _mm512_cvtneps_pbh(x.v);
    return r;
}

INLINE uint16_t tobf16(float const &x) {
    uint16_t storage_;
    union caster_t {
        uint32_t vl;
        float vf;
    };
    caster_t caster;
    caster.vf = x;
    uint32_t rounding_bias = ((caster.vl >> 16) & 1) + UINT32_C(0x7FFF);
    storage_ = static_cast<uint16_t>((caster.vl + rounding_bias) >> 16);
    return storage_;
}
#endif
#endif

#endif
