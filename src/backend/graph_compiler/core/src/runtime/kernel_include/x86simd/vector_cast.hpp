/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VECTOR_CAST_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VECTOR_CAST_HPP

template <>
INLINE vec_s32x4 sc_round_and_cast(const vec_f32x4 &x) {
    return _mm_cvtps_epi32(x.v);
}

template <>
INLINE vec_s32x8 sc_round_and_cast(const vec_f32x8 &x) {
    return _mm256_cvtps_epi32(x.v);
}

INLINE vec_f32x4::operator vec_s32x4() const {
    return _mm_cvttps_epi32(v);
}

INLINE vec_s32x8::operator vec_f32x8() const {
    return _mm256_cvtepi32_ps(v);
}

INLINE vec_f32x8::operator vec_s32x8() const {
    return _mm256_cvttps_epi32(v);
}

#ifdef __AVX512F__
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
