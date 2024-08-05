/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include <array>

#include "common/bit_cast.hpp"
#include "common/dnnl_thread.hpp"
#include "common/float16.hpp"
#include "common/float8.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

float8_e8m0_t &float8_e8m0_t::operator=(float f) {
    // we keep exponent bits and discard sign bit.
    // - support only round to zero to simplify conversion.
    // - no infinities in e8m0, so those become NaN essentially
    uint32_t uf = utils::bit_cast<uint32_t>(f);
    raw_bits_ = (uf >> 23) & 0xff;
    return *this;
}

float8_e8m0_t::operator float() const {
    // no inf, only NaN in e8m0
    // we return Real Indefinite NaN
    if (raw_bits_ == 0xff) return utils::bit_cast<float>(0xffc00000);
    return utils::bit_cast<float>(raw_bits_ << 23);
}

float8_e5m2_t &float8_e5m2_t::operator=(float16_t f) {
    // we just need to apply rounding
    uint16_t fraw = f.raw;
    uint16_t naninf_mask = 0x7c00;

    bool is_special = (fraw & naninf_mask) == naninf_mask;
    bool is_nan = is_special && (fraw & 0x03ff); // one of the lsb is non zero

    // we always set quiet bit for NaN
    if (is_nan) {
        raw_bits_ = (fraw >> 8) | 0x02;
        return *this;
    }

    // if infinity, we just return it as is
    if (is_special) {
        raw_bits_ = fraw >> 8;
        return *this;
    }

    // otherwise we just round and return
    int16_t rounding_nudge = 0x007f + ((fraw & 0x0100) >> 8);
    fraw = fraw + rounding_nudge;
    raw_bits_ = fraw >> 8;
    return *this;
}

float8_e5m2_t &float8_e5m2_t::operator=(float f) {
    float16_t f16 = f;
    float8_e5m2_t f8 = f16;
    raw_bits_ = f8.raw_bits_;
    return *this;
}

float8_e5m2_t::operator float() const {
    float16_t f16 = *this;
    return static_cast<float>(f16);
}

float8_e5m2_t::operator float16_t() const {
    uint16_t snan_mask = 0x7d;
    uint16_t qnan_qbit = 0x02;
    const bool is_snan = (raw_bits_ & snan_mask) == snan_mask;
    const uint8_t raw = is_snan ? raw_bits_ | qnan_qbit : raw_bits_;
    std::array<uint8_t, 2> iraw = {{0, raw}};
    auto f16 = utils::bit_cast<float16_t>(iraw);
    return f16;
}

float8_e4m3_t &float8_e4m3_t::operator=(float16_t f) {
    using namespace utils;
    // Here the idea is to add a large constant to the float16_t to force the
    // proper rounding to f8_e4m3 accuracy.
    int fraw = f.raw;

    // first we extract the sign and make the input positive
    unsigned int s8 = (fraw & 0x8000) >> 8;
    fraw = fraw & 0x7fff;

    // we filter out overflow, nan
    // Note: values in [448;464] round to 448, which is representable
    // So we overflow above 464
    if (fraw > 0x5f40) {
        raw_bits_ = s8 | 0x7f;
        return *this;
    }
    // we filter out underflow when f <= 2^-10
    if (fraw <= 0x1400) {
        raw_bits_ = s8;
        return *this;
    }

    // Compute the rounding shifter by taking its exponent + 7.
    // Lucky us, it does not overflow as fraw <= 448.
    // This allows to discard 7 bits of mantissa during addition,
    // leaving the remaining 3 bits perfectly rounded.
    constexpr bool is_bitcast = true;
    float16_t shifter((fraw & 0x7c00) + 0x1c00, is_bitcast);
    // e8 = e16 - e16_bias + e8_bias = e16 - 15 + 7
    // e8 will be denorm if e8 <= 0 or e16 + 7 < 16
    constexpr int exp_threshold = 0x4000; // raw bits of exponent = 16
    int is_denorm = shifter.raw < exp_threshold;
    if (is_denorm) shifter.raw = exp_threshold;

    float16_t rounded = (float16_t(fraw, is_bitcast) + shifter);
    // separate line to force line above to round to f16
    rounded = rounded - shifter;

    int e8 = ((rounded.raw & 0x7c00) >> 10) - 8;
    uint8_t m8 = (rounded.raw & 0x03ff) >> 7;

    // we need to make the implicit f32 mantissa bit explicit for
    // denorm f8_e4m3
    if (is_denorm) {
        m8 = (m8 | 0x00000008) >> (-e8 + 1);
        e8 = 0;
    }

    raw_bits_ = s8 | (e8 << 3) | m8;
    return *this;
}

float8_e4m3_t &float8_e4m3_t::operator=(float f) {
    float16_t f16 = f;
    float8_e4m3_t f8 = f16;
    raw_bits_ = f8.raw_bits_;
    return *this;
}

float8_e4m3_t::operator float() const {
    float16_t f16 = *this;
    return static_cast<float>(f16);
}

float8_e4m3_t::operator float16_t() const {
    const uint16_t s8 = (raw_bits_ & 0x80) >> 7;
    const uint16_t e8 = (raw_bits_ & 0x78) >> 3;
    const uint16_t m8 = (raw_bits_ & 0x7);
    uint16_t s16 = s8;
    uint16_t e16 = e8 + 8; // 8 = 15 - 7 = f16_bias - f8_e4m3_bias
    uint16_t m16 = m8;

    // Need to convert f8_e4m3 denormal into f16 normal.
    if (e8 == 0 && m8 != 0) {
        uint16_t count = 2;
        count = m8 > 0x1 ? 1 : count;
        count = m8 > 0x3 ? 0 : count;
        e16 -= count;
        m16 = (m16 << (count + 1)) & 0x7;
    } else if (e8 == 0 && m8 == 0) {
        // set correct exponent for zero
        e16 = 0;
    } else if (e8 == 0xf && m8 == 0x7) {
        // set correct exponent and mantissa for NaN input
        e16 = 0x1f;
        m16 = 0x4; // Real Indefinite (a qNaN)
    }
    s16 <<= 15;
    e16 <<= 10;
    m16 <<= 7;

    const uint16_t u16 = s16 | e16 | m16;
    return utils::bit_cast<float16_t>(u16);
}

void cvt_f8_e5m2_to_float(float *out, const float8_e5m2_t *inp, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = inp[i];
}

void cvt_f8_e4m3_to_float(float *out, const float8_e4m3_t *inp, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = inp[i];
}

void cvt_float_to_f8_e5m2(float8_e5m2_t *out, const float *inp, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float8_e5m2_t>(inp[i]);
}

void cvt_float_to_f8_e4m3(float8_e4m3_t *out, const float *inp, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float8_e4m3_t>(inp[i]);
}

void add_floats_and_cvt_to_f8_e5m2(float8_e5m2_t *out, const float *inp0,
        const float *inp1, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float8_e5m2_t>(inp0[i] + inp1[i]);
}

void add_floats_and_cvt_to_f8_e4m3(float8_e4m3_t *out, const float *inp0,
        const float *inp1, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float8_e4m3_t>(inp0[i] + inp1[i]);
}

} // namespace impl
} // namespace dnnl
