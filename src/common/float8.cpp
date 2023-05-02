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

#include <array>

#include "common/bit_cast.hpp"
#include "common/float16.hpp"
#include "common/float8.hpp"

namespace dnnl {
namespace impl {

float8_e5m2_t &float8_e5m2_t::operator=(float f) {
    //TODO(keola): enable more accurate conversion
    // f8_e5m2 is "bottom" of f16, so convert to f16 and truncate
    //    bit     : 0        8
    //    f16     : seeeeemm|mmmmmmmm
    //    f8_e5m2 : seeeeemm|
    float16_t f16 = f;
    // bit cast to array of uint8_t values
    auto iraw = utils::bit_cast<std::array<uint8_t, 2>>(f16);
    // set raw bits to 2nd (little endian) byte: 1s5e2m
    raw_bits_ = iraw[1];
    return *this;
}

float8_e5m2_t::operator float() const {
    std::array<uint8_t, 2> iraw = {{0, raw_bits_}};
    auto f16 = utils::bit_cast<float16_t>(iraw);
    return static_cast<float>(f16);
}

float8_e4m3_t &float8_e4m3_t::operator=(float f) {
    //TODO(keola): enable more accurate conversion
    // f8_e4m3 is nearly a subset of f16, so convert to f16 and truncate
    //    bit     :        8        0
    //    f16     : seeeeemm|mmmmmmmm
    //    f8_e4m3 : seeeemmm|
    //    bit     :        0
    float16_t f16 = f;
    uint16_t u16 = utils::bit_cast<uint16_t>(f16);
    uint16_t s8 = (u16 & 0x8000) >> 8;
    uint16_t e16 = (u16 & 0x7c00) >> 10;
    if (e16 < 8 || 23 < e16) { // if (e8 < 0 || 15 < e8) {
        raw_bits_ = s8 | 0x7f; // overflow to NAN
        return *this;
    }
    uint16_t e8 = (e16 - 8 /* = 7 - 15 = e8_bias - e16_bias */) << 3;
    uint16_t m16 = (u16 & 0x03ff);
    uint16_t m8 = m16 >> 7;
    raw_bits_ = s8 | e8 | m8;
    return *this;
}

float8_e4m3_t::operator float() const {
    uint16_t s16 = (raw_bits_ & 0x80) << 8;
    uint16_t e8 = (raw_bits_ & 0x78) >> 3;
    uint16_t e16 = (e8 + 8 /* 15 - 7 = e16_bias - e8_bias */) << 10;
    uint16_t m8 = (raw_bits_ & 0x7);
    uint16_t m16 = m8 << 7;

    // Need to convert f8_e4m3 denormal into f16 normal.
    if (e8 == 0 && m8 != 0) {
        uint16_t count = 2;
        count = m8 > 0x1 ? 1 : count;
        count = m8 > 0x3 ? 0 : count;
        e16 -= count;
        m16 = (m16 << (count + 1) & 0x7);
    } else if (e8 == 0 && m8 == 0) {
        e16 = 0;
    } /* e8 == 0xf && m == 0x7, when? */

    uint16_t u16 = s16 | e16 | m16;
    auto f16 = utils::bit_cast<float16_t>(u16);
    return static_cast<float>(f16);
}

} // namespace impl
} // namespace dnnl
