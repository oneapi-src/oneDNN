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

#ifndef COMMON_INT4_HPP
#define COMMON_INT4_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dnnl {
namespace impl {

struct uint4_t {
    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<
                    std::is_integral<IntegerType>::value>::type>
    constexpr uint4_t(IntegerType raw) : raw_bits_(static_cast<uint8_t>(raw)) {
#if __cplusplus >= 201402L
        assert(0 <= raw && raw <= std::numeric_limits<uint8_t>::max());
#endif
    }
    uint4_t(float val_f32) {
        uint8_t val_uint8 = static_cast<uint8_t>(val_f32);
        raw_bits_ = val_uint8 & 0xF;
    }

    operator float() const { return (float)raw_bits_; }

    uint8_t raw_bits_;
};

static_assert(sizeof(uint4_t) == 1, "uint4_t must be 1 byte");

struct int4_t {
    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<
                    std::is_integral<IntegerType>::value>::type>
    constexpr int4_t(IntegerType i) : raw_bits_(static_cast<uint8_t>(i)) {}
    int4_t(float val_f32) {
        int8_t val_int8 = static_cast<int8_t>(val_f32);
        bool negative = val_f32 < 0;
        // positive numbers have the most significant bit set to 0
        // negative numbers have the most significant bit set to 1
        raw_bits_ = negative ? (val_int8 & 0xF) | 0x8 : val_int8 & 0x7;
    }

    operator float() const {
        float sign = (raw_bits_ & (1 << 3)) ? -1.f : 1.f;
        return sign * (float)(sign == -1 ? (~raw_bits_ & 0xF) + 1 : raw_bits_);
    }

    uint8_t raw_bits_;
};

static_assert(sizeof(int4_t) == 1, "int4_t must be 1 byte");

} // namespace impl
} // namespace dnnl

#endif
