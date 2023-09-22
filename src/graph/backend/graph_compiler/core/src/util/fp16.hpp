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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_FP16_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_FP16_HPP

#include <cmath>
#include <stdint.h>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// The FP16 datatype implementation, can be cast from/to float
struct fp16_t {
    uint16_t storage_;
    union caster_t {
        uint32_t vl;
        float vf;
    };
    bool operator==(const fp16_t &compare_to) const {
        return storage_ == compare_to.storage_;
    }
    bool operator!=(const fp16_t &compare_to) const {
        return storage_ != compare_to.storage_;
    }

    static uint32_t as_uint(const float x) {
        caster_t ct;
        ct.vf = x;
        return ct.vl;
    }
    static float as_float(const uint32_t x) {
        caster_t ct;
        ct.vl = x;
        return ct.vf;
    }
    static float half_to_float(const uint32_t x) {
        // IEEE-754 16-bit floating-point format (without
        // infinity): 1-5-10, exp-15, +-131008.0,
        // +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
        const uint32_t e = (x & 0x7C00) >> 10; // exponent
        const uint32_t m = (x & 0x03FF) << 13; // mantissa
        const uint32_t v = as_uint((float)m) >> 23;
        // evil log2 bit hack to count leading zeros in
        // denormalized format
        // sign : normalized : denormalized
        return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m)
                | ((e == 0) & (m != 0))
                        * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));
    }
    inline uint16_t float_to_half(const float x) {
        // refrence from
        // https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
        // IEEE-754 16-bit floating-point format (without
        // infinity): 1-5-10, exp-15, +-131008.0,
        // +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
        const uint32_t b = as_uint(x) + 0x00001000;
        // round-to-nearest-even: add last
        // bit after truncated mantissa
        const uint32_t e = (b & 0x7F800000) >> 23; // exponent
        const uint32_t m = b & 0x007FFFFF;
        // mantissa; in line below: 0x007FF000 =
        // 0x00800000-0x00001000 = decimal
        // indicator flag - initial rounding
        // sign : normalized : denormalized : saturate
        return (b & 0x80000000) >> 16
                | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13)
                | ((e < 113) & (e > 101))
                * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1)
                | (e > 143) * 0x7FFF;
    }
    bool is_allbit_1(float f) {
        auto value = as_uint(f);
        return (value == 0x7FFFu) || (value == 0xFFFFu);
    }
    fp16_t(float v) {
        if (is_allbit_1(v)) {
            storage_ = as_uint(v);
        } else {
            auto ret = float_to_half(v);
            storage_ = static_cast<uint16_t>(ret);
        }
    }
    operator float() const {
        auto ret = half_to_float(storage_);
        return ret;
    }
    fp16_t() : storage_(0) {}
    inline static fp16_t from_storage(uint16_t v) {
        fp16_t ret;
        ret.storage_ = v;
        return ret;
    }
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
