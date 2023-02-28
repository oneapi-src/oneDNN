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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_BF16_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_BF16_HPP

#include <cmath>
#include <stdint.h>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// The BFloat16 datatype implementation, can be cast from/to float
struct bf16_t {
    uint16_t storage_;
    union caster_t {
        uint32_t vl;
        float vf;
    };
    operator float() const {
        caster_t val;
        val.vl = uint32_t(storage_) << 16;
        return val.vf;
    }
    bool operator==(const bf16_t &compare_to) const {
        return storage_ == compare_to.storage_;
    }
    bool operator!=(const bf16_t &compare_to) const {
        return storage_ != compare_to.storage_;
    }
    bf16_t(float v) {
        if (std::isnan(v)) {
            storage_ = UINT32_C(0x7FC0);
        } else {
            caster_t caster;
            caster.vf = v;
            uint32_t rounding_bias = ((caster.vl >> 16) & 1) + UINT32_C(0x7FFF);
            storage_ = static_cast<uint16_t>((caster.vl + rounding_bias) >> 16);
        }
    }
    bf16_t() : storage_(0) {}
    inline static bf16_t from_storage(uint16_t v) {
        bf16_t ret;
        ret.storage_ = v;
        return ret;
    }
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
