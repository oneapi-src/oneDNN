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

#ifndef GPU_JIT_CONV_PLAN_UTILS_HPP
#define GPU_JIT_CONV_PLAN_UTILS_HPP

#include <sstream>
#include <string>

#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/ngen/ngen.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct base_plan_t {
    base_plan_t(const hw_t hw = hw_t()) : hw(hw) {}

    int grf_size() const {
        ir_assert(!hw.is_undef());
        return hw.grf_size();
    }

    hw_t hw;
};

inline std::string add_indent(const std::string &tag, const std::string &s) {
    std::ostringstream oss;
    oss << tag << ":" << std::endl;
    oss << ir_utils::add_indent(s, "  ");
    return oss.str();
}

inline layout_t split(const layout_t &layout, int factor) {
    auto tile = layout.split_exact(factor);
    if (tile.is_empty()) return layout_t();
    return layout.map(tile);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
