/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_IR_PLAN_HPP
#define GPU_INTEL_JIT_V2_IR_PLAN_HPP

#include "gpu/intel/jit/ir/hw.hpp"
#include "gpu/intel/jit/v2/ir/plan_utils.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"

#include <sstream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

struct reduce_plan_t : public base_plan_t {
    layout_t src;
    layout_t dst;

    using base_plan_t::base_plan_t;

    reduce_plan_t() = default;
    reduce_plan_t(const hw_t &hw, const layout_t &src, const layout_t &dst)
        : base_plan_t(hw), src(src), dst(dst) {}

    int grf_usage_bytes() const {
        int ret = 0;
        ret += utils::rnd_up(dst.size(), grf_size());
        return ret;
    }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << "src_layout: " << src.str() << std::endl;
        oss << "dst_layout: " << dst.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct reorder_plan_t : public base_plan_t {
    layout_t src;
    layout_t dst;

    using base_plan_t::base_plan_t;

    reorder_plan_t() = default;
    reorder_plan_t(const hw_t &hw, const layout_t &src, const layout_t &dst)
        : base_plan_t(hw), src(src), dst(dst) {}

    int grf_usage_bytes() const {
        int ret = 0;
        ret += utils::rnd_up(dst.size(), grf_size());
        return ret;
    }

    std::string str() const {
        if (!*this) return "(empty)";
        std::ostringstream oss;
        oss << "src_layout: " << src.str() << std::endl;
        oss << "dst_layout: " << dst.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
