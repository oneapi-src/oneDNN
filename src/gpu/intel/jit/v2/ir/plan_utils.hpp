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

#ifndef GPU_INTEL_JIT_V2_IR_PLAN_UTILS_HPP
#define GPU_INTEL_JIT_V2_IR_PLAN_UTILS_HPP

#include "gpu/intel/jit/ir/hw.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

struct base_plan_t {
    base_plan_t(const hw_t &hw = hw_t()) : hw(hw) {}

    explicit operator bool() const { return !hw.is_undef(); }
    int grf_size() const { return hw.grf_size(); }

    hw_t hw;
};

} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
