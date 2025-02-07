/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_PLANNER_PLANNER_HPP
#define GPU_INTEL_JIT_V2_CONV_PLANNER_PLANNER_HPP

#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"
#include "gpu/intel/jit/v2/conv/model.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

enum class planner_mode_t {
    undef,
    trace,
    bench,
    search,
    auto_search,
};

struct planner_params_t {
    planner_mode_t mode = planner_mode_t::undef;
    kernel_desc_t desc;
    model_set_t model_set;
    parse_result_t parse_result;
};

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
