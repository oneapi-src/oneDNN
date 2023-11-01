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

#ifndef GPU_JIT_V2_CONV_PLANNER_MODEL_FIT_HPP
#define GPU_JIT_V2_CONV_PLANNER_MODEL_FIT_HPP

#include "gpu/jit/v2/conv/bench_data.hpp"
#include "gpu/jit/v2/conv/model.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

model_t model_fit(const bench_data_t &bd);

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
