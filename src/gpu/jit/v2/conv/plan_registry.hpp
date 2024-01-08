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

#ifndef GPU_JIT_V2_CONV_PLAN_REGISTRY_HPP
#define GPU_JIT_V2_CONV_PLAN_REGISTRY_HPP

#include "gpu/jit/utils/utils.hpp"
#include "gpu/jit/v2/conv/kernel_desc.hpp"
#include "gpu/jit/v2/conv/model.hpp"

#include <unordered_map>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

class plan_registry_t {
public:
    void set(const kernel_desc_t &desc, const model_t &model) {
        entries_[desc] = model;
    }
    void merge(const plan_registry_t &other);
    kernel_desc_t find_best(const problem_t &prb) const;
    void serialize(std::ostream &out) const {
        ir_utils::serialize(entries_, out);
    }
    void deserialize(std::istream &in) { ir_utils::deserialize(entries_, in); }

private:
    std::unordered_map<kernel_desc_t, model_t,
            ir_utils::hasher_t<kernel_desc_t>>
            entries_;
};

const plan_registry_t &const_plan_registry();
plan_registry_t &plan_registry();
void dump_plan_registry();

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
