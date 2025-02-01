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

#ifndef GPU_INTEL_JIT_V2_CONV_PLAN_REGISTRY_HPP
#define GPU_INTEL_JIT_V2_CONV_PLAN_REGISTRY_HPP

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

class plan_registry_t {
public:
    struct entry_t {
        kernel_desc_t desc;
        model_set_t model_set;

        entry_t() = default;
        entry_t(const kernel_desc_t &desc, const model_set_t &model_set)
            : desc(desc), model_set(model_set) {}
        bool is_empty() const { return desc.is_empty(); }
        void stringify(std::ostream &out) const;
        void parse(std::istream &in);
        std::string str() const;
        std::string registry_str() const;
        IR_DEFINE_DUMP()
    };

    plan_registry_t() = default;
    plan_registry_t(const char **entries);

    void set(const kernel_desc_t &desc, const model_set_t &model_set) {
        entries_.emplace_back(desc, model_set);
    }
    int size() const { return (int)entries_.size(); }
    kernel_desc_t find_best(const problem_t &prb) const;
    void stringify(std::ostream &out) const;
    void parse(std::istream &out);

public:
    std::vector<entry_t> entries_;
};

const plan_registry_t &const_plan_registry();
plan_registry_t &plan_registry();
void dump_plan_registry();

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
