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

#include "gpu/intel/jit/v2/conv/plan_registry.hpp"

#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

const std::vector<uint64_t> &get_plan_registry_data();

void plan_registry_t::merge(const plan_registry_t &other) {
    for (auto &kv : other.entries_) {
        set(kv.first, kv.second);
    }
}

kernel_desc_t plan_registry_t::find_best(const problem_t &prb) const {
    kernel_desc_t best;
    float best_eff = 0;
    for (auto &kv : entries_) {
        auto &desc = kv.first;
        auto &model = kv.second;
        if (!desc.fits(prb)) continue;
        float eff = model.eff(prb);
        if (eff > best_eff) {
            best_eff = eff;
            best = desc;
            best.set_defaults();
        }
    }
    return best;
}

struct plan_registry_instance_t {
    static plan_registry_instance_t &get() {
        static plan_registry_instance_t _instance;
        return _instance;
    }

    plan_registry_instance_t() {
        registry = ir_utils::deserialize_from_data<plan_registry_t>(
                get_plan_registry_data());
#ifdef DNNL_DEV_MODE
        registry_path = getenv_string_user(env_registry_path_name);
        if (!registry_path.empty()) {
            std::ifstream in(registry_path, std::ios::binary);
            if (!in.good()) return;
            plan_registry_t file_registry;
            file_registry.deserialize(in);
            registry.merge(file_registry);
        }
#endif
    }

    void dump() const {
        if (registry_path.empty()) return;
        ir_utils::serialize(registry, registry_path);
        std::vector<std::string> nses;
        nses.emplace_back("dnnl");
        nses.emplace_back("impl");
        nses.emplace_back("gpu");
        nses.emplace_back("jit");
        nses.emplace_back("v2");
        nses.emplace_back("conv");
        ir_utils::serialize_to_file(
                registry, registry_path + ".cpp", "plan_registry_data", nses);
    }

    static const char *env_registry_path_name;
    std::string registry_path;
    plan_registry_t registry;
};

const char *plan_registry_instance_t::env_registry_path_name
        = "GPU_CONV_PLAN_REGISTRY_PATH";

plan_registry_t &plan_registry_impl(bool read_only = true) {
    if (!read_only && plan_registry_instance_t::get().registry_path.empty()) {
        static std::once_flag flag;
        std::call_once(flag, [&] {
            printf("Error: ONEDNN_%s is not set. Exiting...\n",
                    plan_registry_instance_t::env_registry_path_name);
            exit(1);
        });
    }
    return plan_registry_instance_t::get().registry;
}

const plan_registry_t &const_plan_registry() {
    return plan_registry_impl();
}

plan_registry_t &plan_registry() {
    return plan_registry_impl(/*read_only=*/false);
}

void dump_plan_registry() {
    plan_registry_instance_t::get().dump();
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
