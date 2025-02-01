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

#include "gpu/intel/jit/v2/conv/plan_registry.hpp"

#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

const char **get_plan_registry_entries();

plan_registry_t::plan_registry_t(const char **entries) {
    while (*entries) {
        plan_registry_t::entry_t e;
        std::istringstream iss(*entries);
        e.parse(iss);
#ifdef DNNL_DEV_MODE
        {
            std::ostringstream oss;
            e.stringify(oss);
            if (oss.str() != *entries) {
                gpu_warning() << "parsed from:\n  " << *entries
                              << "\nstringified to\n  " << oss.str();
            }
        }
#endif
        entries_.push_back(e);
        entries++;
    }
}

kernel_desc_t plan_registry_t::find_best(const problem_t &prb) const {
    kernel_desc_t best;
    float min_time = std::numeric_limits<float>::max();
    for (auto &e : entries_) {
        if (!e.desc.can_fit(prb)) continue;
        float time = e.model_set.time(prb, e.desc);
        if (time < min_time) {
            min_time = time;
            best = e.desc;
        }
        auto desc = to_stream_k(e.desc);
        if (desc.is_empty() || !desc.can_fit(prb)) continue;
        time = e.model_set.time(prb, desc);
        if (time < min_time) {
            min_time = time;
            best = desc;
        }
    }
    return best;
}

void plan_registry_t::stringify(std::ostream &out) const {
    bool is_first = true;
    for (auto &e : entries_) {
        if (!is_first) out << "\n";
        e.stringify(out);
        is_first = false;
    }
}

void plan_registry_t::parse(std::istream &in) {
    entries_.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        entries_.emplace_back();
        jit::parse(line, entries_.back());
    }
}

void plan_registry_t::entry_t::stringify(std::ostream &out) const {
    jit::stringify(out, desc);
    out << " model=";
    jit::stringify(out, model_set);
}

void plan_registry_t::entry_t::parse(std::istream &in) {
    jit::parse(in, desc);
    stream_match(in, "model=");
    jit::parse(in, model_set);
}

std::string plan_registry_t::entry_t::str() const {
    if (is_empty()) return "(empty)";
    std::ostringstream oss;
    oss << ir_utils::add_tag("Desc", desc.str());
    if (!model_set.is_empty()) {
        oss << std::endl;
        oss << ir_utils::add_tag("Model", model_set.str());
    }
    return oss.str();
}

std::string plan_registry_t::entry_t::registry_str() const {
    gpu_assert(!desc.is_empty() && !model_set.is_empty())
            << "Need both descriptor/model for kernel registry";
    std::ostringstream oss;
    jit::stringify(oss, desc);
    oss << " model=";
    model_set.stringify(oss);
    return oss.str();
}

struct plan_registry_instance_t {
    static plan_registry_instance_t &get() {
        static plan_registry_instance_t _instance;
        return _instance;
    }

    plan_registry_instance_t() {
#ifdef DNNL_DEV_MODE
        registry_path = getenv_string_user(env_registry_path_name);
        if (!registry_path.empty()) {
            std::ifstream in(registry_path);
            if (in.good()) {
                registry.parse(in);
                gpu_info() << "Loaded kernel registry from " << registry_path
                           << " with " << registry.size() << " entries";
                return;
            }
        }
#endif
        registry = plan_registry_t(get_plan_registry_entries());
    }

    void dump() const {
        if (registry_path.empty()) return;
        // Serialize to a text file.
        std::ostringstream oss;
        jit::stringify(oss, registry);

        std::ofstream out(registry_path);
        out << oss.str();

        // Serialize to a .cpp file.
        auto cpp_path = registry_path + ".cpp";
        std::vector<std::string> nses;
        nses.emplace_back("dnnl");
        nses.emplace_back("impl");
        nses.emplace_back("gpu");
        nses.emplace_back("intel");
        nses.emplace_back("jit");
        nses.emplace_back("v2");
        nses.emplace_back("conv");
        auto lines = gpu_utils::split(oss.str(), "\n");
        stringify_to_cpp_file(cpp_path, "plan_registry_entries", nses, lines);
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
