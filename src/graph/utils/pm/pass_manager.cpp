/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"

#include "graph/utils/pm/pass_manager.hpp"

namespace dnnl {
namespace impl {
namespace graph {

using namespace utils;
namespace pass {

bool default_pass_filter(const pass_base_ptr &pass, partition_policy_t policy) {
    // The default pass filter function, which always return true and
    // allows all passes
    UNUSED(pass);
    UNUSED(policy);
    return true;
}

// register a pass
pass_base_t &pass_registry_t::register_pass(const std::string &backend_name,
        const std::string &pass_name, pass_create_fn fn) {
    // create new pass
    auto find = std::find_if(passes_.begin(), passes_.end(),
            [&pass_name](std::list<pass_base_ptr>::value_type &p) -> bool {
                return p->get_pass_name() == pass_name;
            });
    if (find != passes_.end()) {
        return **find;
    } else {
        auto new_pass_ptr = fn(backend_name, pass_name);
        passes_.push_back(new_pass_ptr);
        passes_map_[pass_name] = new_pass_ptr;
        return *new_pass_ptr;
    }
}

pass_base_t &pass_registry_t::register_pass(const pass_base_ptr &pass) {
    passes_.push_back(pass);
    passes_map_[pass->get_pass_name()] = pass;
    return *pass;
}

void pass_registry_t::sort_passes() {
    passes_.sort([](const pass_base_ptr &first, const pass_base_ptr &second) {
        return first->get_priority() > second->get_priority();
    });
}

void pass_manager_t::print_passes(const std::string &pass_config_json) {
    std::ofstream of(pass_config_json);
    assert(of && "can't open file");
    print_passes(&of);
}

void pass_manager_t::print_passes(std::ostream *os) {
    const auto &passes = get_passes();
    json::json_writer_t write(os);
    write.begin_object();
    std::string hash = dnnl_version()->hash;
    std::string version = std::to_string(dnnl_version()->major) + "."
            + std::to_string(dnnl_version()->minor) + "."
            + std::to_string(dnnl_version()->patch);
    write.write_keyvalue("version", version);
    write.write_keyvalue("hash", hash);
    write.write_keyvalue("passes", passes);
    write.end_object();
}

impl::status_t pass_manager_t::run_passes(graph_t &agraph, std::istream *fs,
        partition_policy_t policy, const pass_filter_fn &filter_fn) {
    impl::status_t status = impl::status::success;

    if (*fs) {
        std::list<pass_base_ptr> new_passes;
        json::json_reader_t read(fs);
        json::read_helper_t helper;
        std::string hash;
        std::string version;
        std::list<pass_base_ptr> passes;
        helper.declare_field("hash", &hash);
        helper.declare_field("version", &version);
        helper.declare_field("passes", &passes);
        bool read_json = helper.read_fields(&read);

        // check whether the json string given is valid
        if (read_json && hash == dnnl_version()->hash
                && pass_registry_.get_passes().size() >= passes.size()) {
            // if valid, use passes defined in the given json string
            for (auto &pass : passes) {
                if (pass->get_enable()) {
                    auto &new_pass = get_pass_ptr(pass->get_pass_name());
                    if (typeid(new_pass) == typeid(pass)) {
                        new_pass->set_priority(pass->get_priority());
                        new_passes.push_back(new_pass);
                    }
                }
            }
            new_passes.sort([](const pass_base_ptr &first,
                                    const pass_base_ptr &second) {
                return first->get_priority() > second->get_priority();
            });
            for (auto &pass : new_passes) {
                status = pass->run(agraph);
                if (status != impl::status::success) return status;
                if (agraph.num_unpartitioned_ops() == 0) break;
            }

            return impl::status::success;
        }
        if (read_json) {
            verbose_printf(
                    "graph,warn,pattern,ignore config file for incompatible "
                    "hash id\n");
        } else {
            verbose_printf(
                    "graph,warn,pattern,ignore config file for missing json "
                    "field\n");
        }
    }

    // if no ifstream is given or the string provided is not a valid json,
    // use the passes in the pass manager.
    const std::list<pass_base_ptr> &passes = get_passes();
    for (auto &pass : passes) {
        if (!filter_fn(pass, policy)) continue;
        if (pass->get_enable()) {
            status = pass->run(agraph);
            if (status != impl::status::success) return status;
        }
        if (agraph.num_unpartitioned_ops() == 0) break;
    }

    return impl::status::success;
}

impl::status_t pass_manager_t::run_passes(graph_t &agraph,
        const std::string &pass_config_json, partition_policy_t policy,
        const pass_filter_fn &filter_fn) {
    std::ifstream fs(pass_config_json.c_str());
    return run_passes(agraph, &fs, policy, filter_fn);
}

} // namespace pass
} // namespace graph
} // namespace impl
} // namespace dnnl
