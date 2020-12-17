/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "backend/pass/pass_manager.hpp"

namespace llga {
namespace impl {

using namespace utils;
namespace pass {
void pass_manager::print_passes(const std::string &pass_config_json) {
    std::ofstream of(pass_config_json);
    assert(of && "can't open file");
    print_passes(&of);
}

void pass_manager::print_passes(std::ostream *os) {
    auto passes = get_passes();
    json::json_writer write(os);
    write.begin_object();
    write.write_keyvalue("passes", passes);
    write.end_object();
}

void pass_manager::run_passes(graph &agraph, std::istream *fs) {
    if (*fs) {
        std::list<pass_base_ptr> new_passes;
        json::json_reader read(fs);
        json::read_helper helper;
        std::list<pass_base_ptr> passes;
        helper.declare_field("passes", &passes);
        helper.read_fields(&read);
        for (auto &pass : passes) {
            if (pass->get_enable()) {
                auto &new_pass = get_pass_ptr(pass->get_pass_name());
                if (typeid(new_pass) == typeid(pass)) {
                    new_pass->set_priority(pass->get_priority());
                    new_passes.push_back(new_pass);
                }
            }
        }
        new_passes.sort(
                [](const pass_base_ptr &first, const pass_base_ptr &second) {
                    return first->get_priority() > second->get_priority();
                });
        for (auto &pass : new_passes) {
            pass->run(agraph);
        }
    } else {
        const std::list<pass_base_ptr> &passes = get_passes();
        for (auto &pass : passes) {
            pass->run(agraph);
        }
    }
}
void pass_manager::run_passes(
        graph &agraph, const std::string &pass_config_json) {
    std::ifstream fs(pass_config_json.c_str());
    run_passes(agraph, &fs);
}

} // namespace pass
} // namespace impl
} // namespace llga
