/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef LLGA_BACKEND_PASS_PASS_MANAGER_HPP
#define LLGA_BACKEND_PASS_PASS_MANAGER_HPP

#include <list>
#include <string>

#include "backend/pass/pass_backend.hpp"
#include "backend/pass/pass_base.hpp"
#include "backend/pass/passes/passes.hpp"
#include "utils/json.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace pass {

/*!
 * \brief pass_manager manages the registered passes
 *        as well as backends. It supports pass registration,
 *        pass execution, partition compilation, etc.
 */
class pass_manager {
public:
    pass_manager() { pass_registry::get()->sort_passes(); }

    // get all registered passes
    const std::list<pass_base_ptr> &get_passes() {
        return pass_registry::get()->get_passes();
    }
    // get pass by pass_name
    pass_base_ptr &get_pass_ptr(const std::string &pass_name) {
        return pass_registry::get()->get_pass_ptr(pass_name);
    }
    // print all passes info to json,  e.g. pass name, pass enabled,
    // pass type, pass backend, pass execution priority, etc.
    void print_passes(const std::string &pass_config_json);
    void print_passes(std::ostream *os);

    // run all passes enabled according to passConfig
    void run_passes(graph &agraph, const std::string &pass_config_json);
    void run_passes(graph &agraph, std::istream *fs);
};

} // namespace pass
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
