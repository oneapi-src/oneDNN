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

#include <cstring>
#include <vector>

#include "backend/pass/pass_base.hpp"
#include "backend/pass/pattern_utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace pass {

pass_base::pass_base(pass_type ptype, std::string pbackend, std::string pname)
    : type_(ptype), backend_(std::move(pbackend)), name_(std::move(pname)) {
    index_ = pass_registry::get()->pass_counter++;
}

void transformation_pass::run(graph &agraph) {
    // we can have only one optimized pattern
    FCreateOptPattern optfunc
            = get_attr<FCreateOptPattern>("FCreateOptPattern")[0];
    pattern optimized_pattern;
    optfunc(&optimized_pattern);
    op_t *ostarter = optimized_pattern.get_starter_op();

    // we can have many patterns
    std::vector<FCreatePattern> pfuncs
            = get_attr<FCreatePattern>("FCreatePattern");

    pattern_utils pu;
    for (auto &pfunc : pfuncs) {
        pattern original_pattern;
        pfunc(&original_pattern);
        op_t *pstarter = original_pattern.get_starter_op();
        // for each pattern. match it
        std::vector<std::vector<op_t *>> fusion_ops;
        pu.match(agraph, pstarter, fusion_ops);
        if (!fusion_ops.empty()) {
            // temporary solution here for showing which pattern matched
            char *val = std::getenv("DNNL_GRAPH_DUMP");
            if (val != nullptr && std::strcmp(val, "1") == 0) {
                std::cout << "hit pass " << get_pass_name() << "\n";
            }
            pu.rewrite(agraph, pstarter, ostarter, fusion_ops);
        }
    }
}

// register a pass
pass_base &pass_registry::register_pass(const std::string &backend_name,
        const std::string &pass_name, pass_create_fn fn) {
    // check whether the backend is valid
    if (!pass_backend_registry::get()->has_backend(backend_name)) {
        pass_backend_registry::get()->register_backend(backend_name);
    }
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

void pass_registry::sort_passes() {
    passes_.sort([](const pass_base_ptr &first, const pass_base_ptr &second) {
        return first->get_priority() > second->get_priority();
    });
}

} // namespace pass
} // namespace impl
} // namespace graph
} // namespace dnnl
