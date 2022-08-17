/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_TRANSFORMATION_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_TRANSFORMATION_PATTERN_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pattern_utils.hpp"

#include "utils/pm/nested_matcher.hpp"
#include "utils/pm/pass_base.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {
namespace pass {

/*!
 * \brief transformation_pass matches patterns in the given graph
 *        and creates partitions based on that
 */
class transformation_pass_t : public impl::pass::pass_base {
public:
    explicit transformation_pass_t(std::string pbackend, std::string pname)
        : impl::pass::pass_base(std::move(pbackend), std::move(pname)) {}

    static impl::pass::pass_base_ptr create(
            std::string pbackend, std::string pname) {
        return std::make_shared<transformation_pass_t>(
                std::move(pbackend), std::move(pname));
    }

    impl::status_t run(impl::graph_t &agraph) override {
        std::vector<impl::pass::FCreateV2Pattern> pfuncs
                = get_attr<impl::pass::FCreateV2Pattern>("FCreateV2Pattern");
        pattern_utils_t pu;
        for (auto &pfunc : pfuncs) {
            std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph
                    = std::make_shared<impl::utils::pm::pb_graph_t>("pgraph");
            pfunc(pgraph);

            // match the given pattern in the graph
            std::vector<std::vector<op_t *>> matched_pairs_list;
            pu.match(agraph, pgraph, matched_pairs_list);
            if (!matched_pairs_list.empty()) {
                // temporary solution here for showing which pattern matched
                if (impl::utils::getenv_int_user("DUMP", 0) > 0
                        || impl::utils::check_verbose_string_user(
                                "DUMP", "pattern")) {
                    printf("onednn_graph_verbose,info,pattern,hit,%s\n",
                            get_pass_name().c_str());
                    fflush(stdout);
                }
                pu.set_partitions(agraph, matched_pairs_list, get_kind());
            }
        }
        return impl::status::success;
    }
};

#define DECLARE_PASS_EX(bname, pname, counter) \
    static auto _registered_pass_##pname##_##bname##_##counter##_

#define DECLARE_PASS(bname, pname, counter) \
    DECLARE_PASS_EX(bname, pname, counter)

#define COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS( \
        backend_name, pass_class_name) \
    DECLARE_PASS(backend_name, pass_class_name, __COUNTER__) \
            = registry.register_pass(#backend_name, #pass_class_name, \
                    &transformation_pass_t::create)

#define COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(passes_class_) \
    void register_##passes_class_(impl::pass::pass_registry_t &registry) {
#define COMPILER_BACKEND_REGISTER_PASSES_DEF_END }

#define COMPILER_BACKEND_REGISTER_PASSES_CALL(passes_class_, pass_registry_) \
    pass::register_##passes_class_(pass_registry_);

} // namespace pass
} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
