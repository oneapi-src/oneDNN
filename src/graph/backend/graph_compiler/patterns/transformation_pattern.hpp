/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "graph/utils/pm/nested_matcher.hpp"
#include "graph/utils/pm/pass_base.hpp"
#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {
namespace pass {

/*!
 * \brief transformation_pass matches patterns in the given graph
 *        and creates partitions based on that
 */
class transformation_pass_t : public graph::pass::pass_base {
public:
    explicit transformation_pass_t(std::string pbackend, std::string pname)
        : graph::pass::pass_base(std::move(pbackend), std::move(pname)) {}

    static graph::pass::pass_base_ptr create(
            std::string pbackend, std::string pname) {
        return std::make_shared<transformation_pass_t>(
                std::move(pbackend), std::move(pname));
    }

    graph::status_t run(graph::graph_t &agraph) override {
        // check if current pattern pass can be run on current graph
        engine_kind_t graph_engine_kind = agraph.get_engine_kind();
        if (get_engine_kind() != engine_kind::any_engine
                && get_engine_kind() != graph_engine_kind)
            return impl::status::success;

        std::vector<graph::pass::Pattern> pgraphs
                = get_attr<graph::pass::Pattern>("Pattern");
        pattern_utils_t pu;
        for (const auto &pgraph : pgraphs) {
            // check if min_op_num in the pattern is larger than
            // num_unpartitioned_ops in the graph, if true,
            // no need to run this pattern any more
            if (pgraph->get_min_op_num() > agraph.num_unpartitioned_ops())
                continue;
            // match the given pattern in the graph
            std::vector<std::vector<op_t *>> matched_pairs_list;
            pu.match(agraph, pgraph, matched_pairs_list);
            if (!matched_pairs_list.empty()) {
                // temporary solution here for showing which pattern matched
                if (getenv_int_user("GRAPH_DUMP", 0) > 0
                        || graph::utils::check_verbose_string_user(
                                "DUMP", "pattern")) {
                    printf("onednn_graph_verbose,info,pattern,hit,%s\n",
                            get_pass_name().c_str());
                    fflush(stdout);
                }
                pu.set_partitions(agraph, matched_pairs_list, get_kind(),
                        get_pass_name());
            }
        }
        return graph::status::success;
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
    void register_##passes_class_(graph::pass::pass_registry_t &registry) {
#define COMPILER_BACKEND_REGISTER_PASSES_DEF_END }

#define COMPILER_BACKEND_REGISTER_PASSES_CALL(passes_class_, pass_registry_) \
    pass::register_##passes_class_(pass_registry_);

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
