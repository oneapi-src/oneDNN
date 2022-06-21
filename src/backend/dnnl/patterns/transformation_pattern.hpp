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

#ifndef BACKEND_DNNL_PATTERNS_TRANSFORMATION_PATTERN_HPP
#define BACKEND_DNNL_PATTERNS_TRANSFORMATION_PATTERN_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "utils/pm/nested_matcher.hpp"
#include "utils/pm/pass_base.hpp"
#include "utils/pm/pbuilder.hpp"

#include "backend/dnnl/patterns/pattern_utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

/*!
 * \brief transformation_pass_t generates an optimized graph
 *        when the pass is hit, it can be op replacements,
 *        dead branch elimination, etc.
 */
class transformation_pass_t : public impl::pass::pass_base {
public:
    explicit transformation_pass_t(std::string pbackend, std::string pname)
        : impl::pass::pass_base(impl::pass::pass_type::kTransformation,
                std::move(pbackend), std::move(pname)) {}

    static impl::pass::pass_base_ptr create(
            std::string pbackend, std::string pname) {
        return std::make_shared<transformation_pass_t>(
                std::move(pbackend), std::move(pname));
    }

    // the criteria of pass execution
    void run(impl::graph_t &agraph) override {
        // we can have only one optimized pattern
        std::vector<impl::pass::FCreateV2Pattern> pfuncs
                = get_attr<impl::pass::FCreateV2Pattern>("FCreateV2Pattern");
        impl::pass::FCreateV2FusedOp optfunc
                = get_attr<impl::pass::FCreateV2FusedOp>("FCreateV2FusedOp")[0];
        std::shared_ptr<op_t> fused_op_ptr = optfunc();
        op_t fused_op = *fused_op_ptr;
        pattern_utils_t pu;
        for (auto &pfunc : pfuncs) {
            std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph
                    = std::make_shared<impl::utils::pm::pb_graph_t>("pgraph");
            pfunc(pgraph);

            // for each pattern. match it
            std::vector<std::vector<op_t *>> fusion_ops;
            pu.match(agraph, pgraph, fusion_ops);
            if (!fusion_ops.empty()) {
                // temporary solution here for showing which pattern matched
                if (impl::utils::getenv_int_user("DUMP", 0) > 0) {
                    printf("onednn_graph_verbose,info,pattern,hit,%s\n",
                            get_pass_name().c_str());
                    fflush(stdout);
                }

                // Only fuse not rewrite. Will remove the fuse once dnnl
                // backend support subgraph mode
                pu.fuse(agraph, fusion_ops, fused_op, get_kind());
            }
        }
    }
};

#define DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN( \
        backend_name, pattern_name) \
    registry.register_pass( \
            #backend_name, #pattern_name, &transformation_pass_t::create)

#define DNNL_BACKEND_REGISTER_PATTERN_DECLARE(pattern_class_) \
    void register_##pattern_class_(impl::pass::pass_registry_t &registry);

#define DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(pattern_class_) \
    void register_##pattern_class_(impl::pass::pass_registry_t &registry) {
#define DNNL_BACKEND_REGISTER_PATTERN_DEF_END }

#define DNNL_BACKEND_REGISTER_PATTERN_CALL(pattern_class_, pattern_registry_) \
    pattern::register_##pattern_class_(pattern_registry_);

#define MAX_REPETITION 4
} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
