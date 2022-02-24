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

#include "utils/pm/pass_base.hpp"

#include "pattern_utils.hpp"

#include "utils/pm/nested_matcher.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

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
        if (has_attr("FCreateOptPattern")) {
            impl::pass::FCreateOptPattern optfunc
                    = get_attr<impl::pass::FCreateOptPattern>(
                            "FCreateOptPattern")[0];
            impl::pass::pattern optimized_pattern;
            optfunc(&optimized_pattern);
            op_t *ostarter = optimized_pattern.get_starter_op();

            // we can have many patterns
            std::vector<impl::pass::FCreatePattern> pfuncs
                    = get_attr<impl::pass::FCreatePattern>("FCreatePattern");

            pattern_utils_t pu;
            for (auto &pfunc : pfuncs) {
                impl::pass::pattern original_pattern;
                pfunc(&original_pattern);
                op_t *pstarter = original_pattern.get_starter_op();
                // for each pattern. match it
                std::vector<std::vector<op_t *>> fusion_ops;
                pu.match(agraph, pstarter, fusion_ops);
                if (!fusion_ops.empty()) {
                    // temporary solution here for showing which pattern matched
                    if (impl::utils::getenv_int_user("DUMP", 0) > 0) {
                        printf("onednn_graph_verbose,info,pattern,hit,%s\n",
                                get_pass_name().c_str());
                        fflush(stdout);
                    }

                    // Only fuse not rewrite. Will remove the fuse once dnnl
                    // backend support subgraph mode
                    pu.fuse(agraph, pstarter, ostarter, fusion_ops);
                }
            }
        } else if (has_attr("FCreateV2FusedOp")) {
            std::vector<impl::pass::FCreateV2Pattern> pfuncs
                    = get_attr<impl::pass::FCreateV2Pattern>(
                            "FCreateV2Pattern");
            impl::pass::FCreateV2FusedOp optfunc
                    = get_attr<impl::pass::FCreateV2FusedOp>(
                            "FCreateV2FusedOp")[0];
            std::shared_ptr<op_t> fused_op_ptr = optfunc();
            op_t fused_op = *fused_op_ptr;
            pattern_utils_t pu;
            for (auto &pfunc : pfuncs) {
                std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph
                        = std::make_shared<impl::utils::pm::pb_graph_t>(
                                "pgraph");
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
                    pu.fuse(agraph, fusion_ops, fused_op);
                }
            }
        }
    }
};

#define DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS( \
        backend_name, pass_class_name) \
    registry.register_pass( \
            #backend_name, #pass_class_name, &transformation_pass_t::create)

#define DNNL_BACKEND_REGISTER_PASSES_DECLARE(passes_class_) \
    void register_##passes_class_(impl::pass::pass_registry_t &registry);

#define DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(passes_class_) \
    void register_##passes_class_(impl::pass::pass_registry_t &registry) {
#define DNNL_BACKEND_REGISTER_PASSES_DEF_END }

#define DNNL_BACKEND_REGISTER_PASSES_CALL(passes_class_, pass_registry_) \
    pass::register_##passes_class_(pass_registry_);

#define MAX_REPETITION 4
} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
