/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/kernels/sum.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(sum_fusion)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, sum_fusion)
        .set_priority(8.1f)
        .set_kind(partition_kind_t::binary_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *add_base
                            = pgraph->append_op(graph::op_kind::Add);
                    add_base->append_decision_function(
                            [](op_t *graph_op) -> bool {
                                return !graph_op->has_attr(
                                               op_attr::auto_broadcast)
                                        || graph_op->get_attr<std::string>(
                                                   op_attr::auto_broadcast)
                                        == "none";
                            });

                    auto addgraph = std::make_shared<pb_graph>();
                    pm::pb_op_t *add = addgraph->append_op(graph::op_kind::Add);
                    add->append_decision_function([](op_t *graph_op) -> bool {
                        return !graph_op->has_attr(op_attr::auto_broadcast)
                                || graph_op->get_attr<std::string>(
                                           op_attr::auto_broadcast)
                                == "none";
                    });
                    addgraph->create_input_port(0, add, 0);
                    addgraph->create_input_port(1, add, 1);
                    addgraph->create_output_port(0, add, 0);

                    pgraph->append_repetition(addgraph, {0, 0}, 1,
                            MAX_REPETITION, {in_edge(0, add_base, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel",
                []() -> kernel_ptr { return std::make_shared<sum_t>(); });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
