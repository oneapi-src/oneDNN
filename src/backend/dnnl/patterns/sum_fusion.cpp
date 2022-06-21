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

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/patterns/fusions.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph = pm::pb_graph_t;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

/*!
 * \brief This provides sum-related fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */

DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(sum_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, sum_fusion)
        .set_priority(10.1f)
        .set_kind(impl::partition_kind::binary_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op_t *add_base
                            = pgraph->append_op(impl::op_kind::Add);
                    add_base->append_decision_function(
                            [](op_t *graph_op) -> bool {
                                return !graph_op->has_attr(
                                               op_attr::auto_broadcast)
                                        || graph_op->get_attr<std::string>(
                                                   op_attr::auto_broadcast)
                                        == "none";
                            });

                    auto addgraph = std::make_shared<pb_graph>();
                    pm::pb_op_t *add = addgraph->append_op(impl::op_kind::Add);
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
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::dnnl_sum);
                    fused_op->set_attr<std::string>(op_attr::backend, "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
