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
#include "backend/dnnl/patterns/transformation_pattern.hpp"
#include "backend/dnnl/patterns/utils.hpp"

#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

namespace {
bool check_attributes(op_t *graph_op) {
    if (graph_op->has_attr(op_attr::axes)
            && graph_op->get_attr<std::vector<int64_t>>(op_attr::axes).empty())
        return false;
    return true;
}
} // namespace

/*!
 * \brief This provides reduction fusion.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(reduction_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, reduction_post_ops_fusion)
        .set_priority(8.4f)
        .set_kind(impl::partition_kind::reduction_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *reduction = pgraph->append_alternation(
                            {impl::op_kind::ReduceL1, impl::op_kind::ReduceL2,
                                    impl::op_kind::ReduceMax,
                                    impl::op_kind::ReduceMean,
                                    impl::op_kind::ReduceMin,
                                    impl::op_kind::ReduceProd,
                                    impl::op_kind::ReduceSum});
                    reduction->append_decision_function(check_attributes);

                    auto postop_graph
                            = std::make_shared<pb_graph_t>("postop_graph");
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            get_unary_binary_ops(), "pother_postop");
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    pgraph->append_repetition(postop_graph, {0, 0}, 1,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, reduction, 0)},
                            "prepetition");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::reduction_post_ops_fusion);
                    fused_op->set_attr<std::string>(op_attr::backend, "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
