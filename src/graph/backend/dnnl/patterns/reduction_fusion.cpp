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
#include "graph/backend/dnnl/kernels/reduction.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

namespace {
bool check_attributes(op_t *graph_op) {
    if (graph_op->has_attr(op_attr::axes)
            && graph_op->get_attr<std::vector<int64_t>>(op_attr::axes).empty())
        return false;
    return true;
}
} // namespace

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(reduction_fusion)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, reduction_post_ops_fusion)
        .set_priority(8.4f)
        .set_kind(partition_kind_t::reduction_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *reduction = pgraph->append_alternation(
                            {graph::op_kind::ReduceL1, graph::op_kind::ReduceL2,
                                    graph::op_kind::ReduceMax,
                                    graph::op_kind::ReduceMean,
                                    graph::op_kind::ReduceMin,
                                    graph::op_kind::ReduceProd,
                                    graph::op_kind::ReduceSum});
                    reduction->append_decision_function(check_attributes);

                    auto postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            get_unary_binary_ops());
                    pop->allow_internal_inputs();
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    pgraph->append_repetition(postop_graph, {0, 0}, 0,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, reduction, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_reduction>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
