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

#include "graph/backend/dnnl/kernels/resampling.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

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
bool check_attributes(op_t *op) {
    return op->get_attr<std::string>(op_attr::coordinate_transformation_mode)
            == std::string("half_pixel");
}
} // namespace

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(interpolate_fusion)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, interpolate_post_ops_fusion)
        .set_priority(8.4f)
        .set_kind(partition_kind_t::interpolate_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *interpolate
                            = pgraph->append_op(graph::op_kind::Interpolate);
                    interpolate->append_decision_function(check_attributes);

                    auto postop_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = postop_graph->append_alternation(
                            get_unary_binary_ops());
                    pop->allow_internal_inputs();
                    postop_graph->create_input_port(0, pop, 0);
                    postop_graph->create_input_port(1, pop, 1);
                    postop_graph->create_output_port(0, pop, 0);

                    pgraph->append_repetition(postop_graph, {0, 0}, 1,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, interpolate, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<resampling_fwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
