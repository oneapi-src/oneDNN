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

#include "graph/backend/dnnl/kernels/eltwise.hpp"
#include "graph/backend/dnnl/kernels/large_partition.hpp"
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

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(eltwise_fusion)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, eltwise_binary_fusion)
        .set_priority(8.2f)
        .set_kind(partition_kind_t::unary_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *peltwise
                            = pgraph->append_alternation(get_unary_ops());
                    auto pbinary_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pbinary_op = pbinary_graph->append_alternation(
                            get_binary_ops());
                    pbinary_op->allow_internal_inputs();
                    pbinary_graph->create_input_port(0, pbinary_op, 0);
                    pbinary_graph->create_input_port(1, pbinary_op, 1);
                    pbinary_graph->create_output_port(0, pbinary_op, 0);

                    pgraph->append_repetition(pbinary_graph, {0, 0}, 1,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, peltwise, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_eltwise_fwd>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
