/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include "graph/backend/dnnl/kernels/softmax.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/transformation_pattern.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreateV2Pattern = graph::pass::FCreateV2Pattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(softmax_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, softmax_post_ops_fusion)
        .set_priority(8.2f)
        .set_kind(partition_kind::misc_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *softmax_base
                            = pgraph->append_op(graph::op_kind::SoftMax);

                    // Alt0: Typecast
                    auto ptypecast_graph
                            = std::make_shared<pb_graph_t>("ptypecast_graph");
                    pm::pb_op_t *ptypecast = ptypecast_graph->append_op(
                            graph::op_kind::TypeCast, "typecast");
                    ptypecast_graph->create_input_port(0, ptypecast, 0);
                    ptypecast_graph->create_output_port(0, ptypecast, 0);

                    // Alt1: Quantize
                    auto pquantize_graph
                            = std::make_shared<pb_graph_t>("pquantize_graph");
                    pm::pb_op_t *pquantize = pquantize_graph->append_op(
                            graph::op_kind::Quantize, "quantize");
                    pquantize->append_decision_function(check_zps_values<0>);
                    pquantize_graph->create_input_port(0, pquantize, 0);
                    pquantize_graph->create_output_port(0, pquantize, 0);

                    // Alt2: Typecast + Quantize
                    auto ptcq_graph
                            = std::make_shared<pb_graph_t>("ptcq_graph");
                    pm::pb_op_t *ptc = ptcq_graph->append_op(
                            graph::op_kind::TypeCast, "typecast");
                    pm::pb_op_t *pquant = ptcq_graph->append_op(
                            graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, ptc, 0)}, "quantize");
                    pquant->append_decision_function(check_zps_values<0>);
                    ptcq_graph->create_input_port(0, ptc, 0);
                    ptcq_graph->create_output_port(0, pquant, 0);

                    pgraph->append_alternation(
                            {ptcq_graph, ptypecast_graph, pquantize_graph},
                            in_edges_t {in_edge(0, softmax_base, 0)},
                            "palternation");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<softmax_fwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
