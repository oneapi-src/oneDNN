/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(softmax_post_ops)

/*     
             softmax
                |
        [unary/binary]*[0,4]
                |
          [typecast_out]*
                |
            [quant_out]*
*/
// This pattern supports optional typecast and quant out, so the output of this
// pattern can be float or int8.
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, fp_softmax_post_ops)
        .set_priority(8.2f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *softmax_base
                            = pgraph->append_op(graph::op_kind::SoftMax);

                    // repetition(alternation(unary | binary))
                    auto alt_unary_binary = std::make_shared<pb_graph_t>();
                    auto palt = alt_unary_binary->append_alternation(
                            get_unary_binary_ops());
                    palt->allow_internal_inputs();
                    alt_unary_binary->create_input_port(0, palt, 0);
                    alt_unary_binary->create_output_port(0, palt, 0);
                    auto prep = pgraph->append_repetition(alt_unary_binary,
                            {0, 0}, 0, MAX_REPETITION,
                            in_edges_t {in_edge(0, softmax_base, 0)});

                    // optional typecast
                    auto tc_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *ptypecast
                            = tc_graph->append_op(graph::op_kind::TypeCast);
                    tc_graph->create_input_port(0, ptypecast, 0);
                    tc_graph->create_output_port(0, ptypecast, 0);
                    auto pre_tc = pgraph->append_optional(
                            tc_graph, in_edges_t {in_edge(0, prep, 0)});

                    // optional quantize
                    auto q_graph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pquantize
                            = q_graph->append_op(graph::op_kind::Quantize);
                    pquantize->append_decision_function(check_zps_values<0>);
                    q_graph->create_input_port(0, pquantize, 0);
                    q_graph->create_output_port(0, pquantize, 0);
                    pgraph->append_optional(
                            q_graph, in_edges_t {in_edge(0, pre_tc, 0)});
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
