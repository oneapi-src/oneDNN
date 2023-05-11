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

namespace {
std::shared_ptr<pb_graph_t> make_typecast_quantize_alt() {
    // alternation(TypeCast | Quantize | (TypeCast + Quantize))
    // Alt0: Typecast
    auto tc_graph = std::make_shared<pb_graph_t>();
    pm::pb_op_t *ptypecast = tc_graph->append_op(graph::op_kind::TypeCast);
    tc_graph->create_input_port(0, ptypecast, 0);
    tc_graph->create_output_port(0, ptypecast, 0);

    // Alt1: Quantize
    auto q_graph = std::make_shared<pb_graph_t>();
    pm::pb_op_t *pquantize = q_graph->append_op(graph::op_kind::Quantize);
    pquantize->append_decision_function(check_zps_values<0>);
    q_graph->create_input_port(0, pquantize, 0);
    q_graph->create_output_port(0, pquantize, 0);

    // Alt2: TypeCast + Quantize
    auto tc_q_graph = std::make_shared<pb_graph_t>();
    pm::pb_op_t *ptc = tc_q_graph->append_op(graph::op_kind::TypeCast);
    pm::pb_op_t *pquant = tc_q_graph->append_op(
            graph::op_kind::Quantize, in_edges_t {in_edge(0, ptc, 0)});
    pquant->append_decision_function(check_zps_values<0>);
    tc_q_graph->create_input_port(0, ptc, 0);
    tc_q_graph->create_output_port(0, pquant, 0);

    auto alt_tc_q = std::make_shared<pb_graph_t>();
    auto palt_0 = alt_tc_q->append_alternation({tc_q_graph, tc_graph, q_graph});
    alt_tc_q->create_input_port(0, palt_0, 0);
    alt_tc_q->create_output_port(0, palt_0, 0);

    return alt_tc_q;
}

void make_softmax_post_ops_pattern(const std::shared_ptr<pb_graph_t> &pgraph) {
    pm::pb_op_t *softmax_base = pgraph->append_op(graph::op_kind::SoftMax);

    // repetition(alternation(unary | binary))
    auto alt_unary_binary = std::make_shared<pb_graph_t>();
    auto palt = alt_unary_binary->append_alternation(get_unary_binary_ops());
    palt->allow_internal_inputs();
    alt_unary_binary->create_input_port(0, palt, 0);
    alt_unary_binary->create_output_port(0, palt, 0);
    auto prep = pgraph->append_repetition(alt_unary_binary, {0, 0}, 0,
            MAX_REPETITION, in_edges_t {in_edge(0, softmax_base, 0)});

    auto alt_tc_q = make_typecast_quantize_alt();
    pgraph->append_optional(alt_tc_q, in_edges_t {in_edge(0, prep, 0)});
}

void make_softmax_tc_q_pattern(const std::shared_ptr<pb_graph_t> &pgraph) {
    pm::pb_op_t *softmax_base = pgraph->append_op(graph::op_kind::SoftMax);

    auto alt_tc_q = make_typecast_quantize_alt();
    pgraph->append_optional(alt_tc_q, in_edges_t {in_edge(0, softmax_base, 0)});
}
} // namespace

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(softmax_fusion)

// currently, softmax + eltwise/binary post-ops is not supported on gpu.
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, softmax_post_ops_fusion_cpu)
        .set_priority(8.2f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>(
                "FCreatePattern", make_softmax_post_ops_pattern)
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<softmax_fwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, softmax_tc_q_fusion)
        .set_priority(8.2f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern", make_softmax_tc_q_pattern)
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<softmax_fwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
