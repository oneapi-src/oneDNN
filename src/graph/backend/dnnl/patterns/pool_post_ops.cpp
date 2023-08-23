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

#include "graph/backend/dnnl/kernels/pool.hpp"
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

bool check_avgpool_attributes(op_t *op) {
    return !(op->get_kind() == graph::op_kind::AvgPool
            && op->get_attr<std::string>(graph::op_attr::rounding_type)
                    == "ceil"
            && op->get_attr<bool>(graph::op_attr::exclude_pad) == false);
}

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(pool_post_ops)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, fp_avg_pool)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *avgpool
                            = pgraph->append_op(graph::op_kind::AvgPool);
                    avgpool->append_decision_function(check_avgpool_attributes);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_pooling_fwd>();
        });

/*
                        |
                 [AvgPool/MaxPool]
                        |
                [unary/binary]*[0,4]
                        |
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, fp_pool_post_ops)
        .set_priority(9.9f)
        .set_kind(partition_kind_t::pooling_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto ppool = pgraph->append_alternation(
                            {graph::op_kind::AvgPool, graph::op_kind::MaxPool});
                    ppool->append_decision_function(check_avgpool_attributes);
                    auto post_op_subgraph = std::make_shared<pb_graph_t>();
                    auto palt = post_op_subgraph->append_alternation(
                            get_unary_binary_ops());
                    palt->allow_internal_inputs();
                    post_op_subgraph->create_input_port(0, palt, 0);
                    post_op_subgraph->create_output_port(0, palt, 0);
                    pgraph->append_repetition(post_op_subgraph, {0, 0}, 1,
                            MAX_REPETITION, {in_edge(0, ppool, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_pooling_fwd>();
        });

/*
                        Dequantize
                            |
                    [AvgPool/MaxPool]
                            |
                [StaticReshape/StaticTranspose]*
                            |
                        Quantize
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8_pool_reshape_transpose)
        .set_priority(10.0f)
        .set_kind(partition_kind_t::quantized_pooling_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto pdequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pdequant_data->append_decision_function(
                            check_qtype_equal_to_per_tensor);

                    auto ppool = pgraph->append_alternation(
                            {graph::op_kind::AvgPool, graph::op_kind::MaxPool},
                            {in_edge(0, pdequant_data, 0)});
                    ppool->append_decision_function(check_avgpool_attributes);

                    // (StaticReshape|StaticTranspose)*
                    auto post_op_subgraph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *palt = post_op_subgraph->append_alternation(
                            {graph::op_kind::StaticReshape,
                                    graph::op_kind::StaticTranspose});
                    post_op_subgraph->create_input_port(0, palt, 0);
                    post_op_subgraph->create_output_port(0, palt, 0);
                    auto pop = pgraph->append_optional(post_op_subgraph,
                            in_edges_t {in_edge(0, ppool, 0)});

                    // quantize
                    auto qout = pgraph->append_op(
                            graph::op_kind::Quantize, {in_edge(0, pop, 0)});
                    qout->append_decision_function(
                            check_qtype_equal_to_per_tensor);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_pooling>();
        });

/*
                    Dequantize
                        |
                [AvgPool/MaxPool]
                        |
                [unary/binary]*[0,4]
                        |
                     Quantize
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8_pool_post_ops)
        .set_priority(10.0f)
        .set_kind(partition_kind_t::quantized_pooling_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto pdequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pdequant_data->append_decision_function(
                            check_qtype_equal_to_per_tensor);

                    auto ppool = pgraph->append_alternation(
                            {graph::op_kind::AvgPool, graph::op_kind::MaxPool},
                            {in_edge(0, pdequant_data, 0)});
                    ppool->append_decision_function(check_avgpool_attributes);

                    auto post_op_subgraph = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *pop = post_op_subgraph->append_alternation(
                            get_unary_binary_ops());
                    pop->allow_internal_inputs();
                    post_op_subgraph->create_input_port(0, pop, 0);
                    post_op_subgraph->create_input_port(1, pop, 1);
                    post_op_subgraph->create_output_port(0, pop, 0);

                    auto prep = pgraph->append_repetition(post_op_subgraph,
                            {0, 0}, 0, MAX_REPETITION,
                            in_edges_t {in_edge(0, ppool, 0)});

                    // quantize
                    auto qout = pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, prep, 0)});
                    qout->append_decision_function(
                            check_qtype_equal_to_per_tensor);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_pooling>();
        });

/*
Currently DNNL Backend doesn't support Post-sum/binary with zero points
on GPU, while CPU supports.
matched pattern:
                    Dequantize
                        |
                [AvgPool/MaxPool]   Dequantize
                           \         /
                               Add
                                |
                        [unary/binary]*[0,4]
                                |
                              Quantize
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8_pool_add_post_ops_cpu)
        .set_priority(10.1f)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::quantized_pooling_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto pdequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pdequant_data->append_decision_function(
                            check_qtype_equal_to_per_tensor);

                    auto ppool = pgraph->append_alternation(
                            {graph::op_kind::AvgPool, graph::op_kind::MaxPool},
                            {in_edge(0, pdequant_data, 0)});
                    ppool->append_decision_function(check_avgpool_attributes);

                    // dequantize(rhs) -> add
                    auto postops = post_quantized_add(pgraph, ppool);

                    // quantize
                    auto qout = pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, postops, 0)});
                    qout->append_decision_function(
                            check_qtype_equal_to_per_tensor);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_pooling>();
        });

/*
Currently DNNL Backend doesn't support Post-sum/binary with zero points
on GPU, while CPU supports.
matched pattern:
                    Dequantize
                        |
                (AvgPool|MaxPool)   Dequantize
                           \         /
                               Add
                                |
                        [unary/binary]*[0,4]
                                |
                              Quantize
*/
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, x8_pool_add_post_ops_gpu)
        .set_priority(10.1f)
        .set_engine_kind(engine_kind::gpu)
        .set_kind(partition_kind_t::quantized_pooling_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto pdequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);

                    auto ppool = pgraph->append_alternation(
                            {graph::op_kind::AvgPool, graph::op_kind::MaxPool},
                            {in_edge(0, pdequant_data, 0)});
                    ppool->append_decision_function(check_avgpool_attributes);

                    // dequantize(rhs) -> add
                    auto prep = post_quantized_add(
                            pgraph, ppool, /*check_zps*/ true);

                    // quantize
                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, prep, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_pooling>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
