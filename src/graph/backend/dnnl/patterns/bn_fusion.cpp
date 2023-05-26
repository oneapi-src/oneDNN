/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#include "graph/backend/dnnl/kernels/batchnorm.hpp"
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

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(bn_fusion)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, bn_relu_fusion)
        .set_priority(8.8f)
        .set_kind(partition_kind_t::batch_norm_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bn = pgraph->append_alternation(
                            std::vector<graph::op_kind_t> {
                                    graph::op_kind::BatchNormInference,
                                    graph::op_kind::BatchNormForwardTraining});
                    bn->append_decision_function(
                            check_input_dtype_from_offset<impl::data_type::f32,
                                    1>);
                    pgraph->append_op(
                            graph::op_kind::ReLU, {in_edge(0, bn, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<batchnorm_fwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, int8_bn_fusion)
        .set_priority(9.f)
        .set_kind(partition_kind_t::batch_norm_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto pdequant_data
                            = pgraph->append_op(graph::op_kind::Dequantize);
                    pdequant_data->append_decision_function(
                            check_qtype_equal_to_per_tensor);
                    pdequant_data->append_decision_function(
                            check_zps_values<0>);
                    pdequant_data->append_decision_function(
                            check_input_dtype<impl::data_type::s8>);

                    auto bn = pgraph->append_op(
                            graph::op_kind::BatchNormInference,
                            {in_edge(0, pdequant_data, 0)});
                    bn->append_decision_function(
                            check_input_dtype_from_offset<impl::data_type::f32,
                                    1>);

                    std::shared_ptr<pb_graph_t> relu_graph;
                    {
                        relu_graph = std::make_shared<pb_graph_t>();
                        auto relu = relu_graph->append_op(graph::op_kind::ReLU);
                        relu_graph->create_input_port(0, relu, 0);
                        relu_graph->create_output_port(0, relu, 0);
                    }
                    auto bn_relu = pgraph->append_optional(
                            relu_graph, {in_edge(0, bn, 0)});

                    auto pquant_data = pgraph->append_op(
                            graph::op_kind::Quantize, {in_edge(0, bn_relu, 0)});
                    pquant_data->append_decision_function(
                            check_qtype_equal_to_per_tensor);
                    pquant_data->append_decision_function(check_zps_values<0>);
                    pquant_data->append_decision_function(
                            check_output_dtype<impl::data_type::s8>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<batchnorm_fwd_t>();
        });

#define BATCHNORM_OUTPUT_NUM_CHECK(n1, n2) \
    append_decision_function([](op_t *graph_op) -> bool { \
        return check_output_num<n1>(graph_op) \
                || check_output_num<n2>(graph_op); \
    })

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, bn_bwd_relu_bwd_fusion)
        .set_priority(8.8f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto relu_bwd
                            = pgraph->append_op(graph::op_kind::ReLUBackward);
                    auto bn_bwd = pgraph->append_op(
                            graph::op_kind::BatchNormTrainingBackward,
                            {in_edge(0, relu_bwd, 0)});
                    bn_bwd->append_decision_function(
                            check_input_dtype_from_offset<impl::data_type::f32,
                                    2>);
                    bn_bwd->BATCHNORM_OUTPUT_NUM_CHECK(1, 3);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<batchnorm_bwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
