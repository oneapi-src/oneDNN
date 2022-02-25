/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "backend/dnnl/patterns/fusions.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using pattern = impl::pass::pattern;
using FCreatePattern = impl::pass::FCreatePattern;
using FCreateOptPattern = impl::pass::FCreateOptPattern;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

/*!
 * \brief This provides convtranspose-related fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(convtranspose_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, convtranspose_bias_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *convtranspose
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    convtranspose->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    bias->fill_and_connect_input(0, *convtranspose, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *convtranspose
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    UNUSED(convtranspose);
                    convtranspose->set_attr<int64_t>("num_inputs", 3);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, convtranspose_add_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *convtranspose
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    convtranspose->set_attr<int64_t>("num_inputs", 2);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    add->fill_and_connect_input(0, *convtranspose, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, convtranspose_bias_add_fusion)
        .set_priority(10.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *convtranspose
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    convtranspose->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    bias->fill_and_connect_input(0, *convtranspose, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *convtranspose
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    convtranspose->set_attr<int64_t>("num_inputs", 3);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    add->fill_and_connect_input(0, *convtranspose, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, convtranspose_relu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *convtranspose
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    convtranspose->set_attr<int64_t>("num_inputs", 2);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    relu->fill_and_connect_input(0, *convtranspose, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, convtranspose_bias_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *convtranspose
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    convtranspose->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    bias->fill_and_connect_input(0, *convtranspose, 0);
                    relu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *convtranspose
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    convtranspose->set_attr<int64_t>("num_inputs", 3);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    relu->fill_and_connect_input(0, *convtranspose, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_convtranspose_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *deconv
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    deconv->set_attr<int64_t>("num_inputs", 2);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    deconv->fill_and_connect_input(0, *dequant_data, 0);
                    deconv->fill_and_connect_input(1, *dequant_weight, 0);
                    quant->fill_and_connect_input(0, *deconv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::quantized_convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_convtranspose_bias_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *deconv
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    deconv->set_attr<int64_t>("num_inputs", 3);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    deconv->fill_and_connect_input(0, *dequant_data, 0);
                    deconv->fill_and_connect_input(1, *dequant_weight, 0);
                    quant->fill_and_connect_input(0, *deconv, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *deconv
                            = apattern->create_op(impl::op_kind::ConvTranspose);
                    deconv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    deconv->fill_and_connect_input(0, *dequant_data, 0);
                    deconv->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *deconv, 0);
                    quant->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::quantized_convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_convtranspose_eltwise_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *deconv
                            = pgraph->append_op(impl::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    deconv->append_decision_function(check_input_num<2>);

                    auto eltwise = pgraph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Elu,
                                    impl::op_kind::Exp, impl::op_kind::GELU,
                                    impl::op_kind::HardTanh, impl::op_kind::Log,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sigmoid, impl::op_kind::Sqrt,
                                    impl::op_kind::Square, impl::op_kind::Tanh},
                            in_edges_t {in_edge(0, deconv, 0)});

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, eltwise, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::quantized_convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_convtranspose_bias_eltwise_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *deconv
                            = pgraph->append_op(impl::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    deconv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, deconv, 0)});

                    auto eltwise = pgraph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Elu,
                                    impl::op_kind::Exp, impl::op_kind::GELU,
                                    impl::op_kind::HardTanh, impl::op_kind::Log,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sigmoid, impl::op_kind::Sqrt,
                                    impl::op_kind::Square, impl::op_kind::Tanh},
                            in_edges_t {in_edge(0, bias, 0)});

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, eltwise, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *deconv
                            = pgraph->append_op(impl::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    deconv->append_decision_function(check_input_num<3>);

                    auto eltwise = pgraph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Elu,
                                    impl::op_kind::Exp, impl::op_kind::GELU,
                                    impl::op_kind::HardTanh, impl::op_kind::Log,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sigmoid, impl::op_kind::Sqrt,
                                    impl::op_kind::Square, impl::op_kind::Tanh},
                            in_edges_t {in_edge(0, deconv, 0)});

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, eltwise, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::quantized_convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_convtranspose_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");
                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *deconv
                            = pgraph->append_op(impl::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pdeconv");
                    deconv->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, deconv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, add, 0)}, "pquant");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::quantized_convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_convtranspose_bias_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");
                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *deconv
                            = pgraph->append_op(impl::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pdeconv");
                    deconv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, deconv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, add, 0)}, "pquant");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");
                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *deconv
                            = pgraph->append_op(impl::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pdeconv");
                    deconv->append_decision_function(check_input_num<3>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, deconv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, add, 0)}, "pquant");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::quantized_convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
