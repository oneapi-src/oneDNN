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

#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

namespace {
template <size_t N>
bool check_input_num(op_t *op) {
    return op->num_inputs() == N;
}

bool check_input_all_s8(op_t *op) {
    for (size_t i = 0; i < op->num_inputs(); ++i) {
        logical_tensor_t iport = op->get_input_value(i)->get_logical_tensor();
        if (iport.data_type != impl::data_type::s8) return false;
    }

    return true;
}
} // namespace

/*!
 * \brief This provides conv-related fusion, i.e.
 *        conv-relu fusion, conv-bn fusion, conv-sum fusion, conv-bn-sum fusion, etc.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(conv_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_relu_fusion)
        .set_priority(10.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_relu6_fusion)
        .set_priority(10.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv = pgraph->append_op(
                            impl::op_kind::Convolution, "p-conv");
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)}, "p-add");
                    add->allow_internal_inputs({0, 1});

                    pm::pb_op *relu6
                            = pgraph->append_op(impl::op_kind::HardTanh,
                                    in_edges_t {in_edge(0, add, 0)}, "p-relu6");
                    relu6->set_attr<float>("min", 0.f);
                    relu6->set_attr<float>("max", 6.f);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_add_relu6);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_elu_fusion)
        .set_priority(10.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::Elu,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_add_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_fusion)
        .set_priority(10.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->allow_internal_inputs({0, 1});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bn_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_relu6_fusion)
        .set_priority(10.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0)});
                    add->allow_internal_inputs({0, 1});

                    pm::pb_op *relu6
                            = pgraph->append_op(impl::op_kind::HardTanh,
                                    in_edges_t {in_edge(0, add, 0)});
                    relu6->set_attr<float>("min", 0.f);
                    relu6->set_attr<float>("max", 6.f);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->allow_internal_inputs({0, 1});

                    pm::pb_op *relu6
                            = pgraph->append_op(impl::op_kind::HardTanh,
                                    in_edges_t {in_edge(0, add, 0)});
                    relu6->set_attr<float>("min", 0.f);
                    relu6->set_attr<float>("max", 6.f);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bias_add_relu6);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, bias, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bias_bn_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_sum_fusion)
        .set_priority(9.9f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->allow_internal_inputs({0, 1});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bn_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_sum_fusion)
        .set_priority(9.9f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, bias, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->allow_internal_inputs({0, 1});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->allow_internal_inputs({0, 1});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_bn_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bn, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bn_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, bias, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bn, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bn, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bias_bn_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_elu_fusion)
        .set_priority(10.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::Elu,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::Elu,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bias_add_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_relu_fusion)
        .set_priority(10.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bias_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_relu6_fusion)
        .set_priority(9.9f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *relu6
                            = pgraph->append_op(impl::op_kind::HardTanh,
                                    in_edges_t {in_edge(0, bias, 0)});
                    relu6->set_attr<float>("min", 0.f);
                    relu6->set_attr<float>("max", 6.f);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *relu6
                            = pgraph->append_op(impl::op_kind::HardTanh,
                                    in_edges_t {in_edge(0, conv, 0)});
                    relu6->set_attr<float>("min", 0.f);
                    relu6->set_attr<float>("max", 6.f);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_relu6);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_depthwise_fusion)
        .set_priority(10.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *depthwise
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, conv, 0)});
                    depthwise->append_decision_function(check_input_num<2>);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::dnnl_conv_depthwise);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_conv_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_conv);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_quant_wei_conv_fusion)
        .set_priority(10.6f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "pdequant_data");
                    pm::pb_op *quant_weight = pgraph->append_op(
                            impl::op_kind::Quantize, "pquant_weight");
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)},
                                    "pdequant+weight");
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_quant_wei_conv);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_conv_bias_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_conv_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_conv_relu_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_conv_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_quant_wei_conv_relu_fusion)
        .set_priority(10.6f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_quant_wei_conv_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_conv_bias_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");
                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)}, "pbias");
                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");
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
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<3>);
                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");
                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, add, 0)}, "pquant");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_conv_bias_relu_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bias, 0)});

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_conv_bias_add_relu_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");

                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)}, "pbias");

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)}, "prelu");

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)}, "pquant");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");

                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)}, "prelu");

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)}, "pquant");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_bias_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_conv_bias_add_relu_fusion)
        .set_priority(10.6f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_quant_wei_conv_bias_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_conv_bias_relu_fusion)
        .set_priority(10.6f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bias, 0)});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_quant_wei_conv_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_quant_wei_conv_bias_fusion)
        .set_priority(10.6f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_quant_wei_conv_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_conv_add_relu_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");
                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)}, "prelu");
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)}, "pquant");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_conv_add_relu_fusion)
        .set_priority(10.6f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_quant_wei_conv_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_conv_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::x8s8f32_conv);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_conv_bias_fusion)
        .set_priority(9.9f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<3>);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);

                    pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)}, "pbias");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_conv_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_conv_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)}, "prelu");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_conv_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_conv_bias_relu_fusion)
        .set_priority(10.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)}, "prelu");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)}, "pbias");
                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bias, 0)}, "prelu");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_conv_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_conv_bias_add_relu_fusion)
        .set_priority(10.4f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)}, "pbias");

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)}, "prelu");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)}, "prelu");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_conv_bias_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_conv_add_relu_fusion)
        .set_priority(10.3f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)}, "prelu");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_conv_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_quant_wei_conv_fusion)
        .set_priority(9.9f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_quant_wei_conv);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_conv_bias_fusion)
        .set_priority(10.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<3>);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_quant_wei_conv_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_conv_relu_fusion)
        .set_priority(10.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_quant_wei_conv_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_conv_bias_relu_fusion)
        .set_priority(10.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_quant_wei_conv_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_conv_bias_add_relu_fusion)
        .set_priority(10.3f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_quant_wei_conv_bias_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_conv_add_relu_fusion)
        .set_priority(10.4f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->append_decision_function(
                            check_input_all_s8);
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::x8s8f32_quant_wei_conv_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_fusion)
        .set_priority(10.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0)});
                    add->allow_internal_inputs({0, 1});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->allow_internal_inputs({0, 1});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_elu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Elu,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::Elu,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sigmoid_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Sigmoid,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::Sigmoid,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bias_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_swish_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *sigmoid
                            = pgraph->append_op(impl::op_kind::Sigmoid,
                                    in_edges_t {in_edge(0, bias, 0)});

                    pgraph->append_op(impl::op_kind::Multiply,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, sigmoid, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv = pgraph->append_op(
                            impl::op_kind::Convolution, "p-conv");
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *sigmoid
                            = pgraph->append_op(impl::op_kind::Sigmoid,
                                    in_edges_t {in_edge(0, conv, 0)}, "p-sig");

                    pgraph->append_op(impl::op_kind::Multiply,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, sigmoid, 0)},
                            "p-mul");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_swish);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pgraph->append_op(impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bn);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_bn);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_relu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_relu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_hardtanh_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::HardTanh,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::HardTanh,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bias_hardtanh);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_square_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Square,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::Square,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_square);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_tanh_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Tanh,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::Tanh,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_tanh);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_abs_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Abs,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::Abs,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_abs);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sqrt_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Sqrt,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    pgraph->append_op(impl::op_kind::Sqrt,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_sqrt);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_fusion)
        .set_priority(9.7f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);
                    pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_pass)
        .set_priority(9.7f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pgraph->append_op(impl::op_kind::Convolution);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            impl::op_kind::Convolution);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bwd_f_biasadd_bwd_fusion)
        .set_priority(9.7f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *convbpf = pgraph->append_op(
                            impl::op_kind::ConvolutionBackpropFilters);
                    pgraph->append_op(impl::op_kind::BiasAddBackprop,
                            in_edges_t {in_edge(0, convbpf, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bwd_f_biasadd_bwd);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_simple_resblock_fusion)
        .set_priority(5.f) // low priority to avoid current functionality
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv0
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv0->append_decision_function(check_input_num<2>);
                    pm::pb_op *relu0 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv0, 0)});

                    pm::pb_op *conv1
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, relu0, 0)});
                    conv1->append_decision_function(check_input_num<2>);
                    pm::pb_op *relu1 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv1, 0)});

                    pm::pb_op *conv2
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, relu1, 0)});
                    conv2->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv2, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_simple_resblock);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_conv_bias_relu_conv_bias_relu_conv_bias_add_relu_fusion)
        // .set_priority(10.7f)
        .set_priority(5.f) // low priority to not break current integration
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    // int8_conv_bias_relu
                    pm::pb_op *dequant_src0
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *dequant_wei0
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *conv0
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_src0, 0),
                                            in_edge(1, dequant_wei0, 0)});
                    conv0->append_decision_function(check_input_num<3>);
                    pm::pb_op *relu0 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv0, 0)});
                    pm::pb_op *quant_dst0
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, relu0, 0)});

                    // int8_conv_bias_relu
                    pm::pb_op *dequant_src1
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_dst0, 0)});
                    pm::pb_op *dequant_wei1
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *conv1
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_src1, 0),
                                            in_edge(1, dequant_wei1, 0)});
                    conv1->append_decision_function(check_input_num<3>);
                    pm::pb_op *relu1 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv1, 0)});
                    pm::pb_op *quant_dst1
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, relu1, 0)});

                    // int8_conv_bias_add_relu
                    pm::pb_op *dequant_src2
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_dst1, 0)});
                    pm::pb_op *dequant_wei2
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *conv2
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_src2, 0),
                                            in_edge(1, dequant_wei2, 0)});
                    conv2->append_decision_function(check_input_num<3>);
                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv2, 0),
                                    in_edge(1, dequant_other, 0)});
                    pm::pb_op *relu2 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu2, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_conv_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_conv_bias_relu_conv_bias_relu_fusion)
        // .set_priority(10.7f)
        .set_priority(5.f) // low priority to not break current integration
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    // int8_conv_bias_relu
                    pm::pb_op *dequant_src0
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *dequant_wei0
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *conv0
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_src0, 0),
                                            in_edge(1, dequant_wei0, 0)});
                    conv0->append_decision_function(check_input_num<3>);
                    pm::pb_op *relu0 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv0, 0)});
                    pm::pb_op *quant_dst0
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, relu0, 0)});

                    // int8_conv_bias_relu
                    pm::pb_op *dequant_src1
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_dst0, 0)});
                    pm::pb_op *dequant_wei1
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *conv1
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_src1, 0),
                                            in_edge(1, dequant_wei1, 0)});
                    conv1->append_decision_function(check_input_num<3>);
                    pm::pb_op *relu1 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv1, 0)});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu1, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_conv_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_related_fusion)
        .set_priority(7.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);
                    pm::pb_op *biasadd
                            = pgraph->append_op(impl::op_kind::BiasAdd,
                                    in_edges_t {in_edge(0, conv, 0)});

                    auto pograph_1 = std::make_shared<pb_graph_t>();
                    auto addormul = pograph_1->append_alternation(
                            {impl::op_kind::Add, impl::op_kind::Multiply});
                    pograph_1->create_input_port(0, addormul, 0);
                    pograph_1->create_input_port(1, addormul, 1);
                    pograph_1->create_output_port(0, addormul, 0);

                    auto rep1 = pgraph->append_repetition(pograph_1, {0, 0}, 0,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, biasadd, 0)});

                    auto pograph_2 = std::make_shared<pb_graph_t>();
                    auto eltwise = pograph_2->append_alternation(
                            {impl::op_kind::ReLU, impl::op_kind::Abs,
                                    impl::op_kind::Elu, impl::op_kind::GELU});
                    pograph_2->create_input_port(0, eltwise, 0);
                    pograph_2->create_output_port(0, eltwise, 0);

                    auto rep2 = pgraph->append_repetition(pograph_2, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, rep1, 0)});

                    auto pograph_3 = std::make_shared<pb_graph_t>();
                    auto addormul2 = pograph_3->append_alternation(
                            {impl::op_kind::Add, impl::op_kind::Multiply});
                    pograph_3->create_input_port(0, addormul2, 0);
                    pograph_3->create_output_port(0, addormul2, 0);

                    pgraph->append_repetition(pograph_3, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, rep2, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    auto pograph_1 = std::make_shared<pb_graph_t>();
                    auto addormul = pograph_1->append_alternation(
                            {impl::op_kind::Add, impl::op_kind::Multiply});
                    pograph_1->create_input_port(0, addormul, 0);
                    pograph_1->create_input_port(1, addormul, 1);
                    pograph_1->create_output_port(0, addormul, 0);

                    auto rep1 = pgraph->append_repetition(pograph_1, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, conv, 0)});

                    auto pograph_2 = std::make_shared<pb_graph_t>();
                    auto eltwise = pograph_2->append_alternation(
                            {impl::op_kind::ReLU, impl::op_kind::Abs,
                                    impl::op_kind::Elu, impl::op_kind::GELU});
                    pograph_2->create_input_port(0, eltwise, 0);
                    pograph_2->create_output_port(0, eltwise, 0);

                    auto rep2 = pgraph->append_repetition(pograph_2, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, rep1, 0)});

                    auto pograph_3 = std::make_shared<pb_graph_t>();
                    auto addormul2 = pograph_3->append_alternation(
                            {impl::op_kind::Add, impl::op_kind::Multiply});
                    pograph_3->create_input_port(0, addormul2, 0);
                    pograph_3->create_output_port(0, addormul2, 0);

                    pgraph->append_repetition(pograph_3, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, rep2, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bias_post_ops_chain_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
