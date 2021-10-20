/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#ifndef BACKEND_DNNL_PATTERNS_CONV_FUSION_HPP
#define BACKEND_DNNL_PATTERNS_CONV_FUSION_HPP

#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "backend/dnnl/patterns/transformation_pattern.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph = pm::pb_graph;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

/*!
 * \brief This provides conv-related fusion, i.e.
 *        conv-relu fusion, conv-bn fusion, conv-sum fusion, conv-bn-sum fusion, etc.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(conv_fusion)

#define SET_NUM_INPUTS_CHECK(n) \
    append_decision_function([](op_t *graph_op) -> bool { \
        return graph_op->num_inputs() == (n); \
    })

#define SET_S8_CHECK() \
    append_decision_function([](op_t *graph_op) -> bool { \
        for (size_t i = 0; i < graph_op->num_inputs(); ++i) { \
            logical_tensor_t iport \
                    = graph_op->get_input_value(i)->get_logical_tensor(); \
            if (iport.data_type != impl::data_type::s8) return false; \
        } \
        return true; \
    })

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_relu_fusion)
        .set_priority(10.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv = pgraph->append_op(
                            impl::op_kind::Convolution, "p-conv");
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)}, "p-add");
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0)});
                    add->set_commutative_pair({0, 1});
                    add->allow_internal_inputs({0, 1});

                    pm::pb_op *relu6
                            = pgraph->append_op(impl::op_kind::HardTanh,
                                    in_edges_t {in_edge(0, add, 0)});
                    relu6->set_attr<float>("min", 0.f);
                    relu6->set_attr<float>("max", 6.f);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, bias, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->set_commutative_pair({0, 1});
                    add->allow_internal_inputs({0, 1});

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->set_commutative_pair({0, 1});
                    add->allow_internal_inputs({0, 1});

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, bias, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->set_commutative_pair({0, 1});
                    add->allow_internal_inputs({0, 1});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(
                            impl::op_kind::Add, in_edges_t {in_edge(0, bn, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *bn = pgraph->append_op(
                            impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, bias, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bn, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0)});
                    add->set_commutative_pair({0, 1});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::Elu,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0)});
                    add->set_commutative_pair({0, 1});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *relu6
                            = pgraph->append_op(impl::op_kind::HardTanh,
                                    in_edges_t {in_edge(0, bias, 0)});
                    relu6->set_attr<float>("min", 0.f);
                    relu6->set_attr<float>("max", 6.f);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_conv_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(2);
                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(2);
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});
                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");
                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)}, "pbias");
                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");
                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->set_commutative_pair({0, 1});
                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, add, 0)}, "pquant");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");
                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(3);
                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");
                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bias, 0)});

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)}, "pbias");

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->set_commutative_pair({0, 1});

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)}, "prelu");

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, relu, 0)}, "pquant");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->set_commutative_pair({0, 1});

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)});
                    add->set_commutative_pair({0, 1});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)});
                    add->set_commutative_pair({0, 1});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});
                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, relu, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bias, 0)});
                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->set_commutative_pair({0, 1});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)}, "prelu");
                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)});
                    add->set_commutative_pair({0, 1});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                    pm::pb_op *quant
                            = pgraph->append_op(impl::op_kind::Quantize,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(2);
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(3);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)}, "prelu");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(2);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)}, "pbias");

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->set_commutative_pair({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)}, "prelu");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->set_commutative_pair({0, 1});

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_weight");
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_other");

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "pconv");
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->set_commutative_pair({0, 1});

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(3);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)});
                    add->set_commutative_pair({0, 1});

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, dequant_other, 0)});
                    add->set_commutative_pair({0, 1});

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *quant_weight
                            = pgraph->append_op(impl::op_kind::Quantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quant_weight, 0)});
                    // this pattern requires the weight should be s8
                    dequant_weight->SET_S8_CHECK();
                    pm::pb_op *dequant_other
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)});
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, dequant_other, 0)});
                    add->set_commutative_pair({0, 1});

                    pm::pb_op *relu = pgraph->append_op(impl::op_kind::ReLU,
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, bias, 0)});
                    add->set_commutative_pair({0, 1});
                    add->allow_internal_inputs({0, 1});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv, 0)});
                    add->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Elu,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Sigmoid,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *sigmoid
                            = pgraph->append_op(impl::op_kind::Sigmoid,
                                    in_edges_t {in_edge(0, bias, 0)});

                    pm::pb_op *mul = pgraph->append_op(impl::op_kind::Multiply,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, sigmoid, 0)});
                    mul->set_commutative_pair({0, 1});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv = pgraph->append_op(
                            impl::op_kind::Convolution, "p-conv");
                    conv->SET_NUM_INPUTS_CHECK(3);

                    pm::pb_op *sigmoid
                            = pgraph->append_op(impl::op_kind::Sigmoid,
                                    in_edges_t {in_edge(0, conv, 0)}, "p-sig");

                    pm::pb_op *mul = pgraph->append_op(impl::op_kind::Multiply,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, sigmoid, 0)},
                            "p-mul");
                    mul->set_commutative_pair({0, 1});
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::BatchNormInference,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::HardTanh,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Square,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Tanh,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Abs,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pgraph->append_op(impl::op_kind::Sqrt,
                            in_edges_t {in_edge(0, bias, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);

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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
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

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
