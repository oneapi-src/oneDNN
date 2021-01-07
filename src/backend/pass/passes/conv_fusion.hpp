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
#ifndef LLGA_BACKEND_PASS_PASSES_CONV_FUSION_HPP
#define LLGA_BACKEND_PASS_PASSES_CONV_FUSION_HPP

#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "backend/pass/pass_base.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace pass {

/*!
 * \brief This provides conv-related fusion, i.e.
 *        conv-relu fusion, conv-bn fusion, conv-sum fusion, conv-bn-sum fusion, etc.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused node, update the graph
 */
#define SET_NUM_INPUTS_CHECK(n) \
    set_attr<FRequirement>("FRequirement", [](node_t *graph_node) -> bool { \
        size_t num_inputs = graph_node->num_inputs_tensor(); \
        return num_inputs == n; \
    });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_relu_fusion)
        .set_priority(10.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    add->set_input(0, conv, 0);
                    add->set_input(1, any, 0);
                    relu->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_add_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_relu6_fusion)
        .set_priority(10.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu6 = apattern->create_node(op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    add->set_input(0, conv, 0);
                    add->set_input(1, any, 0);
                    relu6->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_add_relu6);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_elu_fusion)
        .set_priority(10.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *elu = apattern->create_node(op_kind::Elu);
                    add->set_input(0, conv, 0);
                    add->set_input(1, any, 0);
                    elu->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_add_elu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    add->set_input(0, conv, 0);
                    add->set_input(1, any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node
                            = optimized_pattern->create_node(op_kind::conv_add);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bn->set_input(0, conv, 0);
                    add->set_input(0, bn, 0);
                    add->set_input(1, any, 0);
                    relu->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bn_add_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_relu6_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu6 = apattern->create_node(op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    bias->set_input(0, conv, 0);
                    add->set_input(0, bias, 0);
                    add->set_input(1, any, 0);
                    relu6->set_input(0, add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu6 = apattern->create_node(op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    add->set_input(0, conv, 0);
                    add->set_input(1, any, 0);
                    relu6->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_add_relu6);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bias->set_input(0, conv, 0);
                    bn->set_input(0, bias, 0);
                    add->set_input(0, bn, 0);
                    add->set_input(1, any, 0);
                    relu->set_input(0, add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bn->set_input(0, conv, 0);
                    add->set_input(0, bn, 0);
                    add->set_input(1, any, 0);
                    relu->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_bn_add_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_sum_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    bn->set_input(0, conv, 0);
                    add->set_input(0, bn, 0);
                    add->set_input(1, any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bn_add);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_sum_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    bias->set_input(0, conv, 0);
                    bn->set_input(0, bias, 0);
                    add->set_input(0, bn, 0);
                    add->set_input(1, any, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    bn->set_input(0, conv, 0);
                    add->set_input(0, bn, 0);
                    add->set_input(1, any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_bn_add);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bn->set_input(0, conv, 0);
                    relu->set_input(0, bn, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bn_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bias->set_input(0, conv, 0);
                    bn->set_input(0, bias, 0);
                    relu->set_input(0, bn, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bn->set_input(0, conv, 0);
                    relu->set_input(0, bn, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_bn_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_elu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *elu = apattern->create_node(op_kind::Elu);
                    bias->set_input(0, conv, 0);
                    add->set_input(0, bias, 0);
                    add->set_input(1, any, 0);
                    elu->set_input(0, add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *elu = apattern->create_node(op_kind::Elu);
                    add->set_input(0, conv, 0);
                    add->set_input(1, any, 0);
                    elu->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_add_elu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bias->set_input(0, conv, 0);
                    add->set_input(0, bias, 0);
                    add->set_input(1, any, 0);
                    relu->set_input(0, add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    add->set_input(0, conv, 0);
                    add->set_input(1, any, 0);
                    relu->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_add_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_relu6_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *hardtanh = apattern->create_node(op_kind::HardTanh);
                    hardtanh->set_attr<float>("min", 0);
                    hardtanh->set_attr<float>("max", 6);
                    bias->set_input(0, conv, 0);
                    hardtanh->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *hardtanh = apattern->create_node(op_kind::HardTanh);
                    hardtanh->set_attr<float>("min", 0);
                    hardtanh->set_attr<float>("max", 6);
                    hardtanh->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_relu6);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    bias->set_input(0, conv, 0);
                    add->set_input(0, bias, 0);
                    add->set_input(1, any, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    add->set_input(0, conv, 0);
                    add->set_input(1, any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_add);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_elu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *elu = apattern->create_node(op_kind::Elu);
                    bias->set_input(0, conv, 0);
                    elu->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *elu = apattern->create_node(op_kind::Elu);
                    elu->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_elu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sigmoid_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *sigmoid = apattern->create_node(op_kind::Sigmoid);
                    bias->set_input(0, conv, 0);
                    sigmoid->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *sigmoid = apattern->create_node(op_kind::Sigmoid);
                    sigmoid->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_sigmoid);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_swish_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *sigmoid = apattern->create_node(op_kind::Sigmoid);
                    node_t *mul = apattern->create_node(op_kind::Multiply);
                    bias->set_input(0, conv, 0);
                    sigmoid->set_input(0, bias, 0);
                    mul->set_input(0, bias, 0);
                    mul->set_input(1, sigmoid, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *sigmoid = apattern->create_node(op_kind::Sigmoid);
                    node_t *mul = apattern->create_node(op_kind::Multiply);
                    sigmoid->set_input(0, conv, 0);
                    mul->set_input(0, conv, 0);
                    mul->set_input(1, sigmoid, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_swish);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    bn->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node
                            = optimized_pattern->create_node(op_kind::conv_bn);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    bias->set_input(0, conv, 0);
                    bn->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    bn->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_bn);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_relu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    relu->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_relu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bias->set_input(0, conv, 0);
                    relu->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    relu->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_hardtanh_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *hardtanh = apattern->create_node(op_kind::HardTanh);
                    bias->set_input(0, conv, 0);
                    hardtanh->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *hardtanh = apattern->create_node(op_kind::HardTanh);
                    hardtanh->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_hardtanh);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_square_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *square = apattern->create_node(op_kind::Square);
                    bias->set_input(0, conv, 0);
                    square->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *square = apattern->create_node(op_kind::Square);
                    square->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_square);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_tanh_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *tanh = apattern->create_node(op_kind::Tanh);
                    bias->set_input(0, conv, 0);
                    tanh->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *tanh = apattern->create_node(op_kind::Tanh);
                    tanh->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_tanh);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_abs_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *abs = apattern->create_node(op_kind::Abs);
                    bias->set_input(0, conv, 0);
                    abs->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *abs = apattern->create_node(op_kind::Abs);
                    abs->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_abs);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sqrt_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *sqrt = apattern->create_node(op_kind::Sqrt);
                    bias->set_input(0, conv, 0);
                    sqrt->set_input(0, bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                    node_t *sqrt = apattern->create_node(op_kind::Sqrt);
                    sqrt->set_input(0, conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias_sqrt);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    bias->set_input(0, conv, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(3);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bias);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_pass)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *conv = apattern->create_node(op_kind::Convolution);
                    conv->SET_NUM_INPUTS_CHECK(2);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *replace_node = optimized_pattern->create_node(
                            op_kind::Convolution);
                    replace_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bwd_f_biasadd_bwd_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *convbpf = apattern->create_node(
                            op_kind::ConvolutionBackpropFilters);
                    node_t *biasbp
                            = apattern->create_node(op_kind::BiasAddBackprop);
                    biasbp->set_input(0, convbpf, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::conv_bwd_f_biasadd_bwd);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

} // namespace pass
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
