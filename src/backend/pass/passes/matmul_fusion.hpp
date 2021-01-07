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
#ifndef LLGA_BACKEND_PASS_PASSES_MATMUL_FUSION_HPP
#define LLGA_BACKEND_PASS_PASSES_MATMUL_FUSION_HPP

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
 * \brief This provides matmul-related fusion, i.e.
 *        matmul-relu fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused node, update the graph
 */

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_relu_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    relu->set_input(0, matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_elu_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *elu = apattern->create_node(op_kind::Elu);
                    elu->set_input(0, matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_elu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sigmoid_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *sigmoid = apattern->create_node(op_kind::Sigmoid);
                    sigmoid->set_input(0, matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_sigmoid);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_hardtanh_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *hardtanh = apattern->create_node(op_kind::HardTanh);
                    hardtanh->set_input(0, matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_hardtanh);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_gelu_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *gelu = apattern->create_node(op_kind::GELU);
                    gelu->set_input(0, matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_gelu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    bias->set_input(0, matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_sigmoid_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *sigmoid = apattern->create_node(op_kind::Sigmoid);
                    bias->set_input(0, matmul, 0);
                    sigmoid->set_input(0, bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias_sigmoid);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_swish_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *sigmoid = apattern->create_node(op_kind::Sigmoid);
                    node_t *mul = apattern->create_node(op_kind::Multiply);
                    bias->set_input(0, matmul, 0);
                    sigmoid->set_input(0, bias, 0);
                    mul->set_input(0, bias, 0);
                    mul->set_input(1, sigmoid, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias_swish);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_elu_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *elu = apattern->create_node(op_kind::Elu);
                    bias->set_input(0, matmul, 0);
                    elu->set_input(0, bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias_elu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_relu_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bias->set_input(0, matmul, 0);
                    relu->set_input(0, bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_relu6_fusion)
        .set_priority(9.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *relu6 = apattern->create_node(op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    bias->set_input(0, matmul, 0);
                    relu6->set_input(0, bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias_relu6);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_hardtanh_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *hardtanh = apattern->create_node(op_kind::HardTanh);
                    bias->set_input(0, matmul, 0);
                    hardtanh->set_input(0, bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias_hardtanh);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    bias->set_input(0, matmul, 0);
                    add->set_input(0, bias, 0);
                    add->set_input(1, any, 0);
                    relu->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias_add_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_sum_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    bias->set_input(0, matmul, 0);
                    add->set_input(0, bias, 0);
                    add->set_input(1, any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias_add);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_bn_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *bias = apattern->create_node(op_kind::BiasAdd);
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    bias->set_input(0, matmul, 0);
                    bn->set_input(0, bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_bias_bn);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *any = apattern->create_node(op_kind::any);
                    add->set_input(0, matmul, 0);
                    add->set_input(1, any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_add);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_gelu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *gelu = apattern->create_node(op_kind::GELU);
                    add->set_input(0, matmul, 0);
                    add->set_input(1, any, 0);
                    gelu->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_add_gelu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    add->set_input(0, matmul, 0);
                    add->set_input(1, any, 0);
                    relu->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_add_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_sigmoid_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *matmul = apattern->create_node(op_kind::MatMul);
                    node_t *any = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *sigmoid = apattern->create_node(op_kind::Sigmoid);
                    add->set_input(0, matmul, 0);
                    add->set_input(1, any, 0);
                    sigmoid->set_input(0, add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::matmul_add_sigmoid);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

} // namespace pass
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
