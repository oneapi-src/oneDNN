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
#ifndef BACKEND_DNNL_PASSES_MATMUL_FUSION_HPP
#define BACKEND_DNNL_PASSES_MATMUL_FUSION_HPP

#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "backend/dnnl/transformation_pass.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

using pattern = impl::pass::pattern;
using FCreatePattern = impl::pass::FCreatePattern;
using FCreateOptPattern = impl::pass::FCreateOptPattern;

/*!
 * \brief This provides matmul-related fusion, i.e.
 *        matmul-relu fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(matmul_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_relu_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    relu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_elu_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *elu = apattern->create_op(op_kind::Elu);
                    elu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::matmul_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sigmoid_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_hardtanh_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *hardtanh = apattern->create_op(op_kind::HardTanh);
                    hardtanh->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_hardtanh);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_gelu_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *gelu = apattern->create_op(op_kind::GELU);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    bias->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_sigmoid_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_swish_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    op_t *mul = apattern->create_op(op_kind::Multiply);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                    mul->fill_and_connect_input(0, *bias, 0);
                    mul->fill_and_connect_input(1, *sigmoid, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_swish);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_elu_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *elu = apattern->create_op(op_kind::Elu);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    elu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_relu_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    relu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_relu6_fusion)
        .set_priority(9.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *relu6 = apattern->create_op(op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    relu6->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_relu6);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_hardtanh_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *hardtanh = apattern->create_op(op_kind::HardTanh);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    hardtanh->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_hardtanh);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_sum_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_bn_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    bn->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_bn);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *any = apattern->create_op(op_kind::any);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::matmul_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_gelu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *gelu = apattern->create_op(op_kind::GELU);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    gelu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_add_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_sigmoid_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(op_kind::MatMul);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    sigmoid->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_add_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
