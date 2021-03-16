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
#ifndef BACKEND_DNNL_PASSES_CONV_FUSION_HPP
#define BACKEND_DNNL_PASSES_CONV_FUSION_HPP

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
 * \brief This provides conv-related fusion, i.e.
 *        conv-relu fusion, conv-bn fusion, conv-sum fusion, conv-bn-sum fusion, etc.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(conv_fusion)

#define SET_NUM_INPUTS_CHECK(n) \
    set_attr<FRequirement>("FRequirement", [](op_t *graph_op) -> bool { \
        size_t num_inputs = graph_op->num_inputs(); \
        return num_inputs == n; \
    });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_relu_fusion)
        .set_priority(10.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    add->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_relu6_fusion)
        .set_priority(10.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu6 = apattern->create_op(op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    add->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu6->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_add_relu6);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_elu_fusion)
        .set_priority(10.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *elu = apattern->create_op(op_kind::Elu);
                    add->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    elu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_add_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_sum_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    add->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::conv_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bn->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(0, *bn, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bn_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_relu6_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu6 = apattern->create_op(op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    bias->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu6->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu6 = apattern->create_op(op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    add->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu6->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_add_relu6);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bias->fill_and_connect_input(0, *conv, 0);
                    bn->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(0, *bn, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bn->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(0, *bn, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_bn_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_sum_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    bn->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(0, *bn, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bn_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_sum_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    bias->fill_and_connect_input(0, *conv, 0);
                    bn->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(0, *bn, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    bn->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(0, *bn, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_bn_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bn->fill_and_connect_input(0, *conv, 0);
                    relu->fill_and_connect_input(0, *bn, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bn_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bias->fill_and_connect_input(0, *conv, 0);
                    bn->fill_and_connect_input(0, *bias, 0);
                    relu->fill_and_connect_input(0, *bn, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bn->fill_and_connect_input(0, *conv, 0);
                    relu->fill_and_connect_input(0, *bn, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_bn_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_elu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *elu = apattern->create_op(op_kind::Elu);
                    bias->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    elu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *elu = apattern->create_op(op_kind::Elu);
                    add->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    elu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_add_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bias->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    add->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(1, *any, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_relu6_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *hardtanh = apattern->create_op(op_kind::HardTanh);
                    hardtanh->set_attr<float>("min", 0);
                    hardtanh->set_attr<float>("max", 6);
                    bias->fill_and_connect_input(0, *conv, 0);
                    hardtanh->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *hardtanh = apattern->create_op(op_kind::HardTanh);
                    hardtanh->set_attr<float>("min", 0);
                    hardtanh->set_attr<float>("max", 6);
                    hardtanh->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_relu6);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sum_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    bias->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    add->fill_and_connect_input(0, *conv, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_elu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *elu = apattern->create_op(op_kind::Elu);
                    bias->fill_and_connect_input(0, *conv, 0);
                    elu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *elu = apattern->create_op(op_kind::Elu);
                    elu->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sigmoid_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    bias->fill_and_connect_input(0, *conv, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    sigmoid->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_swish_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    op_t *mul = apattern->create_op(op_kind::Multiply);
                    bias->fill_and_connect_input(0, *conv, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                    mul->fill_and_connect_input(0, *bias, 0);
                    mul->fill_and_connect_input(1, *sigmoid, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    op_t *mul = apattern->create_op(op_kind::Multiply);
                    sigmoid->fill_and_connect_input(0, *conv, 0);
                    mul->fill_and_connect_input(0, *conv, 0);
                    mul->fill_and_connect_input(1, *sigmoid, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_swish);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bn_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    bn->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::conv_bn);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_bn_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    bias->fill_and_connect_input(0, *conv, 0);
                    bn->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *bn = apattern->create_op(op_kind::BatchNormInference);
                    bn->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_bn);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_relu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    relu->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::conv_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_relu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    bias->fill_and_connect_input(0, *conv, 0);
                    relu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    relu->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_hardtanh_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *hardtanh = apattern->create_op(op_kind::HardTanh);
                    bias->fill_and_connect_input(0, *conv, 0);
                    hardtanh->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *hardtanh = apattern->create_op(op_kind::HardTanh);
                    hardtanh->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_hardtanh);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_square_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *square = apattern->create_op(op_kind::Square);
                    bias->fill_and_connect_input(0, *conv, 0);
                    square->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *square = apattern->create_op(op_kind::Square);
                    square->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_square);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_tanh_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *tanh = apattern->create_op(op_kind::Tanh);
                    bias->fill_and_connect_input(0, *conv, 0);
                    tanh->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *tanh = apattern->create_op(op_kind::Tanh);
                    tanh->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_tanh);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_abs_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *abs = apattern->create_op(op_kind::Abs);
                    bias->fill_and_connect_input(0, *conv, 0);
                    abs->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *abs = apattern->create_op(op_kind::Abs);
                    abs->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_abs);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_sqrt_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    op_t *sqrt = apattern->create_op(op_kind::Sqrt);
                    bias->fill_and_connect_input(0, *conv, 0);
                    sqrt->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 3);
                    op_t *sqrt = apattern->create_op(op_kind::Sqrt);
                    sqrt->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bias_sqrt);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    conv->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(op_kind::BiasAdd);
                    bias->fill_and_connect_input(0, *conv, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    UNUSED(conv);
                    conv->set_attr<int64_t>("num_inputs", 3);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::conv_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_pass)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *conv = apattern->create_op(op_kind::Convolution);
                    UNUSED(conv);
                    conv->set_attr<int64_t>("num_inputs", 2);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *replace_op = optimized_pattern->create_op(
                            op_kind::Convolution);
                    replace_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bwd_f_biasadd_bwd_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *convbpf = apattern->create_op(
                            op_kind::ConvolutionBackpropFilters);
                    op_t *biasbp
                            = apattern->create_op(op_kind::BiasAddBackprop);
                    biasbp->fill_and_connect_input(0, *convbpf, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::conv_bwd_f_biasadd_bwd);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
