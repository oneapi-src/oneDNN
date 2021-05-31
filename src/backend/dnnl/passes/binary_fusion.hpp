/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#ifndef BACKEND_DNNL_PASSES_BINARY_FUSION_HPP
#define BACKEND_DNNL_PASSES_BINARY_FUSION_HPP

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
 * \brief This provides binary-related fusion, i.e.
 *        binary-relu fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */

DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(binary_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_add_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_add_sigmoid_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    sigmoid->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::add_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_multiply_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *mul = apattern->create_op(op_kind::Multiply);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    add->fill_and_connect_input(0, *mul, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::multiply_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_multiply_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *mul = apattern->create_op(op_kind::Multiply);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    relu->fill_and_connect_input(0, *mul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::multiply_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_mul_sigmoid_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *mul = apattern->create_op(op_kind::Multiply);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    sigmoid->fill_and_connect_input(0, *mul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::multiply_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_maximum_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *max = apattern->create_op(op_kind::Maximum);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    add->fill_and_connect_input(0, *max, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::maximum_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_maximum_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *max = apattern->create_op(op_kind::Maximum);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    relu->fill_and_connect_input(0, *max, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::maximum_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_max_sigmoid_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *max = apattern->create_op(op_kind::Maximum);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    sigmoid->fill_and_connect_input(0, *max, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::maximum_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_minimum_add_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *min = apattern->create_op(op_kind::Minimum);
                    op_t *any = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    add->fill_and_connect_input(0, *min, 0);
                    add->fill_and_connect_input(1, *any, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::minimum_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_minimum_relu_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *min = apattern->create_op(op_kind::Minimum);
                    op_t *relu = apattern->create_op(op_kind::ReLU);
                    relu->fill_and_connect_input(0, *min, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::minimum_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, binary_min_sigmoid_fusion)
        .set_priority(8.2f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *min = apattern->create_op(op_kind::Minimum);
                    op_t *sigmoid = apattern->create_op(op_kind::Sigmoid);
                    sigmoid->fill_and_connect_input(0, *min, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::minimum_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
