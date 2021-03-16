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
#ifndef BACKEND_DNNL_PASSES_GELU_FUSION_HPP
#define BACKEND_DNNL_PASSES_GELU_FUSION_HPP

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
 * \brief This provides GELU fusion.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(gelu_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, gelu_fusion)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *any_1 = apattern->create_op(op_kind::any);
                    op_t *pow = apattern->create_op(op_kind::Pow);
                    op_t *any_2 = apattern->create_op(op_kind::any);
                    op_t *multiply_1 = apattern->create_op(op_kind::Multiply);
                    op_t *any_3 = apattern->create_op(op_kind::any);
                    op_t *add_1 = apattern->create_op(op_kind::Add);
                    op_t *any_4 = apattern->create_op(op_kind::any);
                    op_t *multiply_2 = apattern->create_op(op_kind::Multiply);
                    op_t *tanh = apattern->create_op(op_kind::Tanh);
                    op_t *any_5 = apattern->create_op(op_kind::any);
                    op_t *add_2 = apattern->create_op(op_kind::Add);
                    op_t *any_6 = apattern->create_op(op_kind::any);
                    op_t *multiply_3 = apattern->create_op(op_kind::Multiply);
                    op_t *any_7 = apattern->create_op(op_kind::any);
                    op_t *multiply_4 = apattern->create_op(op_kind::Multiply);
                    pow->fill_and_connect_input(0, *any_1, 0);
                    multiply_1->fill_and_connect_input(0, *pow, 0);
                    multiply_1->fill_and_connect_input(1, *any_2, 0);
                    add_1->fill_and_connect_input(0, *multiply_1, 0);
                    add_1->fill_and_connect_input(1, *any_3, 0);
                    multiply_2->fill_and_connect_input(0, *add_1, 0);
                    multiply_2->fill_and_connect_input(1, *any_4, 0);
                    tanh->fill_and_connect_input(0, *multiply_2, 0);
                    add_2->fill_and_connect_input(0, *tanh, 0);
                    add_2->fill_and_connect_input(1, *any_5, 0);
                    multiply_3->fill_and_connect_input(0, *add_2, 0);
                    multiply_3->fill_and_connect_input(1, *any_6, 0);
                    multiply_4->fill_and_connect_input(0, *multiply_3, 0);
                    multiply_4->fill_and_connect_input(1, *any_7, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *any_1 = apattern->create_op(op_kind::any);
                    op_t *div = apattern->create_op(op_kind::Divide);
                    op_t *erf = apattern->create_op(op_kind::Erf);
                    op_t *any_2 = apattern->create_op(op_kind::any);
                    op_t *add = apattern->create_op(op_kind::Add);
                    op_t *any_3 = apattern->create_op(op_kind::any);
                    op_t *multiply_1 = apattern->create_op(op_kind::Multiply);
                    op_t *any_4 = apattern->create_op(op_kind::any);
                    op_t *multiply_2 = apattern->create_op(op_kind::Multiply);
                    div->fill_and_connect_input(0, *any_1, 0);
                    erf->fill_and_connect_input(0, *div, 0);
                    add->fill_and_connect_input(0, *erf, 0);
                    add->fill_and_connect_input(1, *any_2, 0);
                    multiply_1->fill_and_connect_input(0, *add, 0);
                    multiply_1->fill_and_connect_input(1, *any_3, 0);
                    multiply_2->fill_and_connect_input(0, *multiply_1, 0);
                    multiply_2->fill_and_connect_input(1, *any_4, 0);
                })

        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::GELU);
                    fused_op->set_attr("backend", std::string("dnnl"));
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
