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
#ifndef BACKEND_PASS_PASSES_GELU_FUSION_HPP
#define BACKEND_PASS_PASSES_GELU_FUSION_HPP

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
 * \brief This provides GELU fusion.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused node, update the graph
 */

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, gelu_fusion)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *any_1 = apattern->create_node(op_kind::any);
                    node_t *pow = apattern->create_node(op_kind::Pow);
                    node_t *any_2 = apattern->create_node(op_kind::any);
                    node_t *multiply_1
                            = apattern->create_node(op_kind::Multiply);
                    node_t *any_3 = apattern->create_node(op_kind::any);
                    node_t *add_1 = apattern->create_node(op_kind::Add);
                    node_t *any_4 = apattern->create_node(op_kind::any);
                    node_t *multiply_2
                            = apattern->create_node(op_kind::Multiply);
                    node_t *tanh = apattern->create_node(op_kind::Tanh);
                    node_t *any_5 = apattern->create_node(op_kind::any);
                    node_t *add_2 = apattern->create_node(op_kind::Add);
                    node_t *any_6 = apattern->create_node(op_kind::any);
                    node_t *multiply_3
                            = apattern->create_node(op_kind::Multiply);
                    node_t *any_7 = apattern->create_node(op_kind::any);
                    node_t *multiply_4
                            = apattern->create_node(op_kind::Multiply);
                    pow->set_input(0, any_1, 0);
                    multiply_1->set_input(0, pow, 0);
                    multiply_1->set_input(1, any_2, 0);
                    add_1->set_input(0, multiply_1, 0);
                    add_1->set_input(1, any_3, 0);
                    multiply_2->set_input(0, add_1, 0);
                    multiply_2->set_input(1, any_4, 0);
                    tanh->set_input(0, multiply_2, 0);
                    add_2->set_input(0, tanh, 0);
                    add_2->set_input(1, any_5, 0);
                    multiply_3->set_input(0, add_2, 0);
                    multiply_3->set_input(1, any_6, 0);
                    multiply_4->set_input(0, multiply_3, 0);
                    multiply_4->set_input(1, any_7, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *any_1 = apattern->create_node(op_kind::any);
                    node_t *div = apattern->create_node(op_kind::Divide);
                    node_t *erf = apattern->create_node(op_kind::Erf);
                    node_t *any_2 = apattern->create_node(op_kind::any);
                    node_t *add = apattern->create_node(op_kind::Add);
                    node_t *any_3 = apattern->create_node(op_kind::any);
                    node_t *multiply_1
                            = apattern->create_node(op_kind::Multiply);
                    node_t *any_4 = apattern->create_node(op_kind::any);
                    node_t *multiply_2
                            = apattern->create_node(op_kind::Multiply);
                    div->set_input(0, any_1, 0);
                    erf->set_input(0, div, 0);
                    add->set_input(0, erf, 0);
                    add->set_input(1, any_2, 0);
                    multiply_1->set_input(0, add, 0);
                    multiply_1->set_input(1, any_3, 0);
                    multiply_2->set_input(0, multiply_1, 0);
                    multiply_2->set_input(1, any_4, 0);
                })

        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node
                            = optimized_pattern->create_node(op_kind::GELU);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

} // namespace pass
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
