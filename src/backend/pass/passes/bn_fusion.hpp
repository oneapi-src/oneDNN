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
#ifndef BACKEND_PASS_PASSES_BN_FUSION_HPP
#define BACKEND_PASS_PASSES_BN_FUSION_HPP

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
 * \brief This provides batchnorm-related fusion, i.e.
 *        batchnorm-relu fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused node, update the graph
 */

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, bn_relu_fusion)
        .set_priority(8.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *bn = apattern->create_node(
                            op_kind::BatchNormInference);
                    node_t *relu = apattern->create_node(op_kind::ReLU);
                    relu->set_input(0, bn, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node
                            = optimized_pattern->create_node(op_kind::bn_relu);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(dnnl, bn_bwd_relu_bwd_fusion)
        .set_priority(8.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    node_t *relu_bwd
                            = apattern->create_node(op_kind::ReLUBackprop);
                    node_t *bn_bwd = apattern->create_node(
                            op_kind::BatchNormTrainingBackprop);
                    bn_bwd->set_input(0, relu_bwd, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    node_t *fused_node = optimized_pattern->create_node(
                            op_kind::bn_bwd_relu_bwd);
                    fused_node->set_attr<std::string>("backend", "dnnl");
                });

} // namespace pass
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
