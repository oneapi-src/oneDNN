/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "graph/backend/dnnl/kernels/concat.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

namespace {
bool check_scales_zps_all_equal(op_t *op) {
    auto out_port = op->get_output_value(0);
    if (out_port->get_consumers().empty()) return false;

    auto &out_op = out_port->get_consumers()[0].get_op();
    // We only want to accept int8 concat with inputs using the same scales and
    // zps. Concat does not change range of values so output scales and zps
    // should be same as well.
    if (!out_op.has_attr(op_attr::scales) || !out_op.has_attr(op_attr::zps))
        return false;
    const auto expected_scales
            = out_op.get_attr<std::vector<float>>(op_attr::scales);
    const auto expected_zps
            = out_op.get_attr<std::vector<int64_t>>(op_attr::zps);

    for (size_t i = 0; i < op->num_inputs(); ++i) {
        auto in_port = op->get_input_value(i);
        if (!in_port->has_producer()) return false;

        auto &in_op = in_port->get_producer();
        if (!in_op.has_attr(op_attr::scales) || !in_op.has_attr(op_attr::zps))
            return false;
        auto scales = in_op.get_attr<std::vector<float>>(op_attr::scales);
        auto zps = in_op.get_attr<std::vector<int64_t>>(op_attr::zps);
        if (scales != expected_scales || zps != expected_zps) return false;
    }

    return true;
}
} // namespace

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(concat_fusion)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, int8_concat_fusion)
        .set_priority(8.2f)
        .set_kind(partition_kind_t::misc_quantized_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    in_edges_t input_edges;
                    for (size_t i = 0; i < VARIADIC_INPUT_NUM; ++i) {
                        pm::pb_op_t *dequant
                                = pgraph->append_op(graph::op_kind::Dequantize);
                        input_edges.emplace_back(in_edge(i, dequant, 0));
                    }
                    pm::pb_op_t *concat = pgraph->append_op(
                            graph::op_kind::Concat, input_edges);
                    concat->append_decision_function(
                            check_scales_zps_all_equal);

                    pgraph->append_op(graph::op_kind::Quantize,
                            in_edges_t {in_edge(0, concat, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<quantized_concat>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
