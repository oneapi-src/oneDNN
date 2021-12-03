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
#include "gelu.hpp"
#include <compiler/ir/graph/fusible_op.hpp>

namespace sc {
namespace ops {

gelu_op::gelu_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "gelu";
}

std::shared_ptr<sc_graph_t> gelu_op::get_graph() {
    auto graph = std::make_shared<sc_graph_t>();
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);

    // input
    graph->make_input(inputs);
    bool is_bf16
            = info_.inputs_[0]->details_.dtype_.is_etype(sc_data_etype::BF16);
    sc_op_ptr sqrt_2_over_pi, fitting_const, one, half;
    sqrt_2_over_pi = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {0.79788458f}),
            datatypes::f32, sc_dims {1});
    fitting_const = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {0.044715f}),
            datatypes::f32, sc_dims {1});
    one = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {1.0f}),
            datatypes::f32, sc_dims {1});
    half = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {0.5f}),
            datatypes::f32, sc_dims {1});
    graph_tensor_ptr inputs0 = inputs[0];
    if (is_bf16) {
        auto cast0 = graph->make(
                "cast", {inputs[0]}, {}, {{"dtype", datatypes::f32}});
        inputs0 = cast0->get_outputs()[0];
    }
    auto mul0 = graph->make("mul", {inputs0, inputs0}, {}, {});
    auto mul1 = graph->make("mul",
            {mul0->get_outputs()[0], fitting_const->get_outputs()[0]}, {}, {});
    auto add0 = graph->make(
            "add", {mul1->get_outputs()[0], one->get_outputs()[0]}, {}, {});
    auto mul2 = graph->make("mul", {add0->get_outputs()[0], inputs0}, {}, {});
    auto mul3 = graph->make("mul",
            {mul2->get_outputs()[0], sqrt_2_over_pi->get_outputs()[0]}, {}, {});
    auto tanh0 = graph->make("tanh", {mul3->get_outputs()[0]}, {}, {});
    auto add1 = graph->make(
            "add", {tanh0->get_outputs()[0], one->get_outputs()[0]}, {}, {});
    auto mul4 = graph->make("mul", {inputs0, add1->get_outputs()[0]}, {}, {});
    auto mul5 = graph->make(
            "mul", {mul4->get_outputs()[0], half->get_outputs()[0]}, {}, {});
    if (is_bf16) {
        mul5 = graph->make(
                "cast", mul5->get_outputs(), {}, {{"dtype", datatypes::bf16}});
    }
    // output
    graph->make_output(mul5->get_outputs());
    return graph;
}

void gelu_op::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {}

} // namespace ops

OP_REGISTER(ops::gelu_op, gelu)
} // namespace sc
