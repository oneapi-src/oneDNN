/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include <string>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
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

gelu_backprop_op::gelu_backprop_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT(ins.size() == 2, "Wrong op input size.\n");
    info_.inputs_ = ins;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "gelu_backprop";
}

void gelu_op::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    if (attrs_.get_or_else("gelu_type", std::string("erf")) == "tanh") {
        // tanh impl
        // input
        graph->make_input(inputs);
        bool is_bf16 = info_.inputs_[0]->details_.dtype_.is_etype(
                sc_data_etype::BF16);
        sc_op_ptr sqrt_2_over_pi, fitting_const, one, half;
        sqrt_2_over_pi = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(
                        std::vector<float> {0.79788458f}),
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
                {mul0->get_outputs()[0], fitting_const->get_outputs()[0]}, {},
                {});
        auto add0 = graph->make(
                "add", {mul1->get_outputs()[0], one->get_outputs()[0]}, {}, {});
        auto mul2
                = graph->make("mul", {add0->get_outputs()[0], inputs0}, {}, {});
        auto mul3 = graph->make("mul",
                {mul2->get_outputs()[0], sqrt_2_over_pi->get_outputs()[0]}, {},
                {});
        auto tanh0 = graph->make("tanh", {mul3->get_outputs()[0]}, {}, {});
        auto add1 = graph->make("add",
                {tanh0->get_outputs()[0], one->get_outputs()[0]}, {}, {});
        auto mul4
                = graph->make("mul", {inputs0, add1->get_outputs()[0]}, {}, {});
        auto mul5 = graph->make("mul",
                {mul4->get_outputs()[0], half->get_outputs()[0]}, {}, {});
        if (is_bf16) {
            mul5 = graph->make("cast", mul5->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
        }
        // output
        graph->make_output(mul5->get_outputs());
    } else {
        // erf impl
        // input
        graph->make_input(inputs);
        bool is_bf16 = info_.inputs_[0]->details_.dtype_.is_etype(
                sc_data_etype::BF16);
        sc_op_ptr one_over_sqrt_2, one, half;
        union {
            float v;
            int v2;
        } caster;
        caster.v2 = 0x3f3504f3;
        one_over_sqrt_2 = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {caster.v}),
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
        auto mul0 = graph->make(
                "mul", {inputs0, one_over_sqrt_2->get_outputs()[0]}, {}, {});
        auto erf0 = graph->make("erf", {mul0->get_outputs()[0]}, {}, {});
        auto add0 = graph->make(
                "add", {erf0->get_outputs()[0], one->get_outputs()[0]}, {}, {});
        auto mul1
                = graph->make("mul", {add0->get_outputs()[0], inputs0}, {}, {});
        auto mul2 = graph->make("mul",
                {mul1->get_outputs()[0], half->get_outputs()[0]}, {}, {});
        if (is_bf16) {
            mul2 = graph->make("cast", mul2->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
        }
        // output
        graph->make_output(mul2->get_outputs());
    }
}

void gelu_backprop_op::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);

    // input
    graph->make_input(inputs);
    bool is_bf16
            = info_.inputs_[0]->details_.dtype_.is_etype(sc_data_etype::BF16);
    graph_tensor_ptr inputs0 = inputs[0];
    graph_tensor_ptr inputs1 = inputs[1];

    if (attrs_.get_or_else("gelu_type", std::string("erf")) == "tanh") {
        sc_op_ptr fitting_const1, fitting_const2, fitting_const3,
                fitting_const4, two, one, neg_one, half;
        fitting_const1 = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(
                        std::vector<float> {0.0356774f}),
                datatypes::f32, sc_dims {1});
        fitting_const2 = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {0.797885f}),
                datatypes::f32, sc_dims {1});
        fitting_const3 = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(
                        std::vector<float> {0.0535161f}),
                datatypes::f32, sc_dims {1});
        fitting_const4 = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {0.398942f}),
                datatypes::f32, sc_dims {1});
        two = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {2.0f}),
                datatypes::f32, sc_dims {1});
        one = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {1.0f}),
                datatypes::f32, sc_dims {1});
        neg_one = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {-1.0f}),
                datatypes::f32, sc_dims {1});
        half = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {0.5f}),
                datatypes::f32, sc_dims {1});
        if (is_bf16) {
            auto cast0 = graph->make(
                    "cast", {inputs[0]}, {}, {{"dtype", datatypes::f32}});
            inputs0 = cast0->get_outputs()[0];
        }
        // x*x
        auto mul0 = graph->make("mul", {inputs0, inputs0}, {}, {});
        // 0.0356774x*x
        auto mul1 = graph->make("mul",
                {mul0->get_outputs()[0], fitting_const1->get_outputs()[0]}, {},
                {});
        // 0.0356774x*x+0.797885
        auto add0 = graph->make("add",
                {mul1->get_outputs()[0], fitting_const2->get_outputs()[0]}, {},
                {});
        // (0.0356774x*x+0.797885)*x
        auto mul2
                = graph->make("mul", {add0->get_outputs()[0], inputs0}, {}, {});
        // tanh((0.0356774x*x+0.797885)*x)
        auto tanh0 = graph->make("tanh", {mul2->get_outputs()[0]}, {}, {});
        // 0.5*tanh((0.0356774x*x+0.797885)*x)
        auto mul3 = graph->make("mul",
                {tanh0->get_outputs()[0], half->get_outputs()[0]}, {}, {});

        // 0.053561x*x
        auto mul4 = graph->make("mul",
                {mul0->get_outputs()[0], fitting_const3->get_outputs()[0]}, {},
                {});
        // 0.053561x*x+0.398942
        auto add1 = graph->make("add",
                {mul4->get_outputs()[0], fitting_const4->get_outputs()[0]}, {},
                {});
        // (0.053561x*x+0.398942)*x
        auto mul5
                = graph->make("mul", {add1->get_outputs()[0], inputs0}, {}, {});

        // exp{(0.0356774x*x+0.797885)*x}
        auto exp = graph->make("exp", {mul2->get_outputs()[0]}, {}, {});
        // -(0.0356774x*x+0.797885)*x
        auto mul6 = graph->make("mul",
                {mul2->get_outputs()[0], neg_one->get_outputs()[0]}, {}, {});
        // exp{-(0.0356774x*x+0.797885)*x}
        auto exp1 = graph->make("exp", {mul6->get_outputs()[0]}, {}, {});
        // exp{(0.0356774x*x+0.797885)*x} + exp{-(0.0356774x*x+0.797885)*x}
        auto add2 = graph->make(
                "add", {exp->get_outputs()[0], exp1->get_outputs()[0]}, {}, {});
        // sech((0.0356774x*x+0.797885)*x)
        auto div = graph->make(
                "div", {two->get_outputs()[0], add2->get_outputs()[0]}, {}, {});

        // sech^2((0.0356774x*x+0.797885)*x)
        auto mul7 = graph->make(
                "mul", {div->get_outputs()[0], div->get_outputs()[0]}, {}, {});
        // (0.053561x*x+0.398942)*x*sech^2((0.0356774x*x+0.797885)*x)
        auto mul8 = graph->make("mul",
                {mul5->get_outputs()[0], mul7->get_outputs()[0]}, {}, {});

        auto add3 = graph->make("add",
                {mul3->get_outputs()[0], mul8->get_outputs()[0]}, {}, {});
        auto add4 = graph->make("add",
                {add3->get_outputs()[0], half->get_outputs()[0]}, {}, {});

        if (is_bf16) {
            add4 = graph->make("cast", add4->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
        }
        auto mul10
                = graph->make("mul", {add4->get_outputs()[0], inputs1}, {}, {});
        // output
        graph->make_output(mul10->get_outputs());
    } else {
        auto one = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {1.0f}),
                datatypes::f32, sc_dims {1});
        auto half = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {0.5f}),
                datatypes::f32, sc_dims {1});
        auto neg_half = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {-0.5f}),
                datatypes::f32, sc_dims {1});
        // 1 / sqrt(2*phi)
        auto fitting_const1 = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {0.398942f}),
                datatypes::f32, sc_dims {1});
        auto fitting_const2 = graph->make<constant_op_t>(
                std::make_shared<static_data_t>(std::vector<float> {0.707106f}),
                datatypes::f32, sc_dims {1});
        if (is_bf16) {
            auto cast0 = graph->make(
                    "cast", {inputs[0]}, {}, {{"dtype", datatypes::f32}});
            auto cast1 = graph->make(
                    "cast", {inputs[1]}, {}, {{"dtype", datatypes::f32}});
            inputs0 = cast0->get_outputs()[0];
            inputs1 = cast1->get_outputs()[0];
        }
        // x*x
        auto mul0 = graph->make("mul", {inputs0, inputs0}, {}, {});
        // -0.5f*x*x
        auto mul1 = graph->make("mul",
                {mul0->get_outputs()[0], neg_half->get_outputs()[0]}, {}, {});
        auto exp = graph->make("exp", mul1->get_outputs(), {}, {});
        auto mul2 = graph->make("mul",
                {exp->get_outputs()[0], fitting_const1->get_outputs()[0]}, {},
                {});
        auto mul3
                = graph->make("mul", {mul2->get_outputs()[0], inputs0}, {}, {});
        auto mul4 = graph->make(
                "mul", {inputs0, fitting_const2->get_outputs()[0]}, {}, {});
        auto erf = graph->make("erf", mul4->get_outputs(), {}, {});
        auto add0 = graph->make(
                "add", {erf->get_outputs()[0], one->get_outputs()[0]}, {}, {});
        auto mul5 = graph->make("mul",
                {add0->get_outputs()[0], half->get_outputs()[0]}, {}, {});
        auto add1 = graph->make("add",
                {mul3->get_outputs()[0], mul5->get_outputs()[0]}, {}, {});
        auto mul6
                = graph->make("mul", {add1->get_outputs()[0], inputs1}, {}, {});
        if (is_bf16) {
            mul6 = graph->make("cast", mul6->get_outputs(), {},
                    {{"dtype", datatypes::bf16}});
        }
        // output
        graph->make_output(mul6->get_outputs());
    }
}

void gelu_op::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

void gelu_backprop_op::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

} // namespace ops

OP_REGISTER(ops::gelu_op, gelu)
OP_REGISTER(ops::gelu_backprop_op, gelu_backprop)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
