/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include "pow.hpp"
#include <cmath>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

pow_op::pow_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    beta_ = attrs.get<float>("beta");
    op_name_ = "pow";
}

void pow_op::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    auto pos_beta = std::fabs(beta_);
    sc_op_ptr fast_cal_last_op;
    // input
    graph->make_input(inputs);
    // fast calculation path
    if (pos_beta == 0.f) {
        float one = 1.f;
        auto const_zero = graph->make("constant", {}, {},
                {{"values",
                         std::make_shared<static_data_t>(
                                 &pos_beta, sizeof(pos_beta))},
                        {"dtype", datatypes::f32}, {"plain_dims", sc_dims {1}},
                        {"format", sc_data_format_t()}});
        auto const_one = graph->make("constant", {}, {},
                {{"values", std::make_shared<static_data_t>(&one, sizeof(one))},
                        {"dtype", datatypes::f32}, {"plain_dims", sc_dims {1}},
                        {"format", sc_data_format_t()}});
        auto fmul = graph->make(
                "mul", {inputs[0], const_zero->get_outputs()[0]}, {}, {});
        fast_cal_last_op = graph->make("add",
                {fmul->get_outputs()[0], const_one->get_outputs()[0]}, {}, {});
    } else if (pos_beta == 1.f) {
        fast_cal_last_op = graph->make("tensor_view", inputs, {},
                {{"shape", inputs[0]->details_.get_plain_dims()}});
    } else if (pos_beta == 2.f) {
        fast_cal_last_op = graph->make("square", inputs, {}, {});
    } else if (pos_beta == 3.f) {
        auto fsquare = graph->make("square", inputs, {}, {});
        fast_cal_last_op = graph->make(
                "mul", {fsquare->get_outputs()[0], inputs[0]}, {}, {});
    } else if (pos_beta == 0.5f) {
        fast_cal_last_op = graph->make(
                "squared_root", inputs, {}, {{"reciprocal", beta_ < 0}});
    } else {
        // log(x)
        auto flog = graph->make("log", {inputs[0]}, {}, {});

        // (log(x)*y)
        auto exponent = graph->make("constant", {}, {},
                {{"values",
                         std::make_shared<static_data_t>(
                                 &beta_, sizeof(beta_))},
                        {"dtype", datatypes::f32}, {"plain_dims", sc_dims {1}},
                        {"format", sc_data_format_t()}});
        auto fmul = graph->make("mul",
                {flog->get_outputs()[0], exponent->get_outputs()[0]}, {}, {});
        // pow = exp(log(x)*y)
        auto fexp = graph->make("exp", {fmul->get_outputs()[0]}, {}, {});
        // output
        graph->make_output(fexp->get_outputs());
    }
    if (fast_cal_last_op) {
        // process -0.5f with rsqrt.
        if (beta_ < 0.f && beta_ != -0.5f) {
            fast_cal_last_op = graph->make(
                    "reciprocal", fast_cal_last_op->get_outputs(), {}, {});
        }
        graph->make_output(fast_cal_last_op->get_outputs());
    }
}

void pow_op::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

} // namespace ops

OP_REGISTER(ops::pow_op, pow)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
