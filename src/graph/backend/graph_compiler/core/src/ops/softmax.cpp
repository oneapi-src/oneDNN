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
#include "softmax.hpp"
#include <utility>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

softmax_base_t::softmax_base_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        info_.outputs_ = outs;
        COMPILE_ASSERT(info_.outputs_.size() == 1,
                "softmax op shall have only 1 output.")
        gc::graph::check_logical_tensor_shape_dtype_identical(
                info_.inputs_[0]->details_, info_.outputs_[0]->details_);
    }
    attrs_ = attrs;
    axis_ = attrs_.get_or_else<std::vector<int>>("axis", {1});
}

void softmax_base_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

graph_tensor_ptr softmax_base_t::get_stable_exp_inp(
        const graph_tensor_ptr &input, const std::vector<int> &axis,
        std::shared_ptr<sc_graph_t> &graph) {
    bool numeric_stable = attrs_.get_or_else("numerically_stable", true);
    if (numeric_stable) {
        // x - max(x)
        auto fmax = graph->make("reduce", {input}, {},
                {{"need_mean", false}, {"rd_axis", axis}, {"rd_op", 2}});
        auto fsub = graph->make("sub", {input, fmax->get_outputs()[0]}, {}, {});
        return fsub->get_outputs()[0];
    }
    return input;
}

std::pair<std::shared_ptr<sc_op>, std::shared_ptr<sc_op>>
softmax_op_t::get_exp_reduce(std::shared_ptr<sc_graph_t> &graph,
        const graph_tensor_ptr &input, const std::vector<int> &axis) {
    // The attribute decides whether softmax uses numerically stable process
    // version(do x-max(x) first) or not. Default use the numerically stable
    // version, in some specific cases like mha inference, use the unstable
    // version.
    graph_tensor_ptr fexpinp = get_stable_exp_inp(input, axis, graph);
    bool numeric_stable = attrs_.get_or_else("numerically_stable", true);

    // exp(x)
    auto fexp = graph->make(
            "exp", {fexpinp}, {}, {{"overflow_check", !numeric_stable}});

    // sum(exp(x))
    auto freduce = graph->make("reduce", {fexp->get_outputs()[0]}, {},
            {{"need_mean", false}, {"rd_axis", axis}, {"rd_op", 0}});

    return std::make_pair(fexp, freduce);
}

std::shared_ptr<sc_op> softmax_op_t::get_softmax_result(
        std::shared_ptr<sc_graph_t> &graph, const graph_tensor_ptr &input,
        const std::vector<int> &axis) {
    auto exp_reduce_tuple = get_exp_reduce(graph, input, axis);
    auto f_exp = std::get<0>(exp_reduce_tuple);
    auto f_reduce = std::get<1>(exp_reduce_tuple);
    auto res = graph->make("div",
            {f_exp->get_outputs()[0], f_reduce->get_outputs()[0]}, {}, {});
    return res;
}

void softmax_op_t::make_logical_tensor(std::vector<graph_tensor_ptr> &inputs,
        std::vector<graph_tensor_ptr> &outputs) {
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
}

void softmax_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    make_logical_tensor(inputs, outputs);
    // get axis
    auto &axis = get_axis();
    graph->make_input(inputs);
    // input dtype cast
    graph_tensor_ptr input = cast_input_dtype(inputs[0], graph);
    // calculate
    auto res = get_softmax_result(graph, input, axis);
    // output dtype cast
    res = cast_output_dtype(outputs[0], graph, res);
    graph->make_output(res->get_outputs());
}

void softmax_bwd_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    auto &axis = get_axis();
    // input
    graph->make_input(inputs);
    // dtype cast
    graph_tensor_ptr input0 = cast_input_dtype(inputs[0], graph);
    graph_tensor_ptr input1 = cast_input_dtype(inputs[1], graph);

    auto f_sbr = graph->make("mul", {input0, input1}, {}, {});
    auto f_rd = graph->make("reduce", f_sbr->get_outputs(), {},
            {{"need_mean", false}, {"rd_axis", axis}, {"rd_op", 0}});
    auto f_sub = graph->make("sub", {input0, f_rd->get_outputs()[0]}, {}, {});
    auto f_mul = graph->make("mul", {input1, f_sub->get_outputs()[0]}, {}, {});
    // output
    f_mul = cast_output_dtype(outputs[0], graph, f_mul);

    graph->make_output(f_mul->get_outputs());
}

void log_softmax_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    make_logical_tensor(inputs, outputs);
    auto &axis = get_axis();
    graph->make_input(inputs);
    // input dtype cast
    graph_tensor_ptr input = cast_input_dtype(inputs[0], graph);
    // calculate
    graph_tensor_ptr fexpinp = input;
    auto exp_reduce_pair = get_exp_reduce(graph, input, axis);
    fexpinp = std::get<0>(exp_reduce_pair)->get_inputs()[0];
    auto freduce = std::get<1>(exp_reduce_pair);
    auto f_log = graph->make("log", {freduce->get_outputs()}, {}, {});
    auto f_sub_res
            = graph->make("sub", {fexpinp, f_log->get_outputs()[0]}, {}, {});
    // output dtype cast
    f_sub_res = cast_output_dtype(outputs[0], graph, f_sub_res);
    graph->make_output(f_sub_res->get_outputs());
}

void log_softmax_backward_op_t::get_graph_impl(
        std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    auto &axis = get_axis();
    // input
    graph->make_input(inputs);
    // dtype cast
    graph_tensor_ptr input0 = cast_input_dtype(inputs[0], graph);
    graph_tensor_ptr input1 = cast_input_dtype(inputs[1], graph);

    auto f_sbr = graph->make("reduce", {input0}, {},
            {{"need_mean", false}, {"rd_axis", axis}, {"rd_op", 0}});
    // exp(x)
    auto fexp = graph->make("exp", {input1}, {}, {});

    // dd - expf(d) * reduce(dd)
    auto f_mul = graph->make(
            "mul", {fexp->get_outputs()[0], f_sbr->get_outputs()[0]}, {}, {});
    auto f_sub = graph->make("sub", {input0, f_mul->get_outputs()[0]}, {}, {});

    // output dtype cast
    f_sub = cast_output_dtype(outputs[0], graph, f_sub);

    graph->make_output(f_sub->get_outputs());
}

} // namespace ops

OP_REGISTER(ops::softmax_op_t, softmax)
OP_REGISTER(ops::log_softmax_op_t, log_softmax)
OP_REGISTER(ops::log_softmax_backward_op_t, log_softmax_bwd)
OP_REGISTER(ops::softmax_bwd_op_t, softmax_bwd)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
