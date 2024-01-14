/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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
#include "reduce_graph_op.hpp"
#include <string>
#include <utility>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

graph_reduce_base_t::graph_reduce_base_t(graph_tensor_ptr v,
        const std::vector<int> &rd_axis, const std::string &op_name,
        bool keep_dims)
    : graph_reduce_base_t({std::move(v)}, {}, op_name,
            {{"rd_axis", rd_axis}, {"keep_dims", keep_dims}}) {}

graph_reduce_base_t::graph_reduce_base_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const std::string &op_name,
        const any_map_t &attrs) {
    info_.inputs_ = ins;
    attrs_ = attrs;
    COMPILE_ASSERT(attrs.has_key("rd_axis"),
            "attrs of reduce op should have reduce axis information.");
    if (outs.empty()) {
        auto rd_axis = attrs.get<std::vector<int>>("rd_axis");
        auto keep_dims_ = attrs.get_or_else("keep_dims", true);
        sc_dims out_dims;
        if (keep_dims_) {
            out_dims = ins[0]->details_.get_plain_dims();
            for (size_t i = 0; i < rd_axis.size(); i++) {
                out_dims[rd_axis[i]] = 1;
            }
        } else {
            for (size_t i = 0; i < ins[0]->details_.get_plain_dims().size();
                    i++) {
                if (find(rd_axis.begin(), rd_axis.end(), i) != rd_axis.end()) {
                    continue;
                }
                out_dims.emplace_back(ins[0]->details_.get_plain_dims()[i]);
            }
        }
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this,
                ins[0]->details_.get_format(), out_dims,
                ins[0]->details_.dtype_));
    } else {
        info_.outputs_ = outs;
        if (ins[0]->details_.get_plain_dims()
                == outs[0]->details_.get_plain_dims()) {
            attrs_["keep_dims"] = true;
        }
    }
    op_name_ = op_name;
}

void graph_reduce_base_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

void reduce_mean_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    float item_cnt = 1;
    auto plain_rd_axis_ = attrs_.get<std::vector<int>>("rd_axis");
    for (auto ax : plain_rd_axis_) {
        item_cnt *= inputs[0]->details_.get_plain_dims()[ax];
    }
    // input
    graph_tensor_ptr inputs0 = inputs[0];
    // cast input
    inputs0 = cast_input_dtype(inputs[0], graph);
    auto reduce_num = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {item_cnt}),
            datatypes::f32, sc_dims {1});
    auto reduce_sum = graph->make("reduce_sum", {inputs0}, {}, attrs_);
    auto reduce_sum_div = graph->make("div",
            {reduce_sum->get_outputs()[0], reduce_num->get_outputs()[0]}, {},
            {});
    // output
    sc_op_ptr output_op = reduce_sum_div;
    output_op = cast_output_dtype(outputs[0], graph, output_op);
    graph->make_output(output_op->get_outputs());
}

void reduce_l2_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    auto plain_rd_axis_ = attrs_.get<std::vector<int>>("rd_axis");

    // input
    graph_tensor_ptr inputs0 = inputs[0];
    // cast input
    inputs0 = cast_input_dtype(inputs[0], graph);
    auto square_val = graph->make("square", {inputs0}, {}, {});
    auto reduce_val
            = graph->make("reduce_sum", square_val->get_outputs(), {}, attrs_);
    auto l2_res
            = graph->make("squared_root", reduce_val->get_outputs(), {}, {});
    // cast output
    l2_res = cast_output_dtype(outputs[0], graph, l2_res);
    // output
    graph->make_output(l2_res->get_outputs());
}

void reduce_l1_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    auto plain_rd_axis_ = attrs_.get<std::vector<int>>("rd_axis");

    // input
    graph_tensor_ptr inputs0 = inputs[0];
    // cast input
    inputs0 = cast_input_dtype(inputs[0], graph);
    auto abs_val = graph->make("abs", {inputs0}, {}, {});
    auto reduce_l1
            = graph->make("reduce_sum", abs_val->get_outputs(), {}, attrs_);
    // cast output
    reduce_l1 = cast_output_dtype(outputs[0], graph, reduce_l1);
    // output
    graph->make_output(reduce_l1->get_outputs());
}

OP_REGISTER(reduce_l1_op_t, reduce_l1)
OP_REGISTER(reduce_l2_op_t, reduce_l2)
OP_REGISTER(reduce_mean_op_t, reduce_mean)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
