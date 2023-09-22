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
#include "reduce_mean.hpp"
#include <utility>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

reduce_mean_op_t::reduce_mean_op_t(
        graph_tensor_ptr v, const std::vector<int> &rd_axis, bool keep_dims)
    : reduce_mean_op_t({std::move(v)}, {},
            {{"rd_axis", rd_axis}, {"keep_dims", keep_dims}}) {
    // default is need_allocate
    info_.tensor_share_info_ = {};
}

reduce_mean_op_t::reduce_mean_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
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
    op_name_ = "reduce_mean";
}

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
    auto reduce_num = graph->make<constant_op_t>(
            std::make_shared<static_data_t>(std::vector<float> {item_cnt}),
            datatypes::f32, sc_dims {1});
    auto reduce_sum = graph->make("reduce_sum", inputs, {}, attrs_);
    auto reduce_sum_div = graph->make("div",
            {reduce_sum->get_outputs()[0], reduce_num->get_outputs()[0]}, {},
            {});
    // output
    graph->make_output(reduce_sum_div->get_outputs());
}

void reduce_mean_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {}

OP_REGISTER(reduce_mean_op_t, reduce_mean)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
