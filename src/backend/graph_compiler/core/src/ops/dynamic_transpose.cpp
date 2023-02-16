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

#include <memory>

#include "dynamic_transpose.hpp"
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/statics_table.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
dynamic_transpose_op::dynamic_transpose_op(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : sc_op("dynamic_transpose_op", ins, outs, attrs) {
    COMPILE_ASSERT(
            ins.size() == 2, "Dynamic transpose op shall take 2 inputs.");
    sc_op *order_node = ins[1]->producer_owner_;
    COMPILE_ASSERT(
            order_node->isa<input_op>() || order_node->isa<constant_op_t>(),
            "Dynamic transpose expects input or constant node as the 2nd "
            "input.");
    COMPILE_ASSERT(order_node->attrs_.has_key("values"),
            "Dynamic transpose's 2nd input is expected to have value "
            "attributes.");
    int32_t *order_value = reinterpret_cast<int32_t *>(
            order_node->attrs_.get<std::shared_ptr<static_data_t>>("values")
                    ->data_);
    sc_dims outshape(ins[0]->details_.get_plain_dims().size());
    for (size_t i = 0; i < outshape.size(); ++i) {
        order_.emplace_back(order_value[i]);
        assert(order_.back() >= 0 && order_.back() < (int)outshape.size());
        outshape[i] = ins[0]->details_.get_plain_dims()[order_.back()];
    }

    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(graph_tensor::make(outshape,
                sc_data_format_t(), get_inputs()[0]->details_.dtype_));
    } else {
        COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims() == outshape,
                "Dynamic transpose's output shape does not confirm with the "
                "permutation order.")
    }
}

void dynamic_transpose_op::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    throw std::runtime_error("Not implemented");
}
ir_module_ptr dynamic_transpose_op::get_func(context_ptr ctx) {
    throw std::runtime_error("Not implemented");
}
sc_op_ptr dynamic_transpose_op::constant_optimize(sc_graph_t &graph) {
    // temporarily use contant optimize pass to do the dynamic_transpose ->
    // transpose replacement
    auto new_input = graph.make(
            "transpose", {get_inputs()[0]}, {}, {{"order", order_}});
    this->replace_uses_with_and_remove(new_input);
    return new_input;
}
} // namespace ops

OP_REGISTER(ops::dynamic_transpose_op, dynamic_transpose);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
