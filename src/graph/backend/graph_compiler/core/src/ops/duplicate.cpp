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
#include "duplicate.hpp"
#include <memory>
#include <utility>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

duplicate_op::duplicate_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "duplicate";
}

void duplicate_op::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    // input
    graph->make_input(inputs);
    // Use reorder to do coping.
    auto copy_op = graph->make("reorder", inputs, outputs,
            any_map_t({{"format", inputs[0]->details_.get_format()},
                    {"internal", true}, {"actually_copy", true}}));
    // output
    graph->make_output(copy_op->get_outputs());
}
} // namespace ops

OP_REGISTER(ops::duplicate_op, duplicate)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
