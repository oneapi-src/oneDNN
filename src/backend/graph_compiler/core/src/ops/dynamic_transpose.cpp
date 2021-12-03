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
#include "dynamic_transpose.hpp"
#include <compiler/ir/graph/fusible_op.hpp>

namespace sc {
namespace ops {
dynamic_transpose_op::dynamic_transpose_op(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : sc_op("dynamic_transpose_op", ins, outs, attrs) {
    COMPILE_ASSERT(ins.size() == 2,
            "Do not use llga transpose workaround if you follow "
            "traditional sc definition");
    // Workaround logic for getting transpose axes:
    // since the value of ins[1] is yet not visible in compile stage, we choose
    // to infer transpose axes based on the plain dims of ins[0] + outs[0]
    // Notice: the following logic fails when the swapped dims have the same
    // dimension size (e.g. we cannot use this logic to infer the swapping of 0
    // and 1 axis of logical tensor [1024, 1024] )
    COMPILE_ASSERT(
            !outs.empty(), "transpose's output is not set, cannot infer axes");
    for (int i = 0; i < (int)ins[0]->details_.get_plain_dims().size(); ++i) {
        if (ins[0]->details_.get_plain_dims()[i]
                != outs[0]->details_.get_plain_dims()[i]) {
            real_axes_.emplace_back(i);
        }
    }
    COMPILE_ASSERT(real_axes_.size() == 2,
            "transpose does not support swapping more than 2 axes");
}
void dynamic_transpose_op::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    throw std::runtime_error("Not implemented");
}
ir_module_ptr dynamic_transpose_op::get_func(context_ptr ctx) {
    throw std::runtime_error("Not implemented");
}
sc_op_ptr dynamic_transpose_op::constant_optimize(sc_graph_t &graph) {
    // temporarily use contant optimize pass to do the dynamic_transpose ->
    // transpose replacement
    auto new_input = graph.make(
            "transpose", {get_inputs()[0]}, {}, {{"axes", real_axes_}});
    this->replace_uses_with_and_remove(new_input);
    return new_input;
}
} // namespace ops

OP_REGISTER(::sc::ops::dynamic_transpose_op, dynamic_transpose);
} // namespace sc
