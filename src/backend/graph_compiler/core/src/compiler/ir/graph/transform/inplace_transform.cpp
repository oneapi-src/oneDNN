/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#include "../fusible_op.hpp"
#include "../graph.hpp"
#include "../visitor.hpp"
#include <ops/fusible/memory_movement.hpp>
namespace sc {
void tensor_view_to_copy(sc_graph_t &graph) {
    int count = 0;
    size_t idx = -1;
    for (size_t i = 0; i < graph.ops_.size(); i++) {
        auto &op = graph.ops_[i];
        if (!(op->isa<input_op>() || op->isa<output_op>())) {
            count++;
            idx = i;
            if (count > 1) { return; }
        }
    }
    assert(count == 1);
    if (auto node = graph.ops_[idx]->dyn_cast<tensor_view_op_t>()) {
        auto copy_node = graph.make("reshape", node->get_inputs(), {},
                {{"shape", node->get_shapes()}});
        node->replace_uses_with_and_remove(copy_node);
    }
    graph.reset_op_ids();
}
// currently only support for single tensorview op
void inplace_transform(sc_graph_t &graph, const context_ptr &ctx) {
    tensor_view_to_copy(graph);
}
} // namespace sc
