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

#include "../graph.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <ops/fusible/memory_movement.hpp>

#include <algorithm>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.eliminate_zero_shaped_tensors);

/*
It's possible that the shape of some tensors contains zeros, for example, (4,
32, 0, 128). These tensors are actually empty in size. We need to eliminate them
after infering shape and before other passes.
*/

SC_INTERNAL_API void eliminate_zero_shaped_tensors(
        sc_graph_t &graph, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &op) {
        for (auto in_tsr_itr = op->info_.inputs_.begin();
                in_tsr_itr != op->info_.inputs_.end();) {
            auto &in_tsr = *in_tsr_itr;
            auto &dims = in_tsr->details_.get_plain_dims();
            if (std::all_of(dims.begin(), dims.end(),
                        [](sc_dim d) { return d != 0; })) {
                ++in_tsr_itr;
                continue;
            }
            SC_MODULE_WARN << "Input tensor of op: " << op->op_name_
                           << op->logical_op_id_
                           << " has 0 in shape. If this is not a Concat op, "
                              "the semantics should be checked.";
            auto &uses = in_tsr->uses_;
            if (op->isa<concat_op_t>() || op->get_inputs().size() > 2) {
                SC_MODULE_INFO << "Disconnect this input tensor from op.";
                in_tsr->detach_use(op);
                // Since we will delete this input, the inputs after it should
                // modify their uses_ by decrease index by 1
                for (auto latter_itr = in_tsr_itr + 1;
                        latter_itr != op->info_.inputs_.end(); ++latter_itr) {
                    auto &latter_uses = (*latter_itr)->uses_;
                    for (auto &idx_op : latter_uses) {
                        idx_op.first -= 1;
                    }
                }
                in_tsr_itr = op->info_.inputs_.erase(in_tsr_itr);
            } else if (op->get_inputs().size() == 1) {
                SC_MODULE_INFO
                        << "Disconnect this input tensor from op; Delete this "
                           "op; The children ops directly use this tensor.";
                for (auto &out_tsr : op->get_outputs()) {
                    for (auto &idx_op : out_tsr->uses_) {
                        idx_op.second->replace_input(idx_op.first, in_tsr);
                    }
                }
                op->remove();
            } else if (op->get_inputs().size() == 2) {
                // TODO(niuxiaoguang): Add test to cover this case: only one
                // input tensor is left. Currently in UT, both of inputs to Add
                // op are zero-shaped and are deleted.
                SC_MODULE_INFO << "Disconnect this input tensor from op; "
                                  "Delete this op; The children ops directly "
                                  "use another input tensor.";
                in_tsr->detach_use(op);
                in_tsr_itr = op->info_.inputs_.erase(in_tsr_itr);
                if (!op->info_.inputs_.empty()) { // only one input left
                    auto &input_left = op->info_.inputs_.front();
                    for (auto &out_tsr : op->get_outputs()) {
                        // replace_input may change the uses, we need to copy it
                        auto uses = out_tsr->uses_;
                        for (auto &idx_op : uses) {
                            idx_op.second->replace_input(
                                    idx_op.first, input_left);
                        }
                    }
                }
                op->remove();
                break; // do not check another input tensor
            }
        }
        vis->update_state_for_visited(op);
    });
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
