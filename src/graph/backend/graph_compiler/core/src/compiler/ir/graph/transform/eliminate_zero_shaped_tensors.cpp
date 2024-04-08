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

#include "../graph.hpp"
#include "../visitor.hpp"
#include "transform.hpp"
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/padding.hpp>
#include <ops/fusible/shape_of_tensor.hpp>
#include <ops/fusible/ternary_elemwise.hpp>
#include <ops/graph_convolution.hpp>
#include <ops/matmul.hpp>

#include <algorithm>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.eliminate_zero_shaped_tensors);

static bool is_tensor_zero_shape(const graph_tensor_ptr &gt) {
    const sc_dims dims = gt->details_.get_plain_dims();
    if (std::all_of(
                dims.begin(), dims.end(), [](sc_dim d) { return d != 0; })) {
        return false;
    }
    return true;
}

/*
It's possible that the shape of some tensors contains zeros, for example, (4,
32, 0, 128). These tensors are actually empty in size. We need to eliminate them
after infering shape and before other passes.
*/

SC_INTERNAL_API void eliminate_zero_shaped_tensors(
        sc_graph_t &graph, const context_ptr &ctx) {
    op_visitor_t vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &op) {
        std::vector<bool> is_zero_shape(op->info_.inputs_.size(), false);
        for (size_t idx = 0; idx < op->info_.inputs_.size(); ++idx) {
            if (is_tensor_zero_shape(op->info_.inputs_[idx])) {
                is_zero_shape[idx] = true;
            }
        }
        if (std::all_of(is_zero_shape.begin(), is_zero_shape.end(),
                    [](bool is_zero) { return !is_zero; })) {
            return;
        }
        if (op->isa<concat_op_t>() || op->isa<output_op>()) {
            graph_tensor_ptr input_left = nullptr;
            for (auto in_tsr_itr = op->info_.inputs_.begin();
                    in_tsr_itr != op->info_.inputs_.end();) {
                auto &in_tsr = *in_tsr_itr;
                if (!is_tensor_zero_shape(in_tsr)) {
                    ++in_tsr_itr;
                    continue;
                }
                if (op->info_.inputs_.size() == 1) {
                    // we have 1 tensor left and it is a zero-shape tensor
                    input_left = in_tsr;
                    break;
                }
                SC_MODULE_INFO << "Disconnect input from concat or output op.";
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
            }
            if (input_left) {
                vis->update_state_for_visited(op);
                for (auto &out_tsr : op->get_outputs()) {
                    auto uses = out_tsr->uses_;
                    for (auto &idx_op : uses) {
                        idx_op.second->replace_input(idx_op.first, input_left);
                    }
                }
                op->remove();
            }
        } else if (op->isa<padding_op_t>() || op->isa<shape_of_tensor_op_t>()) {
            COMPILE_ASSERT(0,
                    "padding and shape op do not support zero-shape tensor "
                    "input.");
        } else {
            vis->update_state_for_visited(op);
            if (!op->isa<binary_elementwise_op_impl_t>()
                    && !op->isa<ops::matmul_op>() && !op->isa<select_op_t>()) {
                COMPILE_ASSERT(is_zero_shape[0],
                        "Current op's first input must be a zero tensor.");
            }
            int64_t input_idx_left = -1;
            for (size_t i = 0; i < is_zero_shape.size(); ++i) {
                if (is_zero_shape[i]) {
                    input_idx_left = i;
                    break;
                }
            }
            COMPILE_ASSERT(input_idx_left >= 0,
                    "The current op must have a zero-shape input.");
            int64_t visiting_idx = 0;
            for (auto in_tsr_itr = op->info_.inputs_.begin();
                    in_tsr_itr != op->info_.inputs_.end();) {
                if (visiting_idx == input_idx_left) {
                    in_tsr_itr++;
                } else {
                    (*in_tsr_itr)->detach_use(op);
                    in_tsr_itr = op->info_.inputs_.erase(in_tsr_itr);
                }
                visiting_idx++;
            }
            COMPILE_ASSERT(op->info_.inputs_.size() == 1,
                    "The zero-input op must have only 1 input left.");
            auto &input_left = op->info_.inputs_[0];
            // replace_input may change the uses, we need to copy it
            for (const auto &output : op->info_.outputs_) {
                auto uses = output->uses_;
                for (auto &cld : uses) {
                    cld.second->replace_input(cld.first, input_left);
                }
            }
            op->remove();
        }
    });
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
