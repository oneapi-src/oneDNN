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

#include <vector>
#include "transform.hpp"
#include <ops/fusible/reduce.hpp>
#include <runtime/config.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

void partial_reduce_replace(sc_graph_t &graph, const context_ptr &ctx) {
    auto num_threads = runtime_config_t::get().get_num_threads();
    if (num_threads <= 1 || graph.is_dynamic()) { return; }
    auto ops = graph.ops_;
    for (auto &op : ops) {
        if (auto rdop = op->dyn_cast<reduce_op_t>()) {
            auto rxax = rdop->get_rd_axis();
            bool is_first_axis = false;
            auto &in_dims = rdop->get_inputs()[0]->details_.get_blocking_dims();
            sc_dim reduction_size = 1;
            for (auto ax : rxax) {
                if (ax == 0) { is_first_axis = true; }
                reduction_size *= in_dims[ax];
            }
            if (is_first_axis && reduction_size >= num_threads * 16) {
                rdop->split_op(ctx, graph, num_threads);
            }
        }
    }
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
