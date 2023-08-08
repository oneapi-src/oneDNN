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

#include <fstream>
#include "pass.hpp"
#include <ops/convolution.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_INTERNAL_API void intrusive_opt_level(
        sc_graph_t &graph, const context_ptr &ctx) {
    for (auto &op : graph.ops_) {
        if (ctx->flags_.opt_level_ < sc_opt_level::lv3) {
            if (op->isa<ops::conv_fwd_core_op_t>()) {
                op->attrs_.set("image_affinity", false);
            }
        }
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
