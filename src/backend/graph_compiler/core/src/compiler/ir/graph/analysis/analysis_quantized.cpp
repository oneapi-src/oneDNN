/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#include "../traits.hpp"
#include "analysis.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
void analysis_quantized(sc_graph_t &graph, const context_ptr &ctx) {
    for (auto &op : graph.ops_) {
        if (op->op_name_.find("quantize") != std::string::npos) {
            graph.attrs_[sc_graph_t::attr_key_t::quantize] = true;
            break;
        }
    }
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
