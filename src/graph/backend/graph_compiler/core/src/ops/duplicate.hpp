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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_DUPLICATE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_DUPLICATE_HPP

#include <memory>
#include <vector>
#include <compiler/ir/graph/graph_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
// This op is used to duplicate buffers for the cases one buffer is used for two
// or more inputs/outputs.
class duplicate_op : public graph_op_t, public op_traits::auto_copyable_t {
public:
    duplicate_op(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
};

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
