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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_INTERNAL_INFO_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_INTERNAL_INFO_HPP
#include <memory>
#include <string>
#include <vector>
#include <compiler/ir/graph/tensor_detail.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
struct dyn_internal_info_t {
    // Internal function could be generated from a fused partition. Here we
    // record final arguments for function decl and call.
    std::vector<logical_tensor_t> parti_in_ltsrs_;
    std::vector<logical_tensor_t> parti_out_ltsrs_;
    // Inner(fused op) dispatch table name in fused op.
    std::string dispatch_table_name_;
};
using dyn_internal_info_ptr = std::shared_ptr<dyn_internal_info_t>;
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
