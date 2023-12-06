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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_ATTR_KEYS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_ATTR_KEYS_HPP

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace attr_keys {
// bool. Default false. Applied on local tensors. If true, the tensor must be
// allocated using runtime memory allocator instead of the native stack
constexpr const char *runtime_stack_alloc = "runtime_stack_alloc";
// string. The name of the next function to call. Applied on evaluate-call
// nodes. If true, the next op to be called has no dependency on the current op.
// It can be merged with the previous op to remove the barrier.
constexpr const char *no_post_barrier = "no_post_barrier";
// bool. Default false. It should be applied on tensors, indicating that the
// tensor should be 1xs32 and its element's value is auto-assigned by the
// compiler. The content of the tensor is bind to whether the const cache is
// valid and already initialized.
constexpr const char *is_init_for_const_cache = "is_init_for_const_cache";
// std::shared_ptr<cached_const_graph_tensor>. It should be marked on shared
// constant tensor nodes
constexpr const char *shared_const = "shared_const";
// bool. Default false. Marked on stmt nodes that is related to shared constant
// lazy initialization flag __is_init. module_global_resolver uses this flag to
// find the __is_init related stmt.
constexpr const char *is_shared_const_init_stmt = "is_shared_const_init_stmt";
// size_t. Marked by module_global_resolver on tensor nodes
// ("__shared_const_base_*") that are base tensors of buffers in shared const
// tensor cache. It is the index of the handle to these base tensors. It will be
// used in local_tensor_lower
constexpr const char *shared_const_base_idx = "shared_const_base_idx";
// bool. Default false. Marked "true" on top-level parallel for_loop nodes that
// the user would like to use dynamic parallel dispatching
constexpr const char *dynamic_parallel = "dynamic_parallel";
// bool. Default false. Applied on base tensor.
// If true, the tensor is read only in the funtion.
constexpr const char *read_only_tensor = "read_only_tensor";
// bool. whether codegen enable fast math.
// If false, codegen will not use fast math in the calculation. (Currently only
// works in binary expr calculation of llvm codegen.)
constexpr const char *fast_math = "fast_math";
} // namespace attr_keys

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
