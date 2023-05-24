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
} // namespace attr_keys

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
