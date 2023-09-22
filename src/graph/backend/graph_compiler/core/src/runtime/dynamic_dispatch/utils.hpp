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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_UTILS_HPP

#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
union dispatch_key;
void deep_copy_dynamic_tensor(
        dynamic_tensor_t *out, const dynamic_tensor_t *in);

uint64_t calculate_blocking_dims(void *placeholder, uint64_t *format);

dispatch_key get_impl_dispatch_key(int impl);
} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
