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

#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_CHECKED_PTR_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_CHECKED_PTR_HPP
#include <runtime/aligned_ptr.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
struct checked_ptr_policy_t {
    static void *alloc(size_t sz, size_t alignment);
    static void dealloc(void *ptr, size_t sz);
};

using generic_checked_ptr_t = raii_ptr_t<checked_ptr_policy_t>;
template <typename T>
using checked_ptr_t = aligned_ptr_t<T, generic_checked_ptr_t>;
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
