/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GRAPH_TEST_ALLOCATOR_HPP
#define GRAPH_TEST_ALLOCATOR_HPP

#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#ifdef DNNL_WITH_SYCL
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#endif

namespace dnnl {
namespace graph {
namespace testing {

void *allocate(size_t size, size_t alignment);

void deallocate(void *ptr);

#ifdef DNNL_WITH_SYCL
void *sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx);

void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event);
#endif

} // namespace testing
} // namespace graph
} // namespace dnnl

#endif
