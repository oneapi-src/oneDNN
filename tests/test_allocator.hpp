/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef TEST_ALLOCATOR_HPP
#define TEST_ALLOCATOR_HPP

#include "oneapi/dnnl/dnnl_graph.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

namespace dnnl {
namespace graph {
namespace testing {

void *allocate(size_t n, dnnl::graph::allocator::attribute attr);

void deallocate(void *ptr);

#ifdef DNNL_GRAPH_WITH_SYCL
void *sycl_malloc_wrapper(size_t n, const void *dev, const void *ctx,
        dnnl::graph::allocator::attribute attr);

void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event);
#endif

} // namespace testing
} // namespace graph
} // namespace dnnl

#endif
