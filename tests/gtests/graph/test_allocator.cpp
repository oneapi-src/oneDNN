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

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "test_allocator.hpp"

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

namespace dnnl {
namespace graph {
namespace testing {

void *allocate(size_t size, size_t alignment) {
    (void)alignment;
    return malloc(size);
}

void deallocate(void *ptr) {
    free(ptr);
}

#ifdef DNNL_WITH_SYCL
/// Below functions aim to simulate what integration layer does
/// before creating dnnl::graph::allocator. We expect integration will
/// wrap the real SYCL malloc_shared/free functions like below, then
/// dnnl_graph_allocator will be created with correct allocation/deallocation
/// function pointers.
void *sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    (void)alignment;
    return malloc_shared(size, *static_cast<const ::sycl::device *>(dev),
            *static_cast<const ::sycl::context *>(ctx));
}

void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    // Device is not used in this example, but it may be useful for some users
    // application.
    UNUSED(device);
    // immediate synchronization here is for test purpose for performance, users
    // may need to store the ptr and event and handle them separately
    if (event) {
        auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
        sycl_deps_ptr->wait();
    }
    free(ptr, *static_cast<const ::sycl::context *>(context));
}

simple_sycl_allocator *get_allocator(const ::sycl::context *ctx) {
    static simple_sycl_allocator aallocator(ctx);
    return &aallocator;
}

void *sycl_allocator_malloc(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    simple_sycl_allocator *aallocator
            = get_allocator(static_cast<const ::sycl::context *>(ctx));
    return aallocator->malloc(size, static_cast<const ::sycl::device *>(dev));
}

void sycl_allocator_free(
        void *ptr, const void *device, const void *ctx, void *event) {
    // Device is not used in this example, but it may be useful for some users
    // application.
    UNUSED(device);
    simple_sycl_allocator *aallocator
            = get_allocator(static_cast<const ::sycl::context *>(ctx));
    aallocator->release(ptr, *static_cast<::sycl::event *>(event));
}
#endif

} // namespace testing
} // namespace graph
} // namespace dnnl
