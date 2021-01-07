/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <gtest/gtest.h>

#include "interface/allocator.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

TEST(allocator_test, default_cpu_allocator) {
    dnnl::graph::impl::allocator_t alloc {};

    dnnl::graph::impl::allocator_t::attribute attr {
            dnnl::graph::impl::allocator_lifetime::persistent, 4096};
    void *mem_ptr = alloc.allocate(static_cast<size_t>(16));
    ASSERT_NE(mem_ptr, nullptr);
    alloc.deallocate(mem_ptr);
}

TEST(allocator_test, create_attr) {
    dnnl::graph::impl::allocator_t::attribute attr {
            dnnl::graph::impl::allocator_lifetime::output, 1024};

    ASSERT_EQ(attr.data.alignment, 1024);
    ASSERT_EQ(attr.data.type, dnnl::graph::impl::allocator_lifetime::output);
}

#if DNNL_GRAPH_WITH_SYCL
TEST(allocator_test, default_sycl_allocator) {
    namespace sycl = cl::sycl;
    namespace impl = dnnl::graph::impl;
    impl::allocator_t alloc {};
    sycl::queue q {sycl::gpu_selector {}};

    impl::allocator_t::attribute attr {
            impl::allocator_lifetime::persistent, 4096};
    void *mem_ptr = alloc.allocate(
            static_cast<size_t>(16), q.get_device(), q.get_context(), attr);
    ASSERT_NE(mem_ptr, nullptr);
    alloc.deallocate(mem_ptr, q.get_context());
}

/// Below functions aim to simulate what integration layer does
/// before creating dnnl::graph::allocator. We expect integration will
/// wrap the real SYCL malloc_device/free functions like below, then
/// dnnl_graph_allocator will be created with correct allocation/deallocation
/// function pointers.
void *sycl_malloc_wrapper(size_t n, const void *dev, const void *ctx,
        dnnl::graph::impl::allocator_attr_t attr) {
    // need to handle different allocation types according to attr.type
    if (attr.type == dnnl::graph::impl::allocator_lifetime::persistent) {
        // persistent memory
    } else if (attr.type == dnnl::graph::impl::allocator_lifetime::output) {
        // output tensor memory
    } else {
        // temporary memory
    }

    return malloc_device(n, *static_cast<const cl::sycl::device *>(dev),
            *static_cast<const cl::sycl::context *>(ctx));
}

void sycl_free_wrapper(void *ptr, const void *context) {
    free(ptr, *static_cast<const cl::sycl::context *>(context));
}

TEST(allocator_test, sycl_allocator) {
    namespace sycl = cl::sycl;

    std::unique_ptr<sycl::device> sycl_dev;
    std::unique_ptr<sycl::context> sycl_ctx;

    auto platform_list = sycl::platform::get_platforms();
    dnnl::graph::impl::allocator_t::attribute alloc_attr {
            dnnl::graph::impl::allocator_lifetime::persistent, 1024};
    ASSERT_EQ(alloc_attr.data.type,
            dnnl::graph::impl::allocator_lifetime::persistent);
    ASSERT_EQ(alloc_attr.data.alignment, 1024);
    dnnl::graph::impl::allocator_t sycl_alloc {
            sycl_malloc_wrapper, sycl_free_wrapper};
    for (const auto &plt : platform_list) {
        auto device_list = plt.get_devices();
        for (const auto &dev : device_list) {
            if (dev.is_gpu()) {
                sycl_dev.reset(new sycl::device(dev));
                sycl_ctx.reset(new sycl::context(*sycl_dev));

                auto *mem_ptr = sycl_alloc.allocate(static_cast<size_t>(16),
                        *sycl_dev, *sycl_ctx, alloc_attr);
                ASSERT_NE(mem_ptr, nullptr);
                sycl_alloc.deallocate(mem_ptr, *sycl_ctx);
            }
        }
    }
}
#endif
