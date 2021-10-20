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

#include <mutex>
#include <thread>

#include "interface/allocator.hpp"
#include "interface/c_types_map.hpp"

#include "utils.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

TEST(allocator_test, default_cpu_allocator) {
    dnnl::graph::impl::allocator_t alloc {};

    dnnl::graph::impl::allocator_t::attribute_t attr {
            dnnl::graph::impl::allocator_lifetime::persistent, 4096};
    void *mem_ptr = alloc.allocate(static_cast<size_t>(16));
    ASSERT_NE(mem_ptr, nullptr);
    alloc.deallocate(mem_ptr);
}

TEST(allocator_test, create_attr) {
    dnnl::graph::impl::allocator_t::attribute_t attr {
            dnnl::graph::impl::allocator_lifetime::output, 1024};

    ASSERT_EQ(attr.data.alignment, 1024);
    ASSERT_EQ(attr.data.type, dnnl::graph::impl::allocator_lifetime::output);
}

#if DNNL_GRAPH_WITH_SYCL
TEST(allocator_test, default_sycl_allocator) {
    namespace sycl = cl::sycl;
    namespace impl = dnnl::graph::impl;
    impl::allocator_t alloc {};
    sycl::queue q {sycl::gpu_selector {}, sycl::property::queue::in_order {}};

    impl::allocator_t::attribute_t attr {
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
    dnnl::graph::impl::allocator_t::attribute_t alloc_attr {
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

#ifndef NDEBUG
TEST(allocator_test, monitor) {
    using namespace dnnl::graph::impl;

    const size_t temp_size = 1024, persist_size = 512;

    allocator_t alloc;
    std::vector<void *> persist_bufs;
    std::mutex m;

    auto callee = [&]() {
        // allocate persistent buffer
        void *p_buf = alloc.allocate(persist_size,
                allocator_t::attribute_t {
                        allocator_lifetime::persistent, 4096});
        {
            std::lock_guard<std::mutex> lock(m);
            persist_bufs.emplace_back(p_buf);
        }

        // allocate temporary buffer
        void *t_buf = alloc.allocate(temp_size,
                allocator_t::attribute_t {allocator_lifetime::temp, 4096});
        for (size_t i = 0; i < temp_size; i++) {
            char *ptr = (char *)t_buf + i;
            *ptr = *ptr + 2;
        }
        // deallocate temporary buffer
        alloc.deallocate(t_buf);
    };

    // single thread
    for (size_t iter = 0; iter < 4; iter++) {
        allocator_t::monitor_t::reset_peak_temp_memory(&alloc);
        ASSERT_EQ(allocator_t::monitor_t::get_peak_temp_memory(&alloc), 0);

        callee(); // call the callee to do memory operation

        ASSERT_EQ(allocator_t::monitor_t::get_peak_temp_memory(&alloc),
                temp_size);
        ASSERT_EQ(allocator_t::monitor_t::get_total_persist_memory(&alloc),
                persist_size * (iter + 1));
    }

    for (auto p_buf : persist_bufs) {
        alloc.deallocate(p_buf);
    }
    persist_bufs.clear();

    // multiple threads
    auto thread_func = [&]() {
        allocator_t::monitor_t::reset_peak_temp_memory(&alloc);
        ASSERT_EQ(allocator_t::monitor_t::get_peak_temp_memory(&alloc), 0);
        callee();
        ASSERT_EQ(allocator_t::monitor_t::get_peak_temp_memory(&alloc),
                temp_size);
    };

    std::thread t1(thread_func);
    std::thread t2(thread_func);

    t1.join();
    t2.join();

    // two threads allocated persist buffer
    ASSERT_EQ(allocator_t::monitor_t::get_total_persist_memory(&alloc),
            persist_size * 2);

    for (auto p_buf : persist_bufs) {
        alloc.deallocate(p_buf);
    }
    persist_bufs.clear();
}
#endif
