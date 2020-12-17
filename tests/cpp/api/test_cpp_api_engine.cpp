/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <vector>
#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif

TEST(api_engine, simple_create) {
    using namespace dnnl::graph;
    engine e {engine::kind::cpu, 0};

    allocator alloc {};
    e.set_allocator(alloc);
    ASSERT_EQ(e.get_device_id(), 0);
    ASSERT_EQ(e.get_kind(), engine::kind::cpu);
    ASSERT_FALSE(e.get_device_handle());
}

#if DNNL_GRAPH_WITH_SYCL
/// Below functions aim to simulate what integration layer does
/// before creating dnnl::graph::allocator. We expect integration will
/// wrap the real SYCL malloc_device/free functions like below, then
/// dnnl_graph_allocator will be created with correct allocation/deallocation
/// function pointers.
void *sycl_malloc_wrapper(size_t n, const void *dev, const void *ctx,
        dnnl::graph::allocator::attribute attr) {
    namespace api = dnnl::graph;
    // need to handle different allocation types according to attr.type
    if (attr.type
            == api::allocator::convert_to_c(
                    api::allocator::lifetime::persistent)) {
        // persistent memory
    } else if (attr.type
            == api::allocator::convert_to_c(api::allocator::lifetime::output)) {
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

TEST(api_engine, create_with_sycl) {
    using namespace dnnl::graph;
    using namespace cl::sycl;

    std::unique_ptr<device> sycl_dev;
    std::unique_ptr<context> sycl_ctx;

    auto platform_list = platform::get_platforms();
    for (const auto &plt : platform_list) {
        auto device_list = plt.get_devices();
        for (const auto &dev : device_list) {
            if (dev.is_gpu()) {
                sycl_dev.reset(new device(dev));
                sycl_ctx.reset(new context(*sycl_dev));
                engine e = dnnl::graph::sycl_interop::make_engine(
                        *sycl_dev, *sycl_ctx);
                allocator alloc = dnnl::graph::sycl_interop::make_allocator(
                        sycl_malloc_wrapper, sycl_free_wrapper);
                e.set_allocator(alloc);
                ASSERT_EQ(e.get_device_id(), 0);
            }
        }
    }
}
#endif
