/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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
#include "interface/c_types_map.hpp"

#include "graph/test_allocator.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;

TEST(TestAllocator, DefaultSyclAllocator) {
    graph::engine_kind_t kind = get_test_engine_kind();
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(kind == graph::engine_kind::cpu,
            "skip sycl api test for native cpu runtime.");
#endif
    graph::allocator_t *alloc = new graph::allocator_t();
    sycl::queue q = kind == graph::engine_kind::gpu
            ? sycl::queue {dnnl::impl::sycl::compat::gpu_selector_v,
                    sycl::property::queue::in_order {}}
            : sycl::queue {dnnl::impl::sycl::compat::cpu_selector_v,
                    sycl::property::queue::in_order {}};

    graph::allocator_t::mem_attr_t attr {
            graph::allocator_t::mem_type_t::persistent, 64};
    void *mem_ptr = alloc->allocate(
            static_cast<size_t>(16), q.get_device(), q.get_context(), attr);

    if (mem_ptr == nullptr) {
        // release alloc before asserting.
        delete alloc;
        ASSERT_NE(mem_ptr, nullptr);
    } else {
        sycl::event e;
        alloc->deallocate(mem_ptr, q.get_device(), q.get_context(), e);
        delete alloc;
    }
}

TEST(TestAllocator, SyclAllocator) {
    graph::engine_kind_t kind = get_test_engine_kind();
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(kind == graph::engine_kind::cpu,
            "skip sycl api test for native cpu runtime.");
#endif
    graph::allocator_t::mem_attr_t alloc_attr {
            graph::allocator_t::mem_type_t::persistent, 1024};
    ASSERT_EQ(alloc_attr.type_, graph::allocator_t::mem_type_t::persistent);

    ASSERT_EQ(alloc_attr.alignment_, 1024U);
    graph::allocator_t *sycl_alloc
            = new graph::allocator_t(dnnl::graph::testing::sycl_malloc_wrapper,
                    dnnl::graph::testing::sycl_free_wrapper);
    sycl::device sycl_dev = (kind == graph::engine_kind::gpu)
            ? sycl::device {dnnl::impl::sycl::compat::gpu_selector_v}
            : sycl::device {dnnl::impl::sycl::compat::cpu_selector_v};
    sycl::context sycl_ctx {sycl_dev};

    auto *mem_ptr = sycl_alloc->allocate(
            static_cast<size_t>(16), sycl_dev, sycl_ctx, alloc_attr);

    if (mem_ptr == nullptr) {
        // release sycl_alloc before asserting.
        delete sycl_alloc;
        ASSERT_NE(mem_ptr, nullptr);
    } else {
        sycl::event e;
        sycl_alloc->deallocate(mem_ptr, sycl_dev, sycl_ctx, e);
        delete sycl_alloc;
    }
}
