/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"
#include "test_allocator.hpp"

namespace impl = dnnl::graph::impl;

TEST(TestAllocator, DefaultSyclAllocator) {
    impl::engine_kind_t kind = get_test_engine_kind();
#ifndef DNNL_GRAPH_CPU_SYCL
    SKIP_IF(kind == impl::engine_kind::cpu,
            "skip sycl api test for native cpu runtime.");
#endif
    namespace impl = dnnl::graph::impl;
    impl::allocator_t &alloc = *impl::allocator_t::create();
    sycl::queue q = kind == impl::engine_kind::gpu
            ? sycl::queue {sycl::gpu_selector {},
                    sycl::property::queue::in_order {}}
            : sycl::queue {
                    sycl::cpu_selector {}, sycl::property::queue::in_order {}};

    impl::allocator_t::mem_attr_t attr {
            impl::allocator_t::mem_type_t::persistent, 4096};
    void *mem_ptr = alloc.allocate(
            static_cast<size_t>(16), q.get_device(), q.get_context(), attr);
    ASSERT_NE(mem_ptr, nullptr);
    sycl::event e;
    alloc.deallocate(mem_ptr, q.get_device(), q.get_context(), e);
    alloc.release();
}

TEST(TestAllocator, SyclAllocator) {
    impl::engine_kind_t kind = get_test_engine_kind();
#ifndef DNNL_GRAPH_CPU_SYCL
    SKIP_IF(kind == impl::engine_kind::cpu,
            "skip sycl api test for native cpu runtime.");
#endif
    impl::allocator_t::mem_attr_t alloc_attr {
            impl::allocator_t::mem_type_t::persistent, 1024};
    ASSERT_EQ(alloc_attr.type_, impl::allocator_t::mem_type_t::persistent);
    ASSERT_EQ(alloc_attr.alignment_, 1024);
    impl::allocator_t &sycl_alloc = *impl::allocator_t::create(
            dnnl::graph::testing::sycl_malloc_wrapper,
            dnnl::graph::testing::sycl_free_wrapper);
    sycl::device sycl_dev = (kind == impl::engine_kind::gpu)
            ? sycl::device {sycl::gpu_selector()}
            : sycl::device {sycl::cpu_selector()};
    sycl::context sycl_ctx {sycl_dev};

    auto *mem_ptr = sycl_alloc.allocate(
            static_cast<size_t>(16), sycl_dev, sycl_ctx, alloc_attr);
    ASSERT_NE(mem_ptr, nullptr);
    sycl::event e;
    sycl_alloc.deallocate(mem_ptr, sycl_dev, sycl_ctx, e);
    sycl_alloc.release();
}
