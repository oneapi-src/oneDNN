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

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"
#include "interface/engine.hpp"

namespace impl = dnnl::graph::impl;

TEST(TestEngine, CreateWithDefaultAllocator) {
    impl::engine_kind_t kind = get_test_engine_kind();
#ifndef DNNL_GRAPH_CPU_SYCL
    SKIP_IF(kind == impl::engine_kind::cpu,
            "skip sycl api test for native cpu runtime.");
#endif
    sycl::device dev = (kind == impl::engine_kind::gpu)
            ? sycl::device {sycl::gpu_selector()}
            : sycl::device {sycl::cpu_selector()};
    sycl::context ctx {dev};

    impl::engine_t eng(kind, dev, ctx);

    impl::allocator_t::mem_attr_t attr {
            impl::allocator_t::mem_type_t::temp, 128};
    ASSERT_EQ(attr.type_, impl::allocator_t::mem_type_t::temp);
    ASSERT_EQ(attr.alignment_, 128);

    auto *mem_ptr = eng.get_allocator()->allocate(
            16, eng.sycl_device(), eng.sycl_context(), attr);
    ASSERT_NE(mem_ptr, nullptr);
    sycl::event e;
    eng.get_allocator()->deallocate(
            mem_ptr, eng.sycl_device(), eng.sycl_context(), e);
}
