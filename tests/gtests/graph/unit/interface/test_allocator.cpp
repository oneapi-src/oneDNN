/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"
#include "interface/allocator.hpp"
#include "interface/c_types_map.hpp"

TEST(test_interface_allocator, DefaultCpuAllocator) {
    dnnl::impl::graph::allocator_t *alloc
            = new dnnl::impl::graph::allocator_t();

    void *mem_ptr = alloc->allocate(static_cast<size_t>(16));
    if (mem_ptr == nullptr) {
        delete alloc;
        ASSERT_TRUE(false);
    } else {
        alloc->deallocate(mem_ptr);
        delete alloc;
    }
}

TEST(test_interface_allocator, AllocatorEarlyDestroy) {
    dnnl::impl::graph::allocator_t *alloc
            = new dnnl::impl::graph::allocator_t();
    graph::engine_t *eng = get_engine();
    eng->set_allocator(alloc);
    delete alloc;
    dnnl::impl::graph::allocator_t *engine_alloc
            = reinterpret_cast<dnnl::impl::graph::allocator_t *>(
                    eng->get_allocator());
#ifndef DNNL_WITH_SYCL
    void *mem_ptr = engine_alloc->allocate(static_cast<size_t>(16));
#else
    void *mem_ptr = engine_alloc->allocate(
            static_cast<size_t>(16), get_device(), get_context());
#endif
    if (mem_ptr == nullptr) {
        ASSERT_TRUE(false);
    } else {
#ifndef DNNL_WITH_SYCL
        engine_alloc->deallocate(mem_ptr);
#else
        sycl::event e;
        engine_alloc->deallocate(mem_ptr, get_device(), get_context(), e);
#endif
    }
}
