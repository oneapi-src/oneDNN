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

#include <gtest/gtest.h>

#include "interface/engine.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

TEST(engine_test, simple_engine) {
    auto akind = llga::impl::engine_kind::cpu;
    llga::impl::engine_t engine_cpu(akind, 0);

    ASSERT_EQ(engine_cpu.kind(), akind);
    ASSERT_EQ(engine_cpu.device_id(), 0);

    akind = llga::impl::engine_kind::gpu;
    llga::impl::engine_t engine_gpu(akind, 0);

    ASSERT_EQ(engine_gpu.kind(), akind);
    ASSERT_EQ(engine_gpu.device_id(), 0);
}

#if DNNL_GRAPH_WITH_SYCL
TEST(engine_test, create_with_default_allocator) {
    namespace impl = llga::impl;
    namespace sycl = cl::sycl;

    sycl::queue q {sycl::gpu_selector {}};
    impl::engine_t eng(impl::engine_kind::gpu, q.get_device(), q.get_context());

    impl::allocator::attribute attr {impl::allocator_lifetime::output, 128};
    ASSERT_EQ(attr.data.type, impl::allocator_lifetime::output);
    ASSERT_EQ(attr.data.alignment, 128);

    auto *mem_ptr = eng.get_allocator()->allocate(
            16, eng.sycl_device(), eng.sycl_context(), attr);
    ASSERT_NE(mem_ptr, nullptr);
    eng.get_allocator()->deallocate(mem_ptr, eng.sycl_context());
}
#endif
