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

#include "interface/engine.hpp"

#include <CL/sycl.hpp>

namespace impl = dnnl::graph::impl;
namespace sycl = cl::sycl;

struct test_engine_params {
    impl::engine_kind_t eng_kind_;
};

class TestEngine : public ::testing::TestWithParam<test_engine_params> {
public:
    void sycl_engine() {
        auto param = ::testing::TestWithParam<test_engine_params>::GetParam();

        impl::engine_kind_t kind = param.eng_kind_;

        sycl::device dev = (kind == impl::engine_kind::gpu)
                ? sycl::device {sycl::gpu_selector()}
                : sycl::device {sycl::cpu_selector()};
        sycl::context ctx {dev};

        impl::engine_t eng(kind, dev, ctx);

        impl::allocator_t::attribute_t attr {
                impl::allocator_lifetime::temp, 128};
        ASSERT_EQ(attr.data.type, impl::allocator_lifetime::temp);
        ASSERT_EQ(attr.data.alignment, 128);

        auto *mem_ptr = eng.get_allocator()->allocate(
                16, eng.sycl_device(), eng.sycl_context(), attr);
        ASSERT_NE(mem_ptr, nullptr);
        sycl::event e;
        eng.get_allocator()->deallocate(
                mem_ptr, eng.sycl_device(), eng.sycl_context(), e);
    }
};

TEST_P(TestEngine, CreateWithDefaultAllocator) {
    sycl_engine();
}

#ifdef DNNL_GRAPH_GPU_SYCL
INSTANTIATE_TEST_SUITE_P(SyclEngineGpu, TestEngine,
        ::testing::Values(test_engine_params {impl::engine_kind::gpu}));
#endif

#ifdef DNNL_GRAPH_CPU_SYCL
INSTANTIATE_TEST_SUITE_P(SyclEngineCpu, TestEngine,
        ::testing::Values(test_engine_params {impl::engine_kind::cpu}));
#endif
