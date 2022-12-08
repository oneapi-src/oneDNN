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

#include <vector>
#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "oneapi/dnnl/dnnl_graph_sycl.hpp"

#include "api/test_api_common.hpp"
#include "test_allocator.hpp"

using namespace dnnl::graph;
using namespace sycl;

#ifdef DNNL_WITH_SYCL
TEST(api_engine, create_with_sycl) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(api_test_engine_kind == dnnl_cpu,
            "skip sycl api test for native cpu runtime.");
#endif
    dnnl::engine::kind ekind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);

    queue q = (ekind == dnnl::engine::kind::gpu)
            ? queue(dnnl::impl::sycl::compat::gpu_selector_v,
                    property::queue::in_order {})
            : queue(dnnl::impl::sycl::compat::cpu_selector_v,
                    property::queue::in_order {});

    allocator alloc = dnnl::graph::sycl_interop::make_allocator(
            dnnl::graph::testing::sycl_malloc_wrapper,
            dnnl::graph::testing::sycl_free_wrapper);
    dnnl::engine e = dnnl::graph::sycl_interop::make_engine_with_allocator(
            q.get_device(), q.get_context(), alloc);
    ASSERT_EQ(e.get_kind(), ekind);
}
#endif
