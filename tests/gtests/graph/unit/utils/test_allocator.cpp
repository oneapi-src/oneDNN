/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "interface/c_types_map.hpp"

#include "utils/allocator.hpp"

#include "gtest/gtest.h"

#include "graph/unit/unit_test_common.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif

namespace graph = dnnl::impl::graph;

TEST(Alloctor, SyclAlloctorMallocAndFree) {
    graph::engine_t &engine = *get_engine();
    if (engine.kind() == graph::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        void *p = nullptr;
        ::sycl::event e;
        ASSERT_NO_THROW(
                p = graph::utils::sycl_allocator_t::malloc(1024,
                        graph::utils::sycl_allocator_t::DEFAULT_ALIGNMENT,
                        &engine.sycl_device(), &engine.sycl_context()));
        ASSERT_NO_THROW(graph::utils::sycl_allocator_t::free(
                p, &engine.sycl_device(), &engine.sycl_context(), &e));
#endif

    } else if (engine.kind() == graph::engine_kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
        void *p = nullptr;
        ::sycl::event e;
        ASSERT_NO_THROW(
                p = graph::utils::sycl_allocator_t::malloc(1024,
                        graph::utils::sycl_allocator_t::DEFAULT_ALIGNMENT,
                        &engine.sycl_device(), &engine.sycl_context()));
        ASSERT_NO_THROW(graph::utils::sycl_allocator_t::free(
                p, &engine.sycl_device(), &engine.sycl_context(), &e));
#endif
    }
}
