/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "test_api_common.hpp"
#include "test_allocator.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.h"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif // DNNL_WITH_SYCL

void api_test_dnnl_engine_create(
        dnnl_engine_t *engine, dnnl_engine_kind_t engine_kind) {
    if (engine_kind == dnnl_cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        static ::sycl::device dev {dnnl::impl::sycl::compat::cpu_selector_v};
        static ::sycl::context ctx {dev};
        if (!allocator_handle) {
            ASSERT_EQ(dnnl_graph_sycl_interop_allocator_create(
                              &allocator_handle.allocator,
                              dnnl::graph::testing::sycl_malloc_wrapper,
                              dnnl::graph::testing::sycl_free_wrapper),
                    dnnl_success);

            ASSERT_EQ(dnnl_graph_sycl_interop_allocator_create(
                              &allocator_handle.allocator, nullptr,
                              dnnl::graph::testing::sycl_free_wrapper),
                    dnnl_success);

            ASSERT_EQ(dnnl_graph_sycl_interop_allocator_create(
                              &allocator_handle.allocator, nullptr, nullptr),
                    dnnl_success);

            ASSERT_EQ(dnnl_graph_sycl_interop_make_engine_with_allocator(
                              &engine_handle.engine, &dev, &ctx,
                              allocator_handle.allocator),
                    dnnl_success);
        };
#else
        if (!engine_handle) {
            ASSERT_EQ(dnnl_engine_create(&engine_handle.engine, engine_kind, 0),
                    dnnl_success);
        }
#endif
        *engine = engine_handle.engine;
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        static ::sycl::device dev {dnnl::impl::sycl::compat::gpu_selector_v};
        static ::sycl::context ctx {dev};
        if (!allocator_handle) {
            ASSERT_EQ(dnnl_graph_sycl_interop_allocator_create(
                              &allocator_handle.allocator,
                              dnnl::graph::testing::sycl_malloc_wrapper,
                              dnnl::graph::testing::sycl_free_wrapper),
                    dnnl_success);
            ASSERT_EQ(dnnl_graph_sycl_interop_make_engine_with_allocator(
                              &engine_handle.engine, &dev, &ctx,
                              allocator_handle.allocator),
                    dnnl_success);
        };
        *engine = engine_handle.engine;
#endif
    }
}

void api_test_dnnl_graph_graph_create(
        dnnl_graph_graph_t *graph, dnnl_engine_kind_t engine_kind) {
    ASSERT_EQ(dnnl_graph_graph_create(graph, engine_kind), dnnl_success);
}

dnnl::engine &cpp_api_test_dnnl_engine_create(dnnl::engine::kind engine_kind) {
    if (engine_kind == dnnl::engine::kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        static ::sycl::device dev {dnnl::impl::sycl::compat::cpu_selector_v};
        static ::sycl::context ctx {dev};
        static dnnl::graph::allocator alloc
                = dnnl::graph::sycl_interop::make_allocator(
                        dnnl::graph::testing::sycl_malloc_wrapper,
                        dnnl::graph::testing::sycl_free_wrapper);
        static dnnl::engine eng
                = dnnl::graph::sycl_interop::make_engine_with_allocator(
                        dev, ctx, alloc);
#else
        static dnnl::graph::allocator alloc {};
        static dnnl::engine eng
                = make_engine_with_allocator(engine_kind, 0, alloc);
#endif
        return eng;
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    static ::sycl::device dev {dnnl::impl::sycl::compat::gpu_selector_v};
    static ::sycl::context ctx {dev};
    static dnnl::graph::allocator alloc
            = dnnl::graph::sycl_interop::make_allocator(
                    dnnl::graph::testing::sycl_malloc_wrapper,
                    dnnl::graph::testing::sycl_free_wrapper);
    static dnnl::engine eng
            = dnnl::graph::sycl_interop::make_engine_with_allocator(
                    dev, ctx, alloc);
    return eng;
#else
    static dnnl::graph::allocator alloc {};
    static dnnl::engine eng = make_engine_with_allocator(engine_kind, 0, alloc);
    return eng;
#endif
}

dnnl_engine_kind_t api_test_engine_kind;
