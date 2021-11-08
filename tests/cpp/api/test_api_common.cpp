/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#if DNNL_GRAPH_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.h"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"

void *sycl_alloc(size_t n, const void *dev, const void *ctx,
        dnnl_graph_allocator_attr_t attr) {
    return cl::sycl::malloc_device(n,
            *static_cast<const cl::sycl::device *>(dev),
            *static_cast<const cl::sycl::context *>(ctx));
}

void sycl_free(void *ptr, const void *ctx) {
    return cl::sycl::free(ptr, *static_cast<const cl::sycl::context *>(ctx));
}
#endif // DNNL_GRAPH_WITH_SYCL

void api_test_dnnl_graph_engine_create(
        dnnl_graph_engine_t **engine, dnnl_graph_engine_kind_t engine_kind) {
    if (engine_kind == dnnl_graph_cpu) {
#if DNNL_GRAPH_CPU_SYCL
        static cl::sycl::device dev {cl::sycl::cpu_selector {}};
        static cl::sycl::context ctx {dev};
        if (!allocator_handle) {
            ASSERT_EQ(
                    dnnl_graph_sycl_interop_allocator_create(
                            &allocator_handle.allocator, sycl_alloc, sycl_free),
                    dnnl_graph_result_success);
            ASSERT_EQ(dnnl_graph_sycl_interop_engine_create(
                              &engine_handle.engine, &dev, &ctx),
                    dnnl_graph_result_success);
            ASSERT_EQ(dnnl_graph_engine_set_allocator(
                              engine_handle.engine, allocator_handle.allocator),
                    dnnl_graph_result_success);
        };
#else
        if (!engine_handle) {
            ASSERT_EQ(dnnl_graph_engine_create(
                              &engine_handle.engine, engine_kind, 0),
                    dnnl_graph_result_success);
        }
#endif
        *engine = engine_handle.engine;
    } else {
#if DNNL_GRAPH_GPU_SYCL
        static cl::sycl::device dev {cl::sycl::gpu_selector {}};
        static cl::sycl::context ctx {dev};
        if (!allocator_handle) {
            ASSERT_EQ(
                    dnnl_graph_sycl_interop_allocator_create(
                            &allocator_handle.allocator, sycl_alloc, sycl_free),
                    dnnl_graph_result_success);
            ASSERT_EQ(dnnl_graph_sycl_interop_engine_create(
                              &engine_handle.engine, &dev, &ctx),
                    dnnl_graph_result_success);
            ASSERT_EQ(dnnl_graph_engine_set_allocator(
                              engine_handle.engine, allocator_handle.allocator),
                    dnnl_graph_result_success);
        };
        *engine = engine_handle.engine;
#endif
    }
}

void api_test_dnnl_graph_graph_create(
        dnnl_graph_graph_t **graph, dnnl_graph_engine_kind_t engine_kind) {
    ASSERT_EQ(dnnl_graph_graph_create(graph, engine_kind),
            dnnl_graph_result_success);
}

dnnl::graph::engine &cpp_api_test_dnnl_graph_engine_create(
        dnnl::graph::engine::kind engine_kind) {
    if (engine_kind == dnnl::graph::engine::kind::cpu) {
#if DNNL_GRAPH_CPU_SYCL
        static cl::sycl::device dev {cl::sycl::cpu_selector {}};
        static cl::sycl::context ctx {dev};
        static dnnl::graph::allocator alloc
                = dnnl::graph::sycl_interop::make_allocator(
                        sycl_alloc, sycl_free);
        static dnnl::graph::engine eng
                = dnnl::graph::sycl_interop::make_engine(dev, ctx);
        eng.set_allocator(alloc);
#else
        static dnnl::graph::engine eng(engine_kind, 0);
#endif
        return eng;
    }

#if DNNL_GRAPH_GPU_SYCL
    static cl::sycl::device dev {cl::sycl::gpu_selector {}};
    static cl::sycl::context ctx {dev};
    static dnnl::graph::allocator alloc
            = dnnl::graph::sycl_interop::make_allocator(sycl_alloc, sycl_free);
    static dnnl::graph::engine eng
            = dnnl::graph::sycl_interop::make_engine(dev, ctx);
    eng.set_allocator(alloc);
    return eng;
#else
    static dnnl::graph::engine eng(engine_kind, 0);
    return eng;
#endif
}

dnnl_graph_engine_kind_t api_test_engine_kind;
