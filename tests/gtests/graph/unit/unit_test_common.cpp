/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <iostream>

#include <gtest/gtest.h>

#include "test_allocator.hpp"
#include "unit/unit_test_common.hpp"

namespace graph = dnnl::impl::graph;

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "test_thread.hpp"
#endif

#ifdef DNNL_WITH_SYCL
::sycl::device &get_device() {
    static ::sycl::device dev
            = get_test_engine_kind() == graph::engine_kind::cpu
            ? ::sycl::device {dnnl::impl::sycl::compat::cpu_selector_v}
            : ::sycl::device {dnnl::impl::sycl::compat::gpu_selector_v};
    return dev;
}

::sycl::context &get_context() {
    static ::sycl::context ctx {get_device()};
    return ctx;
}
#endif // DNNL_WITH_SYCL

static dnnl::engine get_dnnl_engine() {
    if (get_test_engine_kind() == graph::engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        static dnnl::graph::allocator alloc
                = dnnl::graph::sycl_interop::make_allocator(
                        dnnl::graph::testing::sycl_malloc_wrapper,
                        dnnl::graph::testing::sycl_free_wrapper);
        static dnnl::engine eng
                = dnnl::graph::sycl_interop::make_engine_with_allocator(
                        get_device(), get_context(), alloc);
#else
        static dnnl::graph::allocator alloc {};
        static dnnl::engine eng
                = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
#endif
        return eng;
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        static dnnl::graph::allocator alloc
                = dnnl::graph::sycl_interop::make_allocator(
                        dnnl::graph::testing::sycl_malloc_wrapper,
                        dnnl::graph::testing::sycl_free_wrapper);
        static dnnl::engine eng
                = dnnl::graph::sycl_interop::make_engine_with_allocator(
                        get_device(), get_context(), alloc);
#else
        assert(!"GPU only support DPCPP runtime now");
        static dnnl::graph::allocator alloc {};
        static dnnl::engine eng
                = make_engine_with_allocator(dnnl::engine::kind::gpu, 0, alloc);
#endif
        return eng;
    }
}

static dnnl::stream get_dnnl_stream() {
    if (get_test_engine_kind() == graph::engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        static ::sycl::queue q {get_context(), get_device(),
                ::sycl::property::queue::in_order {}};
        static dnnl::stream strm
                = dnnl::sycl_interop::make_stream(get_dnnl_engine(), q);
#elif DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        static dnnl::stream strm = dnnl::threadpool_interop::make_stream(
                get_dnnl_engine(), dnnl::testing::get_threadpool());
#else
        static dnnl::stream strm(get_dnnl_engine());
#endif
        return strm;
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        static ::sycl::queue q {get_context(), get_device(),
                ::sycl::property::queue::in_order {}};
        static dnnl::stream strm
                = dnnl::sycl_interop::make_stream(get_dnnl_engine(), q);
#else
        assert(!"GPU only support DPCPP runtime now");
        static dnnl::stream strm(get_dnnl_engine());

#endif
        return strm;
    }
}

graph::engine_t *get_engine() {
    return get_dnnl_engine().get();
}

graph::stream_t *get_stream() {
    return get_dnnl_stream().get();
}

static graph::engine_kind_t test_engine_kind;

graph::engine_kind_t get_test_engine_kind() {
    return test_engine_kind;
}

void set_test_engine_kind(graph::engine_kind_t kind) {
    test_engine_kind = kind;
}
