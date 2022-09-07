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
#include "unit_test_common.hpp"

namespace impl = dnnl::graph::impl;

#ifdef DNNL_GRAPH_WITH_SYCL
::sycl::device &get_device() {
    static ::sycl::device dev = get_test_engine_kind() == impl::engine_kind::cpu
            ? ::sycl::device {::sycl::cpu_selector {}}
            : ::sycl::device {::sycl::gpu_selector {}};
    return dev;
}

::sycl::context &get_context() {
    static ::sycl::context ctx {get_device()};
    return ctx;
}
#endif // DNNL_GRAPH_WITH_SYCL

impl::engine_t &get_engine() {
    if (get_test_engine_kind() == impl::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        static auto sycl_allocator = std::shared_ptr<impl::allocator_t>(
                impl::allocator_t::create(
                        dnnl::graph::testing::sycl_malloc_wrapper,
                        dnnl::graph::testing::sycl_free_wrapper),
                [](impl::allocator_t *alloc) { alloc->release(); });
        static impl::engine_t eng(impl::engine_kind::cpu, get_device(),
                get_context(), sycl_allocator.get());
#else
        static impl::engine_t eng(impl::engine_kind::cpu, 0);
#endif
        return eng;
    } else {
#ifdef DNNL_GRAPH_GPU_SYCL
        static auto sycl_allocator = std::shared_ptr<impl::allocator_t>(
                impl::allocator_t::create(
                        dnnl::graph::testing::sycl_malloc_wrapper,
                        dnnl::graph::testing::sycl_free_wrapper),
                [](impl::allocator_t *alloc) { alloc->release(); });
        static impl::engine_t eng(impl::engine_kind::gpu, get_device(),
                get_context(), sycl_allocator.get());
#else
        assert(!"GPU only support DPCPP runtime now");
        static impl::engine_t eng(impl::engine_kind::gpu, 0);
#endif
        return eng;
    }
}

impl::stream_t &get_stream() {
    if (get_test_engine_kind() == impl::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        static ::sycl::queue q {get_context(), get_device(),
                ::sycl::property::queue::in_order {}};
        static impl::stream_t strm {&get_engine(), q};
#elif DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
        static impl::stream_t strm {
                &get_engine(), dnnl::graph::testing::get_threadpool()};
#else
        static impl::stream_t strm {&get_engine()};
#endif
        return strm;
    } else {
#ifdef DNNL_GRAPH_GPU_SYCL
        static ::sycl::queue q {get_context(), get_device(),
                ::sycl::property::queue::in_order {}};
        static impl::stream_t strm {&get_engine(), q};
#else
        assert(!"GPU only support DPCPP runtime now");
        static impl::stream_t strm {&get_engine()};
#endif
        return strm;
    }
}

static impl::engine_kind_t test_engine_kind;

impl::engine_kind_t get_test_engine_kind() {
    return test_engine_kind;
}

void set_test_engine_kind(impl::engine_kind_t kind) {
    test_engine_kind = kind;
}
