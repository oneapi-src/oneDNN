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

#include "unit_test_common.hpp"

#if DNNL_GRAPH_WITH_SYCL
cl::sycl::device &get_device() {
    static cl::sycl::device dev {cl::sycl::gpu_selector {}};
    return dev;
}

cl::sycl::context &get_context() {
    static cl::sycl::context ctx {get_device()};
    return ctx;
}

void *sycl_alloc(size_t n, const void *dev, const void *ctx,
        impl::allocator_attr_t attr) {
    return cl::sycl::malloc_device(n,
            *static_cast<const cl::sycl::device *>(dev),
            *static_cast<const cl::sycl::context *>(ctx));
}
void sycl_free(void *ptr, const void *ctx) {
    return cl::sycl::free(ptr, *static_cast<const cl::sycl::context *>(ctx));
}
#endif // DNNL_GRAPH_WITH_SYCL

impl::engine_t &get_engine(impl::engine_kind_t engine_kind) {
    UNUSED(engine_kind);
#if DNNL_GRAPH_WITH_SYCL
    static impl::allocator_t sycl_allocator(sycl_alloc, sycl_free);
    static impl::engine_t eng(
            impl::engine_kind::gpu, get_device(), get_context());
    eng.set_allocator(&sycl_allocator);
#else
    static impl::engine_t eng(impl::engine_kind::cpu, 0);
#endif
    return eng;
}

impl::stream &get_stream() {
#if DNNL_GRAPH_WITH_SYCL
    static cl::sycl::queue q {get_context(), get_device()};
    static impl::stream strm {&get_engine(), q};
#else
    static impl::stream strm {&get_engine()};
#endif
    return strm;
}
