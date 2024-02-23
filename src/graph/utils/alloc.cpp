/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "graph/utils/alloc.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/ocl/ocl_usm_utils.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
void *ocl_allocator_t::malloc(size_t size, engine_t *engine) {
    void *p = impl::gpu::ocl::usm::malloc_shared(engine, size);
    assert(p);
    return p;
}

void ocl_allocator_t::free(void *ptr, engine_t *engine) {
    impl::gpu::ocl::usm::free(engine, ptr);
}
#endif

#ifdef DNNL_WITH_SYCL
void *sycl_allocator_t::malloc(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    const size_t align = alignment == 0 ? DEFAULT_ALIGNMENT : alignment;
    return ::sycl::aligned_alloc_shared(align, size,
            *static_cast<const ::sycl::device *>(dev),
            *static_cast<const ::sycl::context *>(ctx));
}

void sycl_allocator_t::free(
        void *ptr, const void *dev, const void *ctx, void *event) {
    UNUSED(dev);
    if (event) {
        auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
        sycl_deps_ptr->wait();
    }
    ::sycl::free(ptr, *static_cast<const ::sycl::context *>(ctx));
}
#endif

void *cpu_allocator_t::malloc(size_t size, size_t alignment) {
    void *ptr = nullptr;
    const size_t align = alignment == 0 ? DEFAULT_ALIGNMENT : alignment;
#ifdef _WIN32
    ptr = _aligned_malloc(size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, align, size);
#endif /* _WIN32 */
    return (rc == 0) ? ptr : nullptr;
}

void cpu_allocator_t::free(void *p) {
#ifdef _WIN32
    _aligned_free((void *)p);
#else
    ::free((void *)p);
#endif /* _WIN32 */
}

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
