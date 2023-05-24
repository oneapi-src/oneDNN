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

#ifndef GRAPH_UTILS_ALLOCATOR_HPP
#define GRAPH_UTILS_ALLOCATOR_HPP

#include "graph/utils/utils.hpp"

#ifdef DNNL_WITH_SYCL
#include "graph/utils/sycl_check.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

/// Default allocator for CPU
class cpu_allocator_t {
public:
    constexpr static size_t DEFAULT_ALIGNMENT = 64;

    static void *malloc(size_t size, size_t alignment) {
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

    static void free(void *p) {
#ifdef _WIN32
        _aligned_free((void *)p);
#else
        ::free((void *)p);
#endif /* _WIN32 */
    }
};

#ifdef DNNL_WITH_SYCL
/// Default allocator for SYCL device
class sycl_allocator_t {
public:
    constexpr static size_t DEFAULT_ALIGNMENT = 64;

    static void *malloc(
            size_t size, size_t alignment, const void *dev, const void *ctx) {
        const size_t align = alignment == 0 ? DEFAULT_ALIGNMENT : alignment;
        return ::sycl::aligned_alloc_shared(align, size,
                *static_cast<const ::sycl::device *>(dev),
                *static_cast<const ::sycl::context *>(ctx));
    }

    static void free(void *ptr, const void *dev, const void *ctx, void *event) {
        UNUSED(dev);
        if (event) {
            auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
            sycl_deps_ptr->wait();
        }
        ::sycl::free(ptr, *static_cast<const ::sycl::context *>(ctx));
    }
};
#endif

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
