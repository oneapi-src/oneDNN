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

#ifndef LLGA_INTERFACE_UTILS_HPP
#define LLGA_INTERFACE_UTILS_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "oneapi/dnnl/dnnl_graph_types.h"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

#ifndef UNUSED
#define UNUSED(x) ((void)x)
#endif

#ifndef assertm
#define assertm(exp, msg) assert(((void)msg, exp))
#endif

namespace llga {
namespace impl {
namespace utils {

template <typename T, typename P>
constexpr bool one_of(T val, P item) {
    return (T)val == item;
}

template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return (T)val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) {
    return one_of(nullptr, ptrs...);
}

template <typename T, typename R = T>
inline R array_product(const T *arr, size_t size) {
    if (size == 0) return 0;

    R prod = 1;
    for (size_t i = 0; i < size; ++i) {
        prod *= arr[i];
    }

    return prod;
}

template <typename T>
inline void array_copy(T *dst, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i)
        dst[i] = src[i];
}

template <typename T, typename U>
inline void array_set(T *arr, const U &val, size_t size) {
    for (size_t i = 0; i < size; ++i)
        arr[i] = static_cast<T>(val);
}

/// Default allocator for CPU
/// now only support the allocation of persistent memory, that means
/// we need to manually free the buffer allocated by this allocator
class cpu_allocator {
public:
    constexpr static size_t DEFAULT_ALIGNMENT = 4096;

    static void *malloc(size_t size, dnnl_graph_allocator_attr_t attr) {
        void *ptr;
        size_t alignment
                = attr.alignment == 0 ? DEFAULT_ALIGNMENT : attr.alignment;
#ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
        int rc = ((ptr) ? 0 : errno);
#else
        int rc = ::posix_memalign(&ptr, alignment, size);
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

#if DNNL_GRAPH_WITH_SYCL
/// Default allocator for SYCL device
/// now only support the allocation of persistent memory, that means
/// we need to manually free the buffer allocated by this allocator.
class sycl_allocator {
public:
    constexpr static size_t DEFAULT_ALIGNMENT = 16;

    static void *malloc(size_t size, const void *dev, const void *ctx,
            dnnl_graph_allocator_attr_t attr) {
        size_t alignment
                = attr.alignment == 0 ? DEFAULT_ALIGNMENT : attr.alignment;
        return cl::sycl::aligned_alloc_device(alignment, size,
                *static_cast<const cl::sycl::device *>(dev),
                *static_cast<const cl::sycl::context *>(ctx));
    }

    static void free(void *ptr, const void *ctx) {
        cl::sycl::free(ptr, *static_cast<const cl::sycl::context *>(ctx));
    }
};
#endif

} // namespace utils
} // namespace impl
} // namespace llga

#endif
