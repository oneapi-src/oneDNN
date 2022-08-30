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

#ifndef UNIT_TEST_COMMON_HPP
#define UNIT_TEST_COMMON_HPP

#include <memory>
#include <vector>

#include "interface/engine.hpp"
#include "interface/partition_cache.hpp"
#include "interface/stream.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#endif

#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
#include "test_thread.hpp"
#endif

#ifdef DNNL_GRAPH_WITH_SYCL
::sycl::device &get_device();
::sycl::context &get_context();
#endif // DNNL_GRAPH_WITH_SYCL

dnnl::graph::impl::engine_t &get_engine();

dnnl::graph::impl::stream_t &get_stream();

dnnl::graph::impl::engine_kind_t get_test_engine_kind();

void set_test_engine_kind(dnnl::graph::impl::engine_kind_t kind);

namespace test {

#ifdef DNNL_GRAPH_WITH_SYCL
constexpr size_t usm_alignment = 16;
#endif

template <typename T>
class TestAllocator {
public:
    typedef T value_type;

    T *allocate(size_t num_elements) {
        namespace impl = dnnl::graph::impl;
        if (get_test_engine_kind() == impl::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
            dev_ = get_device();
            ctx_ = get_context();
            return reinterpret_cast<T *>(::sycl::aligned_alloc(usm_alignment,
                    num_elements * sizeof(T), dev_, ctx_,
                    ::sycl::usm::alloc::shared));
#else
            return reinterpret_cast<T *>(malloc(num_elements * sizeof(T)));
#endif
        } else if (get_test_engine_kind() == impl::engine_kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
            dev_ = get_device();
            ctx_ = get_context();
            return reinterpret_cast<T *>(::sycl::aligned_alloc(usm_alignment,
                    num_elements * sizeof(T), dev_, ctx_,
                    ::sycl::usm::alloc::shared));
#else
            return nullptr;
#endif
        } else {
            return nullptr;
        }
    }

    void deallocate(T *ptr, size_t) {
        namespace impl = dnnl::graph::impl;
        if (!ptr) return;

        if (get_test_engine_kind() == impl::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
            ::sycl::free(ptr, ctx_);
#else
            free(ptr);
#endif
        } else if (get_test_engine_kind() == impl::engine_kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
            ::sycl::free(ptr, ctx_);
#endif
        } else {
        }
    }

    template <typename U>
    struct rebind {
        using other = TestAllocator<U>;
    };

private:
#ifdef DNNL_GRAPH_WITH_SYCL
    // The underlying implementation of ::sycl::device and ::sycl::context
    // are shared ptr. So we can hold a copy of them to avoid be destroyed
    // before we use them.
    ::sycl::device dev_;
    ::sycl::context ctx_;
#endif
};

template <class T, class U>
bool operator==(const TestAllocator<T> &, const TestAllocator<U> &) {
    return true;
}

template <class T, class U>
bool operator!=(const TestAllocator<T> &, const TestAllocator<U> &) {
    return false;
}

template <typename T>
#ifdef DNNL_GRAPH_WITH_SYCL
using vector = std::vector<T, TestAllocator<T>>;
#else
using vector = std::vector<T>;
#endif // DNNL_GRAPH_WITH_SYCL
} // namespace test

inline int get_compiled_partition_cache_size() {
    int result = 0;
    auto status = dnnl::graph::impl::get_compiled_partition_cache_size(&result);
    if (status != dnnl::graph::impl::status::success) return -1;
    return result;
}

inline int set_compiled_partition_cache_capacity(int capacity) {
    if (capacity < 0) return -1;
#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    return dnnl::graph::impl::compiled_partition_cache().set_capacity(capacity);
#endif
    return 0;
}

#endif
