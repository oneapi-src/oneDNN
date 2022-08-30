/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef EXAMPLE_UTILS_HPP
#define EXAMPLE_UTILS_HPP

#include <iostream>
#include <numeric>
#include <unordered_set>

#ifdef DNNL_GRAPH_WITH_SYCL
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#endif

#include "oneapi/dnnl/dnnl_graph.hpp"

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

#define EXAMPLE_SWITCH_TYPE(type_enum, type_key, ...) \
    switch (type_enum) { \
        case dnnl::graph::logical_tensor::data_type::f32: { \
            using type_key = float; \
            __VA_ARGS__ \
        } break; \
        case dnnl::graph::logical_tensor::data_type::f16: { \
            using type_key = int16_t; \
            __VA_ARGS__ \
        } break; \
        case dnnl::graph::logical_tensor::data_type::bf16: { \
            using type_key = uint16_t; \
            __VA_ARGS__ \
        } break; \
        case dnnl::graph::logical_tensor::data_type::u8: { \
            using type_key = uint8_t; \
            __VA_ARGS__ \
        } break; \
        case dnnl::graph::logical_tensor::data_type::s8: { \
            using type_key = int8_t; \
            __VA_ARGS__ \
        } break; \
        default: \
            throw std::runtime_error( \
                    "Not supported data type in current example."); \
    }

struct cpu_deletor {
    cpu_deletor() = default;
    void operator()(void *ptr) {
        if (ptr) free(ptr);
    }
};

#ifdef DNNL_GRAPH_WITH_SYCL
struct sycl_deletor {
    sycl_deletor() = delete;
    ::sycl::context ctx_;
    sycl_deletor(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};
#endif

inline int64_t product(const std::vector<int64_t> &dims) {
    return dims.empty() ? 0
                        : std::accumulate(dims.begin(), dims.end(), (int64_t)1,
                                std::multiplies<int64_t>());
}

dnnl::graph::engine::kind parse_engine_kind(int argc, char **argv) {
    // Returns default engine kind, i.e. CPU, if none given
    if (argc == 1) {
        return dnnl::graph::engine::kind::cpu;
    } else if (argc == 2) {
        // Checking the engine type, i.e. CPU or GPU
        std::string engine_kind_str = argv[1];
        if (engine_kind_str == "cpu") {
            return dnnl::graph::engine::kind::cpu;
        } else if (engine_kind_str == "gpu") {
            return dnnl::graph::engine::kind::gpu;
        } else {
            throw std::runtime_error(
                    "parse_engine_kind: only support cpu or gpu engine");
        }
    }
    // If all above fails, the example should be ran properly
    std::cout << "Inappropriate engine kind." << std::endl
              << "Please run the example like: " << argv[0] << " [cpu|gpu]"
              << "." << std::endl;
    exit(1);
}

// fill the memory according to the given value
//  src -> target memory buffer
//  total_size -> total number of bytes of this buffer
//  val -> fixed value for initialization
template <typename T>
void fill_buffer(void *src, size_t total_size, int val) {
    size_t num_elem = static_cast<size_t>(total_size / sizeof(T));
    T *src_casted = static_cast<T *>(src);
    // can be implemented through OpenMP
    for (size_t i = 0; i < num_elem; ++i)
        *(src_casted + i) = static_cast<T>(val);
}

#ifdef DNNL_GRAPH_WITH_SYCL
template <typename dtype>
void fill_buffer(
        ::sycl::queue &q, void *usm_buffer, size_t length, dtype value) {
    dtype *usm_buffer_casted = static_cast<dtype *>(usm_buffer);
    q.parallel_for(::sycl::range<1>(length), [=](::sycl::id<1> i) {
         int idx = (int)i[0];
         usm_buffer_casted[idx] = value;
     }).wait();
}

void *sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    return malloc_shared(size, *static_cast<const ::sycl::device *>(dev),
            *static_cast<const ::sycl::context *>(ctx));
}

void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    // Device is not used in this example, but it may be useful for some users
    // application.
    UNUSED(device);
    // immediate synchronization here is for test purpose for performance, users
    // may need to store the ptr and event and handle them separately
    if (event) {
        auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
        sycl_deps_ptr->wait();
    }
    free(ptr, *static_cast<const ::sycl::context *>(context));
}
#endif

#endif
