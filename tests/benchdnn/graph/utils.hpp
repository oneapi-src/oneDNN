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

#ifndef BENCHDNN_GRAPH_UTILS_HPP
#define BENCHDNN_GRAPH_UTILS_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#endif

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

namespace graph {

#define GRAPH_SWITCH_TYPE(type_enum, type_key, ...) \
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
        case dnnl::graph::logical_tensor::data_type::s32: { \
            using type_key = int32_t; \
            __VA_ARGS__ \
        } break; \
        case dnnl::graph::logical_tensor::data_type::boolean: { \
            using type_key = bool; \
            __VA_ARGS__ \
        } break; \
        default: \
            throw std::runtime_error( \
                    "Not supported data type in current graph driver."); \
    }

struct cpu_deletor {
    cpu_deletor() = default;
    void operator()(void *ptr) {
        if (ptr) free(ptr);
    }
};

#ifdef DNNL_GRAPH_WITH_SYCL
struct sycl_deletor {
    ::sycl::context ctx_;
    sycl_deletor() = delete;
    sycl_deletor(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};
#endif

dnnl::graph::op::kind opstr2kind(const std::string &kind);

std::string strides2memory_tag(
        const dnnl::graph::logical_tensor::dims_t &strides,
        bool use_x_tag = true);

void skip_unimplemented_data_type(
        const std::vector<dnnl::graph::logical_tensor> &in_out_lts, dir_t dir,
        res_t *res);

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
class init_kernel;

template <typename dtype>
void fill_buffer(sycl::queue &q, void *usm_buffer, size_t length, dtype value) {
    dtype *usm_buffer_casted = static_cast<dtype *>(usm_buffer);
    auto ker = [=](sycl::id<1> i) {
        int idx = (int)i[0];
        usm_buffer_casted[idx] = value;
    };
    q.parallel_for<init_kernel<dtype>>(sycl::range<1>(length), ker).wait();
}
#endif

struct logical_id_manager {
    static logical_id_manager &get() {
        static logical_id_manager id_mgr;
        return id_mgr;
    }

    size_t operator[](const std::string &arg) {
        const auto &it = knots_.find(arg);
        if (it != knots_.end()) return it->second;
        if (frozen_) {
            std::cout << "Unrecognized argument [" << arg << "]!\n";
            std::abort();
        }
        const auto &new_it = knots_.emplace(arg, knots_.size());
        if (new_it.second) {
            return new_it.first->second;
        } else {
            std::cout << "New argument [" << arg
                      << "] is failed to be added to knots.\n";
            std::abort();
        }
    }

    void freeze() { frozen_ = true; }

private:
    logical_id_manager() : frozen_(false) {};

    std::map<std::string, size_t> knots_;
    // indicates that the graph is frozen
    bool frozen_;
};

template <typename T>
void compare_data(T *dst, T *ref, size_t size, float rtol = 1e-5f,
        float atol = 1e-6f, bool equal_nan = false);

} // namespace graph
#endif
