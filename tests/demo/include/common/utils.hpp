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

#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

#include "oneapi/dnnl/dnnl_graph.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
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

// get exact data handle from tensor according to the data type
void *get_handle_from_tensor(const dnnl::graph::tensor &src,
        dnnl::graph::logical_tensor::data_type dtype) {
    switch (dtype) {
        case dnnl::graph::logical_tensor::data_type::f32:
            return src.get_data_handle<float>();
            break;
        case dnnl::graph::logical_tensor::data_type::f16:
            return src.get_data_handle<int16_t>();
            break;
        case dnnl::graph::logical_tensor::data_type::bf16:
            return src.get_data_handle<uint16_t>();
            break;
        case dnnl::graph::logical_tensor::data_type::u8:
            return src.get_data_handle<uint8_t>();
            break;
        case dnnl::graph::logical_tensor::data_type::s8:
            return src.get_data_handle<int8_t>();
            break;
        default:
            throw std::runtime_error(
                    "Not supported data type in current example.");
    }
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
            throw std::runtime_error("only support cpu and gpu engine\n");
        }
    }
    // If all above fails, the example should be ran properly
    std::cout << "Inappropriate engine kind." << std::endl
              << "Please run the example like this: " << argv[0] << " [cpu|gpu]"
              << "." << std::endl;
    exit(1);
}

inline dnnl_graph_dim_t product(const std::vector<int64_t> &dims) {
    return dims.empty()
            ? 0
            : std::accumulate(dims.begin(), dims.end(), (dnnl_graph_dim_t)1,
                    std::multiplies<dnnl_graph_dim_t>());
}

void *allocate(size_t n, dnnl::graph::allocator::attribute attr) {
    (void)attr;
    return malloc(n);
}

void deallocate(void *ptr) {
    free(ptr);
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
static void compare_data(T *dst, T *ref, size_t size, float rtol = 1e-5f,
        float atol = 1e-6f, bool equal_nan = false) {
    auto cal_error = [&](const float dst, const float ref) -> bool {
        const float diff_f32 = dst - ref;
        const float gap = rtol
                        * (std::abs(ref) > std::abs(dst) ? std::abs(ref)
                                                         : std::abs(dst))
                + atol;
        bool good = std::abs(diff_f32) <= gap;
        return good;
    };

    for (size_t i = 0; i < size; ++i) {
        if (std::isfinite(dst[i]) && std::isfinite(ref[i])) {
            const float ref_f32 = static_cast<float>(ref[i]);
            const float dst_f32 = static_cast<float>(dst[i]);
            if (!cal_error(dst_f32, ref_f32)) {
                printf("expected = %s, actual = %s\n",
                        std::to_string(ref[i]).c_str(),
                        std::to_string(dst[i]).c_str());
                throw std::runtime_error(
                        "output result is not equal to excepted "
                        "results");
            }
        } else {
            bool cond = (dst[i] == ref[i]);
            if (equal_nan) { cond = std::isnan(dst[i]) && std::isnan(ref[i]); }
            if (!cond) {
                printf("expected = %s, actual = %s\n",
                        std::to_string(ref[i]).c_str(),
                        std::to_string(dst[i]).c_str());
                throw std::runtime_error(
                        "output result is not equal to excepted "
                        "results");
            }
        }
    }
}

#ifdef DNNL_GRAPH_WITH_SYCL
template <typename dtype>
void fill_buffer(
        cl::sycl::queue &q, void *usm_buffer, size_t length, dtype value) {
    dtype *usm_buffer_casted = static_cast<dtype *>(usm_buffer);
    q.parallel_for(cl::sycl::range<1>(length), [=](cl::sycl::id<1> i) {
         int idx = (int)i[0];
         usm_buffer_casted[idx] = value;
     }).wait();
}

void *sycl_malloc_wrapper(size_t n, const void *dev, const void *ctx,
        dnnl::graph::allocator::attribute attr) {
    return malloc_device(n, *static_cast<const cl::sycl::device *>(dev),
            *static_cast<const cl::sycl::context *>(ctx));
}

void sycl_free_wrapper(void *ptr, const void *context) {
    free(ptr, *static_cast<const cl::sycl::context *>(context));
}
#endif
#endif
