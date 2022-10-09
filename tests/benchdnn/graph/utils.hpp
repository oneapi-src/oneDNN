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

#ifndef BENCHDNN_GRAPH_UTILS_HPP
#define BENCHDNN_GRAPH_UTILS_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#endif

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

namespace graph {

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

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
        default: \
            throw std::runtime_error( \
                    "Not supported data type in current graph driver."); \
    }

#define DNN_GRAPH_SAFE(f, s) \
    do { \
        try { \
            f; \
        } catch (const dnnl::error &e) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, e.what()); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            return FAIL; \
        } \
    } while (0)

typedef std::function<void(const dnnl::engine &,
        const std::vector<dnnl::graph::logical_tensor> &,
        const std::vector<dnnl::graph::logical_tensor> &)>
        cmpl_function_t;

typedef std::function<void(dnnl::stream &,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs)>
        perf_function_t;

const dnnl::engine &get_test_engine();
const dnnl::stream &get_test_stream();

struct cpu_deletor {
    cpu_deletor() = default;
    void operator()(void *ptr) {
        if (ptr) free(ptr);
    }
};

#ifdef DNNL_WITH_SYCL
struct sycl_deletor {
    sycl::context ctx_;
    sycl_deletor() = delete;
    sycl_deletor(const sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) sycl::free(ptr, ctx_);
    }
};

struct scratchpad_mm_mgr {
    void *sycl_alloc_mm(
            size_t size, size_t alignment, const void *dev, const void *ctx);
    void sycl_free_mm(
            void *ptr, const void *device, const void *context, void *event);

private:
    std::unordered_multimap<size_t, std::shared_ptr<void>> map_size_ptr_;
    std::unordered_set<void *> free_ptr_;
};
bool is_sycl_engine();
dnnl::engine &get_engine();
#endif // DNNL_WITH_SYCL

// Engine used to run oneDNN fusion patterns for testing.
inline dnnl_engine_kind_t &get_test_engine_kind() {
    return engine_tgt_kind;
}

void compiled_partition_executor(dnnl::graph::compiled_partition &cp,
        dnnl::stream &stream, const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int measure_perf(timer::timer_t &t,
        std::vector<dnnl::graph::compiled_partition> &cp,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs,
        res_t *res);

inline bool is_plain(dnnl_format_tag_t fmt_tag) {
    return fmt_tag >= dnnl_a && fmt_tag <= dnnl_abcdefghijlk;
}

dnnl::graph::op::kind opstr2kind(const std::string &kind);
dnnl::graph::op::attr attrstr2kind(const std::string &attr_name);

void skip_unimplemented_data_type(
        const std::vector<dnnl::graph::logical_tensor> &in_out_lts, res_t *res);

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

#ifdef DNNL_WITH_SYCL

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
#ifdef DNNL_WITH_SYCL
template <typename dtype>
void fill_buffer(sycl::queue &q, void *usm_buffer, size_t length, dtype value);
#endif

} // namespace graph
#endif
