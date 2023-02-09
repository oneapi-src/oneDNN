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
sycl::queue &get_queue();
#endif // DNNL_WITH_SYCL

void compiled_partition_executor(dnnl::graph::compiled_partition &cp,
        dnnl::stream &stream, const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs);

int execute_and_wait(const std::vector<dnnl::graph::compiled_partition> &cp_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        res_t *res);

int measure_perf(timer::timer_t &t,
        const std::vector<dnnl::graph::compiled_partition> &cp_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        res_t *res);

dnnl::graph::op::kind opstr2kind(const std::string &kind);
dnnl::graph::op::attr attrstr2kind(const std::string &attr_name);

std::string strides2memory_tag(const size_t ndims,
        const dnnl::graph::logical_tensor::dims &strides,
        bool use_x_tag = true);

dnnl::graph::logical_tensor::dims memory_tag2strides(
        const dnnl::graph::logical_tensor::dims &shape, const std::string &tag);

// fill the memory according to the given value
//  src -> target memory buffer
//  total_size -> total number of bytes of this buffer
//  val -> fixed value for initialization
template <typename T>
void fill_buffer(void *src, size_t total_size, int val) {
    size_t num_elem = static_cast<size_t>(total_size / sizeof(T));
    T *src_casted = static_cast<T *>(src);
    // TODO: unify data filler after correctness added.
    // POSTPONE: will be unified with correctness mode introduction.
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

void change_format_to_ncx(dims_t &dims);

template <typename First, typename... Rest>
void change_format_to_ncx(First &first, Rest &...rest) {
    change_format_to_ncx(first);
    change_format_to_ncx(rest...);
}

} // namespace graph
#endif
