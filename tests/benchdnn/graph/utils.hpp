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

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#endif

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif

#include "dnnl_common.hpp"

namespace graph {

struct deserialized_lt;

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

inline bool is_plain(dnnl_format_tag_t fmt_tag) {
    return fmt_tag >= dnnl_a && fmt_tag <= dnnl_abcdefghijlk;
}

dnnl::graph::op::kind opstr2kind(const std::string &kind);
dnnl::graph::op::attr attrstr2kind(const std::string &attr_name);

enum class dnnl_driver_t {
    binary,
    bnorm,
    concat,
    conv,
    custom,
    deconv,
    eltwise,
    lnorm,
    matmul,
    pool,
    prelu,
    reduction,
    reorder,
    resampling,
    softmax,
    others
};

dnnl_driver_t opkind2driver(const dnnl::graph::op::kind &kind);

// permute md based on permutation
void permute_md(dnn_mem_t &mem, std::vector<int64_t> permutation);

void reshape_md(dnn_mem_t &mem, const dnnl::memory::dims &reshaped_dims);

void reshape_md(dnn_mem_t &mem, const dnnl::memory::dims &reshaped_dims,
        const dnnl::memory::dims &reshaped_strides);

// check whether the logical tensor is in NXC format
bool is_nxc_lt_arg(const std::string &kind, const int exec_arg);

// get primitive's arg name according to graph op's output offset
// i.e. If BatchNormForwardTraining's 2-nd output is ReLU's 1-st input
//      the output offset of 2 needs to be mapped to primitive's
//      output arg of DNNL_ARG_VARIANCE
int get_prim_arg_name_from_graph_op_output_offset(
        dnnl::graph::op::kind op_kind, size_t output_offset);
// get primitive's arg name according to graph op's input offset
int get_prim_arg_name_from_graph_op_input_offset(
        dnnl::graph::op::kind op_kind, int input_offset, bool use_dst = false);

/// Get logical tensor layout type based on string
///
/// @param layout_type a string of layout type from deserialized
/// logical tensor
dnnl::graph::logical_tensor::layout_type str2layout(
        const std::string &layout_type);

void change_format_to_ncx(dims_t &dims);

template <typename First, typename... Rest>
void change_format_to_ncx(First &first, Rest &...rest) {
    change_format_to_ncx(first);
    change_format_to_ncx(rest...);
}

struct cpp_stream_t {
    cpp_stream_t(const dnnl::engine &eng,
            dnnl::stream::flags flags = dnnl::stream::flags::default_flags,
            void *interop_obj = nullptr);
    void wait() { stream_.wait(); }
    operator dnnl::stream &() { return stream_; }

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(cpp_stream_t);
    dnnl::stream stream_;
};

// engine used for graph lib, graph lib engine needs allocator to allocate
// memory for constant cache, scratchpad.
struct cpp_engine_t {
    cpp_engine_t();
    dnnl::engine::kind get_kind() const { return engine_.get_kind(); }
    operator dnnl::engine &() { return engine_; }
    operator const dnnl::engine &() const { return engine_; }

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(cpp_engine_t);
    dnnl::engine engine_;
};

// engine used for graph lib, graph lib engine needs allocator to allocate
// memory for constant cache, scratchpad.
inline const cpp_engine_t &get_graph_engine() {
    static const cpp_engine_t instance;
    return instance;
}

} // namespace graph
#endif
