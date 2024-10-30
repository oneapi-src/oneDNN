/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_graph_ocl.hpp"
#endif

#include "common.hpp"
#include "dnnl_common.hpp"

namespace graph {

struct deserialized_lt;

struct bdnn_state_t {
    res_state_t state;
    std::string reason;
};

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
    gnorm,
    others
};

extern bdnn_state_t convert_state(const dnnl_status_t &s);

// Flags that controls the behavior for handling exceptions. The logic
// relies on the fact that the values not intersect with each other.
enum { CRIT = 0x001, WARN = 0x002, NEED_CLEANUP = 0x004 };

#define DNN_GRAPH_SAFE(f, s, ss) \
    do { \
        try { \
            f; \
        } catch (const dnnl::error &e) { \
            if ((s & CRIT) || (s & WARN)) { \
                bdnn_state_t bs = convert_state(e.status); \
                ss->state = bs.state; \
                if (ss->state == res_state_t::SKIPPED) { \
                    ss->reason = bs.reason; \
                } else { \
                    BENCHDNN_PRINT(0, \
                            "Error: Function '%s' at (%s:%d) returned '%s'\n", \
                            __FUNCTION__, __FILE__, __LINE__, e.what()); \
                } \
                fflush(0); \
                if (s & CRIT) exit(2); \
            } \
            if (!(s & NEED_CLEANUP)) return FAIL; \
        } \
    } while (0)

typedef std::function<void(dnnl::stream &,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs)>
        perf_function_t;

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

std::string get_default_tag(size_t length);
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

dnnl_driver_t opkind2driver(const dnnl::graph::op::kind &kind);

// permute md based on permutation
void permute_md(dnn_mem_t &mem, std::vector<int64_t> permutation);

// get primitive's arg name according to graph op's output offset
// i.e. If BatchNormForwardTraining's 2-nd output is ReLU's 1-st input
//      the output offset of 2 needs to be mapped to primitive's
//      output arg of DNNL_ARG_VARIANCE
int get_prim_arg_name_from_graph_op_output_offset(
        dnnl::graph::op::kind op_kind, size_t output_offset);
// get primitive's arg name according to graph op's input offset
int get_prim_arg_name_from_graph_op_input_offset(dnnl::graph::op::kind op_kind,
        size_t input_offset, bool use_dst = false);

/// Get logical tensor layout type based on string
///
/// @param layout_type a string of layout type from deserialized
/// logical tensor
dnnl::graph::logical_tensor::layout_type str2layout(
        const std::string &layout_type);

void change_format_to_ncx(dims_t &dims);

// For a given vector of partitions provide a string with number of ops in
// every partition in format: `{N} {M} ...`.
std::string verbose_partitions_n_ops(
        const std::vector<dnnl::graph::partition> &partitions);

// Returns logical dims as a string object in dims_t format
std::string lt_dims2str(const dnnl::graph::logical_tensor::dims &dims);

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

bool is_gc_backend();

dnnl_data_type_t convert_dt(const dnnl::graph::logical_tensor::data_type dt);

inline double GB(double bytes) {
    return bytes / powf(2, 30);
}

struct graph_fpmath_mode_t {
    graph_fpmath_mode_t() = default;
    graph_fpmath_mode_t(const std::string &mode, bool apply_to_int,
            bool override_json_value)
        : mode_(mode)
        , apply_to_int_(apply_to_int)
        , override_json_value_(override_json_value) {}

    bool operator==(const graph_fpmath_mode_t &rhs) const {
        return mode_ == rhs.mode_ && apply_to_int_ == rhs.apply_to_int_
                && override_json_value_ == rhs.override_json_value_;
    }

    std::string mode_ = "strict";
    bool apply_to_int_ = false;
    // Since fpmath_mode doesn't provide an "undef" value that would indicate
    // it was not set externally to the json case, need to maintain this flag.
    bool override_json_value_ = false;
};

} // namespace graph
#endif
