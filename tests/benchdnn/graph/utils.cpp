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

#include <algorithm>
#include <set>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.h"

#include "cpu/platform.hpp"
#ifdef DNNL_WITH_SYCL
#include "dnnl_sycl.hpp"
#endif
#include "utils.hpp"
#include "utils/timer.hpp"

namespace graph {

void compiled_partition_executor(dnnl::graph::compiled_partition &cp,
        dnnl::stream &stream, const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    if (is_cpu()) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        dnnl::graph::sycl_interop::execute(cp, stream, inputs,
                const_cast<std::vector<dnnl::graph::tensor> &>(outputs));
#else
        cp.execute(stream, inputs, outputs);
#endif
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        dnnl::graph::sycl_interop::execute(cp, stream, inputs,
                const_cast<std::vector<dnnl::graph::tensor> &>(outputs));
#else
        assert(!"GPU only support DPCPP runtime now");
#endif
    }
}

int execute_and_wait(const std::vector<dnnl::graph::compiled_partition> &cp_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        res_t *res) {
    dnnl::stream stream = get_test_stream();
    for (size_t i = 0; i < cp_v.size(); i++) {
        perf_function_t perf_func = std::bind(&compiled_partition_executor,
                cp_v[i], std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3);
        DNN_GRAPH_SAFE(perf_func(stream, inputs_v[i], outputs_v[i]), CRIT);
        DNN_GRAPH_SAFE(stream.wait(), CRIT);
    }
    res->state = EXECUTED;
    return OK;
}

inline int measure_perf_aggregate(timer::timer_t &t, dnnl::stream &stream,
        std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v) {
    const int max_batch_times = 10000;

    // Warm-up run, this is not measured due to possibility the associated
    // kernel has not been built and skews the results.
    auto sz = perf_func_v.size();
    for (size_t i = 0; i < sz; i++) {
        DNN_GRAPH_SAFE(perf_func_v[i](stream, inputs_v[i], outputs_v[i]), WARN);
        DNN_GRAPH_SAFE(stream.wait(), WARN);
    }

    int cur_batch_times
            = fix_times_per_prb ? fix_times_per_prb : min_times_per_prb;

    t.reset();
    reset_gpu_profiling();

    bool is_first_loop = true;
    while (true) {
        for_(size_t i = 0; i < sz; i++)
        for (int j = 0; j < cur_batch_times; j++) {
            DNN_GRAPH_SAFE(
                    perf_func_v[i](stream, inputs_v[i], outputs_v[i]), WARN);
        }
        DNN_GRAPH_SAFE(stream.wait(), WARN);

        if (is_bench_mode(PROF)) {
            uint64_t nsec = 0;
            double freq = 0;
            get_gpu_profiling_info(nsec, freq, 0);
            reset_gpu_profiling();
            t.stamp_with_frequency(cur_batch_times, nsec / 1e6, freq);
        } else {
            t.stamp(cur_batch_times);
        }

        if (should_stop(t)) break;

        // Adjust cur_batch_times after the first batch run
        if (is_first_loop) {
            double ms_min = t.ms(timer::timer_t::min);
            // Heuristic: try to use ~5 batch runs for the whole benchmark
            int batch_times_heuristic = (ms_min == 0.0)
                    ? INT_MAX
                    : MAX2(1,
                            (int)((max_ms_per_prb - t.total_ms()) / ms_min
                                    / 5));
            cur_batch_times = MIN2(max_batch_times, batch_times_heuristic);
            is_first_loop = false;
        }
    }
    return OK;
}

inline int measure_perf_individual(timer::timer_t &t, dnnl::stream &stream,
        std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v) {
    t.reset();
    while (true) {
        auto sz = perf_func_v.size();
        for (size_t i = 0; i < sz; i++) {
            DNN_GRAPH_SAFE(
                    perf_func_v[i](stream, inputs_v[i], outputs_v[i]), WARN);
        }
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

int measure_perf(timer::timer_t &t, std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v) {
    if (is_bench_mode(PERF)) {
        dnnl::stream stream = get_test_stream();
        if (is_cpu() && !is_sycl_engine()) {
            return measure_perf_individual(
                    t, stream, perf_func_v, inputs_v, outputs_v);
        } else {
            return measure_perf_aggregate(
                    t, stream, perf_func_v, inputs_v, outputs_v);
        }
    } else {
        return OK;
    }
}

int measure_perf(timer::timer_t &t,
        const std::vector<dnnl::graph::compiled_partition> &cp_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        res_t *res) {
    std::vector<perf_function_t> perf_func_v;
    for (size_t i = 0; i < cp_v.size(); i++) {
        perf_func_v.emplace_back(std::bind(&compiled_partition_executor,
                cp_v[i], std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3));
    }

    int status = measure_perf(t, perf_func_v, inputs_v, outputs_v);
    if (res) res->state = EXECUTED;

    return status;
}

#ifdef DNNL_WITH_SYCL
void *scratchpad_mm_mgr::sycl_alloc_mm(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    // fake malloc for 0 size
    if (size == 0) return nullptr;

    void *ptr {nullptr};
    bool need_alloc_new_mm = true;
    // find alloc mm with same size
    const auto cnt = map_size_ptr_.count(size);
    if (cnt > 0) {
        const auto Iter = map_size_ptr_.equal_range(size);
        for (auto it = Iter.first; it != Iter.second; ++it) {
            // check if same size mm is free
            if (free_ptr_.find(it->second.get()) != free_ptr_.end()) {
                ptr = it->second.get();
                free_ptr_.erase(ptr);
                need_alloc_new_mm = false;
            }
        }
    }

    if (need_alloc_new_mm) {
        auto sh_ptr = std::shared_ptr<void> {
                malloc_shared(size, *static_cast<const sycl::device *>(dev),
                        *static_cast<const sycl::context *>(ctx)),
                sycl_deletor {*static_cast<const sycl::context *>(ctx)}};
        ptr = sh_ptr.get();
        // record the map of mm size and its ptr for reuse
        map_size_ptr_.emplace(std::make_pair(size, sh_ptr));
    }
    return ptr;
}

void scratchpad_mm_mgr::sycl_free_mm(
        void *ptr, const void *device, const void *context, void *event) {
    free_ptr_.insert(ptr);
}

static scratchpad_mm_mgr s_mm_mgr;

void *test_sycl_malloc_wrapper(
        size_t n, size_t alignment, const void *dev, const void *ctx) {
    return malloc_device(n, *static_cast<const sycl::device *>(dev),
            *static_cast<const sycl::context *>(ctx));
}

void test_sycl_free_wrapper(
        void *ptr, const void *dev, const void *context, void *event) {
    (void)(dev);
    if (event) {
        static_cast<sycl::event *>(const_cast<void *>(event))->wait();
    }
    free(ptr, *static_cast<const sycl::context *>(context));
}

void *sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    void *ptr = is_bench_mode(CORR) || is_cpu()
            ? test_sycl_malloc_wrapper(size, alignment, dev, ctx)
            : s_mm_mgr.sycl_alloc_mm(size, alignment, dev, ctx);

    return ptr;
}

// perf mode, mem will be finally released in s_mm_mgr ~shared_ptr when
// test finished.
void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    if (is_bench_mode(CORR) || is_cpu()) {
        test_sycl_free_wrapper(ptr, device, context, event);
    } else {
        s_mm_mgr.sycl_free_mm(ptr, device, context, event);
    }
}

sycl::queue &get_queue() {
    static dnnl::engine test_eng {::get_test_engine()};
    static sycl::device dev {dnnl::sycl_interop::get_device(test_eng)};
    static sycl::context ctx {dnnl::sycl_interop::get_context(test_eng)};
    static sycl::queue q {ctx, dev, sycl::property::queue::in_order {}};
    return q;
}

const dnnl::engine &get_graph_engine() {
    static dnnl::graph::allocator sycl_allocator {
            dnnl::graph::sycl_interop::make_allocator(
                    sycl_malloc_wrapper, sycl_free_wrapper)};
    static dnnl::engine test_eng {::get_test_engine()};
    static sycl::device dev {dnnl::sycl_interop::get_device(test_eng)};
    static sycl::context ctx {dnnl::sycl_interop::get_context(test_eng)};
    static dnnl::engine eng
            = dnnl::graph::sycl_interop::make_engine_with_allocator(
                    dev, ctx, sycl_allocator);
    return eng;
}

dnnl::stream &get_graph_stream() {
    static dnnl::stream strm {
            dnnl::sycl_interop::make_stream(get_graph_engine(), get_queue())};
    return strm;
}
#endif // DNNL_WITH_SYCL

bool is_sycl_engine() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (is_cpu()) return true;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    if (!is_cpu()) return true;
#endif
    return false;
}

// Engine used to run oneDNN fusion patterns for testing.
const dnnl::engine &get_test_engine() {
    if (is_cpu()) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        static dnnl::engine eng(get_graph_engine());
#else
        static dnnl::graph::allocator alloc {};
        static dnnl::engine eng
                = make_engine_with_allocator(dnnl::engine::kind::cpu,
                        static_cast<size_t>(engine_index), alloc);
#endif
        return eng;
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        static dnnl::engine eng(get_graph_engine());
#else
        assert(!"GPU only support DPCPP runtime now");
        static dnnl::engine eng {
                dnnl::engine::kind::gpu, static_cast<size_t>(engine_index)};
#endif
        return eng;
    }
}

const dnnl::stream &get_test_stream() {
    using stream = dnnl::stream;
    if (is_cpu()) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        static const stream strm(get_graph_stream());
#else
        static const stream strm(get_test_engine());
#endif
        return strm;
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        static const stream strm(get_graph_stream());
#else
        assert(!"GPU only support DPCPP runtime now");
        static const stream strm(get_test_engine());
#endif
        return strm;
    }
}

dnnl::graph::op::kind opstr2kind(const std::string &kind) {
    const std::unordered_map<std::string, dnnl::graph::op::kind> op_map = {
            {"Abs", dnnl::graph::op::kind::Abs},
            {"AbsBackward", dnnl::graph::op::kind::AbsBackward},
            {"Add", dnnl::graph::op::kind::Add},
            {"AvgPool", dnnl::graph::op::kind::AvgPool},
            {"AvgPoolBackward", dnnl::graph::op::kind::AvgPoolBackward},
            {"BatchNormInference", dnnl::graph::op::kind::BatchNormInference},
            {"BatchNormForwardTraining",
                    dnnl::graph::op::kind::BatchNormForwardTraining},
            {"BatchNormTrainingBackward",
                    dnnl::graph::op::kind::BatchNormTrainingBackward},
            {"BiasAdd", dnnl::graph::op::kind::BiasAdd},
            {"BiasAddBackward", dnnl::graph::op::kind::BiasAddBackward},
            {"Clamp", dnnl::graph::op::kind::Clamp},
            {"ClampBackward", dnnl::graph::op::kind::ClampBackward},
            {"Concat", dnnl::graph::op::kind::Concat},
            {"Convolution", dnnl::graph::op::kind::Convolution},
            {"ConvolutionBackwardData",
                    dnnl::graph::op::kind::ConvolutionBackwardData},
            {"ConvolutionBackwardWeights",
                    dnnl::graph::op::kind::ConvolutionBackwardWeights},
            {"ConvTranspose", dnnl::graph::op::kind::ConvTranspose},
            {"ConvTransposeBackwardData",
                    dnnl::graph::op::kind::ConvTransposeBackwardData},
            {"ConvTransposeBackwardWeights",
                    dnnl::graph::op::kind::ConvTransposeBackwardWeights},
            {"Dequantize", dnnl::graph::op::kind::Dequantize},
            {"Divide", dnnl::graph::op::kind::Divide},
            {"DynamicDequantize", dnnl::graph::op::kind::DynamicDequantize},
            {"DynamicQuantize", dnnl::graph::op::kind::DynamicQuantize},
            {"Elu", dnnl::graph::op::kind::Elu},
            {"EluBackward", dnnl::graph::op::kind::EluBackward},
            {"End", dnnl::graph::op::kind::End},
            {"Exp", dnnl::graph::op::kind::Exp},
            {"GELU", dnnl::graph::op::kind::GELU},
            {"GELUBackward", dnnl::graph::op::kind::GELUBackward},
            {"HardSigmoid", dnnl::graph::op::kind::HardSigmoid},
            {"HardSigmoidBackward", dnnl::graph::op::kind::HardSigmoidBackward},
            {"HardSwish", dnnl::graph::op::kind::HardSwish},
            {"HardSwishBackward", dnnl::graph::op::kind::HardSwishBackward},
            {"Interpolate", dnnl::graph::op::kind::Interpolate},
            {"InterpolateBackward", dnnl::graph::op::kind::InterpolateBackward},
            {"LayerNorm", dnnl::graph::op::kind::LayerNorm},
            {"LayerNormBackward", dnnl::graph::op::kind::LayerNormBackward},
            {"LeakyReLU", dnnl::graph::op::kind::LeakyReLU},
            {"Log", dnnl::graph::op::kind::Log},
            {"LogSoftmax", dnnl::graph::op::kind::LogSoftmax},
            {"LogSoftmaxBackward", dnnl::graph::op::kind::LogSoftmaxBackward},
            {"MatMul", dnnl::graph::op::kind::MatMul},
            {"Maximum", dnnl::graph::op::kind::Maximum},
            {"MaxPool", dnnl::graph::op::kind::MaxPool},
            {"MaxPoolBackward", dnnl::graph::op::kind::MaxPoolBackward},
            {"Minimum", dnnl::graph::op::kind::Minimum},
            {"Mish", dnnl::graph::op::kind::Mish},
            {"MishBackward", dnnl::graph::op::kind::MishBackward},
            {"Multiply", dnnl::graph::op::kind::Multiply},
            {"PReLU", dnnl::graph::op::kind::PReLU},
            {"PReLUBackward", dnnl::graph::op::kind::PReLUBackward},
            {"Quantize", dnnl::graph::op::kind::Quantize},
            {"Reciprocal", dnnl::graph::op::kind::Reciprocal},
            {"ReduceL1", dnnl::graph::op::kind::ReduceL1},
            {"ReduceL2", dnnl::graph::op::kind::ReduceL2},
            {"ReduceMax", dnnl::graph::op::kind::ReduceMax},
            {"ReduceMean", dnnl::graph::op::kind::ReduceMean},
            {"ReduceMin", dnnl::graph::op::kind::ReduceMin},
            {"ReduceProd", dnnl::graph::op::kind::ReduceProd},
            {"ReduceSum", dnnl::graph::op::kind::ReduceSum},
            {"ReLU", dnnl::graph::op::kind::ReLU},
            {"ReLUBackward", dnnl::graph::op::kind::ReLUBackward},
            {"Reorder", dnnl::graph::op::kind::Reorder},
            {"Round", dnnl::graph::op::kind::Round},
            {"Sigmoid", dnnl::graph::op::kind::Sigmoid},
            {"SigmoidBackward", dnnl::graph::op::kind::SigmoidBackward},
            {"SoftMax", dnnl::graph::op::kind::SoftMax},
            {"SoftMaxBackward", dnnl::graph::op::kind::SoftMaxBackward},
            {"SoftPlus", dnnl::graph::op::kind::SoftPlus},
            {"SoftPlusBackward", dnnl::graph::op::kind::SoftPlusBackward},
            {"Sqrt", dnnl::graph::op::kind::Sqrt},
            {"SqrtBackward", dnnl::graph::op::kind::SqrtBackward},
            {"Square", dnnl::graph::op::kind::Square},
            {"SquaredDifference", dnnl::graph::op::kind::SquaredDifference},
            {"StaticReshape", dnnl::graph::op::kind::StaticReshape},
            {"StaticTranspose", dnnl::graph::op::kind::StaticTranspose},
            {"Subtract", dnnl::graph::op::kind::Subtract},
            {"Tanh", dnnl::graph::op::kind::Tanh},
            {"TanhBackward", dnnl::graph::op::kind::TanhBackward},
            {"TypeCast", dnnl::graph::op::kind::TypeCast},
            {"Wildcard", dnnl::graph::op::kind::Wildcard}};
    const auto it = op_map.find(kind);
    if (it != op_map.end()) {
        return it->second;
    } else {
        fprintf(stderr, "graph: ERROR: Unsupported opkind: `%s`, exiting...\n",
                kind.c_str());
        exit(2);
    }
}

dnnl::graph::op::attr attrstr2kind(const std::string &attr_name) {
    const std::unordered_map<std::string, dnnl::graph::op::attr> attr_map = {
            // float32 attributes. The value of these attributes can be any single
            // float32 number.
            {"alpha", dnnl::graph::op::attr::alpha},
            {"beta", dnnl::graph::op::attr::beta},
            {"epsilon", dnnl::graph::op::attr::epsilon},
            {"max", dnnl::graph::op::attr::max},
            {"min", dnnl::graph::op::attr::min},
            {"momentum", dnnl::graph::op::attr::momentum},
            // float32 vector attributes. The value of these attributes can be a
            // vector of float32 numbers.
            {"scales", dnnl::graph::op::attr::scales},
            // int64_t attributes. The value of these attributes can be any single
            // int64 number.
            {"axis", dnnl::graph::op::attr::axis},
            {"begin_norm_axis", dnnl::graph::op::attr::begin_norm_axis},
            {"groups", dnnl::graph::op::attr::groups},
            // int64_t vector attributes. The value of these attributes can be a
            // vector of int64 numbers.
            {"axes", dnnl::graph::op::attr::axes},
            {"dilations", dnnl::graph::op::attr::dilations},
            {"weights_shape", dnnl::graph::op::attr::weights_shape},
            {"src_shape", dnnl::graph::op::attr::src_shape},
            {"kernel", dnnl::graph::op::attr::kernel},
            {"order", dnnl::graph::op::attr::order},
            {"output_padding", dnnl::graph::op::attr::output_padding},
            {"dst_shape", dnnl::graph::op::attr::dst_shape},
            {"pads_begin", dnnl::graph::op::attr::pads_begin},
            {"pads_end", dnnl::graph::op::attr::pads_end},
            {"shape", dnnl::graph::op::attr::shape},
            {"sizes", dnnl::graph::op::attr::sizes},
            {"strides", dnnl::graph::op::attr::strides},
            {"zps", dnnl::graph::op::attr::zps},
            // bool attributes. The value of these attributes can be any single bool
            // value.
            {"exclude_pad", dnnl::graph::op::attr::exclude_pad},
            {"keep_dims", dnnl::graph::op::attr::keep_dims},
            {"keep_stats", dnnl::graph::op::attr::keep_stats},
            {"per_channel_broadcast",
                    dnnl::graph::op::attr::per_channel_broadcast},
            {"special_zero", dnnl::graph::op::attr::special_zero},
            {"transpose_a", dnnl::graph::op::attr::transpose_a},
            {"transpose_b", dnnl::graph::op::attr::transpose_b},
            {"use_affine", dnnl::graph::op::attr::use_affine},
            {"use_dst", dnnl::graph::op::attr::use_dst},
            // string attributes. The value of these attributes can be a string.
            {"auto_broadcast", dnnl::graph::op::attr::auto_broadcast},
            {"auto_pad", dnnl::graph::op::attr::auto_pad},
            {"coordinate_transformation_mode",
                    dnnl::graph::op::attr::coordinate_transformation_mode},
            {"data_format", dnnl::graph::op::attr::data_format},
            {"weights_format", dnnl::graph::op::attr::weights_format},
            {"mode", dnnl::graph::op::attr::mode},
            {"qtype", dnnl::graph::op::attr::qtype},
            {"rounding_type", dnnl::graph::op::attr::rounding_type},
    };
    const auto it = attr_map.find(attr_name);
    if (it != attr_map.end()) {
        return it->second;
    } else {
        fprintf(stderr,
                "graph: ERROR: Unsupported attribute: `%s`, exiting...\n",
                attr_name.c_str());
        exit(2);
    }
}

std::string strides2memory_tag(const size_t ndims,
        const dnnl::graph::logical_tensor::dims &strides, bool use_x_tag) {
    std::string template_tag = "abcdefghijk";
    std::vector<std::pair<int64_t, char>> vp;
    bool valid_strides = ndims == strides.size();
    std::string memory_tag;

    // Inserting element in pair vector to keep track of indexes
    for (size_t i = 0; i < strides.size(); ++i) {
        if (strides[i] > 0) {
            vp.emplace_back(strides[i], template_tag.at(i));
        } else {
            valid_strides = false;
        }
    }

    if (valid_strides) {
        // Sort the strides to descending order
        std::sort(vp.begin(), vp.end(),
                [](const std::pair<int64_t, char> &x,
                        const std::pair<int64_t, char> &y) {
                    return x.first > y.first;
                });
        for (size_t i = 0; i < strides.size(); ++i) {
            memory_tag += vp[i].second;
        }
    } else {
        memory_tag = template_tag.substr(0, ndims);
    }

    // translate a, ab, abc, abcd, etc. to abx
    // translate acb, acdb, etc. to axb
    // This is to handle to different memory tag required for partition and
    // primitive. E.g., for group conv, partition tensor has abcd memory tag,
    // primitive weight has abcde memory tag, in order to copy primitive
    // memory to partition, using abx is a easy way.
    if (!use_x_tag) return memory_tag;

    if (memory_tag == "a" || memory_tag == "ab" || memory_tag == "abc"
            || memory_tag == "abcd" || memory_tag == "abcde")
        return "abx";
    if (memory_tag == "acb" || memory_tag == "acdb" || memory_tag == "acdeb")
        return "axb";

    return memory_tag;
}

dnnl::graph::logical_tensor::dims memory_tag2strides(
        const dnnl::graph::logical_tensor::dims &shape,
        const std::string &tag) {
    std::string template_tag = "abcdefghijk";
    // map of {a:0, b:1, c:2, d:3, etc}
    std::unordered_map<char, size_t> char2dim;
    for (size_t i = 0; i < tag.length(); ++i) {
        char2dim[template_tag[i]] = i;
    }

    const size_t ndims = shape.size();
    dnnl::graph::logical_tensor::dims strides(ndims);
    // start from tag's last char, find corresponding dim
    // and set stride
    // example:
    // shape = 1x3x4x4, tag = acdb
    // char2dim[b] = 1, stride[1] = 1
    // char2dim[d] = 3, stride[3] = stride[1] * shape[1] = 3
    // char2dim[c] = 2, stride[2] = stride[3] * shape[3] = 12
    // char2dim[a] = 0, stride[0] = stride[2] * shape[2] = 48
    dnnl_dim_t s = 1;
    for (size_t i = 0; i < ndims; ++i) {
        size_t adim = char2dim[tag[ndims - 1 - i]];
        strides[adim] = s;
        // handle the 0-D tensor case
        if (shape[adim] == 0)
            s = s * 1;
        else
            s = s * shape[adim];
    }
    return strides;
}

void change_format_to_ncx(dims_t &dims) {
    // change format from nxc to ncx
    const auto ndims = static_cast<int>(dims.size());
    dims.insert(dims.begin() + 1, dims[ndims - 1]);
    dims.erase(dims.end() - 1);
}

} // namespace graph
