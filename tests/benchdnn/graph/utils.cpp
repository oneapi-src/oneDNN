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
    cpp_stream_t stream {get_graph_engine()};
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

inline dnnl::stream::flags get_profiling_flags() {
#ifdef DNNL_EXPERIMENTAL_PROFILING
    return dnnl::stream::flags::profiling;
#else
    return static_cast<dnnl::stream::flags>(
            dnnl::impl::stream_flags::profiling);
#endif
}

inline int measure_perf_aggregate(timer::timer_t &t,
        std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v) {
    const int max_batch_times = 4096;
    // Nvidia/AMD don't support profiling.
    const bool use_profiling = is_gpu() && !is_nvidia_gpu() && !is_amd_gpu();
    const dnnl::stream::flags flags = use_profiling
            ? dnnl::stream::flags::default_flags | get_profiling_flags()
            : dnnl::stream::flags::default_flags;
    cpp_stream_t stream {get_graph_engine(), flags};

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
    reset_gpu_profiling(((dnnl::stream)stream).get());

    bool is_first_loop = true;
    size_t prim_num = 1;
    while (true) {
        for_(int i = 0; i < cur_batch_times; i++)
        for (size_t j = 0; j < sz; j++) {
            DNN_GRAPH_SAFE(
                    perf_func_v[j](stream, inputs_v[j], outputs_v[j]), WARN);
        }
        DNN_GRAPH_SAFE(stream.wait(), WARN);

        if (use_profiling) {
            std::vector<uint64_t> nsecs;
            std::vector<uint64_t> cycles;
            get_gpu_profiling_info(((dnnl::stream)stream).get(), nsecs, cycles);
            reset_gpu_profiling(((dnnl::stream)stream).get());

            // Profiling should have information to report, otherwise, stop.
            if (nsecs.empty()) {
                BENCHDNN_PRINT(0, "%s\n",
                        "WARNING: no counters were found during profiling.");
                break;
            }
            // Calculate the number of primitives in a batch
            if (is_first_loop) { prim_num = nsecs.size() / cur_batch_times; }

            for (int i = 0; i < cur_batch_times; i++) {
                int64_t cycles_res = 0;
                double nsecs_res = 0;
                for (size_t j = 0; j < prim_num; j++) {
                    cycles_res += cycles[i * prim_num + j];
                    nsecs_res += nsecs[i * prim_num + j];
                }
                t.stop(1, cycles_res, nsecs_res / 1e6);
            }
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

inline int measure_perf_individual(timer::timer_t &t,
        std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v) {
    const bool use_profiling = is_gpu() && !is_nvidia_gpu() && !is_amd_gpu();
    const dnnl::stream::flags flags = use_profiling
            ? dnnl::stream::flags::default_flags | get_profiling_flags()
            : dnnl::stream::flags::default_flags;
    cpp_stream_t stream {get_graph_engine(), flags};

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
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        // enable GPU profiling, Nvidia/AMD dose not support profiling.
        int ret = OK;
        if (is_cpu() && !is_sycl_engine()) {
            ret = measure_perf_individual(t, perf_func_v, inputs_v, outputs_v);
        } else {
            ret = measure_perf_aggregate(t, perf_func_v, inputs_v, outputs_v);
        }
        return ret;
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
    void *ptr = has_bench_mode_bit(mode_bit_t::corr) || is_cpu()
            ? test_sycl_malloc_wrapper(size, alignment, dev, ctx)
            : s_mm_mgr.sycl_alloc_mm(size, alignment, dev, ctx);

    return ptr;
}

// perf mode, mem will be finally released in s_mm_mgr ~shared_ptr when
// test finished.
void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    if (has_bench_mode_bit(mode_bit_t::corr) || is_cpu()) {
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
            {"Pow", dnnl::graph::op::kind::Pow},
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
            {"Select", dnnl::graph::op::kind::Select},
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

class op_kind_hash_t {
public:
    std::size_t operator()(const dnnl::graph::op::kind &op_kind) const {
        return std::hash<int>()(static_cast<int>(op_kind));
    }
};

dnnl_driver_t opkind2driver(const dnnl::graph::op::kind &kind) {
    const static std::unordered_map<dnnl::graph::op::kind, dnnl_driver_t,
            op_kind_hash_t>
            op_map = {
                    {dnnl::graph::op::kind::Abs, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::AbsBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Add, dnnl_driver_t::binary},
                    {dnnl::graph::op::kind::AvgPool, dnnl_driver_t::pool},
                    {dnnl::graph::op::kind::AvgPoolBackward,
                            dnnl_driver_t::pool},
                    {dnnl::graph::op::kind::BatchNormForwardTraining,
                            dnnl_driver_t::bnorm},
                    {dnnl::graph::op::kind::BatchNormInference,
                            dnnl_driver_t::bnorm},
                    {dnnl::graph::op::kind::BatchNormTrainingBackward,
                            dnnl_driver_t::bnorm},
                    {dnnl::graph::op::kind::BiasAdd, dnnl_driver_t::binary},
                    {dnnl::graph::op::kind::BiasAddBackward,
                            dnnl_driver_t::binary},
                    {dnnl::graph::op::kind::Clamp, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::ClampBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Concat, dnnl_driver_t::concat},
                    {dnnl::graph::op::kind::Convolution, dnnl_driver_t::conv},
                    {dnnl::graph::op::kind::ConvolutionBackwardData,
                            dnnl_driver_t::conv},
                    {dnnl::graph::op::kind::ConvolutionBackwardWeights,
                            dnnl_driver_t::conv},
                    {dnnl::graph::op::kind::ConvTranspose,
                            dnnl_driver_t::deconv},
                    {dnnl::graph::op::kind::ConvTransposeBackwardData,
                            dnnl_driver_t::deconv},
                    {dnnl::graph::op::kind::ConvTransposeBackwardWeights,
                            dnnl_driver_t::deconv},
                    {dnnl::graph::op::kind::Dequantize, dnnl_driver_t::reorder},
                    {dnnl::graph::op::kind::Divide, dnnl_driver_t::binary},
                    {dnnl::graph::op::kind::DynamicDequantize,
                            dnnl_driver_t::reorder},
                    {dnnl::graph::op::kind::DynamicQuantize,
                            dnnl_driver_t::reorder},
                    {dnnl::graph::op::kind::Elu, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::EluBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::End, dnnl_driver_t::others},
                    //{dnnl::graph::op::kind::Erf, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Exp, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::GELU, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::GELUBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::HardSigmoid,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::HardSigmoidBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::HardSwish, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::HardSwishBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Interpolate,
                            dnnl_driver_t::resampling},
                    {dnnl::graph::op::kind::InterpolateBackward,
                            dnnl_driver_t::resampling},
                    {dnnl::graph::op::kind::LayerNorm, dnnl_driver_t::lnorm},
                    {dnnl::graph::op::kind::LayerNormBackward,
                            dnnl_driver_t::lnorm},
                    {dnnl::graph::op::kind::LeakyReLU, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Log, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::LogSoftmax, dnnl_driver_t::softmax},
                    {dnnl::graph::op::kind::LogSoftmaxBackward,
                            dnnl_driver_t::softmax},
                    {dnnl::graph::op::kind::MatMul, dnnl_driver_t::matmul},
                    {dnnl::graph::op::kind::Maximum, dnnl_driver_t::binary},
                    {dnnl::graph::op::kind::MaxPool, dnnl_driver_t::pool},
                    {dnnl::graph::op::kind::MaxPoolBackward,
                            dnnl_driver_t::pool},
                    {dnnl::graph::op::kind::Minimum, dnnl_driver_t::binary},
                    {dnnl::graph::op::kind::Mish, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::MishBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Multiply, dnnl_driver_t::binary},
                    {dnnl::graph::op::kind::PReLU, dnnl_driver_t::prelu},
                    {dnnl::graph::op::kind::PReLUBackward,
                            dnnl_driver_t::prelu},
                    {dnnl::graph::op::kind::Pow, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Quantize, dnnl_driver_t::reorder},
                    {dnnl::graph::op::kind::Reciprocal, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::ReduceL1, dnnl_driver_t::reduction},
                    {dnnl::graph::op::kind::ReduceL2, dnnl_driver_t::reduction},
                    {dnnl::graph::op::kind::ReduceMax,
                            dnnl_driver_t::reduction},
                    {dnnl::graph::op::kind::ReduceMean,
                            dnnl_driver_t::reduction},
                    {dnnl::graph::op::kind::ReduceMin,
                            dnnl_driver_t::reduction},
                    {dnnl::graph::op::kind::ReduceProd,
                            dnnl_driver_t::reduction},
                    {dnnl::graph::op::kind::ReduceSum,
                            dnnl_driver_t::reduction},
                    {dnnl::graph::op::kind::ReLU, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::ReLUBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Reorder, dnnl_driver_t::reorder},
                    {dnnl::graph::op::kind::Round, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Select, dnnl_driver_t::custom},
                    {dnnl::graph::op::kind::Sigmoid, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::SigmoidBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::SoftMax, dnnl_driver_t::softmax},
                    {dnnl::graph::op::kind::SoftMaxBackward,
                            dnnl_driver_t::softmax},
                    {dnnl::graph::op::kind::SoftPlus, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::SoftPlusBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Sqrt, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::SqrtBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::Square, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::SquaredDifference,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::StaticReshape,
                            dnnl_driver_t::custom},
                    {dnnl::graph::op::kind::StaticTranspose,
                            dnnl_driver_t::custom},
                    {dnnl::graph::op::kind::Subtract, dnnl_driver_t::binary},
                    {dnnl::graph::op::kind::Tanh, dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::TanhBackward,
                            dnnl_driver_t::eltwise},
                    {dnnl::graph::op::kind::TypeCast, dnnl_driver_t::reorder},
                    {dnnl::graph::op::kind::Wildcard, dnnl_driver_t::others},
            };
    const auto it = op_map.find(kind);
    if (it != op_map.end()) {
        return it->second;
    } else {
        fprintf(stderr, "graph: ERROR: Unsupported opkind: `%d`, exiting...\n",
                static_cast<int>(kind));
        exit(2);
    }
}

bool is_nxc_lt_arg(const std::string &kind, const int exec_arg) {
    // Mapping from the op kind to a set that indicates which input arg needs
    // reorder
    static const std::unordered_map<std::string, std::unordered_set<int>>
            input_arg_for_reorder = {
                    {"AvgPool", {DNNL_ARG_SRC}},
                    {"AvgPoolBackward", {DNNL_ARG_DIFF_DST}},
                    {"BatchNormInference", {DNNL_ARG_SRC}},
                    {"BatchNormForwardTraining", {DNNL_ARG_SRC}},
                    {"BiasAddBackward", {DNNL_ARG_SRC}},
                    {"Interpolate", {DNNL_ARG_SRC}},
                    {"MaxPool", {DNNL_ARG_SRC}},
                    {"Convolution", {DNNL_ARG_SRC}},
                    {"ConvolutionBackwardData", {DNNL_ARG_DIFF_DST}},
                    {"ConvTranspose", {DNNL_ARG_SRC}},
                    {"ConvTransposeBackwardData", {DNNL_ARG_DIFF_DST}},
                    {"BatchNormTrainingBackward",
                            {DNNL_ARG_SRC, DNNL_ARG_DIFF_DST}},
                    {"BiasAdd", {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1}},
                    {"InterpolateBackward", {DNNL_ARG_DIFF_DST}},
                    {"MaxPoolBackward", {DNNL_ARG_SRC, DNNL_ARG_DIFF_DST}},
                    {"ConvolutionBackwardWeights",
                            {DNNL_ARG_SRC, DNNL_ARG_DIFF_DST}},
                    {"ConvTransposeBackwardWeights",
                            {DNNL_ARG_SRC, DNNL_ARG_DIFF_DST}},
                    {"PReLU", {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS}},
                    {"PReLUBackward",
                            {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS,
                                    DNNL_ARG_DIFF_DST}},
            };

    const auto iter = input_arg_for_reorder.find(kind);
    if (iter != input_arg_for_reorder.end()) {
        const auto &args_to_reorder = iter->second;
        return args_to_reorder.find(exec_arg) != args_to_reorder.end();
    } else {
        return false;
    }
}

std::string strides2memory_tag(const size_t ndims,
        const dnnl::graph::logical_tensor::dims &strides, bool use_x_tag) {
    if (ndims == 0) return "";
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
    const size_t ndims = shape.size();
    // use plain tag as default if the tensor shape rank changed
    std::string real_tag
            = ndims == tag.length() ? tag : template_tag.substr(0, ndims);

    // map of {a:0, b:1, c:2, d:3, etc}
    std::unordered_map<char, size_t> char2dim;
    for (size_t i = 0; i < real_tag.length(); ++i) {
        char2dim[template_tag[i]] = i;
    }

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
        size_t adim = char2dim[real_tag[ndims - 1 - i]];
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

void permute_md(dnn_mem_t &mem, std::vector<int64_t> permutation) {
    std::vector<int> axes(permutation.size(), -1);
    for (int i = 0; i < static_cast<int>(permutation.size()); i++) {
        axes[permutation[i]] = i;
    }
    dnnl::memory::desc md(clone_md(mem.md_));
    dnnl::memory::desc permuted_md = md.permute_axes(axes);
    // Dirty hack to replace md with another one.
    // TODO: replace it with a better solution.
    auto st = dnnl_memory_desc_destroy(mem.md_);
    (void)st;
    assert(st == dnnl_success);
    mem.md_ = clone_md(permuted_md.get());
}
void reshape_md(dnn_mem_t &mem, const dnnl::memory::dims &reshaped_dims,
        const dnnl::memory::dims &reshaped_strides) {

    const auto data_type = static_cast<dnnl::memory::data_type>(mem.dt());
    dnnl::memory::desc md(reshaped_dims, data_type, reshaped_strides);
    // Dirty hack to replace md with another one.
    // TODO: replace it with a better solution.
    auto st = dnnl_memory_desc_destroy(mem.md_);
    (void)st;
    assert(st == dnnl_success);
    mem.md_ = clone_md(md.get());
}

void reshape_md(dnn_mem_t &mem, const dnnl::memory::dims &reshaped_dims) {
    dnnl::memory::desc md(clone_md(mem.md_));
    dnnl::memory::desc reshaped_md = md.reshape(reshaped_dims);
    // Dirty hack to replace md with another one.
    // TODO: replace it with a better solution.
    auto st = dnnl_memory_desc_destroy(mem.md_);
    (void)st;
    assert(st == dnnl_success);
    mem.md_ = clone_md(reshaped_md.get());
}

int get_prim_arg_name_from_graph_op_output_offset(
        dnnl::graph::op::kind op_kind, size_t output_offset) {
    switch (op_kind) {
        case dnnl::graph::op::kind::AbsBackward:
        case dnnl::graph::op::kind::AvgPoolBackward:
        case dnnl::graph::op::kind::ClampBackward:
        case dnnl::graph::op::kind::ConvolutionBackwardData:
        case dnnl::graph::op::kind::ConvTransposeBackwardData:
        case dnnl::graph::op::kind::EluBackward:
        case dnnl::graph::op::kind::GELUBackward:
        case dnnl::graph::op::kind::HardSigmoidBackward:
        case dnnl::graph::op::kind::HardSwishBackward:
        case dnnl::graph::op::kind::InterpolateBackward:
        case dnnl::graph::op::kind::LogSoftmaxBackward:
        case dnnl::graph::op::kind::MaxPoolBackward:
        case dnnl::graph::op::kind::MishBackward:
        case dnnl::graph::op::kind::ReLUBackward:
        case dnnl::graph::op::kind::SigmoidBackward:
        case dnnl::graph::op::kind::SoftMaxBackward:
        case dnnl::graph::op::kind::SoftPlusBackward:
        case dnnl::graph::op::kind::SqrtBackward:
        case dnnl::graph::op::kind::TanhBackward: {
            return DNNL_ARG_DIFF_SRC;
        } break;
        case dnnl::graph::op::kind::BatchNormForwardTraining: {
            if (output_offset == 0)
                return DNNL_ARG_DST;
            else if (output_offset == 1)
                return 0; // running mean is not supported for now
            else if (output_offset == 2)
                return 0; // running var is not supported for now
            else if (output_offset == 3)
                return DNNL_ARG_MEAN;
            else if (output_offset == 4)
                return DNNL_ARG_VARIANCE;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::LayerNorm: {
            if (output_offset == 0)
                return DNNL_ARG_DST;
            else if (output_offset == 1)
                return DNNL_ARG_MEAN;
            else if (output_offset == 2)
                return DNNL_ARG_VARIANCE;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }

        } break;
        case dnnl::graph::op::kind::BatchNormTrainingBackward:
        case dnnl::graph::op::kind::LayerNormBackward: {
            if (output_offset == 0)
                return DNNL_ARG_DIFF_SRC;
            else if (output_offset == 1)
                return DNNL_ARG_DIFF_SCALE;
            else if (output_offset == 2)
                return DNNL_ARG_DIFF_SHIFT;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::BiasAddBackward: {
            return DNNL_ARG_DIFF_BIAS;
        } break;
        case dnnl::graph::op::kind::ConvolutionBackwardWeights:
        case dnnl::graph::op::kind::ConvTransposeBackwardWeights: {
            return DNNL_ARG_DIFF_WEIGHTS;
        } break;
        case dnnl::graph::op::kind::Quantize:
        case dnnl::graph::op::kind::Dequantize:
        case dnnl::graph::op::kind::DynamicDequantize:
        case dnnl::graph::op::kind::DynamicQuantize:
        case dnnl::graph::op::kind::TypeCast:
        case dnnl::graph::op::kind::Reorder: {
            return DNNL_ARG_TO;
        } break;
        case dnnl::graph::op::kind::PReLUBackward: {
            if (output_offset == 0)
                return DNNL_ARG_DIFF_SRC;
            else if (output_offset == 1)
                return DNNL_ARG_DIFF_WEIGHTS;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        static_cast<int>(output_offset));
                assert(false);
                return -1;
            }
        } break;
        default: {
            return DNNL_ARG_DST;
        } break;
    }
}

int get_prim_arg_name_from_graph_op_input_offset(
        dnnl::graph::op::kind op_kind, int input_offset, bool use_dst) {
    switch (op_kind) {
        case dnnl::graph::op::kind::Add:
        case dnnl::graph::op::kind::BiasAdd:
        case dnnl::graph::op::kind::Divide:
        case dnnl::graph::op::kind::Maximum:
        case dnnl::graph::op::kind::Minimum:
        case dnnl::graph::op::kind::Multiply:
        case dnnl::graph::op::kind::Subtract: {
            if (input_offset == 0)
                return DNNL_ARG_SRC_0;
            else if (input_offset == 1)
                return DNNL_ARG_SRC_1;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        }
        case dnnl::graph::op::kind::Concat: {
            return DNNL_ARG_MULTIPLE_SRC + input_offset;
        } break;
        case dnnl::graph::op::kind::Convolution:
        case dnnl::graph::op::kind::ConvTranspose:
        case dnnl::graph::op::kind::MatMul: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_WEIGHTS;
            else if (input_offset == 2)
                return DNNL_ARG_BIAS;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::ConvolutionBackwardData:
        case dnnl::graph::op::kind::ConvTransposeBackwardData: {
            if (input_offset == 0)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 1)
                return DNNL_ARG_WEIGHTS;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::ConvolutionBackwardWeights:
        case dnnl::graph::op::kind::ConvTransposeBackwardWeights: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::PReLU: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_WEIGHTS;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::PReLUBackward: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_WEIGHTS;
            else if (input_offset == 2)
                return DNNL_ARG_DIFF_DST;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::AvgPoolBackward: {
            if (input_offset == 0)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 1) {
                BENCHDNN_PRINT(0,
                        "Error: no support for input %d of Avg Pool Backward",
                        input_offset);
                assert(false);
                return -1;
            } else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::MaxPoolBackward: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 2) {
                BENCHDNN_PRINT(0,
                        "Error: no support for input %d of Max Pool Backward",
                        input_offset);
                assert(false);
                return -1;
            } else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::SoftMaxBackward:
        case dnnl::graph::op::kind::LogSoftmaxBackward: {
            if (input_offset == 0)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 1)
                return DNNL_ARG_DST;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::LayerNorm: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_SCALE;
            else if (input_offset == 2)
                return DNNL_ARG_SHIFT;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::LayerNormBackward: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 2)
                return DNNL_ARG_MEAN;
            else if (input_offset == 3)
                return DNNL_ARG_VARIANCE;
            else if (input_offset == 4)
                return DNNL_ARG_SCALE;
            else if (input_offset == 5)
                return DNNL_ARG_SHIFT;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::BatchNormForwardTraining: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_MEAN;
            else if (input_offset == 2)
                return DNNL_ARG_VARIANCE;
            else if (input_offset == 3)
                return DNNL_ARG_SCALE;
            else if (input_offset == 4)
                return DNNL_ARG_SHIFT;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::BatchNormInference: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_SCALE;
            else if (input_offset == 2)
                return DNNL_ARG_SHIFT;
            else if (input_offset == 3)
                return DNNL_ARG_MEAN;
            else if (input_offset == 4)
                return DNNL_ARG_VARIANCE;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::BatchNormTrainingBackward: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 2)
                return DNNL_ARG_MEAN;
            else if (input_offset == 3)
                return DNNL_ARG_VARIANCE;
            else if (input_offset == 4)
                return DNNL_ARG_SCALE;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::InterpolateBackward: {
            if (input_offset == 0)
                return DNNL_ARG_DIFF_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else if (input_offset == 2) {
                BENCHDNN_PRINT(0,
                        "Error: no support for input %d of Resampling Backward",
                        input_offset);
                assert(false);
                return -1;
            } else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::ReduceL1:
        case dnnl::graph::op::kind::ReduceL2:
        case dnnl::graph::op::kind::ReduceMax:
        case dnnl::graph::op::kind::ReduceMin:
        case dnnl::graph::op::kind::ReduceMean:
        case dnnl::graph::op::kind::ReduceProd:
        case dnnl::graph::op::kind::ReduceSum: {
            if (input_offset == 0)
                return DNNL_ARG_SRC;
            else if (input_offset == 1) {
                BENCHDNN_PRINT(0, "Error: no support for input %d of Reduction",
                        input_offset);
                assert(false);
                return -1;
            } else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::AbsBackward:
        case dnnl::graph::op::kind::ClampBackward:
        case dnnl::graph::op::kind::EluBackward:
        case dnnl::graph::op::kind::GELUBackward:
        case dnnl::graph::op::kind::HardSigmoidBackward:
        case dnnl::graph::op::kind::HardSwishBackward:
        case dnnl::graph::op::kind::MishBackward:
        case dnnl::graph::op::kind::ReLUBackward:
        case dnnl::graph::op::kind::SigmoidBackward:
        case dnnl::graph::op::kind::SoftPlusBackward:
        case dnnl::graph::op::kind::SqrtBackward:
        case dnnl::graph::op::kind::TanhBackward: {
            if (input_offset == 0)
                return use_dst ? DNNL_ARG_DST : DNNL_ARG_SRC;
            else if (input_offset == 1)
                return DNNL_ARG_DIFF_DST;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::Quantize:
        case dnnl::graph::op::kind::Dequantize:
        case dnnl::graph::op::kind::TypeCast:
        case dnnl::graph::op::kind::Reorder: {
            return DNNL_ARG_FROM;
        } break;
        case dnnl::graph::op::kind::DynamicDequantize: {
            if (input_offset == 0)
                return DNNL_ARG_FROM;
            else if (input_offset == 1)
                return DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM;
            else if (input_offset == 2)
                return DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::DynamicQuantize: {
            if (input_offset == 0)
                return DNNL_ARG_FROM;
            else if (input_offset == 1)
                return DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO;
            else if (input_offset == 2)
                return DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_TO;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        case dnnl::graph::op::kind::Select: {
            if (input_offset == 0)
                return DNNL_ARG_WEIGHTS;
            else if (input_offset == 1)
                return DNNL_ARG_SRC_0;
            else if (input_offset == 2)
                return DNNL_ARG_SRC_1;
            else {
                BENCHDNN_PRINT(0, "Error: no matching ARG for offset %d",
                        input_offset);
                assert(false);
                return -1;
            }
        } break;
        default: {
            return DNNL_ARG_SRC;
        } break;
    }
}

dnnl::graph::logical_tensor::layout_type str2layout(
        const std::string &layout_type) {
    if (layout_type == "strided")
        return dnnl::graph::logical_tensor::layout_type::strided;
    else if (layout_type == "any")
        return dnnl::graph::logical_tensor::layout_type::any;
    else if (layout_type == "opaque")
        return dnnl::graph::logical_tensor::layout_type::opaque;
    else
        return dnnl::graph::logical_tensor::layout_type::undef;
}

cpp_stream_t::cpp_stream_t(
        const dnnl::engine &eng, dnnl::stream::flags flags, void *interop_obj) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        auto tp = static_cast<dnnl::threadpool_interop::threadpool_iface *>(
                interop_obj);
        if (tp == nullptr) tp = dnnl::testing::get_threadpool();
        stream_ = dnnl::threadpool_interop::make_stream(eng, tp);
        return;
    }
#endif
    stream_ = dnnl::stream {eng, flags};
}

cpp_engine_t::cpp_engine_t() {
    if (is_cpu()) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        static dnnl::graph::allocator alloc {
                dnnl::graph::sycl_interop::make_allocator(
                        sycl_malloc_wrapper, sycl_free_wrapper)};
#else
        static dnnl::graph::allocator alloc {};
#endif
        engine_ = make_engine_with_allocator(dnnl::engine::kind::cpu,
                static_cast<size_t>(engine_index), alloc);
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        static dnnl::graph::allocator alloc {
                dnnl::graph::sycl_interop::make_allocator(
                        sycl_malloc_wrapper, sycl_free_wrapper)};
        engine_ = make_engine_with_allocator(dnnl::engine::kind::gpu,
                static_cast<size_t>(engine_index), alloc);
#else
        assert(!"GraphAPI GPU only support DPCPP runtime now");
#endif
    }
}
} // namespace graph
