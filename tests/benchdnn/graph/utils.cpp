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

#include <algorithm>
#include <set>
#include <vector>

#include "oneapi/dnnl/dnnl_debug.h"

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
    if (get_test_engine_kind() == dnnl_cpu) {
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

inline bool should_stop(const timer::timer_t &t) {
    const bool stop = false
            || (fix_times_per_prb && t.times() >= fix_times_per_prb)
            || (!fix_times_per_prb && t.total_ms() >= max_ms_per_prb
                    && t.times() >= min_times_per_prb);
    return stop;
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
    maybe_reset_profiling();

    bool is_first_loop = true;
    while (true) {
        for (size_t i = 0; i < sz; i++) {
            for (int j = 0; j < cur_batch_times; j++) {
                DNN_GRAPH_SAFE(
                        perf_func_v[i](stream, inputs_v[i], outputs_v[i]),
                        WARN);
            }
        }
        DNN_GRAPH_SAFE(stream.wait(), WARN);

        uint64_t ticks = 0;
        maybe_reset_profiling(&ticks);
        t.stamp(cur_batch_times, (unsigned long long)ticks);

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
        std::vector<dnnl::graph::compiled_partition> &cp_v,
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
    void *ptr = is_bench_mode(CORR)
            ? test_sycl_malloc_wrapper(size, alignment, dev, ctx)
            : s_mm_mgr.sycl_alloc_mm(size, alignment, dev, ctx);

    return ptr;
}

// perf mode, mem will be finally released in s_mm_mgr ~shared_ptr when
// test finished.
void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    if (is_bench_mode(CORR)) {
        test_sycl_free_wrapper(ptr, device, context, event);
    } else {
        s_mm_mgr.sycl_free_mm(ptr, device, context, event);
    }
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
    static dnnl::engine test_eng {::get_test_engine()};
    static sycl::device dev {dnnl::sycl_interop::get_device(test_eng)};
    static sycl::context ctx {dnnl::sycl_interop::get_context(test_eng)};

    static sycl::queue q {ctx, dev, sycl::property::queue::in_order {}};

    static dnnl::stream strm {
            dnnl::sycl_interop::make_stream(get_graph_engine(), q)};
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
    if (get_test_engine_kind() == dnnl_cpu) {
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
    if (get_test_engine_kind() == dnnl_cpu) {
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
            {"AbsBackprop", dnnl::graph::op::kind::AbsBackprop},
            {"Add", dnnl::graph::op::kind::Add},
            {"AvgPool", dnnl::graph::op::kind::AvgPool},
            {"AvgPoolBackprop", dnnl::graph::op::kind::AvgPoolBackprop},
            {"BatchNormInference", dnnl::graph::op::kind::BatchNormInference},
            {"BatchNormForwardTraining",
                    dnnl::graph::op::kind::BatchNormForwardTraining},
            {"BatchNormTrainingBackprop",
                    dnnl::graph::op::kind::BatchNormTrainingBackprop},
            {"BiasAdd", dnnl::graph::op::kind::BiasAdd},
            {"BiasAddBackprop", dnnl::graph::op::kind::BiasAddBackprop},
            {"Clamp", dnnl::graph::op::kind::Clamp},
            {"ClampBackprop", dnnl::graph::op::kind::ClampBackprop},
            {"Concat", dnnl::graph::op::kind::Concat},
            {"Convolution", dnnl::graph::op::kind::Convolution},
            {"ConvolutionBackpropData",
                    dnnl::graph::op::kind::ConvolutionBackpropData},
            {"ConvolutionBackpropFilters",
                    dnnl::graph::op::kind::ConvolutionBackpropFilters},
            {"ConvTranspose", dnnl::graph::op::kind::ConvTranspose},
            {"ConvTransposeBackpropData",
                    dnnl::graph::op::kind::ConvTransposeBackpropData},
            {"ConvTransposeBackpropFilters",
                    dnnl::graph::op::kind::ConvTransposeBackpropFilters},
            {"Dequantize", dnnl::graph::op::kind::Dequantize},
            {"Divide", dnnl::graph::op::kind::Divide},
            {"DynamicDequantize", dnnl::graph::op::kind::DynamicDequantize},
            {"DynamicQuantize", dnnl::graph::op::kind::DynamicQuantize},
            {"Elu", dnnl::graph::op::kind::Elu},
            {"EluBackprop", dnnl::graph::op::kind::EluBackprop},
            {"End", dnnl::graph::op::kind::End},
            {"Erf", dnnl::graph::op::kind::Erf},
            {"Exp", dnnl::graph::op::kind::Exp},
            {"GELU", dnnl::graph::op::kind::GELU},
            {"GELUBackprop", dnnl::graph::op::kind::GELUBackprop},
            {"HardSwish", dnnl::graph::op::kind::HardSwish},
            {"HardSwishBackprop", dnnl::graph::op::kind::HardSwishBackprop},
            {"Interpolate", dnnl::graph::op::kind::Interpolate},
            {"InterpolateBackprop", dnnl::graph::op::kind::InterpolateBackprop},
            {"LayerNorm", dnnl::graph::op::kind::LayerNorm},
            {"LayerNormBackprop", dnnl::graph::op::kind::LayerNormBackprop},
            {"LeakyReLU", dnnl::graph::op::kind::LeakyReLU},
            {"Log", dnnl::graph::op::kind::Log},
            {"LogSoftmax", dnnl::graph::op::kind::LogSoftmax},
            {"LogSoftmaxBackprop", dnnl::graph::op::kind::LogSoftmaxBackprop},
            {"MatMul", dnnl::graph::op::kind::MatMul},
            {"Maximum", dnnl::graph::op::kind::Maximum},
            {"MaxPool", dnnl::graph::op::kind::MaxPool},
            {"MaxPoolBackprop", dnnl::graph::op::kind::MaxPoolBackprop},
            {"Minimum", dnnl::graph::op::kind::Minimum},
            {"Mish", dnnl::graph::op::kind::Mish},
            {"MishBackprop", dnnl::graph::op::kind::MishBackprop},
            {"Multiply", dnnl::graph::op::kind::Multiply},
            {"PReLU", dnnl::graph::op::kind::PReLU},
            {"PReLUBackprop", dnnl::graph::op::kind::PReLUBackprop},
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
            {"ReLUBackprop", dnnl::graph::op::kind::ReLUBackprop},
            {"Reorder", dnnl::graph::op::kind::Reorder},
            {"Round", dnnl::graph::op::kind::Round},
            {"Sigmoid", dnnl::graph::op::kind::Sigmoid},
            {"SigmoidBackprop", dnnl::graph::op::kind::SigmoidBackprop},
            {"SoftMax", dnnl::graph::op::kind::SoftMax},
            {"SoftMaxBackprop", dnnl::graph::op::kind::SoftMaxBackprop},
            {"SoftPlus", dnnl::graph::op::kind::SoftPlus},
            {"SoftPlusBackprop", dnnl::graph::op::kind::SoftPlusBackprop},
            {"Sqrt", dnnl::graph::op::kind::Sqrt},
            {"SqrtBackprop", dnnl::graph::op::kind::SqrtBackprop},
            {"Square", dnnl::graph::op::kind::Square},
            {"SquaredDifference", dnnl::graph::op::kind::SquaredDifference},
            {"StaticReshape", dnnl::graph::op::kind::StaticReshape},
            {"StaticTranspose", dnnl::graph::op::kind::StaticTranspose},
            {"Subtract", dnnl::graph::op::kind::Subtract},
            {"Tanh", dnnl::graph::op::kind::Tanh},
            {"TanhBackprop", dnnl::graph::op::kind::TanhBackprop},
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
            {"filter_shape", dnnl::graph::op::attr::filter_shape},
            {"input_shape", dnnl::graph::op::attr::input_shape},
            {"kernel", dnnl::graph::op::attr::kernel},
            {"order", dnnl::graph::op::attr::order},
            {"output_padding", dnnl::graph::op::attr::output_padding},
            {"output_shape", dnnl::graph::op::attr::output_shape},
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
            {"filter_format", dnnl::graph::op::attr::filter_format},
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

void skip_unimplemented_data_type(
        const std::vector<dnnl::graph::logical_tensor> &in_out_lts,
        res_t *res) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    using namespace dnnl::impl::cpu::platform;
    // bf16 is supported on AVX512-CORE+
    const bool has_bf16_support
            = is_gpu() || (is_cpu() && has_data_type_support(dnnl_bf16));
    const bool has_f16_support
            = is_gpu() || (is_cpu() && has_data_type_support(dnnl_f16));
#else
    const bool has_bf16_support = is_gpu();
    const bool has_f16_support = is_gpu();
#endif
    for (const auto &in : in_out_lts) {
        bool need_skip = false;
        switch (in.get_data_type()) {
            case dnnl::graph::logical_tensor::data_type::bf16:
                need_skip = !has_bf16_support;
                break;
            case dnnl::graph::logical_tensor::data_type::f16:
                need_skip = !has_f16_support;
                break;
            default: break;
        }
        if (need_skip) {
            res->state = SKIPPED, res->reason = DATA_TYPE_NOT_SUPPORTED;
            return;
        }
    }
}

template <typename T>
void compare_data(
        T *dst, T *ref, size_t size, float rtol, float atol, bool equal_nan) {
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

} // namespace graph
