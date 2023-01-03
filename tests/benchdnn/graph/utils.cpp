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

#include <algorithm>
#include <set>
#include <vector>

#include <oneapi/dnnl/dnnl_debug.h>

#include "src/cpu/platform.hpp"
#include "utils.hpp"

namespace graph {

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
            {"DynamicReshape", dnnl::graph::op::kind::DynamicReshape},
            {"DynamicTranspose", dnnl::graph::op::kind::DynamicTranspose},
            {"DynamicQuantize", dnnl::graph::op::kind::DynamicQuantize},
            {"DynamicDequantize", dnnl::graph::op::kind::DynamicDequantize},
            {"Elu", dnnl::graph::op::kind::Elu},
            {"EluBackprop", dnnl::graph::op::kind::EluBackprop},
            {"Erf", dnnl::graph::op::kind::Erf},
            {"Exp", dnnl::graph::op::kind::Exp},
            {"GELU", dnnl::graph::op::kind::GELU},
            {"GELUBackprop", dnnl::graph::op::kind::GELUBackprop},
            {"HardSigmoid", dnnl::graph::op::kind::HardSigmoid},
            {"HardSwish", dnnl::graph::op::kind::HardSwish},
            {"HardSwishBackprop", dnnl::graph::op::kind::HardSwishBackprop},
            {"Index", dnnl::graph::op::kind::Index},
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
            {"Negative", dnnl::graph::op::kind::Negative},
            {"Pow", dnnl::graph::op::kind::Pow},
            {"PowBackprop", dnnl::graph::op::kind::PowBackprop},
            {"PowBackpropExponent", dnnl::graph::op::kind::PowBackpropExponent},
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
            {"Rsqrt", dnnl::graph::op::kind::Rsqrt},
            {"Select", dnnl::graph::op::kind::Select},
            {"Sigmoid", dnnl::graph::op::kind::Sigmoid},
            {"SigmoidBackprop", dnnl::graph::op::kind::SigmoidBackprop},
            {"Sign", dnnl::graph::op::kind::Sign},
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

std::string strides2memory_tag(const size_t ndims,
        const dnnl::graph::logical_tensor::dims_t &strides, bool use_x_tag) {
    std::string template_tag = "abcdefghijk";
    std::vector<std::pair<dnnl_graph_dim_t, char>> vp;
    bool valid_strides = ndims == strides.size();
    std::string memory_tag;

    // Inserting element in pair vector
    // to keep track of indexes
    for (size_t i = 0; i < strides.size(); ++i) {
        if (strides[i] > 0) {
            vp.push_back(std::make_pair(strides[i], template_tag.at(i)));
        } else {
            valid_strides = false;
        }
    }

    if (valid_strides) {
        // Sort the strides to descending order
        std::sort(vp.begin(), vp.end(),
                [](const std::pair<dnnl_graph_dim_t, char> &x,
                        const std::pair<dnnl_graph_dim_t, char> &y) {
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
    // this is to handle to different memory tag
    // required for partition and primitive.
    // For example, for group conv, partition tensor
    // has abcd memory tag, primitive weight has
    // abcde memory tag, in order to copy primitive
    // memory to partition, using abx is a easy way
    if (!use_x_tag) return memory_tag;

    if (memory_tag == "a" || memory_tag == "ab" || memory_tag == "abc"
            || memory_tag == "abcd" || memory_tag == "abcde")
        return "abx";
    if (memory_tag == "acb" || memory_tag == "acdb" || memory_tag == "acdeb")
        return "axb";

    return memory_tag;
}

void skip_unimplemented_data_type(
        const std::vector<dnnl::graph::logical_tensor> &in_out_lts, dir_t dir,
        res_t *res) {
#if DNNL_GRAPH_CPU_RUNTIME != DNNL_GRAPH_RUNTIME_NONE
    using namespace dnnl::impl::cpu::platform;
    // bf16 is supported on AVX512-CORE+
    const bool has_bf16_support
            = is_gpu() || (is_cpu() && has_data_type_support(dnnl_bf16));
    const bool has_f16_support = (is_gpu() && (dir & FLAG_FWD))
            || (is_cpu() && has_data_type_support(dnnl_f16));
#else
    const bool has_bf16_support = is_gpu();
    const bool has_f16_support = is_gpu() && (dir & FLAG_FWD);
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

#ifdef DNNL_GRAPH_WITH_SYCL
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
                malloc_shared(size, *static_cast<const ::sycl::device *>(dev),
                        *static_cast<const ::sycl::context *>(ctx)),
                sycl_deletor {*static_cast<const ::sycl::context *>(ctx)}};
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
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    UNUSED(alignment);
    return malloc_shared(size, *static_cast<const ::sycl::device *>(dev),
            *static_cast<const ::sycl::context *>(ctx));
}

void test_sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    UNUSED(device);
    // immediate synchronization here is for test purpose for performance, users
    // may need to store the ptr and event and handle them separately
    if (event) {
        auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
        sycl_deps_ptr->wait();
    }
    free(ptr, *static_cast<const ::sycl::context *>(context));
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

const dnnl::graph::engine &get_graph_engine() {
    static auto sycl_allocator {dnnl::graph::sycl_interop::make_allocator(
            sycl_malloc_wrapper, sycl_free_wrapper)};
    static dnnl::engine test_eng {::get_test_engine()};
    static ::sycl::device dev {dnnl::sycl_interop::get_device(test_eng)};
    static ::sycl::context ctx {dnnl::sycl_interop::get_context(test_eng)};
    static dnnl::graph::engine eng {
            dnnl::graph::sycl_interop::make_engine(dev, ctx, sycl_allocator)};
    return eng;
}

#endif // DNNL_GRAPH_WITH_SYCL

const dnnl::graph::engine &get_test_engine() {
    using engine = dnnl::graph::engine;
    if (engine_tgt_kind == dnnl_cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        static engine eng(get_graph_engine());
#else
        static engine eng(engine::kind::cpu, static_cast<int>(engine_index));
#endif
        return eng;
    } else {
#ifdef DNNL_GRAPH_GPU_SYCL
        static engine eng(get_graph_engine());
#else
        assert(!"GPU only support DPCPP runtime now");
        static engine eng(engine::kind::gpu, static_cast<int>(engine_index));
#endif
        return eng;
    }
}

} // namespace graph
