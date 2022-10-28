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

#include <oneapi/dnnl/dnnl_debug.h>

#include "dnnl_graph_common.hpp"
#ifdef DNNL_GRAPH_WITH_SYCL
#include "dnnl_sycl.hpp"
#endif
#include "utils/timer.hpp"

namespace benchdnnext {

void check_known_skipped_case_graph_common(
        const std::vector<dnnl_data_type_t> &v_dt, const std::string &tag,
        const dir_t &dir, res_t *res) {
    // tag::undef not supported for now
    if (tag == tag::undef) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

void check_post_sum_for_bf16in_f32out(const attr_t &attr, res_t *res,
        const std::vector<dnnl::graph::logical_tensor::data_type> &orig_dts) {
    if (orig_dts[0] == graph_dt::bf16 && orig_dts[2] == graph_dt::f32
            && check_has_sum_po(attr.post_ops.entry)) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

void check_graph_eltwise_post_ops(const attr_t &attr, res_t *res) {
    for (const auto &e : attr.post_ops.entry) {
        if (!e.is_eltwise_kind()) continue;

        if (e.eltwise.alg != dnnl_eltwise_swish
                && convert_alg_kind(e.eltwise.alg)
                        == dnnl::graph::op::kind::LastSymbol) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }

        check_graph_eltwise_params(
                res, e.kind, e.eltwise.alpha, e.eltwise.beta);
    }
}

void check_graph_scales_and_zps_support(const attr_t &attr, res_t *res) {
    // oneDNN Graph q/deq supports only following oscale policies
    const std::set<policy_t> supported_policy = {policy_t::PER_OC,
            policy_t::PER_DIM_0, policy_t::PER_DIM_1, policy_t::COMMON};

    bool oscale_ok = attr.oscale.is_def()
            || std::any_of(supported_policy.cbegin(), supported_policy.cend(),
                    [&](const policy_t policy) {
                        return attr.oscale.policy == policy;
                    });
    if (!oscale_ok) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    // currently, only policy_t::COMMON is supported for asymmetric quant
    // for src and dst, other policy is not suppoted by oneDNN Graph.
    for (const int arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        if (!attr.zero_points.is_def(arg)) {
            const auto &zp_e = attr.zero_points.get(arg);
            if (zp_e.policy != policy_t::COMMON) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }
        }
    }
}

// Due to differences between oneDNN and oneDNN graph APIs we need
// to skip cases in which elementwise parameters cannot be set. For
// example, oneDNN graph API doesn't have alpha parameter for ReLU,
// while oneDNN does. Another example is Swish, which is represented
// in oneDNN graph by Multiply+Sigmoid - Sigmoid doesn't accept any
// param, so alpha is fixed and equal to 1.0.
void check_graph_eltwise_params(res_t *res,
        const attr_t::post_ops_t::kind_t alg, const float alpha,
        const float beta) {
    using alg_t = attr_t::post_ops_t::kind_t;

    constexpr float eps = 1.0e-05;
    if (alg == alg_t::RELU_DST) {
        const float expected_alpha = 0.0;
        if (std::fabs(expected_alpha - alpha) > eps) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    } else if (alg == alg_t::SWISH) {
        const float expected_alpha = 1.0;
        if (std::fabs(expected_alpha - alpha) > eps) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

bool check_has_sum_po(
        const std::vector<attr_t::post_ops_t::entry_t> &post_ops) {
    for (const auto &entry : post_ops) {
        if (entry.is_sum_kind()) return true;
    }
    return false;
}

float get_post_eltwise_scale(
        const std::vector<attr_t::post_ops_t::entry_t> &post_ops) noexcept {
    for (const auto &po : post_ops) {
        if (po.is_eltwise_kind()) return po.eltwise.scale;
    }
    return 1.f;
}

dnnl::graph::logical_tensor::data_type convert_dt(
        const dnnl_data_type_t dt) noexcept {
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    switch (dt) {
        case dnnl_f16: return graph_dt::f16;
        case dnnl_bf16: return graph_dt::bf16;
        case dnnl_f32: return graph_dt::f32;
        case dnnl_s32: return graph_dt::s32;
        case dnnl_s8: return graph_dt::s8;
        case dnnl_u8: return graph_dt::u8;
        case dnnl_data_type_undef:
        default: return graph_dt::undef;
    }
}

dnnl_data_type_t convert_dt(
        const dnnl::graph::logical_tensor::data_type dt) noexcept {
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    switch (dt) {
        case graph_dt::f16: return dnnl_f16;
        case graph_dt::bf16: return dnnl_bf16;
        case graph_dt::f32: return dnnl_f32;
        case graph_dt::s32: return dnnl_s32;
        case graph_dt::s8: return dnnl_s8;
        case graph_dt::u8: return dnnl_u8;
        case graph_dt::undef:
        default: return dnnl_data_type_undef;
    }
}

dnnl::graph::op::kind convert_alg_kind(
        const dnnl_alg_kind_t kind, bool is_fwd) noexcept {
    using graph_op = dnnl::graph::op::kind;
    // all options could be easily added later
    if (is_fwd) {
        switch (kind) {
            case dnnl_eltwise_abs: return graph_op::Abs;
            case dnnl_eltwise_clip_v2: return graph_op::Clamp;
            case dnnl_eltwise_elu: return graph_op::Elu;
            case dnnl_eltwise_exp: return graph_op::Exp;
            case dnnl_eltwise_gelu_erf: return graph_op::GELU;
            case dnnl_eltwise_hardsigmoid: return graph_op::HardSigmoid;
            case dnnl_eltwise_hardswish: return graph_op::HardSwish;
            case dnnl_eltwise_log: return graph_op::Log;
            case dnnl_eltwise_logistic: return graph_op::Sigmoid;
            case dnnl_eltwise_mish: return graph_op::Mish;
            case dnnl_eltwise_pow: return graph_op::Pow;
            case dnnl_eltwise_relu: return graph_op::ReLU;
            case dnnl_eltwise_soft_relu: return graph_op::SoftPlus;
            case dnnl_eltwise_round: return graph_op::Round;
            case dnnl_eltwise_sqrt: return graph_op::Sqrt;
            case dnnl_eltwise_square: return graph_op::Square;
            case dnnl_eltwise_tanh: return graph_op::Tanh;
            case dnnl_binary_add: return graph_op::Add;
            case dnnl_binary_div: return graph_op::Divide;
            case dnnl_binary_max: return graph_op::Maximum;
            case dnnl_binary_min: return graph_op::Minimum;
            case dnnl_binary_mul: return graph_op::Multiply;
            case dnnl_binary_sub: return graph_op::Subtract;
            case dnnl_reduction_norm_lp_power_p_sum: return graph_op::ReduceL1;
            case dnnl_reduction_norm_lp_sum: return graph_op::ReduceL2;
            case dnnl_reduction_max: return graph_op::ReduceMax;
            case dnnl_reduction_mean: return graph_op::ReduceMean;
            case dnnl_reduction_min: return graph_op::ReduceMin;
            case dnnl_reduction_mul: return graph_op::ReduceProd;
            case dnnl_reduction_sum: return graph_op::ReduceSum;
            // TODO (damianszw): find nicer way to tell about unsupported type
            case dnnl_eltwise_clip:
            case dnnl_eltwise_clip_v2_use_dst_for_bwd:
            case dnnl_eltwise_elu_use_dst_for_bwd:
            case dnnl_eltwise_exp_use_dst_for_bwd:
            case dnnl_eltwise_gelu_tanh:
            case dnnl_eltwise_linear:
            case dnnl_eltwise_logistic_use_dst_for_bwd:
            case dnnl_eltwise_relu_use_dst_for_bwd:
            case dnnl_eltwise_sqrt_use_dst_for_bwd:
            case dnnl_eltwise_swish:
            case dnnl_eltwise_tanh_use_dst_for_bwd:
            case dnnl_reduction_norm_lp_power_p_max:
            case dnnl_reduction_norm_lp_max:
            default: return graph_op::LastSymbol;
        }
    } else {
        switch (kind) {
            case dnnl_eltwise_abs: return graph_op::AbsBackprop;
            case dnnl_eltwise_clip_v2:
            case dnnl_eltwise_clip_v2_use_dst_for_bwd:
                return graph_op::ClampBackprop;
            case dnnl_eltwise_elu:
            case dnnl_eltwise_elu_use_dst_for_bwd: return graph_op::EluBackprop;
            case dnnl_eltwise_gelu_erf: return graph_op::GELUBackprop;
            case dnnl_eltwise_hardswish: return graph_op::HardSwishBackprop;
            case dnnl_eltwise_logistic:
            case dnnl_eltwise_logistic_use_dst_for_bwd:
                return graph_op::SigmoidBackprop;
            case dnnl_eltwise_mish: return graph_op::MishBackprop;
            case dnnl_eltwise_relu:
            case dnnl_eltwise_relu_use_dst_for_bwd:
                return graph_op::ReLUBackprop;
            case dnnl_eltwise_soft_relu: return graph_op::SoftPlusBackprop;
            case dnnl_eltwise_sqrt:
            case dnnl_eltwise_sqrt_use_dst_for_bwd:
                return graph_op::SqrtBackprop;
            case dnnl_eltwise_tanh:
            case dnnl_eltwise_tanh_use_dst_for_bwd:
                return graph_op::TanhBackprop;
            // Don't support for now
            case dnnl_eltwise_exp:
            case dnnl_eltwise_log:
            case dnnl_eltwise_pow:
            case dnnl_eltwise_round:
            case dnnl_eltwise_square:
            case dnnl_binary_add:
            case dnnl_binary_div:
            case dnnl_binary_max:
            case dnnl_binary_min:
            case dnnl_binary_mul:
            case dnnl_binary_sub:
            case dnnl_reduction_norm_lp_power_p_sum:
            case dnnl_reduction_norm_lp_sum:
            case dnnl_reduction_max:
            case dnnl_reduction_mean:
            case dnnl_reduction_min:
            case dnnl_reduction_mul:
            case dnnl_reduction_sum:
            // TODO (damianszw): find nicer way to tell about unsupported type
            case dnnl_eltwise_clip:
            case dnnl_eltwise_exp_use_dst_for_bwd:
            case dnnl_eltwise_gelu_tanh:
            case dnnl_eltwise_linear:
            case dnnl_eltwise_swish:
            case dnnl_reduction_norm_lp_power_p_max:
            case dnnl_reduction_norm_lp_max:
            default: return graph_op::LastSymbol;
        }
    }
}

dims_t convert_bin_policy(
        const dims_t &lhs_dims, const attr_t::policy_t policy) noexcept {
    using bin_pol = attr_t::policy_t;

    auto rhs_dims = dims_t(lhs_dims.size(), 1);

    switch (policy) {
        case bin_pol::PER_TENSOR: rhs_dims = lhs_dims; break;
        case bin_pol::PER_OC: rhs_dims[1] = lhs_dims[1]; break;
        case bin_pol::PER_DIM_0: rhs_dims[0] = lhs_dims[0]; break;
        case bin_pol::PER_DIM_1: rhs_dims[1] = lhs_dims[1]; break;
        case bin_pol::PER_DIM_01:
            rhs_dims[0] = lhs_dims[0];
            rhs_dims[1] = lhs_dims[1];
            break;
        case bin_pol::COMMON:
        default: break;
    }

    return rhs_dims;
}

std::string convert_attr_policy(attr_t::policy_t policy) noexcept {
    std::string ret_policy;
    switch (policy) {
        case attr_t::policy_t::PER_DIM_0:
        case attr_t::policy_t::PER_OC:
        case attr_t::policy_t::PER_DIM_1: ret_policy = "per_channel"; break;
        case attr_t::policy_t::COMMON: ret_policy = "per_tensor"; break;
        default: assert(!"policy not supported for now."); SAFE_V(FAIL);
    }
    return ret_policy;
}

std::map<std::string, float> convert_eltw_entry(
        const dnnl::graph::op::kind op_kind,
        const attr_t::post_ops_t::entry_t &entry) noexcept {
    using graph_op = dnnl::graph::op::kind;

    std::map<std::string, float> attrs;
    // all options could be easily added later
    switch (op_kind) {
        case graph_op::ReLU: {
            // provide alpha for LeakyReLU
            if (entry.eltwise.alpha != 0.f)
                attrs["alpha"] = entry.eltwise.alpha;
            return attrs;
        }
        case graph_op::Elu: attrs["alpha"] = entry.eltwise.alpha; return attrs;
        case graph_op::Clamp:
            attrs["min"] = entry.eltwise.alpha;
            attrs["max"] = entry.eltwise.beta;
            return attrs;
        case graph_op::HardSigmoid:
            attrs["alpha"] = entry.eltwise.alpha;
            attrs["beta"] = entry.eltwise.beta;
            return attrs;
        default: return attrs;
    }
}

dnnl::graph::graph::fpmath_mode convert_fpmath_mode(
        const dnnl_fpmath_mode_t mode) noexcept {
    using fpmath_mode = dnnl::graph::graph::fpmath_mode;

    switch (mode) {
        case dnnl_fpmath_mode_strict: return fpmath_mode::strict;
        case dnnl_fpmath_mode_bf16: return fpmath_mode::bf16;
        case dnnl_fpmath_mode_f16: return fpmath_mode::f16;
        case dnnl_fpmath_mode_any: return fpmath_mode::any;
        case dnnl_fpmath_mode_tf32: return fpmath_mode::tf32;
        default:
            assert("fpmath policy not supported for now. use strict as "
                   "default");
            return fpmath_mode::strict;
    }
}

int scale_bia(dnn_mem_t &dst, dnn_mem_t &src, const std::vector<float> &scales,
        const int bia_mask) {
    if (scales.empty()) {
        dst = std::move(src);
        return OK;
    }
    float eps = 1.e-9;
    std::vector<float> bia_scales(scales.size(), 0.f);
    std::transform(scales.begin(), scales.end(), bia_scales.begin(),
            [eps](const float scale) { return 1.f / (scale + eps); });
    dnnl_primitive_attr_t bia_attr = nullptr;
    dnnl_primitive_attr_create(&bia_attr);
    dnnl_primitive_attr_set_output_scales_mask(
            bia_attr, bia_mask);
    SAFE(dst.reorder(src, bia_attr), CRIT);
    dnnl_primitive_attr_destroy(bia_attr);

    return OK;
}

dnnl_format_tag_t dnnl_fmt_str2tag(const std::string &fmt_str) {
    dnnl_format_tag_t tag = dnnl_format_tag_undef;
    for (int i = 0; i < dnnl_format_tag_last; ++i) {
        tag = static_cast<dnnl_format_tag_t>(i);
        if (dnnl_fmt_tag2str(tag) == fmt_str) break;
    }
    if (tag == dnnl_format_tag_undef)
        []() {
            SAFE(FAIL, CRIT);
            return 0;
        }();
    return tag;
};

dims_t calculate_strides(dims_t dims, dt dtype, const std::string &tag) {
    dnnl_dims_t dnnl_dims = {0};
    std::copy(dims.begin(), dims.end(), dnnl_dims);

    auto md = dnn_mem_t::init_md(
            (int)dims.size(), dnnl_dims, convert_dt(dtype), tag);

    dims_t strides(query_md_ndims(md));
    std::memcpy(strides.data(), query_md_strides(md),
            strides.size() * sizeof(dnnl_dim_t));
    return strides;
}

std::vector<float> get_scales(const attr_t::scale_t &scales_info,
        const float *raw_scales, int64_t channel_size) {
    const auto q_vals
            = scales_info.policy == policy_t::COMMON ? 1 : channel_size;
    return scales_info.is_def()
            ? std::vector<float>(q_vals, 1.f)
            : std::vector<float>(raw_scales, raw_scales + q_vals);
}

// Get indices, on which post binary ops are located.
std::vector<size_t> get_post_bin_indices(
        const std::vector<attr_t::post_ops_t::entry_t> &po_entry) {
    std::vector<size_t> post_bin_indexes {};
    for (size_t idx = 0; idx < po_entry.size(); idx++) {
        if (po_entry[idx].is_binary_kind()) post_bin_indexes.push_back(idx);
    }
    return post_bin_indexes;
}

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt,
        const dnnl::graph::logical_tensor::data_type &graph_dt,
        const char *atag) {
    using graph_layout = dnnl::graph::logical_tensor::layout_type;

    const auto &dnnl_test_engine = ::get_test_engine();
    const auto dims = lt.get_dims();
    const int ndims = static_cast<int>(dims.size());
    std::string valid_tag = atag ? normalize_tag(atag, ndims) : "abx";

    // NOTE: oneDNN Graph cannot get the concrete format from any-format logical
    //   tensor. Given that some tags in benchdnn is any by default, we should
    //   consider any to be default plain format for oneDNN Graph.
    if (valid_tag == tag::any) valid_tag = normalize_tag("abx", ndims);

    dnnl_dims_t dnnl_dims = {0};
    std::copy(dims.begin(), dims.end(), dnnl_dims);

    const auto ltype = lt.get_layout_type();
    if (graph_layout::undef != ltype) {
        return dnn_mem_t(ndims, dnnl_dims, convert_dt(graph_dt), valid_tag,
                dnnl_test_engine);
    } else {
        []() {
            SAFE(FAIL, CRIT);
            return 0;
        }();
        return dnn_mem_t();
    }
}

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt,
        const dims_t &dims,
        const dnnl::graph::logical_tensor::data_type &graph_dt,
        const char *atag) {
    dnnl::graph::logical_tensor new_lt(
            lt.get_id(), lt.get_data_type(), dims, lt.get_layout_type());
    return make_dnn_mem(new_lt, graph_dt, atag);
}

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt,
        const dims_t &dims, const std::string &atag) {
    dnnl::graph::logical_tensor new_lt(
            lt.get_id(), lt.get_data_type(), dims, lt.get_layout_type());
    return make_dnn_mem(new_lt, atag);
}

dnn_mem_t make_dnn_mem(
        const dnnl::graph::logical_tensor &lt, const std::string &tag) {
    return make_dnn_mem(lt, tag.empty() ? nullptr : tag.c_str());
}

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt, const char *tag) {
    return make_dnn_mem(lt, lt.get_data_type(), tag);
}

void compiled_partition_executor(dnnl::graph::compiled_partition &cp,
        dnnl::graph::stream &stream,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    if (get_test_engine_kind() == dnnl_cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        dnnl::graph::sycl_interop::execute(cp, stream, inputs,
                const_cast<std::vector<dnnl::graph::tensor> &>(outputs));
#else
        cp.execute(stream, inputs, outputs);
#endif
    } else {
#ifdef DNNL_GRAPH_GPU_SYCL
        dnnl::graph::sycl_interop::execute(cp, stream, inputs,
                const_cast<std::vector<dnnl::graph::tensor> &>(outputs));
#else
        assert(!"GPU only support DPCPP runtime now");
#endif
    }
}

int execute_and_wait(perf_function_t &exec_func,
        const dnnl::graph::engine &engine,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    dnnl::graph::stream stream {get_test_stream()};
    BENCHDNNEXT_SAFE(exec_func(stream, inputs, outputs), CRIT);
    BENCHDNNEXT_SAFE(stream.wait(), CRIT);

    return OK;
}

int execute_and_wait(dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs, res_t *res) {
    perf_function_t perf_func
            = std::bind(&compiled_partition_executor, cp, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3);
    const dnnl::graph::engine &engine = get_test_engine();

    int status = execute_and_wait(perf_func, engine, inputs, outputs);
    if (res) res->state = EXECUTED;

    return status;
};

inline int measure_perf_individual(timer::timer_t &t,
        dnnl::graph::stream &stream, perf_function_t &perf_func,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    t.reset();
    while (true) {
        BENCHDNNEXT_SAFE(perf_func(stream, inputs, outputs), WARN);
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

inline int measure_perf_aggregate(timer::timer_t &t,
        dnnl::graph::stream &stream, perf_function_t &perf_func,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    const int max_batch_times = 10000;

    // Warm-up run, this is not measured due to possibility the associated
    // kernel has not been built and skews the results.
    BENCHDNNEXT_SAFE(perf_func(stream, inputs, outputs), WARN);
    BENCHDNNEXT_SAFE(stream.wait(), WARN);

    int cur_batch_times
            = fix_times_per_prb ? fix_times_per_prb : min_times_per_prb;

    t.reset();
    reset_gpu_profiling();

    bool is_first_loop = true;
    while (true) {
        for (int i = 0; i < cur_batch_times; i++) {
            BENCHDNNEXT_SAFE(perf_func(stream, inputs, outputs), WARN);
        }
        BENCHDNNEXT_SAFE(stream.wait(), WARN);

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

int measure_perf(timer::timer_t &t, perf_function_t &perf_func,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    int ret = OK;
    if (is_bench_mode(PERF)) {
        dnnl::graph::stream stream = get_test_stream();

        // For non-DPCPP CPU: measure individual iterations.
        // For DPCPP CPU and GPU: measure iterations in batches to hide driver
        // overhead. DPCPP CPU follows the model of GPU, thus, handled similar.
        if (is_cpu() && !is_sycl_engine())
            ret = measure_perf_individual(
                    t, stream, perf_func, inputs, outputs);
        else
            ret = measure_perf_aggregate(t, stream, perf_func, inputs, outputs);
        return ret;
    } else {
        return ret;
    }
}

#ifdef DNNL_GRAPH_WITH_SYCL
// used as deallocator for shared_ptr
struct sycl_deletor {
    sycl_deletor() = delete;
    ::sycl::context ctx_;
    sycl_deletor(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};

std::shared_ptr<void> alloc_sycl_mem_for_tensor(
        dnnl::graph::tensor &ts, const dnnl::graph::logical_tensor &lt) {
    dnnl::engine test_eng {::get_test_engine()};
    size_t mem_size = lt.get_mem_size();
    std::shared_ptr<void> ts_buff(
            sycl::malloc_shared(mem_size,
                    dnnl::sycl_interop::get_device(test_eng),
                    dnnl::sycl_interop::get_context(test_eng)),
            sycl_deletor {dnnl::sycl_interop::get_context(test_eng)});
    ts.set_data_handle(ts_buff.get());
    return ts_buff;
}

void replace_memory_for_tensor(std::vector<dnnl::graph::tensor> &ts_in,
        std::vector<dnnl::graph::tensor> &ts_out,
        const std::vector<dnnl::graph::logical_tensor> &lt_in,
        const std::vector<dnnl::graph::logical_tensor> &lt_out,
        std::vector<std::shared_ptr<void>> &data_buffers,
        const std::vector<std::pair<size_t, size_t>> &v_pairs) {
    // allocate memory for input tensors
    for (size_t i = 0; i < lt_in.size(); i++) {
        auto ts_buff_ptr = alloc_sycl_mem_for_tensor(ts_in[i], lt_in[i]);
        data_buffers.emplace_back(ts_buff_ptr);
    }

    // allocate memory for output tensors, do not allocate for inplace tensors
    for (size_t i = 0; i < lt_out.size(); i++) {
        size_t lt_id = lt_out[i].get_id();
        // check whether output tensor is inplace with input tensor
        auto iter_inplace_pair = std::find_if(v_pairs.begin(), v_pairs.end(),
                // inplace is pair<size_t lt_id_in, size_t lt_id_out>
                [lt_id](const std::pair<size_t, size_t> &inplace) {
                    return inplace.second == lt_id;
                });
        // when find inplaced tensors, should share mem with input tensor
        if (iter_inplace_pair != v_pairs.end()) {
            for (size_t offset = 0; offset < lt_in.size(); offset++) {
                if (lt_in[offset].get_id() == iter_inplace_pair->first) {
                    ts_out[i].set_data_handle(data_buffers[offset].get());
                }
            }
        } else {
            auto ts_buff_ptr = alloc_sycl_mem_for_tensor(ts_out[i], lt_out[i]);
            data_buffers.emplace_back(ts_buff_ptr);
        }
    }
}
#endif

int measure_perf(timer::timer_t &t, dnnl::graph::compiled_partition &cp,
        std::vector<dnnl::graph::tensor> &inputs,
        std::vector<dnnl::graph::tensor> &outputs,
        const std::vector<dnnl::graph::logical_tensor> &lt_inputs,
        const std::vector<dnnl::graph::logical_tensor> &lt_outputs) {
    perf_function_t perf_func
            = std::bind(&compiled_partition_executor, cp, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3);

#ifdef DNNL_GRAPH_WITH_SYCL
    std::vector<std::shared_ptr<void>> data_buffers;
    const auto v_pairs = cp.get_inplace_ports();
    if (is_bench_mode(PERF) && is_sycl_engine()) {
        replace_memory_for_tensor(
                inputs, outputs, lt_inputs, lt_outputs, data_buffers, v_pairs);
    }
#endif

    return measure_perf(t, perf_func, inputs, outputs);
}

int measure_perf(timer::timer_t &t, dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs, res_t *res) {
    perf_function_t perf_func
            = std::bind(&compiled_partition_executor, cp, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3);
    int status = measure_perf(t, perf_func, inputs, outputs);
    if (res) res->state = EXECUTED;

    return status;
}

inline int measure_perf_aggregate(timer::timer_t &t,
        dnnl::graph::stream &stream, std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v) {
    const int max_batch_times = 10000;

    // Warm-up run, this is not measured due to possibility the associated
    // kernel has not been built and skews the results.
    auto sz = perf_func_v.size();
    for (size_t i = 0; i < sz; i++) {
        BENCHDNNEXT_SAFE(
                perf_func_v[i](stream, inputs_v[i], outputs_v[i]), WARN);
        BENCHDNNEXT_SAFE(stream.wait(), WARN);
    }

    int cur_batch_times
            = fix_times_per_prb ? fix_times_per_prb : min_times_per_prb;

    t.reset();
    reset_gpu_profiling();

    bool is_first_loop = true;
    while (true) {
        for (size_t i = 0; i < sz; i++) {
            for (int j = 0; j < cur_batch_times; j++) {
                BENCHDNNEXT_SAFE(
                        perf_func_v[i](stream, inputs_v[i], outputs_v[i]),
                        WARN);
            }
        }
        BENCHDNNEXT_SAFE(stream.wait(), WARN);

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

inline int measure_perf_individual(timer::timer_t &t,
        dnnl::graph::stream &stream, std::vector<perf_function_t> &perf_func_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v) {
    t.reset();
    while (true) {
        auto sz = perf_func_v.size();
        for (size_t i = 0; i < sz; i++) {
            BENCHDNNEXT_SAFE(
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
        dnnl::graph::stream stream = get_test_stream();
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
        perf_func_v.push_back(std::bind(&compiled_partition_executor, cp_v[i],
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3));
    }

    int status = measure_perf(t, perf_func_v, inputs_v, outputs_v);
    if (res) res->state = EXECUTED;

    return status;
}

int measure_partition_compl(timer::timer_t &ct,
        const dnnl::graph::partition &par,
        const std::vector<dnnl::graph::logical_tensor> &inputs,
        const std::vector<dnnl::graph::logical_tensor> &outputs,
        const dnnl::graph::engine &engine) {
    ct.reset();
    while (true) {
        par.compile(inputs, outputs, engine);
        ct.stamp();
        if (should_stop_ctime(ct)) break;
    }

    return OK;
}

fill_status_t append_graph_with_eltwise(
        const attr_t::post_ops_t::entry_t &eltw_entry) {
    auto eltw_op_kind = convert_alg_kind(eltw_entry.eltwise.alg);
    if (eltw_op_kind == dnnl::graph::op::kind::LastSymbol)
        return fill_status::UNSUPPORTED_OP;

    graph_t &graph = graph_t::get();

    const auto op_id = graph.generate_id_for(entry_kind::ELTWISE);
    const auto src_id = graph.generate_id_for(op_id, lt_kind::SRC, true);
    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

    graph.create_lt(dst_id, graph.get_lt(src_id));

    const auto attrs = convert_eltw_entry(eltw_op_kind, eltw_entry);
    if (eltw_entry.eltwise.alg == dnnl_eltwise_relu
            && attrs.count("alpha") != 0) {
        if (attrs.at("alpha") != 0.f) {
            // here it should be LeakyReLU
            eltw_op_kind = dnnl::graph::op::kind::LeakyReLU;
        }
    }

    dnnl::graph::op eltw_op(op_id, eltw_op_kind, graph.stringify_id(op_id));
    for (const auto &kv : attrs)
        eltw_op.set_attr(kv.first, kv.second);
    // special cases
    if (eltw_entry.kind == attr_t::post_ops_t::kind_t::SRELU)
        eltw_op.set_attr<int64_t>("beta", 1);

    graph.append(op_id, eltw_op, {src_id}, {dst_id});

    return fill_status_t::DONE;
}

std::pair<fill_status_t, size_t> append_graph_with_binary(
        const attr_t::post_ops_t::entry_t &bin_entry) {
    const auto bin_op_kind = convert_alg_kind(bin_entry.binary.alg);
    if (bin_op_kind == dnnl::graph::op::kind::LastSymbol)
        return std::make_pair(fill_status::UNSUPPORTED_OP, 0);

    graph_t &graph = graph_t::get();

    const auto op_id = graph.generate_id_for(entry_kind::BINARY);
    const auto src0_id = graph.generate_id_for(op_id, lt_kind::SRC, true);
    const auto src1_id = graph.generate_id_for(op_id, lt_kind::SRC1);
    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

    const auto bin_src1_dt
            = dequantize_dtype(convert_dt(bin_entry.binary.src1_dt));
    const auto cur_dst_lt = graph.get_lt(src0_id);
    const auto bin_src1_dims = convert_bin_policy(
            cur_dst_lt.get_dims(), bin_entry.binary.policy);

    graph.create_lt(src1_id, bin_src1_dt, bin_src1_dims, bin_entry.binary.tag);
    graph.create_lt(dst_id, cur_dst_lt);

    const std::string auto_broadcast
            = bin_src1_dims == cur_dst_lt.get_dims() ? "none" : "numpy";

    dnnl::graph::op bin_op(op_id, bin_op_kind, graph.stringify_id(op_id));
    bin_op.set_attr("auto_broadcast", auto_broadcast);

    graph.append(op_id, bin_op, {src0_id, src1_id}, {dst_id});

    return std::make_pair(fill_status_t::DONE, src1_id);
}

std::pair<fill_status_t, size_t> append_graph_with_sum(
        const attr_t::post_ops_t::entry_t &bin_entry) {
    graph_t &graph = graph_t::get();

    const auto op_id = graph.generate_id_for(entry_kind::SUM);
    const auto src0_id = graph.generate_id_for(op_id, lt_kind::SRC, true);
    const auto src1_id = graph.generate_id_for(op_id, lt_kind::SRC1);
    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

    const auto cur_dst_lt = graph.get_lt(src0_id);

    graph.create_lt(src1_id, cur_dst_lt);
    graph.create_lt(dst_id, cur_dst_lt);

    dnnl::graph::op sum_op(
            op_id, dnnl::graph::op::kind::Add, graph.stringify_id(op_id));
    sum_op.set_attr("auto_broadcast", std::string("none"));

    graph.append(op_id, sum_op, {src0_id, src1_id}, {dst_id});

    return std::make_pair(fill_status_t::DONE, src1_id);
}

fill_status_t append_graph_with_swish(
        const attr_t::post_ops_t::entry_t &swish_entry, size_t src1_id) {
    attr_t::post_ops_t::entry_t new_eltw_entry = swish_entry;
    // force eltwise handler to use Sigmoid
    new_eltw_entry.eltwise.alg = dnnl_eltwise_logistic;
    fill_status_t status = append_graph_with_eltwise(new_eltw_entry);
    BENCHDNNEXT_VERIFY(status);

    // handle special binary case
    graph_t &graph = graph_t::get();

    const auto op_id = graph.generate_id_for(entry_kind::BINARY);
    const auto src0_id = graph.generate_id_for(op_id, lt_kind::SRC, true);
    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

    graph.create_lt(dst_id, graph.get_lt(src0_id));

    dnnl::graph::op bin_op(
            op_id, dnnl::graph::op::kind::Multiply, graph.stringify_id(op_id));

    graph.append(op_id, bin_op, {src0_id, src1_id}, {dst_id});

    return fill_status::DONE;
}

quant_data_t sum_po_entry2quant_data(const attr_t::post_ops_t::entry_t &e,
        const std::string &tag,
        dnnl::graph::logical_tensor::data_type default_dt) {
    const auto sum_dt = convert_dt(e.sum.dt);
    const auto q_dt = sum_dt == dnnl::graph::logical_tensor::data_type::undef
            ? default_dt
            : sum_dt;
    return quant_data_t(
            q_dt, {e.sum.scale}, {e.sum.zero_point}, "per_tensor", 1, tag);
}

quant_data_t bin_po_entry2quant_data(const attr_t::post_ops_t::entry_t &e,
        const std::string &tag,
        dnnl::graph::logical_tensor::data_type default_dt) {
    const auto bin_dt = convert_dt(e.binary.src1_dt);
    const auto q_dt = bin_dt == dnnl::graph::logical_tensor::data_type::undef
            ? default_dt
            : bin_dt;
    return quant_data_t(q_dt, {1.f}, {0L}, "per_tensor", 1, tag);
}

bool is_dequantize_required_for(const attr_t::post_ops_t::entry_t &e) {
    return e.is_binary_kind()
            && is_low_precision({convert_dt(e.binary.src1_dt)});
}

fill_status_t insert_typecast_after(
        size_t src_id, const dt dst_dt, bool as_constant) {
    graph_t &graph = graph_t::get();

    const auto ptype = as_constant
            ? dnnl::graph::logical_tensor::property_type::constant
            : dnnl::graph::logical_tensor::property_type::undef;

    const auto op_id = graph.generate_id_for(entry_kind::TYPECAST);
    const auto src_lt = graph.get_lt(src_id);

    const auto dst_id = graph.generate_id_for(op_id, lt_kind::DST);

    // create lt for dst
    if (src_lt.get_layout_type()
            == dnnl::graph::logical_tensor::layout_type::strided) {
        graph.create_lt(
                dst_id, dst_dt, src_lt.get_dims(), src_lt.get_strides(), ptype);
    } else {
        graph.create_lt(dst_id, dst_dt, src_lt.get_dims(),
                src_lt.get_layout_id(), ptype);
    }

    dnnl::graph::op tc_op(
            op_id, dnnl::graph::op::kind::TypeCast, graph.stringify_id(op_id));

    graph.append(op_id, tc_op, {src_id}, {dst_id});

    return fill_status::DONE;
}

std::pair<fill_status_t, size_t> insert_typecast_before(
        size_t dst_id, bool as_constant) {
    graph_t &graph = graph_t::get();

    const auto op_id = graph.generate_id_for(entry_kind::TYPECAST);
    const auto src_id = graph.generate_id_for(op_id, lt_kind::SRC);

    const auto ptype = as_constant
            ? dnnl::graph::logical_tensor::property_type::constant
            : dnnl::graph::logical_tensor::property_type::undef;

    const auto dst_lt = graph.get_lt(dst_id);
    const auto src_dt = dnnl::graph::logical_tensor::data_type::f32;
    if (dst_lt.get_layout_type()
            == dnnl::graph::logical_tensor::layout_type::strided)
        graph.create_lt(
                src_id, src_dt, dst_lt.get_dims(), dst_lt.get_strides(), ptype);
    else
        graph.create_lt(src_id, src_dt, dst_lt.get_dims(),
                dst_lt.get_layout_id(), ptype);

    dnnl::graph::op tc_op(
            op_id, dnnl::graph::op::kind::TypeCast, graph.stringify_id(op_id));

    graph.append(op_id, tc_op, {src_id}, {dst_id}, false);

    return std::make_pair(fill_status::DONE, src_id);
}

fill_status_t insert_dequant_before(
        size_t dst_id, const quant_data_t &qdata, bool as_constant) {
    return handle_quant_dequant(
            dst_id, qdata, as_constant, dnnl::graph::op::kind::Dequantize);
}

fill_status_t insert_quant_after(size_t src_id, const quant_data_t &qdata) {
    return handle_quant_dequant(
            src_id, qdata, false, dnnl::graph::op::kind::Quantize);
}

fill_status_t handle_quant_dequant(size_t existing_lt_id,
        const quant_data_t &qdata, bool as_constant,
        dnnl::graph::op::kind op_kind) {
    graph_t &graph = graph_t::get();

    const bool is_quant = op_kind == dnnl::graph::op::kind::Quantize;
    const bool connect_to_previous_block = !is_quant && graph.has_blocks();
    const auto op_id = graph.generate_id_for(
            is_quant ? entry_kind::QUANTIZE : entry_kind::DEQUANTIZE);
    const auto new_lt_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(
                    op_id, is_quant ? lt_kind::DST : lt_kind::SRC);

    const auto ptype = as_constant
            ? dnnl::graph::logical_tensor::property_type::constant
            : dnnl::graph::logical_tensor::property_type::undef;

    const auto existing_lt = graph.get_lt(existing_lt_id);
    if (!qdata.tag.empty())
        graph.create_lt(
                new_lt_id, qdata.dt, existing_lt.get_dims(), qdata.tag, ptype);
    else if (!qdata.strides.empty())
        graph.create_lt(new_lt_id, qdata.dt, existing_lt.get_dims(),
                qdata.strides, ptype);
    else
        return fill_status::UNKNOWN_ERROR;

    dnnl::graph::op q_op(op_id, op_kind, graph.stringify_id(op_id));
    q_op.set_attr("scales", qdata.scales)
            .set_attr("zps", qdata.zps)
            .set_attr("qtype", qdata.qtype)
            .set_attr("axis", qdata.axis);

    if (is_quant)
        graph.append(op_id, q_op, {existing_lt_id}, {new_lt_id});
    else
        graph.append(op_id, q_op, {new_lt_id}, {existing_lt_id}, false);

    return fill_status::DONE;
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

void *sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    void *ptr = is_bench_mode(CORR) || is_cpu()
            ? dnnl::graph::testing::sycl_malloc_wrapper(
                    size, alignment, dev, ctx)
            : s_mm_mgr.sycl_alloc_mm(size, alignment, dev, ctx);

    return ptr;
}

// perf mode, mem will be finally released in s_mm_mgr ~shared_ptr when
// test finished.
void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    if (is_bench_mode(CORR) || is_cpu()) {
        dnnl::graph::testing::sycl_free_wrapper(ptr, device, context, event);
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

dnnl::graph::stream &get_graph_stream() {
    static dnnl::engine test_eng {::get_test_engine()};
    static ::sycl::device dev {dnnl::sycl_interop::get_device(test_eng)};
    static ::sycl::context ctx {dnnl::sycl_interop::get_context(test_eng)};

    static ::sycl::queue q {ctx, dev, ::sycl::property::queue::in_order {}};

    static dnnl::graph::stream strm {
            dnnl::graph::sycl_interop::make_stream(get_graph_engine(), q)};
    return strm;
}
#endif // DNNL_GRAPH_WITH_SYCL

bool is_sycl_engine() {
#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_SYCL
    if (is_cpu()) return true;
#endif

#if DNNL_GRAPH_GPU_RUNTIME == DNNL_GRAPH_RUNTIME_SYCL
    if (!is_cpu()) return true;
#endif
    return false;
}

// Engine used to run oneDNN fusion patterns for testing.
const dnnl::graph::engine &get_test_engine() {
    using engine = dnnl::graph::engine;
    if (get_test_engine_kind() == dnnl_cpu) {
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

const dnnl::graph::stream &get_test_stream() {
    using stream = dnnl::graph::stream;
    if (get_test_engine_kind() == dnnl_cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        static const stream strm(get_graph_stream());
#elif DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
        static const stream strm {dnnl::graph::threadpool_interop::make_stream(
                get_test_engine(), dnnl::graph::testing::get_threadpool())};
#else
        static const stream strm(
                const_cast<dnnl::graph::engine &>(get_test_engine()));
#endif
        return strm;
    } else {
#ifdef DNNL_GRAPH_GPU_SYCL
        static const stream strm(
                const_cast<dnnl::graph::stream &>(get_graph_stream()));
#else
        assert(!"GPU only support DPCPP runtime now");
        static const stream strm(
                const_cast<dnnl::graph::engine &>(get_test_engine()));
#endif
        return strm;
    }
}

std::string entry_kind2str(entry_kind_t ekind) {
    switch (ekind) {
        case entry_kind::NONE: return "NONE";
        case entry_kind::BINARY: return "BINARY";
        case entry_kind::BNORM: return "BNORM";
        case entry_kind::CONCAT: return "CONCAT";
        case entry_kind::CONV: return "CONV";
        case entry_kind::DECONV: return "DECONV";
        case entry_kind::ELTWISE: return "ELTWISE";
        case entry_kind::LNORM: return "LNORM";
        case entry_kind::MATMUL: return "MATMUL";
        case entry_kind::POOL: return "POOL";
        case entry_kind::PRELU: return "PRELU";
        case entry_kind::REDUCTION: return "REDUCTION";
        case entry_kind::REORDER: return "REORDER";
        case entry_kind::RESAMPLING: return "RESAMPLING";
        case entry_kind::SOFTMAX: return "SOFTMAX";
        case entry_kind::SUM: return "SUM";
        case entry_kind::QUANTIZE: return "QUANTIZE";
        case entry_kind::DEQUANTIZE: return "DEQUANTIZE";
        case entry_kind::TYPECAST: return "TYPECAST";
        case entry_kind::RESHAPE: return "RESHAPE";
        case entry_kind::TRANSPOSE: return "TRANSPOSE";
        default: return "";
    }
}

std::string lt_kind2str(lt_kind_t lkind) {
    switch (lkind) {
        case lt_kind::NONE: return "NONE";
        case lt_kind::SRC: return "SRC";
        case lt_kind::SRC1: return "SRC1";
        case lt_kind::WEI: return "WEI";
        case lt_kind::BIA: return "BIA";
        case lt_kind::DST: return "DST";
        case lt_kind::MEAN: return "MEAN";
        case lt_kind::RUN_MEAN: return "RUN_MEAN";
        case lt_kind::BATCH_MEAN: return "BATCH_MEAN";
        case lt_kind::VAR: return "VAR";
        case lt_kind::RUN_VAR: return "RUN_VAR";
        case lt_kind::BATCH_VAR: return "BATCH_VAR";
        case lt_kind::SC: return "SC";
        case lt_kind::SH: return "SH";
        case lt_kind::DIFF_SRC: return "DIFF_SRC";
        case lt_kind::DIFF_WEI: return "DIFF_WEI";
        case lt_kind::DIFF_DST: return "DIFF_DST";
        case lt_kind::DIFF_SC: return "DIFF_SC";
        case lt_kind::DIFF_SH: return "DIFF_SH";
        case lt_kind::SRC_I: return "SRC_I";
        default:
            const auto raw_lkind = static_cast<size_t>(lkind);
            if (raw_lkind > static_cast<size_t>(lt_kind::SRC_I)) return "SRC_I";
            return "";
    }
}

void graph_t::create_lt(size_t aid, dt dtype, const dims_t &adims,
        const std::string &atag,
        dnnl::graph::logical_tensor::property_type ptype) {
    size_t ndims = adims.size();
    const std::string dnnl_fmt_tag_str
            = normalize_tag(atag, static_cast<int>(ndims));
    const dnnl_format_tag_t fmt_tag = dnnl_fmt_str2tag(dnnl_fmt_tag_str);
    if (fmt_tag == dnnl_format_tag_undef) {
        []() {
            SAFE(FAIL, CRIT);
            return 0;
        }();
        return;
    }

    if (fmt_tag == dnnl_format_tag_any) {
        create_lt(aid, dtype, adims, lt::strided, ptype);
        return;
    }

#ifndef DNNL_GRAPH_LAYOUT_DEBUG
    if (!is_plain(fmt_tag)) {
        []() {
            BENCHDNN_PRINT(0, "error: %s\n",
                    "to use dnnl opaque blocked formats, please build the "
                    "library with \"DNNL_GRAPH_LAYOUT_DEBUG=ON\"");
            SAFE(FAIL, CRIT);
            return 0;
        }();
    }
#endif // DNNL_GRAPH_LAYOUT_DEBUG

    const std::string ou_fmt_str = get_ou_format(dnnl_fmt_tag_str);

    static_assert(DNNL_GRAPH_MAX_NDIMS == DNNL_MAX_NDIMS,
            "Maximum number of dimensions of primitive and graph is not "
            "the same.");

    if (is_plain(fmt_tag)) {
        dims_t strides(ndims, 0);
        dim_t acc = 1;
        // if the layout can be described with strides, let's calculate
        // the strides according to the given tag.
        for (int d = static_cast<int>(ndims) - 1; d >= 0; --d) {
            const size_t coord = static_cast<size_t>(ou_fmt_str[d] - 'a');
            strides[coord] = acc;
            acc *= adims[coord];
        }
        create_lt(aid, dtype, adims, strides, ptype);
    } else {
        const size_t dnnl_layout_id
                = encode_dnnl_layout(static_cast<size_t>(fmt_tag));
        dnnl::graph::logical_tensor t(aid, dtype, adims, dnnl_layout_id, ptype);
        lts_.emplace(aid, t);
    }
}

} // namespace benchdnnext
