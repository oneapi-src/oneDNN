/*******************************************************************************
* Copyright 2025 Intel Corporation
* Copyright 2026 SpacemiT Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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
#include <math.h>

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/rvv_layer_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

rvv_layer_normalization_fwd_t::rvv_layer_normalization_fwd_t(const pd_t *apd)
    : primitive_t(apd) {
    if (pd()->src_md()->data_type == data_type::f16) {
        // The widening sum is profitable on the tested VLEN <= 256 systems
        // only when the normalized axis contains at least 129 elements.
        constexpr dim_t widening_sum_min_len = 129;
        const bool use_widening_sum = get_platform_vlen() <= 256
                && pd()->norm_axis() >= widening_sum_min_len;
        const bool weights_f16
                = pd()->weights_md()->data_type == data_type::f16;
        f16_fused_kernel_.reset(
                new jit_rvv_layernorm_f16_fused_kernel_t(pd()->use_scale(),
                        pd()->use_shift(), weights_f16, use_widening_sum));
    } else {
        fused_kernel_.reset(new jit_rvv_layernorm_fused_kernel_t(
                pd()->use_scale(), pd()->use_shift()));
        data_kernel_.reset(new jit_rvv_layernorm_data_kernel_t(
                pd()->use_scale(), pd()->use_shift()));
    }
}

status_t rvv_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    float *mean_ptr = nullptr;
    float *variance_ptr = nullptr;
    auto scratchpad = ctx.get_scratchpad_grantor();
    if (pd()->use_tmp_stats()) {
        mean_ptr = scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance_ptr = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean_ptr = pd()->stats_are_src()
                ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
                : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
        variance_ptr = pd()->stats_are_src()
                ? const_cast<float *>(
                          CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
                : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);
    }

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();
    const float eps = pd()->desc()->layer_norm_epsilon;
    const bool save_stats = pd()->is_training();
    const bool calculate_stats = !pd()->stats_are_src();

    if (pd()->src_md()->data_type == data_type::f16) {
        const auto *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        auto *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        const void *scale = pd()->use_scale()
                ? CTX_IN_MEM(const void *, DNNL_ARG_SCALE)
                : nullptr;
        const void *shift = pd()->use_shift()
                ? CTX_IN_MEM(const void *, DNNL_ARG_SHIFT)
                : nullptr;
        const size_t dt_size = types::data_type_size(data_type::f16);

        parallel_nd(N, [&](dim_t n) {
            jit_rvv_layernorm_f16_fused_kernel_t::call_params_t p;
            p.src = reinterpret_cast<const char *>(src) + n * C * dt_size;
            p.dst = reinterpret_cast<char *>(dst) + n * C * dt_size;
            p.scale = scale;
            p.shift = shift;
            p.len = C;
            p.eps = eps;
            p.mean = save_stats ? mean_ptr + n : nullptr;
            p.variance = save_stats ? variance_ptr + n : nullptr;
            (*f16_fused_kernel_)(&p);
        });
        return status::success;
    }

    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
    const float *scale = pd()->use_scale()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SCALE)
            : nullptr;
    const float *shift = pd()->use_shift()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SHIFT)
            : nullptr;

    if (!calculate_stats) {
        parallel_nd(N, [&](dim_t n) {
            const float *src_row = src + n * C;
            float *dst_row = dst + n * C;
            const float mean_val = mean_ptr[n];
            const float inv_std = 1.f / sqrtf(variance_ptr[n] + eps);

            jit_rvv_layernorm_data_kernel_t::call_params_t data_p;
            data_p.src = src_row;
            data_p.dst = dst_row;
            data_p.scale = scale;
            data_p.shift = shift;
            data_p.len = C;
            data_p.mean = mean_val;
            data_p.inv_std = inv_std;
            (*data_kernel_)(&data_p);
        });
        return status::success;
    }

    parallel_nd(N, [&](dim_t n) {
        const float *src_row = src + n * C;
        float *dst_row = dst + n * C;
        jit_rvv_layernorm_fused_kernel_t::call_params_t fused_p;
        fused_p.src = src_row;
        fused_p.dst = dst_row;
        fused_p.scale = scale;
        fused_p.shift = shift;
        fused_p.len = C;
        fused_p.eps = eps;
        fused_p.mean = save_stats ? mean_ptr + n : nullptr;
        fused_p.variance = save_stats ? variance_ptr + n : nullptr;
        (*fused_kernel_)(&fused_p);
    });
    return status::success;
}

namespace {
template <bool scale, bool stats, bool weights>
const jit_rvv_layernorm_bwd_kernel_t &get_kernel() {
    static const jit_rvv_layernorm_bwd_kernel_t kernel(scale, stats, weights);
    return kernel;
}
template <bool weights>
const jit_rvv_layernorm_bwd_kernel_t &get_kernel(bool scale, bool stats) {
    if (scale)
        return stats ? get_kernel<true, true, weights>()
                     : get_kernel<true, false, weights>();
    return stats ? get_kernel<false, true, weights>()
                 : get_kernel<false, false, weights>();
}
} // namespace

status_t jit_uni_layer_normalization_bwd_t::pd_t::init(const engine_t *engine) {
    UNUSED(engine);
    using namespace data_type;
    VDISPATCH_LNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_LNORM(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
#if !(defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1)
    VDISPATCH_LNORM(false, VERBOSE_UNSUPPORTED_ISA);
#endif
    VDISPATCH_LNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src");
    VDISPATCH_LNORM(!skip_mean(), VERBOSE_UNSUPPORTED_FEATURE, "RMSNorm");
    VDISPATCH_LNORM(src_md()->data_type == f32
                    && diff_dst_md()->data_type == f32
                    && diff_src_md()->data_type == f32
                    && stat_md()->data_type == f32,
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_LNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    for (const auto *md : {src_md(), diff_dst_md(), diff_src_md()}) {
        const memory_desc_wrapper d(md);
        VDISPATCH_LNORM(d.is_plain() && d.is_dense()
                        && d.blocking_desc().strides[ndims() - 1] == 1,
                VERBOSE_UNSUPPORTED_TAG);
    }
    const memory_desc_wrapper stats_d(stat_md());
    VDISPATCH_LNORM(stats_d.is_blocking_desc(), VERBOSE_UNSUPPORTED_TAG);
    diff_weights_ = desc()->prop_kind == prop_kind::backward
            && (use_scale() || use_shift());
    nthr_ = static_cast<int>(
            nstl::min<dim_t>(dnnl_get_max_threads(), across_axis()));
    if (diff_weights_) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.template book<float>(
                memory_tracking::names::key_lnorm_reduction,
                2 * norm_axis() * nthr_);
    }
    return status::success;
}

jit_uni_layer_normalization_bwd_t::jit_uni_layer_normalization_bwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

status_t jit_uni_layer_normalization_bwd_t::init(engine_t *engine) {
    UNUSED(engine);
    const auto *kernel = pd()->diff_weights_
            ? &get_kernel<true>(pd()->use_scale(), !pd()->stats_are_src())
            : &get_kernel<false>(pd()->use_scale(), !pd()->stats_are_src());
    CHECK(kernel->status());
    kernel_ = kernel;
    return status::success;
}

status_t jit_uni_layer_normalization_bwd_t::execute(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    const auto dd = CTX_IN_MEM(const float *, DNNL_ARG_DIFF_DST);
    const auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    const auto mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
    const auto var = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    status_t status = status::success;
    const auto ds = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DIFF_SRC, status);
    CHECK(status);
    const auto dg = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DIFF_SCALE, status);
    CHECK(status);
    const auto db = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DIFF_SHIFT, status);
    CHECK(status);
    const memory_desc_wrapper src_d(pd()->src_md()), dd_d(pd()->diff_dst_md()),
            ds_d(pd()->diff_src_md()), stat_d(pd()->stat_md());
    const dim_t C = pd()->norm_axis(), N = pd()->across_axis();
    const int max_nthr = pd()->nthr_;
    float *reduce = pd()->diff_weights_
            ? ctx.get_scratchpad_grantor().template get<float>(
                      memory_tracking::names::key_lnorm_reduction)
            : nullptr;
    // Fixed per-thread regions remain valid even if the runtime supplies fewer
    // threads than the primitive's scratchpad capacity.
    if (reduce) std::fill(reduce, reduce + 2 * C * max_nthr, 0.f);
    parallel(max_nthr, [&](int ithr, int nthr) {
        dim_t begin, end;
        balance211(N, nthr, ithr, begin, end);
        for (dim_t n = begin; n < end; ++n) {
            const dim_t stat_off = stat_d.off_l(n);
            const float inv = 1.f
                    / sqrtf(var[stat_off] + pd()->desc()->layer_norm_epsilon);
            const jit_rvv_layernorm_bwd_kernel_t::call_params_t p {
                    src + src_d.off_l(n * C), dd + dd_d.off_l(n * C),
                    ds + ds_d.off_l(n * C), scale,
                    reduce ? reduce + 2 * C * ithr : nullptr,
                    reduce ? reduce + 2 * C * ithr + C : nullptr, C,
                    mean[stat_off], inv, static_cast<float>(C)};
            (*kernel_)(&p);
        }
    });
    if (reduce) {
        parallel_nd(C, [&](dim_t c) {
            float sum_g = 0.f, sum_b = 0.f;
            for (int t = 0; t < max_nthr; ++t) {
                sum_g += reduce[2 * C * t + c];
                sum_b += reduce[2 * C * t + C + c];
            }
            if (dg) dg[c] = sum_g;
            if (db) db[c] = sum_b;
        });
    }
    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
