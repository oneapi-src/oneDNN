/*******************************************************************************
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
#include <cmath>
#include <cstddef>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/rvv_layer_normalization_bwd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

struct jit_rvv_layernorm_bwd_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        const float *diff_dst;
        float *diff_src;
        const float *scale;
        float *diff_scale;
        float *diff_shift;
        dim_t len;
        float mean;
        float inv_std;
        float channels;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_layernorm_bwd_kernel_t)
    jit_rvv_layernorm_bwd_kernel_t(bool scale, bool stats, bool weights)
        : jit_generator_t("jit_rvv_layernorm_bwd_kernel")
        , scale_(scale)
        , stats_(stats)
        , weights_(weights) {
        create_kernel();
    }
    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

private:
    bool scale_, stats_, weights_;
    void generate() override {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
        const Reg src = a1, dd = a2, ds = a3, gamma = a4;
        const Reg dg = a5, db = a6, len = a7;
        const Reg vl = t0, bytes = t1;
        const FReg mean = ft0, inv = ft1, channels = ft2;
        const FReg sum = ft3, sum_x = ft4, zero = ft5;
        const VReg x(0), dy(1), u(2), tmp(3), acc(4), g(5);
        const VReg v_sum(8), v_sum_x(9), reduction(10);
#define BWD_OFF(field) static_cast<int32_t>(offsetof(call_params_t, field))
        auto load_row = [&]() {
            ld(src, a0, BWD_OFF(src));
            ld(dd, a0, BWD_OFF(diff_dst));
            ld(len, a0, BWD_OFF(len));
            if (scale_) ld(gamma, a0, BWD_OFF(scale));
        };
        flw(mean, a0, BWD_OFF(mean));
        flw(inv, a0, BWD_OFF(inv_std));
        flw(channels, a0, BWD_OFF(channels));
        fmv_w_x(zero, x0);
        if (stats_ || weights_) {
            load_row();
            if (weights_) {
                ld(dg, a0, BWD_OFF(diff_scale));
                ld(db, a0, BWD_OFF(diff_shift));
            }
            // Initialize all lanes, preserving inactive accumulator lanes in
            // the final partial vector before reducing at VLMAX.
            if (stats_) {
                vsetvli(vl, x0, SEW::e32, LMUL::m1, VTA::tu, VMA::ma);
                vmv_v_i(v_sum, 0);
                vmv_v_i(v_sum_x, 0);
            }
            Label reduce_loop;
            L(reduce_loop);
            vsetvli(vl, len, SEW::e32, LMUL::m1, VTA::tu, VMA::ma);
            vle32_v(x, src);
            vle32_v(dy, dd);
            vfsub_vf(x, x, mean);
            if (weights_) {
                vfmul_vv(tmp, x, dy);
                vfmul_vf(tmp, tmp, inv);
                vle32_v(acc, dg);
                vfadd_vv(acc, acc, tmp);
                vse32_v(acc, dg);
                vle32_v(acc, db);
                vfadd_vv(acc, acc, dy);
                vse32_v(acc, db);
            }
            if (stats_) {
                if (scale_) {
                    vle32_v(g, gamma);
                    vfmul_vv(u, dy, g);
                } else {
                    vmv_v_v(u, dy);
                }
                vfadd_vv(v_sum, v_sum, u);
                vfmul_vv(tmp, u, x);
                vfadd_vv(v_sum_x, v_sum_x, tmp);
            }
            slli(bytes, vl, 2);
            add(src, src, bytes);
            add(dd, dd, bytes);
            if (scale_ && stats_) add(gamma, gamma, bytes);
            if (weights_) {
                add(dg, dg, bytes);
                add(db, db, bytes);
            }
            sub(len, len, vl);
            bnez(len, reduce_loop);
            if (stats_) {
                vsetvli(vl, x0, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
                vfmv_s_f(reduction, zero);
                vfredusum_vs(reduction, v_sum, reduction);
                vfmv_f_s(sum, reduction);
                vfmv_s_f(reduction, zero);
                vfredusum_vs(reduction, v_sum_x, reduction);
                vfmv_f_s(sum_x, reduction);
                fdiv_s(sum, sum, channels);
                fmul_s(sum_x, sum_x, inv);
            }
        }
        load_row();
        ld(ds, a0, BWD_OFF(diff_src));
        Label apply_loop;
        L(apply_loop);
        vsetvli(vl, len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        vle32_v(dy, dd);
        if (scale_) {
            vle32_v(g, gamma);
            vfmul_vv(dy, dy, g);
        }
        if (stats_) {
            vle32_v(x, src);
            vfsub_vf(x, x, mean);
            vfmul_vf(x, x, sum_x);
            vfmul_vf(x, x, inv);
            vfdiv_vf(x, x, channels);
            vfsub_vf(dy, dy, sum);
            vfsub_vv(dy, dy, x);
        }
        vfmul_vf(dy, dy, inv);
        vse32_v(dy, ds);
        slli(bytes, vl, 2);
        add(src, src, bytes);
        add(dd, dd, bytes);
        add(ds, ds, bytes);
        if (scale_) add(gamma, gamma, bytes);
        sub(len, len, vl);
        bnez(len, apply_loop);
#undef BWD_OFF
#endif
        ret();
    }
};

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
    : primitive_t(apd) {
    kernel_ = pd()->diff_weights_
            ? &get_kernel<true>(pd()->use_scale(), !pd()->stats_are_src())
            : &get_kernel<false>(pd()->use_scale(), !pd()->stats_are_src());
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
