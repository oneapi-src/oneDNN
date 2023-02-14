/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2021-2023 FUJITSU LIMITED
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

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/jit_uni_batch_normalization_s8.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

using data_t = int8_t;

struct jit_uni_bnorm_s8_call_params_t {
    // keep int sizes at 8 bytes -- jit code expects this
    size_t channel_offt_count, spat_offt_count;
    float eps;
    const float *scale, *shift, *mean, *var;
    const data_t *src, *dst;
};

template <cpu_isa_t isa>
struct jit_bnorm_base_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_s8_t)

    const int vlen = cpu_isa_traits<isa>::vlen;

    const batch_normalization_pd_t *pd_;

    XReg reg_param = abi_param1;

    XReg reg_scale = x3;
    XReg reg_shift = x4;
    XReg reg_mean = x5;

    XReg reg_channel_offt_count = x8;
    XReg reg_spat_offt = x9;
    XReg reg_spat_offt_count = x10;
    XReg reg_tmp = x11;
    XReg reg_src = x12;
    XReg reg_dst = x13;
    XReg reg_var = x14;
    XReg reg_channel_offt_1byte = x15;
    XReg reg_channel_offt_4byte = x1;
    WReg reg_relu_alpha = WReg(abi_not_param1.getIdx());

    PReg kstore_mask = p1;

    ZReg vzero = z29;
    ZReg vone = z30;
    ZReg veps = z31;
    ZReg vmm_aux = z28;
    ZReg z_tmp0 = z25;

    size_t simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);
    size_t c_in_xmm_ = 16;
    size_t chan_data_offt_;
    size_t num_c_blocks_;
    size_t c_tail_;
    bool with_relu_;
    bool has_alpha_value_;

    PReg p_lsb_128 = p6;
    PReg p_tmp0 = p5;

    XReg xreg_addr(const XReg &base, const XReg &off = XReg(DUMMY_IDX),
            const int disp = 0) {
        XReg x_addr = base;
        uint32_t offIdx = off.getIdx();

        if (offIdx <= SP_IDX) {
            add(X_DEFAULT_ADDR, base, off);
            x_addr = X_DEFAULT_ADDR;
        }
        if (disp) {
            add_imm(X_DEFAULT_ADDR, x_addr, disp, X_TMP_0);
            x_addr = X_DEFAULT_ADDR;
        }

        return x_addr;
    }

    void uni_fmax(const ZReg &dst, const ZReg &src, const ZReg &src2) {
        mov(z_tmp0.d, src2.d);
        fmaxnm(z_tmp0.s, P_ALL_ONE, src.s);
        fmax(z_tmp0.s, P_ALL_ONE, src.s);
        mov(dst.d, z_tmp0.d);
    }

    void compute_predefined_variables() {
        chan_data_offt_ = pd_->C() * sizeof(float);
        num_c_blocks_ = pd_->C() / c_in_xmm_;
        c_tail_ = pd_->C() % c_in_xmm_;
        with_relu_ = (pd_->with_relu_post_op(false) || pd_->fuse_norm_relu())
                && pd_->is_fwd();
        has_alpha_value_ = with_relu_ && pd_->with_relu_post_op(false)
                && pd_->alpha() != 0;
    }

    void load_common_params() {
        fmov(vone.s, 1.0);

#define PARAM_OFF(x) (int32_t)(offsetof(jit_uni_bnorm_s8_call_params_t, x))
        ld1rw(veps.s, P_ALL_ONE / T_z, ptr(reg_param, PARAM_OFF(eps)));
        uni_eor(vzero, vzero, vzero);

        ldr(reg_channel_offt_count,
                ptr(reg_param, PARAM_OFF(channel_offt_count)));
        ldr(reg_spat_offt_count, ptr(reg_param, PARAM_OFF(spat_offt_count)));
        ldr(reg_src, ptr(reg_param, PARAM_OFF(src)));
        ldr(reg_dst, ptr(reg_param, PARAM_OFF(dst)));
        ldr(reg_mean, ptr(reg_param, PARAM_OFF(mean)));
        ldr(reg_scale, ptr(reg_param, PARAM_OFF(scale)));
        ldr(reg_shift, ptr(reg_param, PARAM_OFF(shift)));
        ldr(reg_var, ptr(reg_param, PARAM_OFF(var)));
#undef PARAM_OFF

        if (has_alpha_value_) {
            mov_imm(reg_relu_alpha, float2int(pd_->alpha()));
        }
    }

    XReg mean_ptr(size_t offt = 0) {
        return xreg_addr(reg_mean, reg_channel_offt_4byte, offt);
    }

    XReg var_ptr(size_t offt = 0) {
        return xreg_addr(reg_var, reg_channel_offt_4byte, offt);
    }

    XReg scale_ptr(size_t offt = 0) {
        return xreg_addr(reg_scale, reg_channel_offt_4byte, offt);
    }

    XReg shift_ptr(size_t offt = 0) {
        return xreg_addr(reg_shift, reg_channel_offt_4byte, offt);
    }

    XReg src_ptr(size_t offt = 0) {
        return xreg_addr(reg_src, reg_spat_offt, offt);
    }

    XReg dst_ptr(size_t offt = 0) {
        return xreg_addr(reg_dst, reg_spat_offt, offt);
    }

    virtual void prepare_tail_mask() {}
    virtual void load_mean_and_var(const ZReg &vmean, const ZReg &vsqrtvar,
            size_t offt, bool need_tail) {}
    virtual void load_scale(const ZReg &vscale, size_t offt, bool need_tail) {}
    virtual void load_shift(const ZReg &vshift, size_t offt, bool need_tail) {}
    virtual void compute_dst(bool need_tail) {}

    // Precomputes vscale and vshift for following
    // `vdst = vscale * vsrc + vshift`
    void compute_vscaleshift(const ZReg &vscale, const ZReg &vshift,
            const ZReg &vmean, const ZReg &vsqrtvar, size_t offt,
            bool need_tail) {
        load_mean_and_var(vmean, vsqrtvar, offt, need_tail);
        fadd(vsqrtvar.s, vsqrtvar.s, veps.s);
        fsqrt(vsqrtvar.s, P_ALL_ONE / T_m, vsqrtvar.s);

        if (pd_->use_scale() && pd_->use_shift()) {
            load_scale(vscale, offt, need_tail);
            uni_fdiv(vscale.s, vscale.s, vsqrtvar.s, z_tmp0.s, P_ALL_ONE);
            load_shift(vshift, offt, need_tail);
            fmls(vshift.s, P_ALL_ONE / T_m, vmean.s, vscale.s);
        } else if (pd_->use_scale()) {
            load_scale(vscale, offt, need_tail);
            uni_fdiv(vscale.s, vscale.s, vsqrtvar.s, z_tmp0.s, P_ALL_ONE);
            fmul(vmean.s, vmean.s, vscale.s);
            uni_fsub(vshift.s, vzero.s, vmean.s);
        } else if (pd_->use_shift()) {
            uni_fdiv(vscale.s, vone.s, vsqrtvar.s, z_tmp0.s, P_ALL_ONE);
            load_shift(vshift, offt, need_tail);
            fmls(vshift.s, P_ALL_ONE / T_m, vmean.s, vscale.s);
        } else {
            uni_fdiv(vscale.s, vone.s, vsqrtvar.s, z_tmp0.s, P_ALL_ONE);
            fmul(vmean.s, vmean.s, vscale.s);
            uni_fsub(vshift.s, vzero.s, vmean.s);
        }
    }

    void forward() {
        eor(reg_channel_offt_1byte, reg_channel_offt_1byte,
                reg_channel_offt_1byte);
        eor(reg_channel_offt_4byte, reg_channel_offt_4byte,
                reg_channel_offt_4byte);
        mov(WReg(reg_tmp.getIdx()), sizeof(data_t) * c_in_xmm_);

        if (num_c_blocks_) compute_dst(false);
        if (c_tail_) compute_dst(true);
    }

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void generate() override {
        preamble();

        // Only SVE512 is supported.
        assert(isa == sve_512);
        if (isa == sve_512) { ptrue(p_lsb_128.b, VL16); }

        compute_predefined_variables();
        load_common_params();
        prepare_tail_mask();
        forward();
        postamble();
    }

    jit_bnorm_base_t(const batch_normalization_pd_t *pd) : pd_(pd) {}
};

template <cpu_isa_t isa>
struct jit_bnorm_s8_t;

template <>
struct jit_bnorm_s8_t<sve_512> : public jit_bnorm_base_t<sve_512> {
    PReg tail_opmask = PReg(1); // f32 mask for channel math

    void prepare_tail_mask() override {
        if (!c_tail_) return;

        set_preg(tail_opmask.s, c_tail_, X_TMP_0, X_TMP_1);
    }

    void load_mean_and_var(const ZReg &vmean, const ZReg &vsqrtvar, size_t offt,
            bool need_tail) override {
        if (need_tail) {
            ld1w(vmean.s, tail_opmask / T_z, ptr(mean_ptr(offt)));
            ld1w(vsqrtvar.s, tail_opmask / T_z, ptr(var_ptr(offt)));
        } else {
            ldr(vmean, ptr(mean_ptr(offt)));
            ldr(vsqrtvar, ptr(var_ptr(offt)));
        }
    }

    void load_scale(const ZReg &vscale, size_t offt, bool need_tail) override {
        if (need_tail) {
            ld1w(vscale.s, tail_opmask / T_z, ptr(scale_ptr(offt)));
        } else {
            ldr(vscale, ptr(scale_ptr(offt)));
        }
    }

    void load_shift(const ZReg &vshift, size_t offt, bool need_tail) override {
        if (need_tail) {
            ld1w(vshift.s, tail_opmask / T_z, ptr(shift_ptr(offt)));
        } else {
            ldr(vshift, ptr(shift_ptr(offt)));
        }
    }

    void process_relu_alpha(ZReg vmm_dst) {
        dup(vmm_aux.s, reg_relu_alpha);
        fcmge(kstore_mask.s, P_ALL_ONE / T_z, vmm_dst.s, 0.f);
        fmul(vmm_aux.s, kstore_mask / T_m, vmm_dst.s);
        mov(vmm_dst.s, kstore_mask / T_m, vmm_aux.s);
    }

    void compute_dst(bool need_tail) override {
        Label c_loop;
        L(c_loop);
        {
            ZReg v = ZReg(0);
            ZReg vscale = ZReg(1);
            ZReg vshift = ZReg(2);
            ZReg vmean = ZReg(3);
            ZReg vsqrtvar = ZReg(4);

            // compute single vscale and vshift vectors...
            compute_vscaleshift(vscale, vshift, vmean, vsqrtvar, 0, need_tail);

            // ... then process all spatial loop with it and move to the
            // next channel chunk
            mov(reg_spat_offt, reg_channel_offt_1byte);
            Label mb_sp_loop;
            L(mb_sp_loop);
            {
                if (need_tail) {
                    set_preg(p_tmp0.s, c_tail_, X_TMP_0, X_TMP_1);
                    ld1sb(v.s, p_tmp0 / T_z, ptr(src_ptr()));
                } else {
                    ld1sb(v.s, P_ALL_ONE / T_z, ptr(src_ptr()));
                }

                scvtf(v.s, P_ALL_ONE / T_m, v.s);

                fmad(v.s, P_ALL_ONE, vscale.s, vshift.s);
                if (with_relu_) {
                    if (has_alpha_value_)
                        process_relu_alpha(v);
                    else
                        uni_fmax(v, v, vzero);
                }

                frinti(v.s, P_ALL_ONE / T_m, v.s);
                fcvtzs(v.s, P_ALL_ONE / T_m, v.s);
                smin(v.s, 127);
                smax(v.s, -128);
                if (need_tail)
                    st1b(v.s, p_tmp0, ptr(dst_ptr()));
                else
                    st1b(v.s, P_ALL_ONE, ptr(dst_ptr()));

                add(reg_spat_offt, reg_spat_offt, reg_channel_offt_count);
                cmp(reg_spat_offt, reg_spat_offt_count);
                b(LT, mb_sp_loop);
            }

            // reg_tmp checks c_in_xmm_ channels ahead for further tail process
            add(reg_tmp, reg_tmp, sizeof(data_t) * c_in_xmm_);
            add(reg_channel_offt_1byte, reg_channel_offt_1byte,
                    sizeof(data_t) * c_in_xmm_);
            add(reg_channel_offt_4byte, reg_channel_offt_4byte,
                    sizeof(float) * c_in_xmm_);
            cmp(reg_tmp, reg_channel_offt_count);
            b(LE, c_loop);
        }
    }

    jit_bnorm_s8_t(const batch_normalization_pd_t *pd)
        : jit_bnorm_base_t<sve_512>(pd) {}
};

namespace bnorm_s8_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {
    driver_t(const batch_normalization_pd_t *pd) : pd_(pd), ker_(pd_) {}
    ~driver_t() = default;

    // TODO: for problems where thread pieces don't fit L2 cache, add spatial
    // re-balance using less pieces.
    void exec(int ithr, int nthr, const data_t *src, data_t *dst,
            const float *scale, const float *shift, const float *mean,
            const float *var) {
        dim_t N = pd_->MB();
        dim_t C = pd_->C();
        dim_t D = pd_->D();
        dim_t H = pd_->H();
        dim_t W = pd_->W();
        dim_t SP = D * H * W;

        jit_uni_bnorm_s8_call_params_t p;

        p.eps = pd_->desc()->batch_norm_epsilon;

        p.scale = scale;
        p.shift = shift;
        p.mean = mean;
        p.var = var;

        dim_t work_amount {N * SP}, start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        p.channel_offt_count = C;
        p.spat_offt_count = (end - start) * p.channel_offt_count;
        p.src = src + start * p.channel_offt_count;
        p.dst = dst + start * p.channel_offt_count;

        if (p.spat_offt_count != 0) ker_(&p);
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    const batch_normalization_pd_t *pd_;

    jit_bnorm_s8_t<isa> ker_;
};

} // namespace bnorm_s8_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::pd_t::init(
        engine_t *engine) {
    auto desired_fmt_tag = (ndims() == 4) ? nhwc : ndhwc;

    bool ok = true && mayiuse(isa) && is_fwd() && !has_zero_dim_memory()
            && one_of(ndims(), 4, 5) && stats_is_src()
            && src_md()->data_type == s8 && check_scale_shift_data_type()
            && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
            && (attr()->has_default_values() || this->with_relu_post_op(false))
            && set_default_formats_common()
            && memory_desc_wrapper(src_md()) == memory_desc_wrapper(dst_md());
    if (!ok) return status::unimplemented;

    // BN+Add+Relu fusion is not currently implemented
    if (fuse_norm_add_relu()) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_s8_fwd_t<isa>::jit_uni_batch_normalization_s8_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            bnorm_driver_, new bnorm_s8_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);
    auto mean = const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN));
    auto var
            = const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE));
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    // do sequential if the problem is less than one 4K memory page
    const bool force_sequential
            = pd()->MB() * pd()->C() * pd()->D() * pd()->H() * pd()->W()
            <= 4096;

    parallel(force_sequential ? 1 : 0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, dst, scale, shift, mean, var);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_s8_fwd_t<
        isa>::~jit_uni_batch_normalization_s8_fwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_s8_fwd_t<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
