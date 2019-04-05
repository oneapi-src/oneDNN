/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "jit_uni_batch_normalization_s8.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {

using namespace Xbyak;

typedef int8_t data_t;

template <cpu_isa_t isa>
struct jit_bnorm_t: public jit_generator {
    struct call_params_t {
        // keep int sizes at 8 bytes -- jit code expects this
        size_t coff_max, soff_max;
        float eps, one;
        const float *scale_shift, *mean, *var;
        const data_t *src, *dst;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_t)

    using Vmm = typename utils::conditional<isa == avx2, Ymm, Zmm>::type;
    const AddressFrame &vmmword = (isa == avx2) ? yword : zword;

    const int vlen = cpu_isa_traits<isa>::vlen;

    const batch_normalization_pd_t *bdesc_;

    void (*ker)(const call_params_t *);
    void operator()(const call_params_t *p) { (*ker)(p); }

    Reg64 reg_param = abi_param1;

    Reg64 reg_scale_shift = rbx;
    Reg64 reg_mean = rbp;

    Reg64 reg_coff_max = r8;
    Reg64 reg_soff = r9;
    Reg64 reg_soff_max = r10;
    Reg64 reg_tmp = r11;
    Reg64 reg_src = r12;
    Reg64 reg_dst = r13;
    Reg64 reg_var = r14;

    // channel tail processing
    Opmask ktail_mask_s8 = Opmask(1);
    Opmask ktail_mask_f32 = Opmask(2);

    Vmm vtail_mask_f32 = Vmm(27);
    Vmm vtail_mask_s8 = Vmm(28);
    Vmm vzero = Vmm(29);
    Vmm vone = Vmm(30);
    Vmm veps = Vmm(31);

    bool with_relu_;
    size_t simd_w_;
    size_t unroll_regs_;
    size_t chan_data_offt_;
    size_t num_c_blocks_;
    size_t c_tail_;

    void compute_predefined_variables() {
        chan_data_offt_ = bdesc_->C() * sizeof(float);
        num_c_blocks_ = bdesc_->C() / simd_w_;
        c_tail_ = bdesc_->C() % simd_w_;
        unroll_regs_ = 4;
        with_relu_ = (bdesc_->with_relu_post_op() || bdesc_->fuse_bn_relu())
            && bdesc_->is_fwd();
    }

    void load_common_params() {
#       define PARAM_OFF(x) offsetof(call_params_t, x)
        uni_vbroadcastss(vone, vmmword[reg_param + PARAM_OFF(one)]);
        uni_vbroadcastss(veps, vmmword[reg_param + PARAM_OFF(eps)]);
        uni_vpxor(vzero, vzero, vzero);

        mov(reg_coff_max, ptr[reg_param + PARAM_OFF(coff_max)]);
        mov(reg_soff_max, ptr[reg_param + PARAM_OFF(soff_max)]);
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_scale_shift, ptr[reg_param + PARAM_OFF(scale_shift)]);
        mov(reg_var, ptr[reg_param + PARAM_OFF(var)]);
#       undef PARAM_OFF
    }

    void prepare_tail_mask_avx512() {
        if (!c_tail_) return;

        size_t tail_4byte_chunks = c_tail_ / sizeof(float);
        const int mask_s8 = (1 << tail_4byte_chunks) - 1;
        const int mask_f32 = (1 << c_tail_) - 1;

        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask_s8);
        kmovw(ktail_mask_s8, regw_tmp);

        mov(regw_tmp, mask_f32);
        kmovw(ktail_mask_f32, regw_tmp);
    }

    void prepare_tail_mask_avx2() {
        if (!c_tail_) return;

        static const uint32_t mask_f32[16] = {0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0, 0, 0, 0, 0, 0, 0, 0};

        static const uint32_t mask_s8[16] = {0xffffffff, 0xffffffff, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp, reinterpret_cast<size_t>(&mask_f32[8 - c_tail_]));
        vmovups(vtail_mask_f32, ptr[reg_tmp]);

        mov(reg_tmp, reinterpret_cast<size_t>(
                    &mask_s8[2 - c_tail_ / sizeof(float)]));
        vmovups(vtail_mask_s8, ptr[reg_tmp]);
    }

    template <typename T>
    void uni_vmovups_tail_avx2(const Operand &dst, const Operand &src) {
        if (sizeof(T) == sizeof(float)) {
            if (dst.isMEM())
                vmaskmovps(dst.getAddress(), vtail_mask_f32, Vmm(src.getIdx()));
            else
                vmaskmovps(Vmm(dst.getIdx()), vtail_mask_f32, src.getAddress());
        } else if (sizeof(T) == sizeof(data_t)) {
            if (dst.isMEM())
                vmaskmovps(dst.getAddress(), vtail_mask_s8, Xmm(src.getIdx()));
            else
                vmaskmovps(Xmm(dst.getIdx()), vtail_mask_s8, src.getAddress());
        } else
            assert(!"unsupported data type");
    }

    template <typename T>
    void uni_vmovups_tail_avx512(const Operand &dst, const Operand &src) {
        if (sizeof(T) == sizeof(float)) {
            if (dst.isMEM())
                vmovups(dst.getAddress() | ktail_mask_f32, Vmm(src.getIdx()));
            else
                vmovups(Vmm(dst.getIdx()) | ktail_mask_f32 | T_z, src.getAddress());
        } else if (sizeof(T) == sizeof(data_t)) {
            if (dst.isMEM()) {
                vmovups(dst.getAddress() | ktail_mask_s8, Xmm(src.getIdx()));
            }
            else {
                vmovups(Xmm(dst.getIdx()) | ktail_mask_s8 | T_z, src.getAddress());
            }
        } else
            assert(!"unsupported data type");
    }

    template <typename T>
    void uni_vmovups_tail(const Operand &dst, const Operand &src) {
        if (isa == avx512_core)
            uni_vmovups_tail_avx512<T>(dst, src);
        else if (isa == avx2)
            uni_vmovups_tail_avx2<T>(dst, src);
    }

    Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + offt];
    }

    Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + offt];
    }

    Address scale_ptr(size_t offt = 0) {
        return vmmword[reg_scale_shift + offt + 0 * chan_data_offt_];
    }

    Address shift_ptr(size_t offt = 0) {
        return vmmword[reg_scale_shift + offt + 1 * chan_data_offt_];
    }

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + reg_soff + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_soff + offt];
    }

    template <typename body_t, typename tail_t>
    void channel_loop(size_t C16_blocks, size_t c_tail, size_t unroll_regs,
            body_t body, tail_t tail) {
        size_t num_loops = C16_blocks / unroll_regs;
        size_t loop_tail = C16_blocks - num_loops * unroll_regs;

        for (size_t n_l = 0; n_l < num_loops; n_l++) {
            for (size_t i_block = 0; i_block < unroll_regs; i_block++) {
                body(i_block, n_l * unroll_regs + i_block);
            }
        }

        if (loop_tail) {
            for (size_t i_block = 0; i_block < loop_tail; i_block++) {
                body(i_block, num_loops * unroll_regs + i_block);
            }
        }

        // Unroll c_tail with i_block = 4 as loop_tail has 3 iterations at most
        if (c_tail)
            tail(3, C16_blocks);
    }

    void forward() {
        xor_(reg_soff, reg_soff);
        Label mb_sp_loop;
        L(mb_sp_loop); {

            // fills vscale and vshift with values so that algorithm performs
            // vdst = vscale * vsrc + vbeta next;
            auto compute_vscaleshift = [=](const Vmm &vscale, const Vmm &vshift,
                    size_t base_reg, size_t c_block, bool need_tail = false) {
                size_t coff_f32 = c_block * simd_w_ * sizeof(float);

                // register numeration should be aligned with body and tail
                Vmm vsqrtvar = Vmm(base_reg + 3*unroll_regs_);
                Vmm vmean = Vmm(base_reg + 4*unroll_regs_);

                if (need_tail) {
                    uni_vmovups_tail<float>(vmean, mean_ptr(coff_f32));
                    uni_vmovups_tail<float>(vsqrtvar, var_ptr(coff_f32));
                } else {
                    uni_vmovups(vmean, mean_ptr(coff_f32));
                    uni_vmovups(vsqrtvar, var_ptr(coff_f32));
                }
                uni_vaddps(vsqrtvar, vsqrtvar, veps);
                uni_vsqrtps(vsqrtvar, vsqrtvar);

                if (bdesc_->use_scaleshift()) {
                    if (need_tail) {
                        uni_vmovups_tail<float>(vscale, scale_ptr(coff_f32));
                        uni_vmovups_tail<float>(vshift, shift_ptr(coff_f32));
                    } else {
                        uni_vmovups(vscale, scale_ptr(coff_f32));
                        uni_vmovups(vshift, shift_ptr(coff_f32));
                    }
                    vdivps(vscale, vscale, vsqrtvar);
                    uni_vfnmadd231ps(vshift, vmean, vscale);
                } else {
                    vdivps(vscale, vone, vsqrtvar);
                    uni_vmulps(vmean, vmean, vscale);
                    uni_vsubps(vshift, vzero, vmean);
                }
            };

            channel_loop(num_c_blocks_, c_tail_, unroll_regs_,
                    [=](size_t base_reg, size_t c_block) {
                        size_t coff_s8 = c_block * simd_w_ * sizeof(data_t);

                        Vmm v = Vmm(base_reg + 0*unroll_regs_);
                        Vmm vscale = Vmm(base_reg + 1*unroll_regs_);
                        Vmm vshift = Vmm(base_reg + 2*unroll_regs_);

                        compute_vscaleshift(vscale, vshift, base_reg, c_block);

                        // up convert
                        // TODO: try to load 64 bytes and create 4 zmms from 'em
                        vpmovsxbd(v, src_ptr(coff_s8));
                        vcvtdq2ps(v, v);

                        uni_vfmadd213ps(v, vscale, vshift);
                        if (with_relu_)
                            uni_vmaxps(v, v, vzero);

                        // down convert
                        vcvtps2dq(v, v);
                        vpmovsdb(dst_ptr(coff_s8), v);
                    },
                    [=](size_t base_reg, size_t c_block) {
                        size_t coff_s8 = c_block * simd_w_ * sizeof(data_t);
                        size_t tail_1byte_left = c_tail_ % sizeof(float);

                        Xmm x = Xmm(base_reg + 0*unroll_regs_);
                        Vmm v = Vmm(base_reg + 0*unroll_regs_);
                        Vmm vscale = Vmm(base_reg + 1*unroll_regs_);
                        Vmm vshift = Vmm(base_reg + 2*unroll_regs_);

                        // load 4-byte chunks, then 1-byte chunks
                        uni_vmovups_tail<data_t>(x, src_ptr(coff_s8));
                        for (size_t tl = 0; tl < tail_1byte_left; tl++) {
                            size_t byte_off = c_tail_ - tail_1byte_left + tl;
                            vpinsrb(x, x, src_ptr(coff_s8 + byte_off), byte_off);
                        }

                        compute_vscaleshift(vscale, vshift, base_reg, c_block,
                                true);

                        // up convert
                        vpmovsxbd(v, x);
                        vcvtdq2ps(v, v);

                        uni_vfmadd213ps(v, vscale, vshift);
                        if (with_relu_)
                            uni_vmaxps(v, v, vzero);

                        // down convert
                        vcvtps2dq(v, v);
                        vpmovsdb(x, v);

                        // store 4-byte chunks, then 1-byte chunks
                        uni_vmovups_tail<data_t>(dst_ptr(coff_s8), x);
                        for (size_t tl = 0; tl < tail_1byte_left; tl++) {
                            size_t byte_off = c_tail_ - tail_1byte_left + tl;
                            vpextrb(dst_ptr(coff_s8 + byte_off), x, byte_off);
                        }
                    });

            add(reg_soff, reg_coff_max);
            cmp(reg_soff, reg_soff_max);
            jl(mb_sp_loop);
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *bdesc): bdesc_(bdesc) {
        static_assert(isa == avx2 || isa == avx512_core, "unsupported isa");

        simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);

        preamble();
        compute_predefined_variables();

        if (isa == avx512_core)
            prepare_tail_mask_avx512();
        else if (isa == avx2)
            prepare_tail_mask_avx2();

        load_common_params();
        forward();
        postamble();

        ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
                    this->getCode()));
    }
};

}

namespace bnorm_s8_impl {

template <cpu_isa_t isa>
struct driver_t: public c_compatible {
    driver_t(const batch_normalization_pd_t *bdesc)
        : bdesc_(bdesc), ker_(bdesc_) {}
    ~driver_t() {}

    void exec(int ithr, int nthr, const data_t *src, data_t *dst,
            const float *scale_shift, const float *mean, const float *var) {
        dim_t N = bdesc_->MB();
        dim_t C = bdesc_->C();
        dim_t D = bdesc_->D();
        dim_t H = bdesc_->H();
        dim_t W = bdesc_->W();
        dim_t SP = D * H * W;

        typename jit_bnorm_t<isa>::call_params_t p;

        p.eps = bdesc_->desc()->batch_norm_epsilon;
        p.one = 1.0f;

        p.scale_shift = scale_shift;
        p.mean = mean;
        p.var = var;

        /* Naive balancing: allows unrolling and handle tails nicely */
        dim_t work_amount{N*SP}, start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        p.coff_max = C;
        p.soff_max = (end - start) * p.coff_max;
        p.src = src + start * p.coff_max;
        p.dst = dst + start * p.coff_max;

        if (p.soff_max != 0)
            ker_(&p);
    }

private:
    const batch_normalization_pd_t *bdesc_;

    jit_bnorm_t<isa> ker_;
};

}

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::pd_t::init() {
    auto desired_fmt_tag = (ndims() == 4) ? nhwc : ndhwc;

    bool ok = true
        && mayiuse(isa)
        && is_fwd()
        && !has_zero_dim_memory()
        && one_of(ndims(), 4, 5)
        && stats_is_src()
        && src_md()->data_type == s8
        && IMPLICATION(use_scaleshift(), weights_md()->data_type == f32)
        && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
        && (attr()->has_default_values() || this->with_relu_post_op());
    if (!ok) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_s8_fwd_t<isa>::jit_uni_batch_normalization_s8_fwd_t(
        const pd_t *apd): cpu_primitive_t(apd) {
    bnorm_driver_ = new bnorm_s8_impl::driver_t<isa>(pd());
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_s8_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto scale_shift = CTX_IN_MEM(const float *, MKLDNN_ARG_SCALE_SHIFT);
    auto mean = const_cast<float *>(CTX_IN_MEM(const float *, MKLDNN_ARG_MEAN));
    auto var = const_cast<float *>(CTX_IN_MEM(const float *,
                MKLDNN_ARG_VARIANCE));
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

    // do sequential if the problem is less than one 4K memory page
    const bool force_sequential = pd()->MB() * pd()->C() * pd()->D() * pd()->H()
        * pd()->W() <= 4096;

    parallel(force_sequential ? 1 : 0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, dst, scale_shift, mean, var);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_s8_fwd_t<isa>::
~jit_uni_batch_normalization_s8_fwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_s8_fwd_t<avx512_core>;
/* TODO: add avx2 version */

}
}
}
