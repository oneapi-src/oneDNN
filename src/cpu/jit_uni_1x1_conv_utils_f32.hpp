/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef JIT_UNI_1x1_CONV_UTILS_F32_HPP
#define JIT_UNI_1x1_CONV_UTILS_F32_HPP

#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

/* 1x1-kernel does not support non-unit strides so far, so the idea is:
 *  - for fwd or bwd_weights: to copy src to a scratch memory (with strides
 *    equal to 1) and then call the kernel
 *  - for bwd_data: reduce the problem to the one with unit stride by
 *    performing computations in a scratch memory (with strides equal to 1)
 *    and then copy the result to diff_src */
template <typename conv_pd_t>
inline void rtus_prepare(conv_pd_t *self, const convolution_desc_t *&conv_d,
        const memory_desc_t *&src_d, const memory_desc_t *dst_d) {
    const bool is_bwd_data = self->cdesc()->prop_kind
        == prop_kind::backward_data;

    bool rtus_applicable = true
        && (conv_d->strides[0] != 1 || conv_d->strides[1] != 1)
        && src_d->format == memory_format::nChw8c;
    for (int d = 2; d < 4; ++d) {
        /* TODO: relax these conditions (by improving reducer) */
        rtus_applicable = rtus_applicable
            && conv_d->padding[0][d - 2] == 0
            && dst_d->dims[d] * conv_d->strides[d - 2] == src_d->dims[d];
    }

    if (rtus_applicable) {
        self->rtus_.reduce_src_ = true;
        conv_d = &(self->rtus_.conv_d_ = *conv_d);
        self->rtus_.conv_d_.strides[0] = self->rtus_.conv_d_.strides[1] = 1;
        utils::array_set(self->rtus_.conv_d_.padding[0], 0, 2);
        utils::array_set(self->rtus_.conv_d_.padding[1], 0, 2);
        const int ic = src_d->dims[1];
        if (is_bwd_data) {
            src_d = &(self->rtus_.conv_d_.diff_src_desc = *dst_d);
            self->rtus_.conv_d_.diff_src_desc.dims[1] = ic;
            memory_desc_wrapper::compute_blocking(
                    self->rtus_.conv_d_.diff_src_desc);
        } else {
            src_d = &(self->rtus_.conv_d_.src_desc = *dst_d);
            self->rtus_.conv_d_.src_desc.dims[1] = ic;
            memory_desc_wrapper::compute_blocking(
                    self->rtus_.conv_d_.src_desc);
        }
    }
}

template <cpu_isa_t isa>
struct rtus_driver_f32_t: public jit_generator {
    using data_t = float;

    struct call_params_t {
        const data_t *ws; /* reduced image (w/ strides = 1) */
        const data_t *src; /* source image (w/ non-unit strides) */
        size_t icb;
        size_t os;
        size_t iw_start;
    };

    void (*ker_)(const call_params_t *p);

    /* cpu specific part */
    using Vmm =
        typename utils::conditional<isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const Xbyak::AddressFrame &vmmword = (isa == avx2) ? yword : zword;
    void uni_vpxor(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
            const Xbyak::Operand& op)
    { if (isa == avx2) vpxor(x1, x2, op); else vpxord(x1, x2, op); }
    const int vlen = cpu_isa_trait<isa>::vlen;
    const int typesize = sizeof(float);

    Xbyak::Reg64 reg_ws = abi_param1;
    Xbyak::Reg64 reg_src = abi_not_param1;
    Xbyak::Reg64 reg_icb = rdx;
    Xbyak::Reg64 reg_os = r11;
    Xbyak::Reg64 reg_iw_start = r8;

    Xbyak::Reg64 reg_cur_os = rax;
    Xbyak::Reg64 reg_cur_iw = r9;
    Xbyak::Reg64 reg_cur_src = r10;

    Vmm reg_zero = Vmm(0);
    Vmm reg_v = Vmm(1);

    int simd_w = vlen / typesize;
    int iw_, stride_w_;
    int src_step_h_, src_step_icb_, ws_step_icb_;
    bool src_to_ws_;

    rtus_driver_f32_t(int iw, int stride_w, int src_step_h,
            int src_step_icb, int ws_step_icb, bool src_to_ws)
        : iw_(iw), stride_w_(stride_w), src_step_h_(src_step_h)
        , src_step_icb_(src_step_icb), ws_step_icb_(ws_step_icb)
        , src_to_ws_(src_to_ws)
    { generate(); }

    void loop_is() {
        using namespace Xbyak;

        mov(reg_cur_src, reg_src);
        mov(reg_cur_iw, reg_iw_start);
        mov(reg_cur_os, reg_os);

        Label is_loop, skip_h_step;
        L(is_loop);

        if (src_to_ws_) {
            vmovups(reg_v, ptr[reg_cur_src]);
            vmovups(ptr[reg_ws], reg_v);
        } else {
            vmovups(reg_v, ptr[reg_ws]);
            vmovups(ptr[reg_cur_src], reg_v);
            for (int w = 1; w < stride_w_; ++w)
                vmovups(ptr[reg_cur_src + w * simd_w * typesize], reg_zero);
        }

        add(reg_ws, simd_w * typesize);

        add(reg_cur_iw, stride_w_);
        add(reg_cur_src, stride_w_ * simd_w * typesize);

        cmp(reg_cur_iw, iw_);
        jl(skip_h_step);

        if (src_to_ws_) {
            add(reg_cur_src, (src_step_h_ - iw_) * simd_w * typesize);
        } else {
            Xbyak::Reg64 reg_cur_src_fin = reg_cur_iw; /* just reuse */
            mov(reg_cur_src_fin, reg_cur_src);
            add(reg_cur_src_fin, (src_step_h_ - iw_) * simd_w * typesize);
            Label ih_loop;
            L(ih_loop);

            for (int w = 0; w < stride_w_; ++w)
                vmovups(ptr[reg_cur_src + w * simd_w * typesize], reg_zero);

            add(reg_cur_src, stride_w_ * simd_w * typesize);
            cmp(reg_cur_src, reg_cur_src_fin);
            jl(ih_loop);
        }
        xor_(reg_cur_iw, reg_cur_iw);

        L(skip_h_step);

        sub(reg_cur_os, simd_w * typesize);
        jnz(is_loop);

        /* restore dst */
        sub(reg_ws, reg_os);
    }

    void generate() {
        using namespace Xbyak;
        assert(isa == avx2 || isa == avx512_mic);

#if defined(_WIN32)
        push(rdi);
#endif

#define READ_PARAM(what) \
        mov(reg_ ## what, ptr[abi_param1 + offsetof(call_params_t, what)])
        READ_PARAM(src);
        READ_PARAM(icb);
        READ_PARAM(os);
        READ_PARAM(iw_start);

        assert(reg_ws == abi_param1);
        READ_PARAM(ws); /* reg_ws should always be read the last */
#undef  READ_PARAM

        shl(reg_os, cpu_isa_trait<isa>::vlen_shift);

        if (!src_to_ws_)
            uni_vpxor(reg_zero, reg_zero, reg_zero);

        Label icb_loop;
        L(icb_loop);

        loop_is();

        add(reg_ws, ws_step_icb_ * simd_w * typesize);
        add(reg_src, src_step_icb_ * simd_w * typesize);

        dec(reg_icb);
        jnz(icb_loop, T_NEAR);

#if defined(_WIN32)
        pop(rdi);
#endif

        ret();
        this->ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(
                    this->getCode()));
    }
};

template <cpu_isa_t isa, typename conv_t>
inline void init_rtus_driver_f32(conv_t *self) {
    using data_t = float;
    const auto &conf = self->conf_;
    const auto &cd = *conf.cdesc();
    const bool is_bwd_data = cd.prop_kind == prop_kind::backward_data;

    if (!conf.rtus_.reduce_src_) return;

    const int max_threads = omp_get_max_threads();
    size_t factor = 0;
    switch (cd.prop_kind) {
    case prop_kind::forward_training: case prop_kind::forward_inference:
        factor = conf.jcp_.nb_reduce; break;
    case prop_kind::backward_data:
        factor = conf.jcp_.nb_load_blocking_max; break;
    case prop_kind::backward_weights:
        factor = conf.jcp_.nb_bcast_blocking; break;
    default: assert(!"unsupported prop_kind");
    }

    self->ws_per_thread_ = factor * conf.jcp_.is * conf.jcp_.ic_block;
    self->scratch_ = (data_t *)malloc(
            max_threads * self->ws_per_thread_ * sizeof(data_t), 64);

    const int stride_h = cd.strides[0];
    const int stride_w = cd.strides[1];

    const auto &src_d = is_bwd_data ? cd.diff_src_desc : cd.src_desc;
    const int ih = src_d.dims[2];
    const int iw = src_d.dims[3];

    const int src_step_h = stride_h * iw;
    const int src_step_icb = ih * iw;
    const int ws_step_icb = conf.jcp_.is;
    const bool src_to_ws = !is_bwd_data;
    self->rtus_driver_ = new rtus_driver_f32_t<isa>(iw, stride_w, src_step_h,
            src_step_icb, ws_step_icb, src_to_ws);
}

}
}
}

#endif
