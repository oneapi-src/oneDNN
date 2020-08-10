/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef CPU_AARCH64_JIT_UNI_1X1_CONV_UTILS_HPP
#define CPU_AARCH64_JIT_UNI_1X1_CONV_UTILS_HPP

#include "common/convolution_pd.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/primitive_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

struct reduce_to_unit_stride_t {
    convolution_desc_t conv_d_;
    bool reduce_src_;
    size_t space_per_thread_;
};

/* 1x1-kernel does not support non-unit strides so far, so the idea is:
 *  - for fwd or bwd_weights: to copy src to a scratch memory (with strides
 *    equal to 1) and then call the kernel
 *  - for bwd_data: reduce the problem to the one with unit stride by
 *    performing computations in a scratch memory (with strides equal to 1)
 *    and then copy the result to diff_src */
template <typename conv_pd_t>
inline void rtus_prepare(conv_pd_t *self, const convolution_desc_t *&conv_d,
        const memory_desc_t *&src_d, const memory_desc_t *dst_d) {
#if 1
    return;
#else
    const int ndims = src_d->ndims;

    bool rtus_applicable = utils::one_of(ndims, 3, 4);
    if (ndims == 3)
        rtus_applicable = rtus_applicable && conv_d->strides[0] != 1
                && conv_d->src_desc.data_type != data_type::s32;
    else
        rtus_applicable = rtus_applicable
                && (conv_d->strides[0] != 1 || conv_d->strides[1] != 1);
    for (int d = 2; d < ndims; ++d) {
        /* TODO: relax these conditions (by improving reducer) */
        rtus_applicable = rtus_applicable && conv_d->padding[0][d - 2] == 0
                && dst_d->dims[d] * conv_d->strides[d - 2] == src_d->dims[d];
    }
    if (!rtus_applicable) return;

    const auto dat_tag = ndims == 3
            ? memory_desc_wrapper(src_d).matches_one_of_tag(
                    format_tag::nCw8c, format_tag::nCw16c, format_tag::nwc)
            : memory_desc_wrapper(src_d).matches_one_of_tag(
                    format_tag::nChw8c, format_tag::nChw16c, format_tag::nhwc);
    if (dat_tag == format_tag::undef) return;

    const bool is_nspc
            = utils::one_of(dat_tag, format_tag::nwc, format_tag::nhwc);
    if (is_nspc && !mayiuse(avx2)) return;

    // rtus is applicable, configure it.
    self->rtus_.reduce_src_ = true;
    conv_d = &(self->rtus_.conv_d_ = *conv_d);
    self->rtus_.conv_d_.strides[0] = 1;
    if (ndims == 4) self->rtus_.conv_d_.strides[1] = 1;
    utils::array_set(self->rtus_.conv_d_.padding[0], 0, 2);
    if (ndims == 4) utils::array_set(self->rtus_.conv_d_.padding[1], 0, 2);
    const int ic = src_d->dims[1];
    if (self->desc()->prop_kind == prop_kind::backward_data) {
        data_type_t data_type = self->rtus_.conv_d_.diff_src_desc.data_type;
        src_d = &(self->rtus_.conv_d_.diff_src_desc = *dst_d);
        self->rtus_.conv_d_.diff_src_desc.dims[1] = ic;
        self->rtus_.conv_d_.diff_src_desc.data_type = data_type;
        memory_desc_wrapper::compute_blocking(
                self->rtus_.conv_d_.diff_src_desc, dat_tag);
    } else {
        data_type_t data_type = self->rtus_.conv_d_.src_desc.data_type;
        src_d = &(self->rtus_.conv_d_.src_desc = *dst_d);
        self->rtus_.conv_d_.src_desc.dims[1] = ic;
        self->rtus_.conv_d_.src_desc.data_type = data_type;
        memory_desc_wrapper::compute_blocking(
                self->rtus_.conv_d_.src_desc, dat_tag);
    }
#endif
}

template <typename conv_pd_t>
inline void rtus_prepare_space_info(conv_pd_t *self,
        memory_tracking::registrar_t &scratchpad, int max_threads) {
    if (!self->rtus_.reduce_src_) return;
    const auto &jcp = self->jcp_;
    const bool is_nspc
            = utils::one_of(jcp.src_tag, format_tag::nhwc, format_tag::nwc);

    const size_t factor = utils::pick_by_prop_kind(self->desc()->prop_kind,
            jcp.nb_reduce, jcp.nb_load_blocking_max, jcp.nb_bcast_blocking);
    size_t typesize
            = types::data_type_size(self->invariant_src_md()->data_type);

    self->rtus_.space_per_thread_
            = is_nspc ? jcp.is * jcp.ic : factor * jcp.is * jcp.ic_block;
    scratchpad.book(memory_tracking::names::key_conv_rtus_space,
            max_threads * self->rtus_.space_per_thread_, typesize);
}

template <cpu_isa_t isa>
struct rtus_driver_t : public jit_generator {
// TODO
#if 0
    struct call_params_t {
        const void *ws; /* reduced image (w/ strides = 1) */
        const void *src; /* source image (w/ non-unit strides) */
        size_t icb;
        size_t os;
        size_t iw_start;
    };

    void (*ker_)(const call_params_t *p);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(rtus_driver_t)
    using reg64_t = const Xbyak::Xbyak_aarch64::XReg;

    reg64_t reg_ws             = x16; //r12;
    reg64_t reg_src            = x17; //r13;
    reg64_t reg_icb            = x18; //rdx;
    reg64_t reg_os             = x19; //r11;
    reg64_t reg_iw_start       = x21; //r8;

    reg64_t reg_cur_os         = x22; //rax;
    reg64_t reg_cur_iw         = x23; //r9;
    reg64_t reg_cur_src        = x24; //r10;
    reg64_t reg_cur_src_fin    = reg_cur_iw; /* just reuse */

    //Xbyak::Opmask tail_mask   = k2;
    xa::PReg tail_mask          = p1;
    
    // nspc section
    reg64_t reg_cur_icb         = x22; //rax;
    reg64_t reg_tail_mask       = x25; //r14;
    reg64_t reg_icb_remainder   = x26; //rcx;
    reg64_t reg_ws_copy         = x27; //r15;

    reg64_t reg_tmp_ofs         = x28;
    reg64_t reg_tmp_imm         = x29;

    int iw_, stride_w_;
    int src_step_h_, src_step_icb_, ws_step_icb_, vlen_, vlen_shift_;
    bool src_to_ws_;
    size_t typesize_;
    int ic_, ic_tail_;
    bool is_nspc_;

    xa::ZReg reg_zero;  //Xyak::Xmm reg_zero;
    xa::XReg reg_v;     //Xbyak::Xmm reg_v;

    rtus_driver_t(int iw, int stride_w, int src_step_h, int src_step_icb,
            int ws_step_icb, bool src_to_ws, size_t typesize, int ic,
            bool is_nspc = false)
        : iw_(iw)
        , stride_w_(stride_w)
        , src_step_h_(src_step_h)
        , src_step_icb_(src_step_icb)
        , ws_step_icb_(ws_step_icb)
        , src_to_ws_(src_to_ws)
        , typesize_(typesize)
        , ic_(ic)
        , is_nspc_(is_nspc) {
        using namespace Xbyak::Xbyak_aarch64;

        assert(ic_ > 0);

        /*FIXME: derive Vmm type on compile time.
         * changing register type  on runtime
         * seems dangerous,and some xbyak functions might
         * fail to work on reg_v, reg_zero because of this
         * data_type change, e.g. uni_vpxor doen't
         * work on reg_zero now*/
        auto VReg = [=](int idx, size_t typesize) {
            ZReg res;
            if (is_nspc_) {
                switch (isa) {
                    case sve: res = ZReg(idx); break;
                    default: assert(!"Not supported isa"); res = Xmm(idx);
                }
                return res;
            }
            switch (isa) {
                case sve:
                    switch (typesize) {
                        case 4: res = ZReg(idx); break;
                        case 2: //res = Ymm(idx); break;
                        case 1: //res = Xmm(idx); break;
                        default:
                            assert(!"Not supported typesize");
                            res = ZReg(idx);
                    }
            }
            return res;
        };

        reg_zero = VReg(0, typesize);
        reg_v = VReg(1, typesize);

        vlen_ = reg_v.getBit() / 8;
        vlen_shift_ = 0;

        int tvlen = is_nspc_ ? typesize_ : vlen_;
        while (tvlen > 1) {
            tvlen /= 2;
            vlen_shift_++;
        }

        const int simd_w = vlen_ / sizeof(float);
        ic_tail_ = ic_ % simd_w;

        generate();
    }

    void loop_is() {
        using namespace Xbyak::Xbyak_aarch64;

        CGA64::mov(reg_cur_src, reg_src);
        CGA64::mov(reg_cur_iw, reg_iw_start);
        CGA64::mov(reg_cur_os, reg_os);

        xa::LabelAArch64 is_loop;
        CGA64::L_aarch64(is_loop);

        if (src_to_ws_) {
            CGA64::ldr(reg_v, xa::ptr(reg_cur_src));
            CGA64::str(reg_v, xa::ptr(reg_ws));
        } else {
            CGA64::ldr(reg_v, xa::ptr(reg_ws));
            CGA64::str(reg_v, xa::ptr(reg_cur_src));
            for (int w = 1; w < stride_w_; ++w){
                // TODO
                if((w*vlen_) == 0){
                    CGA64::str(reg_zero, xa::ptr(reg_cur_src);
                }else{
                    CGA64::add_imm(reg_tmp_ofs, reg_cur_src, w*vlen_, reg_tmp_imm);
                    CGA64::str(reg_zero, xa::ptr(reg_tmp_ofs);
                }
            }
        }

        CGA64::add_imm(reg_ws, reg_ws, vlen_, reg_tmp_imm);
        CGA64::add_imm(reg_cur_src, reg_cur_src, stride_w_ * vlen_, reg_tmp_imm);

        // for 1d or stride_h=1 convolutions the loop over h should be skipped
        if (!(src_step_icb_ == iw_ || src_step_h_ == iw_)) {
            xa::LabelAArch64 skip_h_step;
            CGA64::add_imm(reg_cur_iw, reg_cur_iw, stride_w_, reg_tmp_imm);
            CGA64::cmp(reg_cur_iw, iw_);
            CGA64::b(xa::LT, skip_h_step); //jl(skip_h_step);

            if (src_to_ws_) {
                CGA64::add_imm(reg_cur_src, reg_cur_src,
                                (src_step_h_ - iw_) * vlen_, reg_tmp_imm);
            } else {
                CGA64::mov(reg_cur_src_fin, reg_cur_src);
                CGA64::add_imm(reg_cur_src_fin, reg_cur_src_fin,
                                (src_step_h_ - iw_) * vlen_, reg_tmp_imm);
                xa::LabelAArch64 ih_loop;
                CGA64::L_aarch64(ih_loop);

                for (int w = 0; w < stride_w_; ++w){
                    if( (w*vlen_) == 0){
                        CGA64::str(reg_zero, xa::ptr(reg_cur_src));
                    }else{
                        CGA64::add_imm(reg_tmp_ofs, reg_cur_src, w*vlen_, reg_tmp_imm);
                        CGA64::str(reg_zero, xa::ptr(reg_tmp_ofs));
                    }
                }

                CGA64::add_imm(reg_cur_src, reg_cur_src, stride_w_ * vlen_, reg_tmp_imm);
                CGA64::cmp(reg_cur_src, reg_cur_src_fin);
                CGA64::b(xa::LT, ih_loop); //jl(ih_loop);
            }
            CGA64::mov(reg_cur_iw, 0);
            CGA64::L_aarch64(skip_h_step);
        }

        CGA64::subs_imm(reg_cur_os, reg_cur_os, vlen_, reg_tmp_imm);
        CGA64::b(xa::NE, is_loop); //jnz(is_loop);

        /* restore dst */
        CGA64::subs(reg_ws, reg_ws, reg_os);
    }

    void loop_is_nspc() {
        using namespace Xbyak::Xbyak_aarch64;

        assert(is_nspc_);

        CGA64::mov(reg_cur_src, reg_src);
        CGA64::mov(reg_cur_iw, reg_iw_start);

        if (isa == sve) {
            //push(rcx); // preserve rcx, used for shift
            CGA64::mov(reg_icb_remainder, reg_icb);
            CGA64::and_imm(reg_icb_remainder, reg_icb_remainder,
                    (vlen_ / typesize_) - 1, reg_tmp_imm); // # of elements in tail
            CGA64::mov(reg_tail_mask, 1);
            CGA64::lsl(reg_tail_mask, reg_icb_remainder.cvt8());
            CGA64::sub(reg_tail_mask, 1); //dec(reg_tail_mask);
            //pop(rcx);

            switch (typesize_) {
                case 4: kmovw(tail_mask, reg_tail_mask.cvt32()); break;
                case 2: kmovd(tail_mask, reg_tail_mask.cvt32()); break;
                case 1: kmovq(tail_mask, reg_tail_mask); break;
                default: assert(!"Unsupported typesize");
            }
        }

        auto load_reg = [=](const xa::ZReg &vreg, const xa::XReg &reg,
                                const int64_t offset, const int load_size) {
            if (isa == sve) {
                CGA64::add_imm(reg_tmp_ofs, reg, offset, reg_tmp_imm);
                CGA64::ldr(vreg, xa::ptr(reg_tmp_ofs));
                //const Address &addr = ptr[reg + offset];
                //switch (typesize_) {
                //    case 4: vmovups(vreg, addr); break;
                //    case 2: vmovdqu16(vreg, addr); break;
                //    case 1: vmovdqu8(vreg, addr); break;
                //    default: assert(!"Unsupported typesize");
                //}
            } else {
                assert(!"Unsupported load_bytes");
                //load_bytes(vreg, reg, offset, load_size);
            }
        };

        auto store_reg = [=](const xa::XReg &reg, const xa::ZReg &vreg,
                                 const int64_t offset, const int store_size) {
            if (isa == sve) {
                CGA64::add_imm(reg_tmp_ofs, reg, offset, reg_tmp_imm);
                CGA64::str(vreg, xa::ptr(reg_tmp_ofs));
                //const Address &addr = ptr[reg + offset];
                //switch (typesize_) {
                //    case 4: vmovups(addr, vreg); break;
                //    case 2: vmovdqu16(addr, vreg); break;
                //    case 1: vmovdqu8(addr, vreg); break;
                //    default: assert(!"Unsupported typesize");
                //}
            } else {
                assert(!"Unsupported store_bytes");
                //store_bytes(vreg, reg, offset, store_size);
            }
        };

        CGA64::mov(reg_ws_copy, reg_ws);
        CGA64::lsl(reg_icb, vlen_shift_);

        const size_t w_step_factor = ic_ * typesize_;
        const size_t max_load_store_bytes = 32;
        const size_t load_store_size
                = isa == sve ? vlen_ : max_load_store_bytes;
        size_t load_store_tail_size = (typesize_ == 1 ? max_load_store_bytes
                                                      : ic_tail_ * typesize_);

        xa::LabelAArch64 is_loop, ic_loop, ic_loop_tail, ic_loop_finish;
        CGA64::L_aarch64(is_loop);
        {
            CGA64::mov(reg_cur_src, reg_src);
            CGA64::mov(reg_ws, reg_ws_copy);
            CGA64::mov(reg_cur_icb, reg_icb);

            CGA64::L_aarch64(ic_loop);
            {
                CGA64::cmp(reg_cur_icb, load_store_size);
                CGA64::b(xa::LT, ic_loop_tail); //jl(ic_loop_tail);

                if (src_to_ws_) {
                    load_reg(reg_v, reg_cur_src, 0, load_store_size);
                    store_reg(reg_ws, reg_v, 0, load_store_size);
                } else {
                    load_reg(reg_v, reg_ws, 0, load_store_size);
                    store_reg(reg_cur_src, reg_v, 0, load_store_size);
                    for (int w = 1; w < stride_w_; ++w)
                        store_reg(reg_cur_src, reg_zero, w * w_step_factor,
                                load_store_size);
                }
                CGA64::add_imm(reg_ws, reg_ws, load_store_size, reg_tmp_imm);
                CGA64::add_imm(reg_cur_src, reg_cur_src, load_store_size, reg_tmp_imm);

                CGA64::subs_imm(reg_cur_icb, reg_cur_icb, load_store_size, reg_tmp_imm);
                CGA64::b(ic_loop);
            }

            CGA64::L_aarch64(ic_loop_tail);
            {
                CGA64::cmp(reg_cur_icb, 0);
                CGA64::b(xa::EQ, ic_loop_finish); //je(ic_loop_finish);

                if (src_to_ws_) {
                    load_reg(reg_v | tail_mask, reg_cur_src, 0,
                            load_store_tail_size);
                    store_reg(
                            reg_ws, reg_v | tail_mask, 0, load_store_tail_size);
                } else {
                    load_reg(
                            reg_v | tail_mask, reg_ws, 0, load_store_tail_size);
                    store_reg(reg_cur_src, reg_v | tail_mask, 0,
                            load_store_tail_size);
                    for (int w = 1; w < stride_w_; ++w)
                        store_reg(reg_cur_src, reg_zero | tail_mask,
                                w * w_step_factor, load_store_tail_size);
                }
            }
            CGA64::L_aarch64(ic_loop_finish);

            CGA64::add_imm(reg_ws_copy, reg_ws_copy, w_step_factor, reg_tmp_imm);
            CGA64::add_imm(reg_src, reg_src, stride_w_ * w_step_factor, reg_tmp_imm);

            // for 1d or stride_h=1 convolutions the loop over h should be skipped
            const bool skip_oh_step = src_step_h_ == iw_;
            if (!skip_oh_step) {
                CGA64::mov(reg_cur_src, reg_src);
                xa::LabelAArch64 skip_h_step;
                CGA64::add_imm(reg_cur_iw, reg_cur_iw, stride_w_, reg_tmp_imm);
                CGA64::cmp(reg_cur_iw, reg_cur_iw, iw_, reg_tmp_imm);
                CGA64::b(xa::LT, skip_h_step); //jl(skip_h_step, T_NEAR);

                if (src_to_ws_) {
                    CGA64::add_imm(reg_src, reg_src, (src_step_h_ - iw_) * w_step_factor, reg_tmp_imm);
                } else {
                    CGA64::mov(reg_cur_src_fin, reg_cur_src);
                    CGA64::add_imm(reg_cur_src_fin, reg_cur_src_fin, (src_step_h_ - iw_) * w_step_factor, reg_tmp_imm);
                    xa::LabelAArch64 ih_loop_nhwc, ic_ih_loop_nhwc, ic_tail_ih_loop_nhwc,
                            ic_finish_ih_loop_nhwc;
                    CGA64::L_aarch64(ih_loop_nhwc);
                    CGA64::mov(reg_cur_src, reg_src);
                    CGA64::mov(reg_cur_icb, reg_icb);
                    CGA64::L_aarch64(ic_ih_loop_nhwc);
                    CGA64::cmp(reg_cur_icb, load_store_size);
                    CGA64::b(xa::LT, ic_tail_ih_loop_nhwc); //jl(ic_tail_ih_loop_nhwc);

                    for (int w = 0; w < stride_w_; ++w)
                        store_reg(reg_cur_src, reg_zero, w * w_step_factor,
                                load_store_size);

                    CGA64::add_imm(reg_cur_src, reg_cur_src, load_store_size, reg_tmp_imm);
                    CGA64::subs_imm(reg_cur_icb, reg_cur_icb, load_store_size, reg_tmp_imm);
                    CGA64::b(xa::NE, ic_ih_loop_nhwc); //jnz(ic_ih_loop_nhwc);

                    CGA64::L_aarch64(ic_tail_ih_loop_nhwc);
                    CGA64::cmp(reg_cur_icb, 0);
                    CGA64::b(xa::LE, ic_finish_ih_loop_nhwc); //jle(ic_finish_ih_loop_nhwc);

                    for (int w = 0; w < stride_w_; ++w)
                        store_reg(reg_cur_src, reg_zero | tail_mask,
                                w * w_step_factor, load_store_tail_size);

                    CGA64::L_aarch64(ic_finish_ih_loop_nhwc);

                    CGA64::add_imm(reg_src, reg_src, stride_w_ * w_step_factor, reg_tmp_imm);
                    CGA64::cmp(reg_src, reg_cur_src_fin);
                    CGA64::b(xa::LT, ih_loop_nhwc); //jl(ih_loop_nhwc);
                }
                CGA64::mov(reg_cur_iw, 0);
                CGA64::L_aarch64(skip_h_step);
            }

            CGA64::subs(reg_os, 1);
            CGA64::b(xa::NE, is_loop); //jnz(is_loop);
        }
    }

    void generate() {
        using namespace Xbyak::Xbyak_aarch64;
        assert(isa == sve);

        preamble();
#define READ_PARAM(what) \
    mov(reg_##what, ptr[abi_param1_aarch64 + offsetof(call_params_t, what)])
        READ_PARAM(src);
        READ_PARAM(icb);
        READ_PARAM(os);
        READ_PARAM(iw_start);
        READ_PARAM(ws);
#undef READ_PARAM

        if (!src_to_ws_) {
            switch (reg_zero.getBit() / 8) {
                case 64 /*sve*/: {
                    xa::ZReg zreg(reg_zero.getIdx());
                    CGA64::fmov(zreg); // zero clear
                    break;
                }
                default: assert(!"rtus kernel failure");
            }
        }
        if (is_nspc_) {
            loop_is_nspc();
        } else {
            CGA64::lsl(reg_os, vlen_shift_);

            xa::LabelAArch64 icb_loop;
            CGA64::L_aarch64(icb_loop);

            loop_is();

            CGA64::add_imm(reg_ws, reg_ws, ws_step_icb_ * vlen_, reg_tmp_imm);
            CGA64::add_imm(reg_src, reg_src, src_step_icb_ * vlen_, reg_tmp_imm);

            CGA64::subs_imm(reg_icb, reg_icb, vlen_ / typesize_, reg_tmp_imm);
            CGA64::b(xa::NE, icb_loop); //jnz(icb_loop, T_NEAR);
        }

        postamble();

        //uni_vzeroupper();
        ret();
        this->ker_ = reinterpret_cast<decltype(ker_)>(
                const_cast<uint32_t *>(this->getCode32()));
    }
#endif
};

template <cpu_isa_t isa, typename conv_t>
inline void init_rtus_driver(conv_t *self) {
    const auto &conf = *self->pd();
    if (!conf.rtus_.reduce_src_) return;

    const auto &cd = *conf.desc();
    const int ndims = conf.ndims();
    const int stride_h = (conf.ndims() == 3) ? 1 : cd.strides[0];
    const int stride_w = cd.strides[ndims - 3];

    const bool is_bwd_data = cd.prop_kind == prop_kind::backward_data;
    const auto &src_d = is_bwd_data ? *conf.diff_src_md() : *conf.src_md();

    const int ih = ndims == 3 ? 1 : src_d.dims[2];
    const int iw = src_d.dims[ndims - 1];
    const int ic = src_d.dims[1];

    const auto src_tag = memory_desc_wrapper(src_d).matches_one_of_tag(
            format_tag::nhwc, format_tag::nwc);
    const bool is_nspc = src_tag != format_tag::undef;
    const int src_step_h = stride_h * iw;
    const int src_step_icb = !is_nspc ? ih * iw : 1;
    const int ws_step_icb = !is_nspc ? conf.jcp_.is : 1;
    const bool src_to_ws = !is_bwd_data;
    const size_t typesize
            = types::data_type_size(self->pd()->invariant_src_md()->data_type);

    self->rtus_driver_ = new rtus_driver_t<isa>(iw, stride_w, src_step_h,
            src_step_icb, ws_step_icb, src_to_ws, typesize, ic, is_nspc);
}

inline int best_divider(int value, int min_divider, int max_divider,
        bool find_max, int step = 1) {
    using namespace dnnl::impl::utils;
    max_divider = nstl::max(1, nstl::min(max_divider, value));
    min_divider = nstl::max(1, nstl::min(min_divider, max_divider));

    auto loss_ratio = [](int total, int chunk) {
        return float(rnd_up(total, chunk) - total) / rnd_up(total, chunk);
    };

    float min_loss = FLT_MAX;
    int x_divider = max_divider;
    for (int divider = max_divider; divider >= min_divider; divider -= step) {
        const float loss = loss_ratio(value, divider);
        if ((find_max && loss < min_loss) || (!find_max && loss <= min_loss)) {
            min_loss = loss;
            x_divider = divider;
        }
    }
    return x_divider;
}

typedef jit_1x1_conv_conf_t jcp_t;

inline bool is_bcast_layout_nxc(const jcp_t &jcp) {
    switch (jcp.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
        case prop_kind::backward_weights:
            return utils::one_of(jcp.src_tag, format_tag::ndhwc,
                    format_tag::nhwc, format_tag::nwc);
        case prop_kind::backward_data:
            return utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                    format_tag::nhwc, format_tag::nwc);
        default: assert(!"invalid prop_kind"); return false;
    }
}

inline bool is_load_layout_nxc(const jcp_t &jcp) {
    return jcp.prop_kind == prop_kind::backward_weights
            && utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                    format_tag::nwc);
}

inline bool is_out_layout_nxc(const jcp_t &jcp) {
    switch (jcp.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            return utils::one_of(jcp.dst_tag, format_tag::ndhwc,
                    format_tag::nhwc, format_tag::nwc);
        case prop_kind::backward_data:
            return utils::one_of(jcp.src_tag, format_tag::ndhwc,
                    format_tag::nhwc, format_tag::nwc);
        case prop_kind::backward_weights: return false;
        default: assert(!"invalid prop_kind"); return false;
    }
}

inline size_t get_bcast_u_offset(const jcp_t &jcp) {
    return is_bcast_layout_nxc(jcp) ? jcp.ic : jcp.ic_block;
}

inline size_t get_bcast_j_offset(const jcp_t &jcp) {
    return is_bcast_layout_nxc(jcp) ? jcp.reduce_dim : jcp.reduce_loop_unroll;
}

inline size_t get_bcast_offset(const jcp_t &jcp, int u, int j) {
    size_t offset;
    if (utils::one_of(jcp.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference, prop_kind::backward_data)) {
        assert(jcp.reduce_loop_unroll == jcp.reduce_block);
        if (is_bcast_layout_nxc(jcp) || u != jcp.reduce_loop_unroll) {
            offset = j * get_bcast_j_offset(jcp) + u;
        } else {
            offset = (jcp.bcast_dim + j) * get_bcast_j_offset(jcp);
        }
    } else {
        offset = u * get_bcast_u_offset(jcp) + j;
    }
    return sizeof(float) * offset;
}

inline size_t get_load_u_offset(const jcp_t &jcp) {
    return is_load_layout_nxc(jcp) ? jcp.oc : jcp.oc_block;
}

inline size_t get_load_i_offset(const jcp_t &jcp) {
    return is_load_layout_nxc(jcp) ? jcp.oc_block : jcp.os;
}

inline size_t get_load_bwd_w_offset(const jcp_t &jcp, int i, int u0) {
    if (is_load_layout_nxc(jcp)) {
        return i * get_load_i_offset(jcp) + u0 * get_load_u_offset(jcp);
    } else {
        return (i * get_load_i_offset(jcp) + u0) * get_load_u_offset(jcp);
    }
}

inline size_t get_output_i_offset(const jcp_t &jcp) {
    if (is_out_layout_nxc(jcp)) {
        return jcp.load_block;
    } else {
        return (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) * jcp.load_block;
    }
}

inline size_t get_output_j_offset(const jcp_t &jcp) {
    return is_out_layout_nxc(jcp) ? jcp.load_dim : jcp.load_block;
}

inline size_t get_load_loop_output_fwd_offset(
        const jcp_t &jcp, int load_loop_blk) {
    size_t offset = load_loop_blk * jcp.oc_block * sizeof(float);
    if (!is_out_layout_nxc(jcp)) {
        offset *= jcp.with_dw_conv ? jcp.ow : jcp.os;
    }
    return offset;
}

inline size_t get_load_loop_output_bwd_d_offset(
        const jcp_t &jcp, int load_loop_blk) {
    size_t offset = load_loop_blk * jcp.ic_block * sizeof(float);
    if (!is_out_layout_nxc(jcp)) { offset *= jcp.os; }
    return offset;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
