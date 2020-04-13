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

#ifndef JIT_UNI_1X1_CONV_UTILS_HPP
#define JIT_UNI_1X1_CONV_UTILS_HPP

#include "common/convolution_pd.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/primitive_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

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
                    format_tag::nCw8c, format_tag::nCw16c)
            : memory_desc_wrapper(src_d).matches_one_of_tag(
                    format_tag::nChw8c, format_tag::nChw16c);
    if (dat_tag == format_tag::undef) return;

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
}

template <typename conv_pd_t>
inline void rtus_prepare_space_info(conv_pd_t *self,
        memory_tracking::registrar_t &scratchpad, int max_threads) {
    if (!self->rtus_.reduce_src_) return;
    const auto &jcp = self->jcp_;

    const size_t factor = utils::pick_by_prop_kind(self->desc()->prop_kind,
            jcp.nb_reduce, jcp.nb_load_blocking_max, jcp.nb_bcast_blocking);
    size_t typesize
            = types::data_type_size(self->invariant_src_md()->data_type);

    self->rtus_.space_per_thread_ = factor * jcp.is * jcp.ic_block;
    scratchpad.book(memory_tracking::names::key_conv_rtus_space,
            typesize * max_threads * self->rtus_.space_per_thread_);
}

template <cpu_isa_t isa>
struct rtus_driver_t : public jit_generator {

    struct call_params_t {
        const void *ws; /* reduced image (w/ strides = 1) */
        const void *src; /* source image (w/ non-unit strides) */
        size_t icb;
        size_t os;
        size_t iw_start;
    };

    void (*ker_)(const call_params_t *p);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(rtus_driver_t)

    Xbyak::Reg64 reg_ws = abi_param1;
    Xbyak::Reg64 reg_src = abi_not_param1;
    Xbyak::Reg64 reg_icb = rdx;
    Xbyak::Reg64 reg_os = r11;
    Xbyak::Reg64 reg_iw_start = r8;

    Xbyak::Reg64 reg_cur_os = rax;
    Xbyak::Reg64 reg_cur_iw = r9;
    Xbyak::Reg64 reg_cur_src = r10;

    int iw_, stride_w_;
    int src_step_h_, src_step_icb_, ws_step_icb_, vlen_, vlen_shift_;
    bool src_to_ws_;
    size_t typesize_;

    Xbyak::Xmm reg_zero;
    Xbyak::Xmm reg_v;

    rtus_driver_t(int iw, int stride_w, int src_step_h, int src_step_icb,
            int ws_step_icb, bool src_to_ws, size_t typesize)
        : iw_(iw)
        , stride_w_(stride_w)
        , src_step_h_(src_step_h)
        , src_step_icb_(src_step_icb)
        , ws_step_icb_(ws_step_icb)
        , src_to_ws_(src_to_ws)
        , typesize_(typesize) {
        using namespace Xbyak;

        /*FIXME: derive Vmm type on compile time.
         * changing register type  on runtime
         * seems dangerous,and some xbyak functions might
         * fail to work on reg_v, reg_zero because of this
         * data_type change, e.g. uni_vpxor doen't
         * work on reg_zero now*/
        auto Vmm = [=](int idx, int typesize) {
            Xmm res;
            switch (isa) {
                case avx2:
                    switch (typesize) {
                        case 4: res = Ymm(idx); break;
                        case 2: res = Xmm(idx); break;
                        default:
                            assert(!"Not supported typesize");
                            res = Ymm(idx);
                    }
                    break;
                case avx512_common:
                case avx512_core:
                case avx512_mic:
                    switch (typesize) {
                        case 4: res = Zmm(idx); break;
                        case 2: res = Ymm(idx); break;
                        case 1: res = Xmm(idx); break;
                        default:
                            assert(!"Not supported typesize");
                            res = Zmm(idx);
                    }
            }
            return res;
        };

        reg_zero = Vmm(0, typesize);
        reg_v = Vmm(1, typesize);

        vlen_ = reg_v.getBit() / 8;
        vlen_shift_ = 0;

        int tvlen = vlen_;
        while (tvlen > 1) {
            tvlen /= 2;
            vlen_shift_++;
        }
        generate();
    }

    void loop_is() {
        using namespace Xbyak;

        mov(reg_cur_src, reg_src);
        mov(reg_cur_iw, reg_iw_start);
        mov(reg_cur_os, reg_os);

        Label is_loop;
        L(is_loop);

        if (src_to_ws_) {
            vmovups(reg_v, ptr[reg_cur_src]);
            vmovups(ptr[reg_ws], reg_v);
        } else {
            vmovups(reg_v, ptr[reg_ws]);
            vmovups(ptr[reg_cur_src], reg_v);
            for (int w = 1; w < stride_w_; ++w)
                vmovups(ptr[reg_cur_src + w * vlen_], reg_zero);
        }

        add(reg_ws, vlen_);
        add(reg_cur_src, stride_w_ * vlen_);

        // for 1d or stride_h=1 convolutions the loop over h should be skipped
        if (!(src_step_icb_ == iw_ || src_step_h_ == iw_)) {
            Label skip_h_step;
            add(reg_cur_iw, stride_w_);
            cmp(reg_cur_iw, iw_);
            jl(skip_h_step);

            if (src_to_ws_) {
                add(reg_cur_src, (src_step_h_ - iw_) * vlen_);
            } else {
                Xbyak::Reg64 reg_cur_src_fin = reg_cur_iw; /* just reuse */
                mov(reg_cur_src_fin, reg_cur_src);
                add(reg_cur_src_fin, (src_step_h_ - iw_) * vlen_);
                Label ih_loop;
                L(ih_loop);

                for (int w = 0; w < stride_w_; ++w)
                    vmovups(ptr[reg_cur_src + w * vlen_], reg_zero);

                add(reg_cur_src, stride_w_ * vlen_);
                cmp(reg_cur_src, reg_cur_src_fin);
                jl(ih_loop);
            }
            xor_(reg_cur_iw, reg_cur_iw);
            L(skip_h_step);
        }

        sub(reg_cur_os, vlen_);
        jnz(is_loop);

        /* restore dst */
        sub(reg_ws, reg_os);
    }

    void generate() {
        using namespace Xbyak;
        assert(isa == avx2 || isa == avx512_common || isa == avx512_core
                || isa == avx512_mic);

#if defined(_WIN32)
        assert(reg_src == abi_not_param1 && abi_not_param1 == rdi);
        push(rdi);
#endif

#define READ_PARAM(what) \
    mov(reg_##what, ptr[abi_param1 + offsetof(call_params_t, what)])
        READ_PARAM(src);
        READ_PARAM(icb);
        READ_PARAM(os);
        READ_PARAM(iw_start);

        assert(reg_ws == abi_param1);
        READ_PARAM(ws); /* reg_ws should always be read the last */
#undef READ_PARAM

        shl(reg_os, vlen_shift_);

        if (!src_to_ws_) {
            switch (reg_zero.getBit() / 8) {
                case 16 /*xmm*/: uni_vpxor(reg_zero, reg_zero, reg_zero); break;
                case 32 /*ymm*/: {
                    Xbyak::Ymm ymm_z(reg_zero.getIdx());
                    uni_vpxor(ymm_z, ymm_z, ymm_z);
                    break;
                }
                case 64 /*zmm*/: {
                    Xbyak::Zmm zmm_z(reg_zero.getIdx());
                    uni_vpxor(zmm_z, zmm_z, zmm_z);
                    break;
                }
                default: assert(!"rtus kernel failure");
            }
        }

        Label icb_loop;
        L(icb_loop);

        loop_is();

        add(reg_ws, ws_step_icb_ * vlen_);
        add(reg_src, src_step_icb_ * vlen_);

        dec(reg_icb);
        jnz(icb_loop, T_NEAR);

#if defined(_WIN32)
        pop(rdi);
#endif

        uni_vzeroupper();
        ret();
        this->ker_ = reinterpret_cast<decltype(ker_)>(
                const_cast<uint8_t *>(this->getCode()));
    }
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

    const int src_step_h = stride_h * iw;
    const int src_step_icb = ih * iw;
    const int ws_step_icb = conf.jcp_.is;
    const bool src_to_ws = !is_bwd_data;
    const size_t typesize
            = types::data_type_size(self->pd()->invariant_src_md()->data_type);

    self->rtus_driver_ = new rtus_driver_t<isa>(iw, stride_w, src_step_h,
            src_step_icb, ws_step_icb, src_to_ws, typesize);
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

inline status_t get_depthwise_conv_desc(convolution_desc_t &cd_dw,
        const memory_desc_t &src_dw_md, const primitive_attr_t &attr_1x1,
        primitive_attr_t &attr_dw, int dw_po_index) {

    const memory_desc_wrapper src_dw_d(src_dw_md);
    const int ndims = src_dw_d.ndims();
    if (ndims != 4) return status::unimplemented;

    if (dw_po_index == -1 || dw_po_index >= attr_1x1.post_ops_.len_
            || !attr_1x1.post_ops_.entry_[dw_po_index].is_convolution())
        return status::invalid_arguments;

    // Create new attributes with scales from depthwise post-op and copy
    // post-ops after depthwise post-op.
    auto &dw_po = attr_1x1.post_ops_.entry_[dw_po_index].depthwise_conv;
    if (utils::one_of(
                dw_po.dst_dt, data_type::u8, data_type::s8, data_type::s32)
            && dw_po.count) {
        CHECK(attr_dw.output_scales_.set(
                dw_po.count, dw_po.mask, dw_po.scales));
    }

    auto &len = attr_dw.post_ops_.len_;
    for (int i = dw_po_index + 1; i < attr_1x1.post_ops_.len_; ++i) {
        attr_dw.post_ops_.entry_[len++] = attr_1x1.post_ops_.entry_[i];
    }

    attr_dw.scratchpad_mode_ = attr_1x1.scratchpad_mode_;

    const bool with_bias = dw_po.bias_dt != data_type::undef;

    const auto n = src_dw_d.dims()[0];
    const auto oc = src_dw_d.dims()[1];
    const auto g = src_dw_d.dims()[1];
    const auto ih = src_dw_d.dims()[ndims - 2];
    const auto iw = src_dw_d.dims()[ndims - 1];
    const auto stride = dw_po.stride;

    const dims_t weights_tz = {g, 1, 1, 3, 3};

    const dims_t dst_tz
            = {n, oc, utils::div_up(ih, stride), utils::div_up(iw, stride)};

    const dims_t bias_tz = {oc};
    const dims_t pad_tz = {1, 1};
    const dims_t stride_tz = {stride, stride};

    memory_desc_t src_md, weights_md, bias_md, dst_md;

    dnnl_memory_desc_init_by_tag(&src_md, ndims, src_dw_md.dims,
            src_dw_md.data_type, format_tag::any);

    dnnl_memory_desc_init_by_tag(
            &weights_md, ndims + 1, weights_tz, dw_po.wei_dt, format_tag::any);

    if (with_bias)
        dnnl_memory_desc_init_by_tag(
                &bias_md, 1, bias_tz, dw_po.bias_dt, format_tag::a);

    dnnl_memory_desc_init_by_tag(
            &dst_md, ndims, dst_tz, dw_po.dst_dt, format_tag::any);

    CHECK(conv_desc_init(&cd_dw, prop_kind::forward_inference,
            alg_kind::convolution_auto, &src_md, &weights_md,
            with_bias ? &bias_md : nullptr, &dst_md, stride_tz, nullptr, pad_tz,
            pad_tz));

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
