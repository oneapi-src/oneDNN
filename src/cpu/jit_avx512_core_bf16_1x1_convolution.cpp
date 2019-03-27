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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx512_core_bf16_1x1_convolution.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

#define data_blk_off(f, n, c, h, w) \
    ((ndims == 3) \
    ? (f).blk_off(n, c, w) \
    : (f).blk_off(n, c, h, w))

namespace {
template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end,
    T nx, T &nx_start, T &nx_end, T nx_divider)
{
    const T grp_size = utils::div_up(nthr, nx_divider);
    const T grp_count = utils::div_up(nthr, grp_size);

    T grp = ithr / grp_size;
    T grp_ithr = ithr % grp_size;
    T grp_nthr = grp_size;
    T first_grps = nthr % grp_count;
    if (first_grps > 0 && grp >= first_grps) {
        ithr -= first_grps * grp_size;
        grp_nthr--;
        grp = ithr / grp_nthr + first_grps;
        grp_ithr = ithr % grp_nthr;
    }
    balance211(nx, grp_count, grp, nx_start, nx_end);
    balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}
}
/* convolution forward */

template <data_type_t dst_type>
void _jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>::execute_forward()
const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights =
        reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const float *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    auto scratchpad = this->scratchpad();

    const auto &jcp = kernel_->jcp;
    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad.template get<float>(
                memory_tracking::names::key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;
    }

    parallel(0, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, dst, scratchpad);
    });

    if (pd()->wants_zero_pad_dst())
        output_memory_primitive(0)->zero_pad();
}

template <data_type_t dst_type>
void _jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>::execute_forward_thr(
            const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const float *bias, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad)
const {
    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const int ndims = src_d.ndims();
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];
    const int pad_t = (ndims == 3) ? 0 : pd()->desc()->padding[0][0];
    const int pad_l = pd()->desc()->padding[0][ndims - 3];

    const auto &jcp = kernel_->jcp;
    auto rtus_space = scratchpad.template get<src_data_t>(
            memory_tracking::names::key_conv_rtus_space);
    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();
    const int nb_oc = jcp.nb_load;
    const int nb_ic = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;

    int bcast_start{0}, bcast_end{0}, ocb_start{0}, ocb_end{0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
        jcp.nb_load, ocb_start, ocb_end, jcp.load_grp_count);

    auto init_bcast = [&](int iwork, int &n, int &g, int &bcast_step,
            int &oh, int &ow, int &ih, int &iw)
    {
        int osb{0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
            jcp.nb_bcast);
        bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                jcp.nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        oh = os / jcp.ow;
        ow = os % jcp.ow;

        ih = nstl::max(oh * stride_h - pad_t, 0);
        iw = nstl::max(ow * stride_w - pad_l, 0);
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os,
            bcast_step * os_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](int ocb, int &load_step)
    {
        load_step = step(jcp.nb_load_blocking, ocb_end - ocb,
            jcp.nb_load_blocking_max);
        p.load_dim = this_block_size(ocb * jcp.oc_block,
            ocb_end * jcp.oc_block, load_step * jcp.oc_block);
    };

    auto init_reduce = [&]()
    {
        p.reduce_dim = this_block_size(0, jcp.ic, jcp.ic);
        rp.icb = p.reduce_dim / jcp.reduce_block;
    };

    auto inner_ker = [&](int ocb, int n, int g, int oh, int ow,
        int ih, int iw)
    {
        const int icb = 0;
        const int _ocb = g * nb_oc + ocb;
        const size_t dst_off = data_blk_off(dst_d, n, _ocb , oh, ow);

        p.output_data = &dst[dst_off];
        p.bias_data = &bias[_ocb * jcp.oc_block];
        p.load_data = &weights[pd()->with_groups()
            ? weights_d.blk_off(g, ocb, icb)
            : weights_d.blk_off(ocb, icb)];

        const int _icb = g * nb_ic + icb;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_
                + _icb * jcp.is * jcp.ic_block;
            if (ocb == ocb_start) {
                rp.src = src + data_blk_off(src_d, n, _icb, ih, iw);
                rtus_driver_->ker_(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src + data_blk_off(src_d, n, _icb, ih, iw);

        kernel_->jit_ker(&p);
    };

    if (jcp.loop_order == loop_rlb) {
        init_reduce();
        int ocb = ocb_start;
        while (ocb < ocb_end) {
            int load_step;
            init_load(ocb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            ocb += load_step;
        }
    } else if (jcp.loop_order == loop_lbr) {
        int ocb = ocb_start;
        while (ocb < ocb_end) {
            int load_step;
            init_load(ocb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                init_reduce();
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            ocb += load_step;
        }
    } else if (jcp.loop_order == loop_rbl) {
        init_reduce();
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, load_step);
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                ocb += load_step;
            }
            iwork += bcast_step;
        }
    } else if (jcp.loop_order == loop_blr) {
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, load_step);
                init_reduce();
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                ocb += load_step;
            }
            iwork += bcast_step;
        }
    } else {
        assert(!"unsupported loop order");
    }
}


template struct _jit_avx512_core_bf16_1x1_convolution_fwd_t<data_type::f32>;
template struct _jit_avx512_core_bf16_1x1_convolution_fwd_t<data_type::bf16>;

template <data_type_t diff_src_type>
void _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<diff_src_type>::execute_backward_data()
const {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>
        (this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>
        (this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t *>(this->memory());
    auto scratchpad = this->scratchpad();
    parallel(0, [&](const int ithr, const int nthr) {
        execute_backward_data_thr(ithr, nthr, diff_dst, weights, diff_src,
            scratchpad);
    });
}

template <data_type_t diff_src_type>
void _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<diff_src_type>::execute_backward_data_thr(
        const int ithr, const int nthr,
        const diff_dst_data_t *diff_dst, const wei_data_t *weights,
        diff_src_data_t *diff_src,
        const memory_tracking::grantor_t &scratchpad)
const {

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const int ndims = diff_src_d.ndims();
    const auto &jcp = kernel_->jcp;
    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto rtus_space = scratchpad.template get<diff_src_data_t>(
            memory_tracking::names::key_conv_rtus_space);

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();
    const int nb_ic = jcp.nb_load;
    const int nb_oc = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;

    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];
    const int pad_t = (ndims == 3) ? 0 : pd()->desc()->padding[0][0];
    const int pad_l = pd()->desc()->padding[0][ndims - 3];

    int bcast_start{0}, bcast_end{0}, icb_start{0}, icb_end{0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
        jcp.nb_load, icb_start, icb_end, jcp.load_grp_count);

    auto init_bcast = [&](const int icb, int iwork, int &n, int &g, int &bcast_step,
            int &oh, int &ow, int &ih, int &iw)
    {
        int osb{0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
            jcp.nb_bcast);
        bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                jcp.nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        p.bcast_dim = this_block_size(os, jcp.os,
            bcast_step * os_block);
        rp.os = p.bcast_dim;

        oh = os / jcp.ow;
        ow = os % jcp.ow;
        ih = nstl::max(oh * stride_h - pad_t, 0);
        iw = nstl::max(ow * stride_w - pad_l, 0);
        rp.iw_start = iw;
    };

    auto init_load = [&](int icb, int &load_step)
    {
        load_step = step(jcp.nb_load_blocking, icb_end - icb,
            jcp.nb_load_blocking_max);
        p.load_dim = this_block_size(icb * jcp.ic_block,
            icb_end * jcp.ic_block, load_step * jcp.ic_block);
        rp.icb = p.load_dim / jcp.ic_block;
    };

    auto init_reduce = [&]()
    {
        p.reduce_dim = this_block_size(0, jcp.oc, jcp.oc);
    };

    auto inner_ker = [&](int icb, int n, int g, int oh, int ow,
        int ih, int iw)
    {
        const int ocb = 0;
        const int _icb = g * nb_ic + icb;
        const size_t diff_src_off = data_blk_off(diff_src_d, n, _icb , ih, iw);

        rp.src = diff_src + diff_src_off;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_;
            p.output_data = rp.ws;
        } else
            p.output_data = rp.src;
        p.load_data = &weights[pd()->with_groups()
            ? weights_d.blk_off(g, ocb, icb)
            : weights_d.blk_off(ocb, icb)];

        const int _ocb = g * nb_oc + ocb;
        p.bcast_data = diff_dst + data_blk_off(diff_dst_d, n, _ocb, oh, ow);

        kernel_->jit_ker(&p);
        if (pd()->rtus_.reduce_src_)
            rtus_driver_->ker_(&rp);
    };

    if (jcp.loop_order == loop_rlb) {
        init_reduce();
        int icb = icb_start;
        while (icb < icb_end) {
            int load_step;
            init_load(icb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n{0}, g{0}, bcast_step, oh, ow, ih, iw;
                init_bcast(0, iwork, n, g, bcast_step, oh, ow, ih, iw);
                inner_ker(icb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            icb += load_step;
        }
        //XXX: this is the loop order for strided
    } else if (jcp.loop_order == loop_lbr) {
        int icb = icb_start;
        while (icb < icb_end) {
            int load_step;
            init_load(icb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(icb, iwork, n, g, bcast_step, oh, ow, ih, iw);
                init_reduce();
                inner_ker(icb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            icb += load_step;
        }
    } else if (jcp.loop_order == loop_rbl) {
        init_reduce();
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(0, iwork, n, g, bcast_step, oh, ow, ih, iw);
            int icb = icb_start;
            while (icb < icb_end) {
                int load_step;
                init_load(icb, load_step);
                inner_ker(icb, n, g, oh, ow, ih, iw);
                icb += load_step;
            }
            iwork += bcast_step;
        }
    } else if (jcp.loop_order == loop_blr) {
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(0, iwork, n, g, bcast_step, oh, ow, ih, iw);
            int icb = icb_start;
            while (icb < icb_end) {
                int load_step;
                init_load(icb, load_step);
                init_reduce();
                inner_ker(icb, n, g, oh, ow, ih, iw);
                icb += load_step;
            }
            iwork += bcast_step;
        }
    } else {
        assert(!"unsupported loop order");
    }

}

template struct _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<data_type::f32>;
template struct _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<data_type::bf16>;
}
}
}
