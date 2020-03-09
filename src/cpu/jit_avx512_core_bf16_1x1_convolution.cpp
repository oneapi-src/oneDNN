/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "dnnl_types.h"

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "jit_avx512_core_bf16_1x1_convolution.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::prop_kind;

#define data_blk_off(f, n, c, d, h, w) \
    ((ndims == 3) ? (f).blk_off(n, c, w) \
                  : ((ndims == 4) ? (f).blk_off(n, c, h, w) \
                                  : (f).blk_off(n, c, d, h, w)))

namespace {
/*TODO: investigate why common balance2D defined in dnnl_thread.hpp
 * not used here ?*/
template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end, T nx, T &nx_start,
        T &nx_end, T nx_divider) {
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
} // namespace

/* convolution forward */
template <data_type_t dst_type>
void jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(const char *, DNNL_ARG_DST);
    auto weights_dw = CTX_IN_MEM(
            const dw_wei_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);

    auto scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = kernel_->jcp;
    if (pd()->wants_padded_bias()) {
        const size_t bia_dt_size = pd()->jcp_.typesize_bia;
        auto padded_bias = scratchpad.template get<char>(key_conv_padded_bias);
        utils::array_copy(
                padded_bias, bias, bia_dt_size * jcp.oc_without_padding);
        utils::array_set(padded_bias + bia_dt_size * jcp.oc_without_padding,
                0.f, bia_dt_size * (jcp.oc - jcp.oc_without_padding));
        bias = padded_bias;
    }

    float *bias_dw = nullptr;
    auto &jcp_dw = pd()->jcp_dw_;
    if (pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS)->data_type
            == data_type::bf16) {
        memory_tracking::grantor_t dw_scratchpad(
                scratchpad, memory_tracking::names::prefix_fusion);
        auto bias_in = CTX_IN_MEM(
                const src_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
        bias_dw = dw_scratchpad.template get<float>(
                key_conv_bias_bf16_convert_wsp);
        cvt_bfloat16_to_float(bias_dw, bias_in, jcp_dw.oc_without_padding);
        utils::array_set(bias_dw + jcp_dw.oc_without_padding, 0.f,
                jcp_dw.oc - jcp_dw.oc_without_padding);
    } else {
        auto bias_in = CTX_IN_MEM(
                const float *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
        bias_dw = const_cast<float *>(bias_in);
    }

    parallel(0, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, weights_dw, bias_dw,
                dst, scratchpad);
    });

    if (pd()->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad();
}

template <data_type_t dst_type>
void jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>::execute_forward_thr(
        const int ithr, const int nthr, const src_data_t *src,
        const wei_data_t *weights, const char *bias,
        const dw_wei_data_t *weights_dw, const float *bias_dw, const char *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper dw_weights_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS));
    const memory_desc_wrapper dw_bias_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS));

    const auto &jcp = kernel_->jcp;
    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<src_data_t>(key_conv_rtus_space)
            : NULL;
    float *store_buffer = scratchpad.template get<float>(key_conv_store_wsp);

    const int ndims = src_d.ndims();
    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();

    const int nb_oc = jcp.nb_load;
    const int nb_ic = jcp.nb_reduce;
    const int nb_ic_blocking = jcp.nb_reduce_blocking;

    // override some constants for fused dw_conv
    const int os_block = jcp.with_dw_conv ? jcp.ow : jcp.bcast_block;
    const int nb_bcast = jcp.with_dw_conv ? jcp.oh : jcp.nb_bcast;
    const int nb_bcast_blocking = jcp.with_dw_conv ? 1 : jcp.nb_bcast_blocking;
    const int nb_bcast_blocking_max
            = jcp.with_dw_conv ? 1 : jcp.nb_bcast_blocking_max;
    const int nb_load_blocking = jcp.nb_load_blocking;
    const int nb_load_blocking_max = jcp.with_dw_conv
            ? jcp.nb_load_blocking
            : jcp.nb_load_blocking_max;

    // Begin: declare Variables needed for dw conv.
    auto &jcp_dw = pd()->jcp_dw_;
    dst_data_t *pbuf; //bf16->bf16 fusion
    size_t row_offset;
    const int nb_buffer = jcp.nb_load_blocking;
    std::vector<decltype(pbuf)> addrs;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto init_bcast = [&](int iwork, int bcast_end, int &n, int &g,
                              int &bcast_step, int &od, int &oh, int &ow,
                              int &id, int &ih, int &iw) {
        int osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, nb_bcast);
        bcast_step = step(
                nb_bcast_blocking, nb_bcast - osb, nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        od = os / (jcp.oh * jcp.ow);
        int os_2d = os % (jcp.oh * jcp.ow);
        oh = os_2d / jcp.ow;
        ow = os_2d % jcp.ow;

        id = od * stride_d;
        ih = oh * stride_h;
        iw = ow * stride_w;
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](int ocb, int ocb_end, int &load_step) {
        load_step = step(nb_load_blocking, ocb_end - ocb, nb_load_blocking_max);
        p.load_dim = this_block_size(ocb * jcp.oc_block, ocb_end * jcp.oc_block,
                load_step * jcp.oc_block);
    };

    auto init_reduce = [&](int icb) {
        const int nb_ic_blocking_step
                = nstl::min(icb + nb_ic_blocking, nb_ic) - icb;
        p.first_last_flag = 0 | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                | (icb + nb_ic_blocking_step >= nb_ic ? FLAG_REDUCE_LAST : 0);

        p.reduce_dim = this_block_size(
                icb * jcp.ic_block, jcp.ic, nb_ic_blocking_step * jcp.ic_block);
        rp.icb = p.reduce_dim / jcp.reduce_block;
    };

    auto ker_1x1 = [&](int ocb, int ocb_start, int icb, int n, int g, int od,
                           int oh, int ow, int id, int ih, int iw) {
        const int _ocb = g * nb_oc + ocb;

        void *output_data = jcp.with_dw_conv
                ? (void *)(pbuf + (oh % jcp_dw.kh) * row_offset)
                : (void *)(&dst[data_blk_off(dst_d, n, _ocb, od, oh, ow)
                        * dst_d.data_type_size()]);
        p.output_data = output_data;

        p.bias_data = &bias[_ocb * jcp.oc_block * jcp.typesize_bia];
        p.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];

        const int _icb = g * nb_ic + icb;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_
                    + _icb * jcp.is * jcp.ic_block;
            if (ocb == ocb_start) {
                rp.src = src + data_blk_off(src_d, n, _icb, id, ih, iw);
                rtus_driver_->ker_(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src + data_blk_off(src_d, n, _icb, id, ih, iw);

        const size_t grp_count = utils::div_up(
                jcp.nthr, utils::div_up(jcp.nthr, jcp.load_grp_count));
        const size_t max_load_per_thread
                = rnd_up((jcp.load_dim / grp_count), jcp.load_block);
        const size_t str_size = jcp.bcast_dim * max_load_per_thread;
        p.store_buffer = store_buffer + ithr * str_size
                + data_blk_off(dst_d, 0, 0, od, oh, ow);

        kernel_->jit_ker(&p);
    };

    auto conv_1x1 = [&](int bcast_start, int bcast_end, int ocb_start,
                            int ocb_end) {
        if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;
        if (jcp.loop_order == loop_lbr) {
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, ocb_end, load_step);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n, g, bcast_step, od, oh, ow, id, ih, iw;
                    init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow,
                            id, ih, iw);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                    }
                    iwork += bcast_step;
                }
                ocb += load_step;
            }
        } else if (jcp.loop_order == loop_blr) {
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, od, oh, ow, id, ih, iw;
                init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow, id,
                        ih, iw);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, ocb_end, load_step);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                    }
                    ocb += load_step;
                }
                iwork += bcast_step;
            }
        } else {
            assert(!"unsupported loop order");
        }
    };

    auto ker_dw = [&](int n, int ocb_start, int load_step, int &dw_oh) {
        int oh_1x1 = nstl::max(dw_oh * jcp_dw.stride_h - jcp_dw.t_pad, 0);

        for (int i = 0; i < jcp_dw.kh; ++i)
            addrs[i] = pbuf + ((oh_1x1++) % jcp_dw.kh) * row_offset;

        const auto ocb_end = ocb_start + load_step;
        const auto wch_stride
                = jcp_dw.iw * jcp_dw.nb_ch_blocking * jcp_dw.ch_block;

        const int dil_h = jcp_dw.dilate_h + 1;
        const int str_h = jcp_dw.stride_h;
        const int ch_num = jcp_dw.nb_ch_blocking;

        for (int ch = ocb_start; ch < ocb_end; ch += jcp_dw.nb_ch_blocking) {

            const int i_t_overflow
                    = nstl::max(0, (int)(jcp_dw.t_pad - dw_oh * str_h));
            const int i_b_overflow
                    = nstl::max(jcp_dw.ih,
                              (int)(dw_oh * str_h + (jcp_dw.kh - 1) * dil_h
                                      - jcp_dw.t_pad + 1))
                    - jcp_dw.ih;

            const int kh = div_up(i_t_overflow, dil_h);
            const int kh_padding = jcp_dw.kh - div_up(i_t_overflow, dil_h)
                    - div_up(i_b_overflow, dil_h);

            const int ow = 0;
            const int kw = 0;
            jit_conv_call_s par_conv_dw;

            par_conv_dw.src = addrs.data();
            par_conv_dw.dst = &dst[dst_d.blk_off(n, ch, dw_oh, ow)
                    * dst_d.data_type_size()];

            par_conv_dw.filt
                    = &weights_dw[dw_weights_d.blk_off(ch, 0, 0, kh, kw)];
            if (bias)
                par_conv_dw.bias
                        = &bias_dw[dw_bias_d.blk_off(ch * jcp_dw.ch_block)];

            par_conv_dw.kh_padding = (size_t)nstl::max(0, kh_padding);

            par_conv_dw.ch_blocks = nstl::min(ch + ch_num, jcp_dw.nb_ch) - ch;

            kernel_dw_->jit_ker(&par_conv_dw);

            for (int i = 0; i < jcp_dw.kh; ++i)
                addrs[i] += wch_stride;
        }
    };

    auto conv_dw = [&]() {
        // Set variables
        memory_tracking::grantor_t dw_scratchpad(
                scratchpad, memory_tracking::names::prefix_fusion);
        const auto dw_conv_buffer
                = dw_scratchpad.get<dst_data_t>(key_fusion_inout_buffer);

        const auto dw_conv_buffer_size_
                = jcp_dw.kh * jcp.ow * nb_buffer * jcp.oc_block;
        pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;
        row_offset = dw_conv_buffer_size_ / jcp_dw.kh;
        addrs.resize(jcp_dw.kh);

        int bcast_start {0}, bcast_end {0}, ocb_start, ocb_end;
        balance2D(nthr, ithr, jcp.mb * jcp.ngroups * jcp_dw.oh, bcast_start,
                bcast_end, nb_oc, ocb_start, ocb_end, jcp.load_grp_count);

        while (ocb_start < ocb_end) {
            int load_step;
            init_load(ocb_start, ocb_end, load_step);

            int oh_1x1 = 0;
            auto bcast_iter = bcast_start;
            while (bcast_iter < bcast_end) {
                int n, g, oh_dw;
                nd_iterator_init(bcast_iter, n, jcp.mb, g, jcp.ngroups, oh_dw,
                        jcp_dw.oh);
                if (oh_dw == 0) oh_1x1 = 0; // Reset over mb boundary
                const int oh_1x1_range = oh_dw * jcp_dw.stride_h - jcp_dw.t_pad;
                const int oh_1x1_begin = nstl::max(oh_1x1_range, 0);
                const int oh_1x1_end
                        = nstl::min(oh_1x1_range + jcp_dw.kh, jcp.oh);
                oh_1x1 = nstl::max(
                        oh_1x1_begin, oh_1x1); // Skip rows computed previously

                // dw_spatial to 1x1 spatial conversion. if jcp.oh != jcp_dw.oh
                const int bcast_start_1x1
                        = n * jcp.ngroups * jcp.oh + g * jcp.oh + oh_1x1;
                const int bcast_end_1x1 = bcast_start_1x1 - oh_1x1 + oh_1x1_end;

                conv_1x1(bcast_start_1x1, bcast_end_1x1, ocb_start,
                        ocb_start + load_step);
                oh_1x1 = oh_1x1_end;
                ker_dw(n, g * nb_oc + ocb_start, load_step, oh_dw);

                bcast_iter += nb_bcast_blocking;
            }
            ocb_start += load_step;
        }
    };

    if (jcp.with_dw_conv) {
        conv_dw();
    } else {
        const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
        int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
                ocb_start, ocb_end, jcp.load_grp_count);

        conv_1x1(bcast_start, bcast_end, ocb_start, ocb_end);
    }
}

template struct jit_avx512_core_bf16_1x1_convolution_fwd_t<data_type::f32>;
template struct jit_avx512_core_bf16_1x1_convolution_fwd_t<data_type::bf16>;

template <data_type_t diff_src_type>
void jit_avx512_core_bf16_1x1_convolution_bwd_data_t<
        diff_src_type>::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);
    auto scratchpad = ctx.get_scratchpad_grantor();
    parallel(0, [&](const int ithr, const int nthr) {
        execute_backward_data_thr(
                ithr, nthr, diff_dst, weights, diff_src, scratchpad);
    });
}

template <data_type_t diff_src_type>
void jit_avx512_core_bf16_1x1_convolution_bwd_data_t<
        diff_src_type>::execute_backward_data_thr(const int ithr,
        const int nthr, const diff_dst_data_t *diff_dst,
        const wei_data_t *weights, diff_src_data_t *diff_src,
        const memory_tracking::grantor_t &scratchpad) const {

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const auto &jcp = kernel_->jcp;

    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.template get<diff_src_data_t>(key_conv_rtus_space)
            : NULL;
    float *store_buffer = scratchpad.template get<float>(key_conv_store_wsp);
    const int ndims = diff_src_d.ndims();
    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();
    const int nb_ic = jcp.nb_load;
    const int nb_oc = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;
    const int nb_oc_blocking = jcp.nb_reduce_blocking;

    int bcast_start {0}, bcast_end {0}, icb_start {0}, icb_end {0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
            icb_start, icb_end, jcp.load_grp_count);

    auto init_bcast = [&](int iwork, int &n, int &g, int &bcast_step, int &od,
                              int &oh, int &ow, int &id, int &ih, int &iw) {
        int osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, jcp.nb_bcast);
        bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                jcp.nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        od = os / (jcp.oh * jcp.ow);
        const int os_2d = os % (jcp.oh * jcp.ow);
        oh = os_2d / jcp.ow;
        ow = os_2d % jcp.ow;
        id = od * stride_d;
        ih = oh * stride_h;
        iw = ow * stride_w;
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](int icb, int &load_step) {
        load_step = step(
                jcp.nb_load_blocking, icb_end - icb, jcp.nb_load_blocking_max);
        p.load_dim = this_block_size(icb * jcp.ic_block, icb_end * jcp.ic_block,
                load_step * jcp.ic_block);
        rp.icb = p.load_dim / jcp.ic_block;
    };

    auto init_reduce = [&](int ocb) {
        const int nb_oc_blocking_step
                = nstl::min(ocb + nb_oc_blocking, nb_oc) - ocb;
        p.first_last_flag = 0 | (ocb == 0 ? FLAG_REDUCE_FIRST : 0)
                | (ocb + nb_oc_blocking_step >= nb_oc ? FLAG_REDUCE_LAST : 0);

        p.reduce_dim = this_block_size(
                ocb * jcp.oc_block, jcp.oc, nb_oc_blocking_step * jcp.oc_block);
    };

    auto inner_ker = [&](int icb, int ocb, int n, int g, int od, int oh, int ow,
                             int id, int ih, int iw) {
        const int _icb = g * nb_ic + icb;
        const size_t diff_src_off
                = data_blk_off(diff_src_d, n, _icb, id, ih, iw);

        rp.src = diff_src + diff_src_off;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_;
            p.output_data = rp.ws;
        } else
            p.output_data = rp.src;
        p.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];

        const int _ocb = g * nb_oc + ocb;
        p.bcast_data = diff_dst + data_blk_off(diff_dst_d, n, _ocb, od, oh, ow);

        const size_t grp_count = utils::div_up(
                jcp.nthr, utils::div_up(jcp.nthr, jcp.load_grp_count));
        const size_t max_load_per_thread
                = rnd_up((jcp.load_dim / grp_count), jcp.load_block);
        const size_t str_size = jcp.bcast_dim * max_load_per_thread;
        p.store_buffer = store_buffer + ithr * str_size
                + data_blk_off(diff_src_d, 0, 0, id, ih, iw);
        kernel_->jit_ker(&p);
        if (pd()->rtus_.reduce_src_) rtus_driver_->ker_(&rp);
    };

    if (jcp.loop_order == loop_lbr) {
        int icb = icb_start;
        while (icb < icb_end) {
            int load_step;
            init_load(icb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, od, oh, ow, id, ih, iw;
                init_bcast(iwork, n, g, bcast_step, od, oh, ow, id, ih, iw);
                for (int ocb = 0; ocb < nb_oc; ocb += nb_oc_blocking) {
                    init_reduce(ocb);
                    inner_ker(icb, ocb, n, g, od, oh, ow, id, ih, iw);
                }
                iwork += bcast_step;
            }
            icb += load_step;
        }
    } else {
        assert(!"unsupported loop order");
    }
}

template struct jit_avx512_core_bf16_1x1_convolution_bwd_data_t<data_type::f32>;
template struct jit_avx512_core_bf16_1x1_convolution_bwd_data_t<
        data_type::bf16>;

/* convolution backward wtr weights */

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

template <data_type_t diff_weights_type>
jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<diff_weights_type>::
        jit_avx512_core_bf16_1x1_convolution_bwd_weights_t(const pd_t *apd)
    : primitive_impl_t(apd)
    , kernel_(nullptr)
    , acc_ker_(nullptr)
    , reducer_bias_(nullptr)
    , rtus_driver_(nullptr)
    , tr_reorder_(nullptr) {
    kernel_ = new jit_avx512_core_bf16_1x1_conv_kernel(
            pd()->jcp_, *pd()->attr());

    reducer_bias_ = new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_);
    init_rtus_driver<avx512_common>(this);

    acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();

    if (!pd()->jcp_.uses_permw_transposition)
        tr_reorder_ = new jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t();
}

template <data_type_t diff_weights_type>
void jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<diff_weights_type>::
        execute_backward_weights(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(diff_wei_data_t *, DNNL_ARG_DIFF_WEIGHTS);

    auto scratchpad = ctx.get_scratchpad_grantor();
    float *diff_bias = nullptr;
    if (pd()->jcp_.bia_dt == data_type::bf16) {
        diff_bias = scratchpad.template get<float>(
                key_conv_bias_bf16_convert_wsp);
    } else
        diff_bias = pd()->wants_padded_bias()
                ? scratchpad.template get<float>(key_conv_padded_bias)
                : CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = pd()->jcp_;

    auto rtus_space = scratchpad.template get<src_data_t>(key_conv_rtus_space);
    auto wei_reduction = scratchpad.template get<float>(key_conv_wei_reduction);

    auto tr_src_buffer = !jcp.uses_permw_transposition
            ? scratchpad.template get<src_data_t>(key_conv_tr_src)
            : nullptr;
    auto tr_diff_buffer = !jcp.uses_permw_transposition
            ? scratchpad.template get<diff_dst_data_t>(key_conv_tr_diff_dst)
            : nullptr;
    auto d_dst_f32_buffer
            = scratchpad.template get<float>(key_conv_dst_bf16_convert_wsp);

    const int ndims = src_d.ndims();
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic;

    simple_barrier::ctx_t reduction_barrier;
    if (dnnl_thr_syncable()) simple_barrier::ctx_init(&reduction_barrier);

    const auto reducer_bia_scratchpad
            = memory_tracking::grantor_t(scratchpad, prefix_reducer_bia);
    auto rb = this->reducer_bias_;
    rb->init(reducer_bia_scratchpad);

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int nb_ic = jcp.nb_bcast;
    const int nb_ic_blocking = jcp.nb_bcast_blocking;

    const int nb_oc = jcp.nb_load;
    const int nb_oc_blocking = jcp.nb_load_blocking;

    const int sp_nb = jcp.nb_reduce;
    const int mb_sp_work = jcp.mb * sp_nb;

    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto ker = [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);

        const int ithr_ic_b = ithr % jcp.nthr_ic_b;
        const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
        const int ithr_g = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
        const int ithr_mb = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b / jcp.nthr_g;

        /* reduction dimension */
        int mb_sp_b_start {0}, mb_sp_b_end {0};
        balance211(
                mb_sp_work, jcp.nthr_mb, ithr_mb, mb_sp_b_start, mb_sp_b_end);

        /* independent dimensions */
        int g_start {0}, oc_b_start {0}, ic_b_start {0};
        int g_end {0}, oc_b_end {0}, ic_b_end {0};

        balance211(jcp.ngroups, jcp.nthr_g, ithr_g, g_start, g_end);
        balance211(jcp.nb_load, jcp.nthr_oc_b, ithr_oc_b, oc_b_start, oc_b_end);
        balance211(
                jcp.nb_bcast, jcp.nthr_ic_b, ithr_ic_b, ic_b_start, ic_b_end);

        float *diff_wei;
        if (diff_weights_type == data_type::bf16) {
            diff_wei = wei_reduction + (ithr_mb)*wei_size;
        } else {
            diff_wei = ithr_mb == 0
                    ? (float *)diff_weights
                    : (float *)wei_reduction + (ithr_mb - 1) * wei_size;
        }

        int sp_b_step = 0;
        for (int mb_sp_b = mb_sp_b_start; mb_sp_b < mb_sp_b_end;
                mb_sp_b += sp_b_step) {
            int img {0}, sp_b {0};
            nd_iterator_init(mb_sp_b, img, jcp.mb, sp_b, sp_nb);
            sp_b_step = step(jcp.nb_reduce_blocking,
                    nstl::min(sp_nb - sp_b, mb_sp_b_end - mb_sp_b),
                    jcp.nb_reduce_blocking_max);

            for (int g = g_start; g < g_end; ++g) {
                int load_step = 0;
                int bcast_step = 0;
                for (int ic_b = ic_b_start; ic_b < ic_b_end;
                        ic_b += bcast_step) {
                    bcast_step = step(nb_ic_blocking, ic_b_end - ic_b,
                            jcp.nb_bcast_blocking_max);
                    for (int oc_b = oc_b_start; oc_b < oc_b_end;
                            oc_b += load_step) {
                        load_step = step(nb_oc_blocking, oc_b_end - oc_b,
                                jcp.nb_load_blocking_max);
                        const int _ic_b = g * nb_ic + ic_b;
                        const int _oc_b = g * nb_oc + oc_b;

                        float *store_to;

                        const size_t off
                                = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                        store_to = diff_wei + off;

                        const src_data_t *diff_src
                                = &src[src_d.blk_off(img, _ic_b)];

                        const diff_dst_data_t *pdiff_dst
                                = &diff_dst[diff_dst_d.blk_off(img, _oc_b)];
                        const src_data_t *local_src = diff_src;

                        auto p = jit_1x1_conv_call_s();
                        auto rp = rtus_driver_t<avx512_common>::call_params_t();

                        p.output_stride
                                = jcp.ic * jcp.oc_block * jcp.typesize_out;

                        p.load_dim = load_step * jcp.oc_block;

                        p.bcast_dim = bcast_step * jcp.ic_block;
                        rp.icb = bcast_step;
                        p.output_data = store_to;

                        p.reduce_dim = sp_b_step * jcp.reduce_block;
                        if (!jcp.uses_permw_transposition)
                            p.reduce_dim = nstl::min(p.reduce_dim,
                                    (size_t)jcp.reduce_dim
                                            - sp_b * jcp.reduce_block);

                        rp.os = p.reduce_dim;

                        p.first_last_flag = 0
                                | (mb_sp_b == mb_sp_b_start ? FLAG_REDUCE_FIRST
                                                            : 0);

                        int sp = sp_b * jcp.reduce_block;
                        p.load_data = pdiff_dst + sp * jcp.oc_block;

                        if (pd()->rtus_.reduce_src_) {
                            const int oh = sp / jcp.ow;
                            const int ow = sp % jcp.ow;

                            const int ih = oh * stride_h;
                            const int iw = ow * stride_w;
                            rp.iw_start = iw;

                            rp.ws = rtus_space
                                    + ithr * pd()->rtus_.space_per_thread_
                                    + sp * jcp.ic_block;

                            if (ndims == 3)
                                rp.src = local_src
                                        + iw * src_d.blocking_desc().strides[2];
                            else
                                rp.src = local_src
                                        + ih * src_d.blocking_desc().strides[2]
                                        + iw * src_d.blocking_desc().strides[3];
                            rtus_driver_->ker_(&rp);

                            p.bcast_data = rp.ws;
                        } else
                            p.bcast_data = local_src + sp * jcp.ic_block;
                        if (!jcp.uses_permw_transposition) {
                            bf16_support::jit_call_t ptr;
                            ptr.nelems = p.reduce_dim;
                            int thr_src_block_size = rnd_up(jcp.reduce_dim, 2)
                                    * jcp.ic_block * jcp.nb_bcast_blocking_max;
                            src_data_t *tr_src
                                    = &tr_src_buffer[ithr * thr_src_block_size];
                            for (int bs = 0; bs < bcast_step; bs++) {
                                size_t src_off
                                        = bs * jcp.reduce_dim * jcp.ic_block;
                                size_t src_tr_off = bs
                                        * rnd_up(jcp.reduce_dim, 2)
                                        * jcp.ic_block;
                                src_data_t *curr_inp = &(
                                        (src_data_t *)p.bcast_data)[src_off];
                                src_data_t *curr_out = &tr_src[src_tr_off];
                                ptr.inp = (void *)curr_inp;
                                ptr.out = (void *)curr_out;
                                tr_reorder_->jit_ker(&ptr);
                            }

                            p.bcast_data = (void *)tr_src;

                            int thr_dst_block_size = rnd_up(jcp.reduce_dim, 2)
                                    * jcp.oc_block * jcp.nb_load_blocking_max;
                            diff_dst_data_t *tr_diff_dst = &tr_diff_buffer[ithr
                                    * thr_dst_block_size];
                            for (int ls = 0; ls < load_step; ls++) {
                                size_t ddst_off = ls * jcp.os * jcp.oc_block;
                                size_t ddst_tr_off = ls
                                        * rnd_up(jcp.reduce_dim, 2)
                                        * jcp.oc_block;
                                diff_dst_data_t *curr_inp
                                        = &((diff_dst_data_t *)
                                                        p.load_data)[ddst_off];
                                diff_dst_data_t *curr_out
                                        = &tr_diff_dst[ddst_tr_off];
                                ptr.inp = (void *)curr_inp;
                                ptr.out = (void *)curr_out;
                                tr_reorder_->jit_ker(&ptr);
                            }
                            p.load_data = (void *)tr_diff_dst;
                        }
                        kernel_->jit_ker(&p);
                    }
                }
            }
        }
    };

    auto ker_reduce_and_convert_diff_wei = [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);

        const int ithr_ic_b = ithr % jcp.nthr_ic_b;
        const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
        const int ithr_g = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
        const int ithr_mb = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b / jcp.nthr_g;

        /* independent dimensions */
        int g_start {0}, oc_b_start {0}, ic_b_start {0};
        int g_end {0}, oc_b_end {0}, ic_b_end {0};

        balance211(jcp.ngroups, jcp.nthr_g, ithr_g, g_start, g_end);
        balance211(jcp.nb_load, jcp.nthr_oc_b, ithr_oc_b, oc_b_start, oc_b_end);
        balance211(
                jcp.nb_bcast, jcp.nthr_ic_b, ithr_ic_b, ic_b_start, ic_b_end);

        const int g_work = g_end - g_start;
        const int oc_b_work = oc_b_end - oc_b_start;
        const int ic_b_work = ic_b_end - ic_b_start;

        const int _start_nthr_mb = 1;
        const bool is_bf16_out = diff_weights_type == data_type::bf16;
        /* diff_weights[:] += sum(ws_reduction_[thr_mb][:]) */
        if (jcp.nthr_mb > _start_nthr_mb) {
            if (dnnl_thr_syncable())
                simple_barrier::barrier(&reduction_barrier, jcp.nthr);
            const int work = g_work * oc_b_work * ic_b_work;
            int start {0}, end {0};
            balance211(work, jcp.nthr_mb, ithr_mb, start, end);
            if (start == end) return;

            for (int thr_mb = _start_nthr_mb; thr_mb < jcp.nthr_mb; ++thr_mb) {
                int w = start;
                int sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_start {0};
                nd_iterator_init(w, sub_g_start, g_work, sub_oc_b_start,
                        oc_b_work, sub_ic_b_start, ic_b_work);
                while (w < end) {
                    const int g = g_start + sub_g_start;
                    const int oc_b = oc_b_start + sub_oc_b_start;
                    const int ic_b = ic_b_start + sub_ic_b_start;

                    const size_t acc_size = (size_t)jcp.ic_block * jcp.oc_block
                            * nstl::min(end - w, ic_b_work - sub_ic_b_start);

                    const size_t off
                            = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                    float *wei_reduced = is_bf16_out
                            ? wei_reduction + off
                            : (float *)diff_weights + off;

                    int thr_mb_buffer_idx = is_bf16_out ? thr_mb : thr_mb - 1;
                    float *wei_to_reduce = wei_reduction
                            + thr_mb_buffer_idx * wei_size + off;
                    if (is_bf16_out && thr_mb == jcp.nthr_mb - 1)
                        // the last iteration for bfloat16 requires conversion
                        // and store to diff_weights array
                        add_floats_and_cvt_to_bfloat16(
                                (bfloat16_t *)(diff_weights + off), wei_reduced,
                                wei_to_reduce, acc_size);
                    else
                        acc_ker_->accumulate(
                                wei_reduced, wei_to_reduce, acc_size);

                    nd_iterator_jump(w, end, sub_g_start, g_work,
                            sub_oc_b_start, oc_b_work, sub_ic_b_start,
                            ic_b_work);
                }
            }
        } else if (is_bf16_out) {
            for_(int g = g_start; g < g_end; g++)
            for (int oc_b = oc_b_start; oc_b < oc_b_end; oc_b++) {
                const size_t acc_size
                        = (size_t)ic_b_work * jcp.ic_block * jcp.oc_block;
                const size_t off
                        = wht_blk_off(diff_weights_d, g, oc_b, ic_b_start);

                cvt_float_to_bfloat16((bfloat16_t *)(diff_weights + off),
                        (const float *)(wei_reduction + off), acc_size);
            }
        }
    };

    auto ker_bias = [&](int ithr, int nthr) {
        assert(nthr == rb->balancer().nthr_);

        const int batch_job_start = rb->balancer().ithr_job_off(ithr);
        const int batch_njobs = rb->balancer().ithr_njobs(ithr);

        if (batch_njobs == 0) return;

        /* reduction dimension */
        int img_start {0}, img_end {0};

        balance211(jcp.mb, rb->balancer().nthr_per_group_,
                rb->balancer().id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start {0}, ocb_start {0};
        nd_iterator_init(
                batch_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_load);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int batch_job_loc = 0; batch_job_loc < batch_njobs;
                    ++batch_job_loc) {
                const size_t _oc = g * jcp.nb_load + ocb;

                const diff_dst_data_t *d_dst
                        = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                float *d_bias = &rb->get_local_ptr(
                        ithr, diff_bias, reducer_bia_scratchpad)[batch_job_loc
                        * rb->balancer().job_size_];

                const size_t d_dst_f32_size = (size_t)jcp.os * jcp.oc_block;
                auto dst_ws = d_dst_f32_buffer + d_dst_f32_size * ithr;

                cvt_bfloat16_to_float(dst_ws, d_dst, d_dst_f32_size);

                if (img == img_start)
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] = 0.;

                for (int hw = 0; hw < jcp.os; ++hw) {
                    PRAGMA_OMP_SIMD()
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] += dst_ws[o];
                    dst_ws += 16;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_load);
            }
        }

        if (dnnl_thr_syncable())
            rb->reduce(ithr, diff_bias, reducer_bia_scratchpad);
    };

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        ker(ithr, jcp.nthr);
        if (dnnl_thr_syncable())
            ker_reduce_and_convert_diff_wei(ithr, jcp.nthr);
        if (pd()->with_bias()) ker_bias(ithr, jcp.nthr);
    });

    if (!dnnl_thr_syncable()) {
        parallel(jcp.nthr, [&](const int ithr, const int nthr) {
            ker_reduce_and_convert_diff_wei(ithr, jcp.nthr);
            if (pd()->with_bias()) {
                auto rb = this->reducer_bias_;
                assert(nthr == rb->balancer().nthr_);
                MAYBE_UNUSED(nthr);
                if (rb->balancer().ithr_njobs(ithr) == 0) return;
                rb->reduce_nolock(ithr, diff_bias, reducer_bia_scratchpad);
            }
        });
    }

    /* TODO: put this in ker_bias */
    if (pd()->jcp_.bia_dt == data_type::bf16) {
        auto diff_bias_f32 = scratchpad.template get<float>(
                key_conv_bias_bf16_convert_wsp);
        auto diff_bias_in = CTX_OUT_MEM(
                prec_traits<data_type::bf16>::type *, DNNL_ARG_DIFF_BIAS);
        cvt_float_to_bfloat16(diff_bias_in, diff_bias_f32,
                pd()->jcp_.oc_without_padding * pd()->jcp_.ngroups);

    } else if (pd()->wants_padded_bias()) {
        auto diff_bias_in = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);
        utils::array_copy(diff_bias_in, diff_bias, jcp.oc_without_padding);
    }
}

template struct jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<
        data_type::f32>;
template struct jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<
        data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
