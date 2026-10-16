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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx512_common_1x1_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

#define data_blk_off(f, n, c, d, h, w) \
    ((ndims == 3) ? (f).blk_off(n, c, w) \
                  : ((ndims == 4) ? (f).blk_off(n, c, h, w) \
                                  : (f).blk_off(n, c, d, h, w)))
/* convolution forward */

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_1x1_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward(const exec_ctx_t &ctx) const {
    const auto &jcp = kernel_->jcp;
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    auto weights_dw = CTX_IN_MEM(
            const wei_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
    auto bias_dw = CTX_IN_MEM(
            const dst_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(pd()->jcp_.post_ops, ctx);
    const auto &post_ops_binary_rhs_arg_vec_dw = pd()->dw_conv_pd_
            ? binary_injector::prepare_binary_args(
                      pd()->dw_conv_pd_->jcp_.post_ops, ctx,
                      static_cast<unsigned>(
                              pd()->jcp_.post_ops.entry_.size() + 1))
            : std::vector<const void *> {};

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    if (pd()->wants_padded_bias()) {
        auto padded_bias
                = scratchpad.template get<dst_data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;
    }

    parallel(jcp.nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, weights_dw, bias_dw,
                dst, scratchpad, post_ops_binary_rhs_arg_vec.data(),
                post_ops_binary_rhs_arg_vec_dw.data());
    });

    if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_1x1_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_thr(const int ithr, const int nthr,
        const src_data_t *src, const wei_data_t *weights,
        const dst_data_t *bias, const wei_data_t *weights_dw,
        const dst_data_t *bias_dw, dst_data_t *dst,
        const memory_tracking::grantor_t &scratchpad,
        const void *post_ops_binary_rhs_arg_vec,
        const void *post_ops_binary_rhs_arg_vec_dw) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_1x1_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper dw_dst_d(pd()->dst_md());
    const memory_desc_wrapper dw_weights_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS));
    const memory_desc_wrapper dw_bias_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS));

    const auto &jcp = kernel_->jcp;
    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<src_data_t>(key_conv_rtus_space)
            : nullptr;

    const int ndims = src_d.ndims();
    const dim_t stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const dim_t stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const dim_t stride_w = pd()->desc()->strides[ndims - 3];

    auto step = [](dim_t default_step, dim_t remaining, dim_t tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_args_t();

    auto rp = rtus_driver_t<avx512_core>::call_params_t();

    const dim_t nb_oc = jcp.nb_load;
    const dim_t nb_ic = jcp.nb_reduce;
    const dim_t nb_ic_blocking = jcp.nb_reduce_blocking;

    // override some constants for fused dw_conv
    const dim_t os_block = jcp.with_dw_conv ? jcp.ow : jcp.bcast_block;
    const dim_t nb_bcast = jcp.with_dw_conv ? jcp.oh : jcp.nb_bcast;
    const dim_t nb_bcast_blocking
            = jcp.with_dw_conv ? 1 : jcp.nb_bcast_blocking;
    const dim_t nb_bcast_blocking_max
            = jcp.with_dw_conv ? 1 : jcp.nb_bcast_blocking_max;
    const dim_t nb_load_blocking = jcp.nb_load_blocking;
    const dim_t nb_load_blocking_max = jcp.with_dw_conv
            ? jcp.nb_load_blocking
            : jcp.nb_load_blocking_max;
    const bool is_dst_layout_nxc = utils::one_of(
            jcp.dst_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    const bool is_src_layout_nxc = utils::one_of(
            jcp.src_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    // Begin: declare Variables needed for dw conv.
    memory_tracking::grantor_t dw_scratchpad(
            scratchpad, memory_tracking::names::prefix_fusion);
    dst_data_t *pbuf;
    dim_t row_offset;
    const dim_t nb_buffer = jcp.nb_load_blocking;
    std::vector<dst_data_t *> addrs;
    // End

    auto init_bcast = [&](dim_t iwork, dim_t bcast_end, dim_t &n, dim_t &g,
                              dim_t &bcast_step, dim_t &od, dim_t &oh,
                              dim_t &ow, dim_t &id, dim_t &ih, dim_t &iw) {
        dim_t osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, nb_bcast);
        bcast_step = step(
                nb_bcast_blocking, nb_bcast - osb, nb_bcast_blocking_max);
        bcast_step = nstl::min<dim_t>(bcast_step, bcast_end - iwork);

        const dim_t os = osb * os_block;
        od = os / (jcp.oh * jcp.ow);
        const dim_t os_2d = os % (jcp.oh * jcp.ow);
        oh = os_2d / jcp.ow;
        ow = os_2d % jcp.ow;

        id = od * stride_d;
        ih = oh * stride_h;
        iw = ow * stride_w;
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](dim_t ocb, dim_t ocb_end, dim_t &load_step) {
        load_step = step(nb_load_blocking, ocb_end - ocb, nb_load_blocking_max);
        const auto max_oc = nstl::min<dim_t>(
                ocb_end * jcp.oc_block, jcp.oc_without_padding);
        p.load_dim = this_block_size(
                ocb * jcp.oc_block, max_oc, load_step * jcp.oc_block);
    };

    auto init_reduce = [&](dim_t icb) {
        const dim_t nb_ic_blocking_step
                = nstl::min(icb + nb_ic_blocking, nb_ic) - icb;
        p.first_last_flag = 0 | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                | (icb + nb_ic_blocking_step >= nb_ic ? FLAG_REDUCE_LAST : 0);

        p.reduce_dim = this_block_size(
                icb * jcp.ic_block, jcp.ic, nb_ic_blocking_step * jcp.ic_block);
        rp.icb = p.reduce_dim;
    };

    auto ker_1x1 = [&](dim_t ocb, dim_t ocb_start, dim_t icb, dim_t n, dim_t g,
                           dim_t od, dim_t oh, dim_t ow, dim_t id, dim_t ih,
                           dim_t iw) {
        const dim_t oc_off_idx = is_dst_layout_nxc
                ? g * jcp.oc + ocb * jcp.oc_block
                : g * nb_oc + ocb;
        const dim_t dst_off = data_blk_off(dst_d, n, oc_off_idx, od, oh, ow);

        p.output_data = jcp.with_dw_conv
                ? pbuf + (oh % pd()->dw_conv_pd_->jcp_.kh) * row_offset
                : &dst[dst_off];
        p.bias_data = bias
                ? &bias[oc_off_idx * (is_dst_layout_nxc ? 1 : jcp.oc_block)]
                : nullptr;

        p.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];
        const dim_t ic_off_idx = is_src_layout_nxc
                ? g * jcp.ic + icb * jcp.ic_block
                : g * nb_ic + icb;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_
                    + (is_src_layout_nxc ? ic_off_idx
                                         : jcp.is * ic_off_idx * jcp.ic_block);
            if (ocb == ocb_start) {
                rp.src = src + data_blk_off(src_d, n, ic_off_idx, id, ih, iw);
                (*rtus_driver_)(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src + data_blk_off(src_d, n, ic_off_idx, id, ih, iw);

        p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;
        p.dst_orig = static_cast<const float *>(p.output_data) - dst_off;

        (*kernel_)(&p);
    };

    auto conv_1x1 = [&](dim_t bcast_start, dim_t bcast_end, dim_t ocb_start,
                            dim_t ocb_end) {
        if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;

        if (jcp.loop_order == loop_rlb) {
            for (dim_t icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                dim_t ocb = ocb_start;
                while (ocb < ocb_end) {
                    dim_t load_step;
                    init_load(ocb, ocb_end, load_step);
                    dim_t iwork = bcast_start;
                    while (iwork < bcast_end) {
                        dim_t n {0}, g {0}, od {0}, oh {0}, ow {0}, id {0},
                                ih {0}, iw {0};
                        dim_t bcast_step {0};
                        init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh,
                                ow, id, ih, iw);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                        iwork += bcast_step;
                    }
                    ocb += load_step;
                }
            }
        } else if (jcp.loop_order == loop_lbr) {
            dim_t ocb = ocb_start;
            while (ocb < ocb_end) {
                dim_t load_step;
                init_load(ocb, ocb_end, load_step);
                dim_t iwork = bcast_start;
                while (iwork < bcast_end) {
                    dim_t n {0}, g {0}, od {0}, oh {0}, ow {0}, id {0}, ih {0},
                            iw {0};
                    dim_t bcast_step {0};
                    init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow,
                            id, ih, iw);
                    for (dim_t icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                    }
                    iwork += bcast_step;
                }
                ocb += load_step;
            }
        } else if (jcp.loop_order == loop_rbl) {
            for (dim_t icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                dim_t iwork = bcast_start;
                while (iwork < bcast_end) {
                    dim_t n {0}, g {0}, od {0}, oh {0}, ow {0}, id {0}, ih {0},
                            iw {0};
                    dim_t bcast_step {0};
                    init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow,
                            id, ih, iw);
                    dim_t ocb = ocb_start;
                    while (ocb < ocb_end) {
                        dim_t load_step;
                        init_load(ocb, ocb_end, load_step);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                        ocb += load_step;
                    }
                    iwork += bcast_step;
                }
            }
        } else if (jcp.loop_order == loop_blr) {
            dim_t iwork = bcast_start;
            while (iwork < bcast_end) {
                dim_t n {0}, g {0}, od {0}, oh {0}, ow {0}, id {0}, ih {0},
                        iw {0};
                dim_t bcast_step {0};
                init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow, id,
                        ih, iw);
                dim_t ocb = ocb_start;
                while (ocb < ocb_end) {
                    dim_t load_step;
                    init_load(ocb, ocb_end, load_step);
                    for (dim_t icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
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

    auto ker_dw = [&](dim_t n, dim_t ocb_start, dim_t load_step, dim_t &dw_oh) {
        auto &jcp_dw = pd()->dw_conv_pd_->jcp_;
        dim_t oh_1x1
                = nstl::max<dim_t>(dw_oh * jcp_dw.stride_h - jcp_dw.t_pad, 0);

        for (int i = 0; i < jcp_dw.kh; ++i)
            addrs[i] = pbuf + ((oh_1x1++) % jcp_dw.kh) * row_offset;

        const auto ocb_end = ocb_start + load_step;
        const auto wch_stride = (is_src_layout_nxc ? 1 : jcp_dw.iw)
                * jcp_dw.nb_ch_blocking * jcp_dw.ch_block;
        const dim_t dil_h = jcp_dw.dilate_h + 1;
        const dim_t str_h = jcp_dw.stride_h;
        const dim_t ch_num = jcp_dw.nb_ch_blocking;
        const dim_t ow = 0;
        const dim_t kw = 0;

        for (dim_t ch = ocb_start; ch < ocb_end; ch += jcp_dw.nb_ch_blocking) {

            const dim_t i_t_overflow
                    = nstl::max<dim_t>(0, jcp_dw.t_pad - dw_oh * str_h);
            const dim_t i_b_overflow
                    = nstl::max<dim_t>(jcp_dw.ih,
                              dw_oh * str_h + (jcp_dw.kh - 1) * dil_h
                                      - jcp_dw.t_pad + 1)
                    - jcp_dw.ih;

            const dim_t kh = div_up(i_t_overflow, dil_h);
            const dim_t kh_padding = jcp_dw.kh - div_up(i_t_overflow, dil_h)
                    - div_up(i_b_overflow, dil_h);

            jit_conv_args_t par_conv_dw;

            par_conv_dw.src = addrs.data();

            const dim_t ch_step = is_dst_layout_nxc
                    ? jcp_dw.ch_block
                    : dw_dst_d.blk_off(0, 1, 0, 0);
            par_conv_dw.dst
                    = &dst[dw_dst_d.blk_off(n, 0, dw_oh, ow) + ch * ch_step];

            par_conv_dw.filt
                    = &weights_dw[dw_weights_d.blk_off(ch, 0, 0, kh, kw)];
            if (bias)
                par_conv_dw.bias
                        = &bias_dw[dw_bias_d.blk_off(ch * jcp_dw.ch_block)];

            par_conv_dw.kh_padding = nstl::max<dim_t>(0, kh_padding);

            par_conv_dw.load_work
                    = (nstl::min<dim_t>(ch + ch_num, jcp_dw.nb_ch) - ch)
                    * jcp_dw.ch_block;

            par_conv_dw.post_ops_binary_rhs_arg_vec
                    = post_ops_binary_rhs_arg_vec_dw;
            par_conv_dw.dst_orig = dst;

            (*kernel_dw_)(&par_conv_dw);

            for (int i = 0; i < jcp_dw.kh; ++i)
                addrs[i] += wch_stride;
        }
    };

    auto conv_dw = [&]() {
        // Set variables
        auto dw_conv_buffer
                = dw_scratchpad.get<dst_data_t>(key_fusion_inout_buffer);
        auto &jcp_dw = pd()->dw_conv_pd_->jcp_;

        const dim_t dw_conv_buffer_size
                = jcp_dw.kh * jcp.ow * nb_buffer * jcp.oc_block;
        pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size;
        row_offset = dw_conv_buffer_size / jcp_dw.kh;
        addrs.resize(jcp_dw.kh);

        dim_t bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
        balance2D(nthr, ithr, jcp.mb * jcp.ngroups * jcp_dw.oh, bcast_start,
                bcast_end, nb_oc, ocb_start, ocb_end,
                static_cast<dim_t>(jcp.load_grp_count));

        while (ocb_start < ocb_end) {
            dim_t load_step;
            init_load(ocb_start, ocb_end, load_step);

            dim_t oh_1x1 = 0;
            auto bcast_iter = bcast_start;
            while (bcast_iter < bcast_end) {
                dim_t n {0}, g {0}, oh_dw {0};
                nd_iterator_init(bcast_iter, n, jcp.mb, g, jcp.ngroups, oh_dw,
                        jcp_dw.oh);
                if (oh_dw == 0) oh_1x1 = 0; // Reset over mb boundary
                const dim_t oh_1x1_range
                        = oh_dw * jcp_dw.stride_h - jcp_dw.t_pad;
                const dim_t oh_1x1_begin = nstl::max<dim_t>(oh_1x1_range, 0);
                const dim_t oh_1x1_end
                        = nstl::min<dim_t>(oh_1x1_range + jcp_dw.kh, jcp.oh);
                oh_1x1 = nstl::max<dim_t>(
                        oh_1x1_begin, oh_1x1); // Skip rows computed previously

                // dw_spatial to 1x1 spatial conversion. if jcp.oh != jcp_dw.oh
                const dim_t bcast_start_1x1
                        = n * jcp.ngroups * jcp.oh + g * jcp.oh + oh_1x1;
                const dim_t bcast_end_1x1
                        = bcast_start_1x1 - oh_1x1 + oh_1x1_end;

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

        const dim_t work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
        dim_t bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
                ocb_start, ocb_end, static_cast<dim_t>(jcp.load_grp_count));

        conv_1x1(bcast_start, bcast_end, ocb_start, ocb_end);
    }
}

REG_AVX512_ISA(template struct jit_avx512_common_1x1_convolution_fwd_t<
        data_type::f32>);
/* convolution backward wtr data */

template <data_type_t diff_dst_type, data_type_t wei_type,
        data_type_t diff_src_type>
void jit_avx512_common_1x1_convolution_bwd_data_t<diff_dst_type, wei_type,
        diff_src_type>::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const auto &jcp = kernel_->jcp;
    auto rtus_space = pd()->rtus_.reduce_src_
            ? ctx.get_scratchpad_grantor().template get<diff_src_data_t>(
                      key_conv_rtus_space)
            : nullptr;

    const int ndims = diff_src_d.ndims();

    assert(jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1);

    const dim_t stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const dim_t stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const dim_t stride_w = pd()->desc()->strides[ndims - 3];

    const dim_t nb_ic = jcp.nb_load;
    const dim_t nb_oc = jcp.nb_reduce;
    const dim_t os_block = jcp.bcast_block;
    const dim_t nb_oc_blocking = jcp.nb_reduce_blocking;

    const dim_t work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto step = [](dim_t default_step, dim_t remaining, dim_t tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    parallel(jcp.nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        auto p = jit_1x1_conv_args_t();
        auto rp = rtus_driver_t<avx512_core>::call_params_t();

        dim_t bcast_start {0}, bcast_end {0}, icb_start {0}, icb_end {0};
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
                icb_start, icb_end, static_cast<dim_t>(jcp.load_grp_count));

        bool reduce_outer
                = (jcp.loop_order == loop_rbl || jcp.loop_order == loop_rlb);
        const dim_t nboc_outer = reduce_outer ? nb_oc : 1;
        const dim_t ocb_outer_step = reduce_outer ? nb_oc_blocking : 1;

        const dim_t nboc_inner = reduce_outer ? 1 : nb_oc;
        const dim_t ocb_inner_step = reduce_outer ? 1 : nb_oc_blocking;
        const dim_t max_ic = nstl::min<dim_t>(icb_end * jcp.ic_block, jcp.ic);

        for (dim_t ocb_outer = 0; ocb_outer < nboc_outer;
                ocb_outer += ocb_outer_step) {
            dim_t cur_ocb_outer
                    = nstl::min(ocb_outer + ocb_outer_step, nboc_outer)
                    - ocb_outer;

            dim_t load_step = 0;
            for (dim_t icb = icb_start; icb < icb_end; icb += load_step) {
                load_step = step(jcp.nb_load_blocking, jcp.nb_load - icb,
                        jcp.nb_load_blocking_max);

                p.load_dim = this_block_size(
                        icb * jcp.ic_block, max_ic, load_step * jcp.ic_block);
                rp.icb = p.load_dim;

                dim_t bcast_step;
                for (dim_t iwork = bcast_start; iwork < bcast_end;
                        iwork += bcast_step) {
                    dim_t n {0}, g {0}, osb {0};
                    nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
                            jcp.nb_bcast);

                    bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                            jcp.nb_bcast_blocking_max);
                    bcast_step
                            = nstl::min<dim_t>(bcast_step, bcast_end - iwork);

                    const dim_t os = osb * os_block;
                    p.bcast_dim = this_block_size(
                            os, jcp.os, bcast_step * os_block);
                    rp.os = p.bcast_dim;

                    const dim_t od = os / (jcp.oh * jcp.ow);
                    const dim_t os_2d = os % (jcp.oh * jcp.ow);
                    const dim_t oh = os_2d / jcp.ow;
                    const dim_t ow = os_2d % jcp.ow;
                    const dim_t id = od * stride_d;
                    const dim_t ih = oh * stride_h;
                    const dim_t iw = ow * stride_w;
                    rp.iw_start = iw;
                    const bool is_dsrc_layout_nxc
                            = utils::one_of(jcp.src_tag, format_tag::nwc,
                                    format_tag::nhwc, format_tag::ndhwc);
                    const dim_t ic_off_idx = is_dsrc_layout_nxc
                            ? g * jcp.ic + icb * jcp.ic_block
                            : g * nb_ic + icb;
                    rp.src = diff_src
                            + data_blk_off(
                                    diff_src_d, n, ic_off_idx, id, ih, iw);
                    if (pd()->rtus_.reduce_src_) {
                        rp.ws = rtus_space
                                + ithr * pd()->rtus_.space_per_thread_;
                        p.output_data = rp.ws;
                    } else
                        p.output_data = rp.src;

                    for (dim_t ocb_inner = 0; ocb_inner < nboc_inner;
                            ocb_inner += ocb_inner_step) {
                        const dim_t cur_ocb_inner
                                = nstl::min(ocb_inner + ocb_inner_step,
                                          nboc_inner)
                                - ocb_inner;

                        const dim_t ocb = reduce_outer ? ocb_outer : ocb_inner;
                        const dim_t nb_oc_blocking_step
                                = reduce_outer ? cur_ocb_outer : cur_ocb_inner;
                        const bool is_ddst_layout_nxc
                                = utils::one_of(jcp.dst_tag, format_tag::nwc,
                                        format_tag::nhwc, format_tag::ndhwc);
                        const dim_t oc_off_idx = is_ddst_layout_nxc
                                ? g * jcp.oc + ocb * jcp.oc_block
                                : g * nb_oc + ocb;
                        dim_t diff_dst_off = data_blk_off(
                                diff_dst_d, n, oc_off_idx, od, oh, ow);
                        p.bcast_data = &diff_dst[diff_dst_off];

                        p.load_data = &weights[pd()->with_groups()
                                        ? weights_d.blk_off(g, ocb, icb)
                                        : weights_d.blk_off(ocb, icb)];

                        p.first_last_flag = ocb == 0 ? FLAG_REDUCE_FIRST : 0;

                        p.reduce_dim = this_block_size(ocb * jcp.oc_block,
                                jcp.oc, nb_oc_blocking_step * jcp.oc_block);

                        (*kernel_)(&p);
                    }
                    if (pd()->rtus_.reduce_src_) (*rtus_driver_)(&rp);
                }
            }
        }
    });
}

REG_AVX512_ISA(template struct jit_avx512_common_1x1_convolution_bwd_data_t<
        data_type::f32>);

/* convolution backward wtr weights */

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

status_t jit_avx512_common_1x1_convolution_bwd_weights_t ::init(
        engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_,
            new jit_avx512_common_1x1_conv_kernel_t(
                    pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
    CHECK(safe_ptr_assign(
            acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
    CHECK(safe_ptr_assign(reducer_bias_,
            new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_)));
    CHECK(kernel_->create_kernel());
    CHECK(acc_ker_->create_kernel());
    CHECK(reducer_bias_->create_kernel());

    CHECK(init_rtus_driver<avx512_core>(this));
    return status::success;
}

void jit_avx512_common_1x1_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias_in = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_BIAS);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<data_t>(key_conv_rtus_space)
            : nullptr;
    const bool is_bias_padded
            = pd()->with_bias() && jcp.oc_without_padding % jcp.oc_block != 0;

    data_t *diff_bias = is_bias_padded
            ? scratchpad.get<data_t>(key_conv_padded_bias)
            : diff_bias_in;
    auto wei_reduction = scratchpad.get<data_t>(key_conv_wei_reduction);

    const int ndims = src_d.ndims();
    const dim_t wei_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block)
            * rnd_up(jcp.ic, jcp.ic_block);

    simple_barrier::ctx_t *reduction_barrier
            = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_wei_reduction_bctx);
    if (dnnl_thr_syncable() && jcp.nthr_mb > 1)
        simple_barrier::ctx_init(reduction_barrier);

    memory_tracking::grantor_t reducer_bia_scratchpad(
            scratchpad, prefix_reducer_bia);
    auto rb = this->reducer_bias_.get();
    rb->init(reducer_bia_scratchpad);

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const dim_t nb_ic = jcp.nb_bcast;
    const dim_t nb_ic_blocking = jcp.nb_bcast_blocking;

    const dim_t nb_oc = jcp.nb_load;
    const dim_t nb_oc_blocking = jcp.nb_load_blocking;

    const dim_t sp_nb = jcp.nb_reduce;
    const dim_t mb_sp_work = jcp.mb * sp_nb;

    const dim_t stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const dim_t stride_w = pd()->desc()->strides[ndims - 3];

    auto step = [](dim_t default_step, dim_t remaining, dim_t tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    const bool is_src_layout_nxc = utils::one_of(
            jcp.src_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    const bool is_ddst_layout_nxc = utils::one_of(
            jcp.dst_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    auto maybe_zero_icpad
            = [= COMPAT_THIS_CAPTURE](const dim_t g_start, const dim_t g_end,
                      const dim_t ocb_start, const dim_t ocb_end) {
        // write zeros to IC padded region.
        const dim_t ic_tail = jcp.ic_without_padding % jcp.ic_block;
        if (is_ddst_layout_nxc && ic_tail != 0) {
            for_(dim_t g = g_start; g < g_end; ++g)
            for (dim_t z_ocb = ocb_start; z_ocb < ocb_end; ++z_ocb) {
                const dim_t z_icb = nb_ic - 1;
                const dim_t off = wht_blk_off(diff_weights_d, g, z_ocb, z_icb)
                        + ic_tail * jcp.oc_block;
                data_t *z_wei = diff_weights + off;
                const dim_t zero_work
                        = (nb_ic * jcp.ic_block - jcp.ic_without_padding)
                        * jcp.oc_block;
                PRAGMA_OMP_SIMD()
                for (dim_t o = 0; o < zero_work; ++o) {
                    z_wei[o] = 0;
                }
            }
        }
    };

    auto ker = [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);

        const int ithr_ic_b = ithr % jcp.nthr_ic_b;
        const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
        const int ithr_g = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
        const int ithr_mb = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b / jcp.nthr_g;

        /* reduction dimension */
        dim_t mb_sp_b_start {0}, mb_sp_b_end {0};
        balance211(
                mb_sp_work, jcp.nthr_mb, ithr_mb, mb_sp_b_start, mb_sp_b_end);

        /* independent dimensions */
        dim_t g_start {0}, oc_b_start {0}, ic_b_start {0};
        dim_t g_end {0}, oc_b_end {0}, ic_b_end {0};

        balance211(jcp.ngroups, jcp.nthr_g, ithr_g, g_start, g_end);
        balance211(jcp.nb_load, jcp.nthr_oc_b, ithr_oc_b, oc_b_start, oc_b_end);
        balance211(
                jcp.nb_bcast, jcp.nthr_ic_b, ithr_ic_b, ic_b_start, ic_b_end);

        const dim_t g_work = g_end - g_start;
        const dim_t oc_b_work = oc_b_end - oc_b_start;
        const dim_t ic_b_work = ic_b_end - ic_b_start;
        const bool cache_aliasing
                = (jcp.ic * jcp.ngroups * sizeof(float)) % 1024 == 0;
        dim_t reduce_step = jcp.nb_reduce_blocking;
        dim_t reduce_step_max = jcp.nb_reduce_blocking_max;
        if (is_src_layout_nxc && cache_aliasing) {
            // Experiments show 4 is a magic number with the tested shapes.
            // TODO: maybe tune for shapes with sp_dim%4 != 0
            reduce_step = nstl::min<dim_t>(4, reduce_step);
            reduce_step_max = reduce_step;
        }

        data_t *diff_wei = ithr_mb == 0
                ? diff_weights
                : wei_reduction + (ithr_mb - 1) * wei_size;

        dim_t sp_b_step = 0;
        for (dim_t mb_sp_b = mb_sp_b_start; mb_sp_b < mb_sp_b_end;
                mb_sp_b += sp_b_step) {
            dim_t img {0}, sp_b {0};
            nd_iterator_init(mb_sp_b, img, jcp.mb, sp_b, sp_nb);
            sp_b_step = step(reduce_step,
                    nstl::min(sp_nb - sp_b, mb_sp_b_end - mb_sp_b),
                    reduce_step_max);

            for (dim_t g = g_start; g < g_end; ++g) {
                dim_t load_step = 0;
                dim_t bcast_step = 0;
                for (dim_t ic_b = ic_b_start; ic_b < ic_b_end;
                        ic_b += bcast_step) {
                    if (is_src_layout_nxc && cache_aliasing) {
                        bcast_step = ic_b_work;
                    } else {
                        bcast_step = step(nb_ic_blocking, ic_b_end - ic_b,
                                jcp.nb_bcast_blocking_max);
                    }

                    for (dim_t oc_b = oc_b_start; oc_b < oc_b_end;
                            oc_b += load_step) {
                        load_step = step(nb_oc_blocking, oc_b_end - oc_b,
                                jcp.nb_load_blocking_max);
                        const dim_t _ic_b = g * nb_ic + ic_b;
                        const dim_t oc_off_idx = is_ddst_layout_nxc
                                ? g * jcp.oc + oc_b * jcp.oc_block
                                : g * nb_oc + oc_b;

                        data_t *store_to;

                        const dim_t off
                                = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                        store_to = diff_wei + off;

                        const dim_t ic_off_idx
                                = (is_src_layout_nxc ? jcp.ic_block : 1)
                                * _ic_b;
                        const data_t *diff_src
                                = &src[src_d.blk_off(img, ic_off_idx)];

                        const dim_t sp_b_end = sp_b + sp_b_step;
                        const data_t *pdiff_dst = &diff_dst[diff_dst_d.blk_off(
                                img, oc_off_idx)];
                        const data_t *local_src = diff_src;

                        auto p = jit_1x1_conv_args_t();
                        auto rp = rtus_driver_t<avx512_core>::call_params_t();

                        p.output_stride = utils::rnd_up(jcp.ic, jcp.ic_block)
                                * jcp.oc_block * jcp.typesize_out;

                        p.load_dim = this_block_size(oc_b * jcp.oc_block,
                                jcp.oc, load_step * jcp.oc_block);

                        p.bcast_dim = this_block_size(ic_b * jcp.ic_block,
                                jcp.ic, bcast_step * jcp.ic_block);
                        rp.icb = p.bcast_dim;
                        p.output_data = store_to;

                        p.reduce_dim = sp_b_step * jcp.reduce_block;
                        rp.os = p.reduce_dim;

                        p.first_last_flag = 0
                                | (mb_sp_b == mb_sp_b_start ? FLAG_REDUCE_FIRST
                                                            : 0)
                                | (sp_b_end == sp_nb ? FLAG_SP_LAST : 0);

                        dim_t sp = sp_b * jcp.reduce_block;
                        dim_t oc_mult
                                = is_ddst_layout_nxc ? jcp.oc : jcp.oc_block;
                        p.load_data = pdiff_dst + sp * oc_mult;

                        if (pd()->rtus_.reduce_src_) {
                            const dim_t oh = sp / jcp.ow;
                            const dim_t ow = sp % jcp.ow;

                            const dim_t ih = oh * stride_h;
                            const dim_t iw = ow * stride_w;
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
                            (*rtus_driver_)(&rp);

                            p.bcast_data = rp.ws;
                        } else {
                            dim_t ic_mult
                                    = is_src_layout_nxc ? jcp.ic : jcp.ic_block;
                            p.bcast_data = local_src + sp * ic_mult;
                        }

                        (*kernel_)(&p);
                    }
                }
            }
        }

        if (ithr_mb == 0 && ic_b_end >= jcp.nb_bcast) {
            maybe_zero_icpad(g_start, g_end, oc_b_start, oc_b_end);
        }

        /* diff_weights[:] += sum(wei_reduction[thr_mb][:]) */
        if (dnnl_thr_syncable() && jcp.nthr_mb > 1) {
            simple_barrier::barrier(reduction_barrier, jcp.nthr);
            const dim_t work = g_work * oc_b_work * ic_b_work;
            dim_t start {0}, end {0};
            balance211(work, jcp.nthr_mb, ithr_mb, start, end);
            if (start == end) return;

            for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
                dim_t w = start;
                dim_t sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_start {0};
                nd_iterator_init(w, sub_g_start, g_work, sub_oc_b_start,
                        oc_b_work, sub_ic_b_start, ic_b_work);
                while (w < end) {
                    const dim_t g = g_start + sub_g_start;
                    const dim_t oc_b = oc_b_start + sub_oc_b_start;
                    const dim_t ic_b = ic_b_start + sub_ic_b_start;
                    const int ic_to_accumulate = static_cast<int>(
                            nstl::min<dim_t>(
                                    end - w, ic_b_work - sub_ic_b_start)
                            * jcp.ic_block);
                    const int acc_size = static_cast<int>(
                            this_block_size(ic_b * jcp.ic_block,
                                    jcp.ic_without_padding, ic_to_accumulate)
                            * jcp.oc_block);

                    const dim_t off
                            = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                    data_t *d = diff_weights + off;
                    data_t *s = wei_reduction + (thr_mb - 1) * wei_size + off;

                    acc_ker_->accumulate(d, s, acc_size);

                    nd_iterator_jump(w, end, sub_g_start, g_work,
                            sub_oc_b_start, oc_b_work, sub_ic_b_start,
                            ic_b_work);
                }
            }
        }
    };

    auto ker_bias = [=](int ithr, int nthr) {
        assert(nthr == rb->balancer().nthr_);

        const int b_job_start = rb->balancer().ithr_job_off(ithr);
        const int b_njobs = rb->balancer().ithr_njobs(ithr);

        if (b_njobs == 0) return;

        /* reduction dimension */
        dim_t img_start {0}, img_end {0};

        balance211(jcp.mb, rb->balancer().nthr_per_group_,
                rb->balancer().id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start {0}, ocb_start {0};
        nd_iterator_init(
                b_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_load);

        for (dim_t img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const dim_t oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb * jcp.oc_block
                        : g * jcp.nb_load + ocb;
                const data_t *d_dst
                        = &diff_dst[diff_dst_d.blk_off(img, oc_off_idx)];

                data_t *d_bias = rb->get_local_ptr(ithr, diff_bias,
                                         reducer_bia_scratchpad)
                        + b_job_loc * rb->balancer().job_size_;
                const dim_t sp_shift = is_ddst_layout_nxc ? jcp.ngroups * jcp.oc
                                                          : jcp.oc_block;
                const auto max_oc = this_block_size(
                        ocb * jcp.oc_block, jcp.oc, jcp.oc_block);
                if (img == img_start)
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] = 0.f;

                for (dim_t os = 0; os < jcp.os; ++os) {
                    PRAGMA_OMP_SIMD()
                    for (dim_t o = 0; o < max_oc; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += sp_shift;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_load);
            }
        }

        if (dnnl_thr_syncable())
            rb->reduce(ithr, diff_bias, reducer_bia_scratchpad);
    };

    if (dnnl_thr_syncable()) {
        parallel(jcp.nthr,
                [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
            ker(ithr, jcp.nthr);
            if (pd()->with_bias()) ker_bias(ithr, jcp.nthr);
        });
    } else {
        parallel(jcp.nthr, [=](int ithr, int nthr) { ker(ithr, nthr); });
        if (jcp.nthr_mb > 1)
            parallel(jcp.nthr, [= COMPAT_THIS_CAPTURE](int ithr, int nthr) {
                assert(nthr == jcp.nthr);

                const int ithr_ic_b = ithr % jcp.nthr_ic_b;
                const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
                const int ithr_g
                        = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
                const int ithr_mb
                        = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b / jcp.nthr_g;

                /* independent dimensions */
                dim_t g_start {0}, oc_b_start {0}, ic_b_start {0};
                dim_t g_end {0}, oc_b_end {0}, ic_b_end {0};

                balance211(jcp.ngroups, jcp.nthr_g, ithr_g, g_start, g_end);
                balance211(jcp.nb_load, jcp.nthr_oc_b, ithr_oc_b, oc_b_start,
                        oc_b_end);
                balance211(jcp.nb_bcast, jcp.nthr_ic_b, ithr_ic_b, ic_b_start,
                        ic_b_end);

                const dim_t g_work = g_end - g_start;
                const dim_t oc_b_work = oc_b_end - oc_b_start;
                const dim_t ic_b_work = ic_b_end - ic_b_start;

                const dim_t work = g_work * oc_b_work * ic_b_work;
                dim_t start {0}, end {0};
                balance211(work, jcp.nthr_mb, ithr_mb, start, end);
                if (start == end) return;

                for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
                    dim_t w = start;
                    dim_t sub_g_start {0}, sub_oc_b_start {0},
                            sub_ic_b_start {0};
                    nd_iterator_init(w, sub_g_start, g_work, sub_oc_b_start,
                            oc_b_work, sub_ic_b_start, ic_b_work);
                    while (w < end) {
                        const dim_t g = g_start + sub_g_start;
                        const dim_t oc_b = oc_b_start + sub_oc_b_start;
                        const dim_t ic_b = ic_b_start + sub_ic_b_start;
                        const dim_t ic_to_accumulate
                                = nstl::min(end - w, ic_b_work - sub_ic_b_start)
                                * jcp.ic_block;
                        const dim_t acc_size
                                = this_block_size(ic_b * jcp.ic_block,
                                          jcp.ic_without_padding,
                                          ic_to_accumulate)
                                * jcp.oc_block;

                        const dim_t off
                                = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                        data_t *d = diff_weights + off;
                        data_t *s
                                = wei_reduction + (thr_mb - 1) * wei_size + off;

                        acc_ker_->accumulate(d, s, acc_size);

                        nd_iterator_jump(w, end, sub_g_start, g_work,
                                sub_oc_b_start, oc_b_work, sub_ic_b_start,
                                ic_b_work);
                    }
                }
            });
        if (pd()->with_bias()) {
            parallel(jcp.nthr,
                    [=](int ithr, int nthr) { ker_bias(ithr, nthr); });
            parallel(jcp.nthr, [=](int ithr, int nthr) {
                assert(nthr == rb->balancer().nthr_);
                MAYBE_UNUSED(nthr);
                if (rb->balancer().ithr_njobs(ithr) == 0) return;
                rb->reduce_nolock(ithr, diff_bias, reducer_bia_scratchpad);
            });
        }
    }

    parallel(1, [=](int ithr, int nthr) {
        /* TODO: put this in ker_bias */
        if (is_bias_padded) {
            assert(IMPLICATION(!is_ddst_layout_nxc, jcp.ngroups == 1));
            const dim_t padded_stride = rnd_up(jcp.oc, jcp.oc_block);
            const dim_t stride = jcp.oc_without_padding;
            for (dim_t g = 0; g < jcp.ngroups; ++g) {
                utils::array_copy(diff_bias_in + g * stride,
                        diff_bias + g * padded_stride, stride);
            }
        }
    });
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
