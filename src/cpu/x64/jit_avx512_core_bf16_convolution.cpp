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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "common/bfloat16.hpp"
#include "cpu/x64/jit_avx512_core_bf16_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

void jit_avx512_core_bf16_convolution_fwd_t::prepare_padded_bias(
        const char *&bias, const memory_tracking::grantor_t &scratchpad) const {
    if (!pd()->wants_padded_bias()) return;

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;
    auto padded_bias = scratchpad.template get<char>(
            memory_tracking::names::key_conv_padded_bias);
    utils::array_copy(
            padded_bias, bias, bia_dt_size * pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + bia_dt_size * pd()->jcp_.oc_without_padding,
            0.f, bia_dt_size * (pd()->jcp_.oc - pd()->jcp_.oc_without_padding));
    bias = padded_bias;
}

void jit_avx512_core_bf16_convolution_fwd_t::execute_forward_1d(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    dim_t work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.nb_ow;

    int nthr = jcp.aligned_threads ? jcp.aligned_threads : jcp.nthr;
    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();

        int n {0}, g {0}, occ {0}, owb {0};

        if (jcp.loop_order == loop_cwgn) {
            int dummy {0};
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, g,
                    jcp.ngroups, n, jcp.mb, dummy, 1);
        } else if (jcp.loop_order == loop_gncw) {
            int dummy {0};
            nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, dummy, 1);
        } else
            assert(!"unsupported loop order");

        while (start < end) {
            int ocb = occ * jcp.nb_oc_blocking;
            int g_ocb = g * jcp.nb_oc + ocb;
            int g_oc = g_ocb * jcp.oc_block;
            int g_icb = g * jcp.nb_ic;

            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            auto bias_w = bias ? bias + g_oc * bia_dt_size : nullptr;
            const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::nwc;
            const int oc_idx = (is_dst_layout_nxc ? jcp.oc_block : 1) * g_ocb;
            auto dst_w
                    = dst + jcp.typesize_out * dst_d.blk_off(n, oc_idx, ow_s);
            const bool is_src_layout_nxc = jcp.src_tag == format_tag::nwc;
            const int ic_idx = (is_src_layout_nxc ? jcp.ic_block : 1) * g_icb;
            auto src_w = src + src_d.blk_off(n, ic_idx, iw_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb);

            par_conv.src = src_w;
            par_conv.dst = dst_w;
            par_conv.filt = wht_w;
            par_conv.bias = bias_w;
            par_conv.owb = owb;
            kernel_->jit_ker(&par_conv);

            if (jcp.loop_order == loop_cwgn) {
                int dummy {0};
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, g,
                        jcp.ngroups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == loop_gncw) {
                int dummy {0};
                nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, dummy, 1);
            } else
                assert(!"unsupported loop order");
        }
    });
}

void jit_avx512_core_bf16_convolution_fwd_t::execute_forward_2d(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    dim_t work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh * jcp.nb_ow;

    int nthr = jcp.aligned_threads ? jcp.aligned_threads : jcp.nthr;
    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();

        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        int n {0}, g {0}, occ {0}, oh_s {0}, owb {0};

        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, g,
                    jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");

        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g_ocb = g * jcp.nb_oc + ocb;
            int g_oc = g_ocb * jcp.oc_block;
            int g_icb = g * jcp.nb_ic;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            auto bias_w = bias ? bias + bia_dt_size * g_oc : nullptr;

            const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::nhwc;
            const int oc_idx = (is_dst_layout_nxc ? jcp.oc_block : 1) * g_ocb;
            auto dst_w = dst
                    + jcp.typesize_out * dst_d.blk_off(n, oc_idx, oh_s, ow_s);
            const bool is_src_layout_nxc = jcp.src_tag == format_tag::nhwc;
            const int ic_idx = (is_src_layout_nxc ? jcp.ic_block : 1) * g_icb;
            auto src_w = src + src_d.blk_off(n, ic_idx, ih_s, iw_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0);

            for (int oj = oh_s, ij = ih_s; oj < oh_e;
                    ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = div_up(max(0, -ij), dilate_h);
                int i_b_overflow = div_up(
                        max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                        dilate_h);
                int kh_padding
                        = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow);
                auto aux_src = src_w + i_t_overflow * dilate_h * src_h_stride;
                auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                par_conv.src = aux_src;
                par_conv.dst = dst_w;
                par_conv.filt = aux_wht;
                par_conv.bias = bias_w;
                par_conv.kh_padding = kh_padding;
                par_conv.owb = owb;
                kernel_->jit_ker(&par_conv);

                src_w += src_h_stride * jcp.stride_h;
                dst_w += jcp.typesize_out * dst_h_stride;
            }
            if (jcp.loop_order == loop_cwgn)
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, g,
                        jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    });
}

void jit_avx512_core_bf16_convolution_fwd_t::execute_forward_3d(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    dim_t work_amount
            = jcp.mb * jcp.ngroups * oc_chunks * jcp.od * jcp.oh * jcp.nb_ow;

    int nthr = jcp.aligned_threads ? jcp.aligned_threads : jcp.nthr;
    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();

        size_t src_d_stride = src_d.blk_off(0, 0, 1);
        size_t src_h_stride = src_d.blk_off(0, 0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 0, 1);
        size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);

        int n {0}, g {0}, occ {0}, od_s {0}, oh_s {0}, owb {0};

        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, g,
                    jcp.ngroups, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, od_s, jcp.od, oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");

        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g_ocb = g * jcp.nb_oc + ocb;
            int g_oc = g_ocb * jcp.oc_block;
            int g_icb = g * jcp.nb_ic;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            int id_s = -jcp.f_pad + od_s * jcp.stride_d;
            int dilate_d = jcp.dilate_d + 1;
            int d_t_overflow = div_up(max(0, -id_s), dilate_d);
            int d_b_overflow = div_up(
                    max(0, id_s - jcp.id + (jcp.kd - 1) * dilate_d + 1),
                    dilate_d);
            int kd_padding = nstl::max(0, jcp.kd - d_t_overflow - d_b_overflow);

            auto bias_w = bias ? bias + g_oc * bia_dt_size : nullptr;

            const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
            const int oc_idx = (is_dst_layout_nxc ? jcp.oc_block : 1) * g_ocb;
            auto dst_w = dst
                    + jcp.typesize_out
                            * dst_d.blk_off(n, oc_idx, od_s, oh_s, ow_s);
            const bool is_src_layout_nxc = jcp.src_tag == format_tag::ndhwc;
            const int ic_idx = (is_src_layout_nxc ? jcp.ic_block : 1) * g_icb;
            auto src_w = src + src_d.blk_off(n, ic_idx, id_s, ih_s, iw_s)
                    + d_t_overflow * dilate_d * src_d_stride;
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0)
                    + d_t_overflow * wht_d_stride;

            for (int oj = oh_s, ij = ih_s; oj < oh_e;
                    ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = div_up(max(0, -ij), dilate_h);
                int i_b_overflow = div_up(
                        max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                        dilate_h);
                int kh_padding
                        = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow);
                auto aux_src = src_w + i_t_overflow * dilate_h * src_h_stride;
                auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                par_conv.src = aux_src;
                par_conv.dst = dst_w;
                par_conv.filt = aux_wht;
                par_conv.bias = bias_w;
                par_conv.kh_padding = kh_padding;
                par_conv.kd_padding = kd_padding;
                par_conv.owb = owb;
                kernel_->jit_ker(&par_conv);

                src_w += src_h_stride * jcp.stride_h;
                dst_w += jcp.typesize_out * dst_h_stride;
            }
            if (jcp.loop_order == loop_cwgn)
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, g,
                        jcp.ngroups, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, od_s, jcp.od, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    });
}

void jit_avx512_core_bf16_convolution_bwd_data_t ::execute_backward_data_3d(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_amount = jcp.ngroups * jcp.mb * ic_chunks * jcp.id * jcp.ih;
        balance211(work_amount, nthr, ithr, start, end);

        auto par_conv = jit_conv_call_s();

        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);

        bool is_fast_path_d = jcp.dilate_d == 0 && jcp.stride_d == 1;
        bool is_fast_path_h = jcp.dilate_h == 0 && jcp.stride_h == 1;

        int n {0}, g {0}, icc {0}, id_s {0}, ih_s {0};
        if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, icc, ic_chunks, g, jcp.ngroups, n, jcp.mb,
                    id_s, jcp.id, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_gnc)
            nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, icc, ic_chunks,
                    id_s, jcp.id, ih_s, jcp.ih);
        else
            assert(!"unsupported loop order");

        while (start < end) {
            int icb = icc * jcp.nb_ic_blocking;
            int g_icb = g * jcp.nb_ic + icb;
            int g_ocb = g * jcp.nb_oc;

            int work_rem = end - start;
            int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;

            int od_s = 0, kd_len = 0, kd_lo = 0;
            if (is_fast_path_d) {
                int d_t_overflow = max(0, jcp.kd - 1 - id_s - jcp.f_pad);
                int d_b_overflow
                        = max(0, jcp.kd - jcp.id + id_s - jcp.back_pad);
                kd_len = jcp.kd - d_t_overflow - d_b_overflow;
                kd_lo = d_b_overflow;
                od_s = id_s + jcp.f_pad - d_b_overflow;
            } else if (jcp.dilate_d != 0) { // stride == 1
                int dilate_d = jcp.dilate_d + 1;
                // Note: use div_up to account for "holes" in filter
                int d_t_overflow = div_up(
                        max(0, (jcp.kd - 1) * dilate_d - id_s - jcp.f_pad),
                        dilate_d);
                int d_b_overflow
                        = div_up(max(0,
                                         (jcp.kd - 1) * dilate_d + 1 - jcp.id
                                                 + id_s - jcp.back_pad),
                                dilate_d);
                kd_len = jcp.kd - d_t_overflow - d_b_overflow;
                kd_lo = d_b_overflow;
                od_s = id_s + jcp.f_pad - d_b_overflow * dilate_d;
            } else { // dilate == 0
                int d_t_overflow = max(
                        0, (jcp.kd - 1 - id_s - jcp.f_pad) / jcp.stride_d);
                int d_b_overflow = max(0,
                        (jcp.kd - jcp.id + id_s - jcp.back_pad) / jcp.stride_d);
                int overflow_kd_hi = jcp.kd - 1
                        - modulo(
                                jcp.id - 1 + jcp.back_pad - id_s, jcp.stride_d);
                int overflow_kd_lo = (id_s + jcp.f_pad) % jcp.stride_d;

                kd_len = (overflow_kd_hi - overflow_kd_lo) / jcp.stride_d + 1
                        - d_t_overflow - d_b_overflow;
                kd_lo = overflow_kd_lo + d_b_overflow * jcp.stride_d;
                od_s = (id_s + jcp.f_pad - kd_lo) / jcp.stride_d;
            }
            assert(kd_len >= 0);

            const bool is_dsrc_layout_nxc = jcp.src_tag == format_tag::ndhwc;
            const int ic_idx = (is_dsrc_layout_nxc ? jcp.ic_block : 1) * g_icb;
            auto diff_src_w = diff_src
                    + jcp.typesize_out * diff_src_d.blk_off(n, ic_idx, id_s);
            const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
            const int oc_idx = (is_ddst_layout_nxc ? jcp.oc_block : 1) * g_ocb;
            auto diff_dst_w = diff_dst + diff_dst_d.blk_off(n, oc_idx, od_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, 0, icb, kd_lo);

            for (int ij = ih_s; ij < ih_e; ++ij) {
                int oj, kh_len, kh_lo;
                if (is_fast_path_h) { // dilate == 0 && stride == 1
                    int i_t_overflow = max(0, jcp.kh - 1 - ij - jcp.t_pad);
                    int i_b_overflow = max(0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                    kh_len = jcp.kh - i_t_overflow - i_b_overflow;
                    kh_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow;
                } else if (jcp.dilate_h != 0) { // stride == 1
                    int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    int i_t_overflow = div_up(
                            max(0, (jcp.kh - 1) * dilate_h - ij - jcp.t_pad),
                            dilate_h);
                    int i_b_overflow
                            = div_up(max(0,
                                             (jcp.kh - 1) * dilate_h + 1
                                                     - jcp.ih + ij - jcp.b_pad),
                                    dilate_h);
                    kh_len = jcp.kh - i_t_overflow - i_b_overflow;
                    kh_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                } else { // dilate == 0
                    int i_t_overflow = max(
                            0, (jcp.kh - 1 - ij - jcp.t_pad) / jcp.stride_h);
                    int i_b_overflow = max(0,
                            (jcp.kh - jcp.ih + ij - jcp.b_pad) / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1
                            - modulo(jcp.ih - 1 + jcp.b_pad - ij, jcp.stride_h);
                    int overflow_kh_lo = (ij + jcp.t_pad) % jcp.stride_h;

                    kh_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h
                            + 1 - i_t_overflow - i_b_overflow;
                    kh_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                    oj = (ij + jcp.t_pad - kh_lo) / jcp.stride_h;
                }
                assert(kh_len >= 0);

                par_conv.src = diff_src_w
                        + jcp.typesize_out * ij * diff_src_h_stride;
                par_conv.dst = diff_dst_w + oj * diff_dst_h_stride;
                par_conv.filt = wht_w + kh_lo * wht_h_stride;
                par_conv.kh_padding = kh_len;
                par_conv.kd_padding = kd_len;

                kernel_->jit_ker(&par_conv);
            }

            if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end, icc, ic_chunks, g, jcp.ngroups, n,
                        jcp.mb, id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, icc,
                        ic_chunks, id_s, jcp.id, ih_s, jcp.ih);
            else
                assert(!"unsupported loop order");
        }
    });
}

void jit_avx512_core_bf16_convolution_bwd_data_t ::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_amount = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih * jcp.nb_iw;
        balance211(work_amount, nthr, ithr, start, end);

        auto par_conv = jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        bool is_fast_path = jcp.dilate_h == 0 && jcp.stride_h == 1;

        int n {0}, g {0}, icc {0}, ih_s {0}, iwb {0};
        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, icc, ic_chunks, iwb, jcp.nb_iw, g,
                    jcp.ngroups, n, jcp.mb, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, icc, ic_chunks,
                    iwb, jcp.nb_iw, ih_s, jcp.ih);
        else
            assert(!"unsupported loop order");

        while (start < end) {
            int icb = icc * jcp.nb_ic_blocking;
            int g_icb = g * jcp.nb_ic + icb;
            int g_ocb = g * jcp.nb_oc;

            int work_rem = end - start;
            int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
            int iw_s = iwb * jcp.iw_block;
            int ow_s = iw_s / jcp.stride_w;

            auto diff_src_w = diff_src;
            auto diff_dst_w = diff_dst;
            const bool is_ddst_layout_nxc = utils::one_of(
                    jcp.dst_tag, format_tag::nwc, format_tag::nhwc);
            const int oc_idx = (is_ddst_layout_nxc ? jcp.oc_block : 1) * g_ocb;
            const bool is_dsrc_layout_nxc = utils::one_of(
                    jcp.src_tag, format_tag::nwc, format_tag::nhwc);
            const int ic_idx = (is_dsrc_layout_nxc ? jcp.ic_block : 1) * g_icb;
            if (jcp.ndims == 3) {
                diff_src_w += jcp.typesize_out
                        * diff_src_d.blk_off(n, ic_idx, iw_s);
                diff_dst_w += diff_dst_d.blk_off(n, oc_idx, ow_s);
            } else {
                diff_src_w += jcp.typesize_out
                        * diff_src_d.blk_off(n, ic_idx, 0, iw_s);
                diff_dst_w += diff_dst_d.blk_off(n, oc_idx, 0, ow_s);
            }
            auto wht_w = weights + wht_blk_off(weights_d, g, 0, icb);

            for (int ij = ih_s; ij < ih_e; ++ij) {
                int oj, k_len, k_lo;
                if (is_fast_path) { // dilate == 0 && stride == 1
                    int i_t_overflow = max(0, jcp.kh - 1 - ij - jcp.t_pad);
                    int i_b_overflow = max(0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                    k_len = jcp.kh - i_t_overflow - i_b_overflow;
                    k_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow;
                } else if (jcp.dilate_h != 0) { // stride == 1
                    int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    int i_t_overflow = div_up(
                            max(0, (jcp.kh - 1) * dilate_h - ij - jcp.t_pad),
                            dilate_h);
                    int i_b_overflow
                            = div_up(max(0,
                                             (jcp.kh - 1) * dilate_h + 1
                                                     - jcp.ih + ij - jcp.b_pad),
                                    dilate_h);
                    k_len = jcp.kh - i_t_overflow - i_b_overflow;
                    k_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                } else { // dilate == 0
                    int i_t_overflow = max(
                            0, (jcp.kh - 1 - ij - jcp.t_pad) / jcp.stride_h);
                    int i_b_overflow = max(0,
                            (jcp.kh - jcp.ih + ij - jcp.b_pad) / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1
                            - modulo(jcp.ih - 1 + jcp.b_pad - ij, jcp.stride_h);
                    int overflow_kh_lo = (ij + jcp.t_pad) % jcp.stride_h;

                    k_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h + 1
                            - i_t_overflow - i_b_overflow;
                    k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                    oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                }
                assert(k_len >= 0);

                par_conv.src = diff_src_w
                        + jcp.typesize_out * ij * diff_src_h_stride;
                par_conv.dst = diff_dst_w + oj * diff_dst_h_stride;
                par_conv.filt = wht_w + k_lo * wht_h_stride;
                par_conv.kh_padding = k_len;
                par_conv.iwb = iwb;

                kernel_->jit_ker(&par_conv);
            }

            if (jcp.loop_order == loop_cwgn)
                nd_iterator_jump(start, end, icc, ic_chunks, iwb, jcp.nb_iw, g,
                        jcp.ngroups, n, jcp.mb, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, icc,
                        ic_chunks, iwb, jcp.nb_iw, ih_s, jcp.ih);
            else
                assert(!"unsupported loop order");
        }
    });
}

jit_avx512_core_bf16_convolution_bwd_weights_t ::
        jit_avx512_core_bf16_convolution_bwd_weights_t(const pd_t *apd)
    : primitive_t(apd)
    , kernel_(nullptr)
    , acc_ker_(nullptr)
    , reducer_bias_(nullptr)
    , trans_kernel_(nullptr)
    , trans_dst_kernel_(nullptr) {
    const auto &j = pd()->jcp_;

    nthr_ = j.nthr;
    nthr_mb_ = j.nthr_mb;
    nthr_g_ = j.nthr_g;
    nthr_oc_b_ = j.nthr_oc_b;
    nthr_ic_b_ = j.nthr_ic_b;

    kernel_ = new jit_avx512_core_bf16_conv_bwd_weights_kernel_f32(j);

    if (j.transpose_src) { trans_kernel_ = create_trans_src(&j); }
    if (j.transpose_dst) { trans_dst_kernel_ = create_trans_dst(&j); }

    if (nthr_mb_ > 1) acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();

    reducer_bias_ = new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_);
}

struct jit_avx512_core_bf16_convolution_bwd_weights_t ::thread_info_t {
    const src_data_t *src;
    const diff_dst_data_t *diff_dst;
    const void *diff_weights;
    float *diff_bias;

    const memory_tracking::grantor_t scratchpad;

    src_data_t *tr_src = nullptr;
    diff_dst_data_t *tr_diff_dst = nullptr;
    simple_barrier::ctx_t *tr_src_bctx = nullptr;
    simple_barrier::ctx_t *tr_diff_dst_bctx = nullptr;

    float *wei_bia_reduction;
    simple_barrier::ctx_t *wei_bia_reduction_bctx;

    int ithr;
    int ithr_ic_b, ithr_oc_b, ithr_g, ithr_mb;
    int ithr_but_oc;
    int ithr_but_ic;

    int img_start = 0, img_end = 0, img_work;
    int g_start = 0, g_end = 0, g_work;
    int oc_b_start = 0, oc_b_end = 0, oc_b_work;
    int ic_b_start = 0, ic_b_end = 0, ic_b_work;

    thread_info_t(const jit_avx512_core_bf16_convolution_bwd_weights_t *self,
            const exec_ctx_t &ctx, int ithr)
        : scratchpad(ctx.get_scratchpad_grantor()), ithr(ithr) {
        diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
        src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
        diff_weights = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS);

        const auto &jcp = self->kernel_->jcp;

        if (self->pd()->jcp_.bia_dt == data_type::bf16) {
            diff_bias = scratchpad.template get<float>(
                    key_conv_bias_bf16_convert_wsp);
        } else
            diff_bias = self->pd()->wants_padded_bias()
                    ? scratchpad.template get<float>(key_conv_padded_bias)
                    : CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);

        if (jcp.transpose_src) {
            tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);

            if (dnnl_thr_syncable())
                tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                        key_conv_tr_src_bctx);
        }
        if (jcp.transpose_dst) {
            tr_diff_dst = scratchpad.template get<diff_dst_data_t>(
                    key_conv_tr_diff_dst);

            if (dnnl_thr_syncable())
                tr_diff_dst_bctx
                        = scratchpad.template get<simple_barrier::ctx_t>(
                                key_conv_tr_diff_dst_bctx);
        }
        wei_bia_reduction
                = scratchpad.template get<float>(key_conv_wei_bia_reduction);
        if (dnnl_thr_syncable())
            wei_bia_reduction_bctx
                    = scratchpad.template get<simple_barrier::ctx_t>(
                            key_conv_wei_bia_reduction_bctx);

        ithr_ic_b = ithr % self->nthr_ic_b_;
        ithr_oc_b = ithr / self->nthr_ic_b_ % self->nthr_oc_b_;
        ithr_g = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ % self->nthr_g_;
        ithr_mb = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ / self->nthr_g_;

        ithr_but_oc = (ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_ic_b_
                + ithr_ic_b;

        ithr_but_ic = (ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_oc_b_
                + ithr_oc_b;

        int work_amount = jcp.nthr_mb_work;
        /* reduction dimension */
        balance211(work_amount, self->nthr_mb_, ithr_mb, img_start, img_end);
        img_work = img_end - img_start;

        /* independent dimensions */
        balance211(jcp.ngroups, self->nthr_g_, ithr_g, g_start, g_end);
        g_work = g_end - g_start;

        balance211(
                jcp.nb_oc, self->nthr_oc_b_, ithr_oc_b, oc_b_start, oc_b_end);
        oc_b_work = oc_b_end - oc_b_start;

        balance211(
                jcp.nb_ic, self->nthr_ic_b_, ithr_ic_b, ic_b_start, ic_b_end);
        ic_b_work = ic_b_end - ic_b_start;
    }
};

size_t jit_avx512_core_bf16_convolution_bwd_weights_t::tr_src_buf_number(
        const thread_info_t *ti, int g, int ic) const {
    const jit_conv_conf_t jcp = this->kernel_->jcp;
    return dnnl_thr_syncable()
            ? ti->ithr_mb * jcp.nb_ic * jcp.ngroups + g * jcp.nb_ic + ic
            : ti->ithr;
}
size_t jit_avx512_core_bf16_convolution_bwd_weights_t::tr_diff_dst_buf_number(
        const thread_info_t *ti, int g, int oc) const {
    const jit_conv_conf_t jcp = this->kernel_->jcp;
    return dnnl_thr_syncable()
            ? ti->ithr_mb * jcp.nb_oc * jcp.ngroups + g * jcp.nb_oc + oc
            : ti->ithr;
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::trans_src(
        src_data_t *tr_src, const src_data_t *src, int row_count) const {
    const jit_conv_conf_t jcp = this->kernel_->jcp;
    const int pf_depth = 2;
    struct {
        const src_data_t *src;
        src_data_t *tr_src;
    } pf_circ_buf_src[pf_depth];

    assert(jcp.ic_block == 16 || jcp.is_1stconv);
    const int src_stride = jcp.iw * jcp.ic_block;
    const int tr_src_stride = jcp.tr_iw * jcp.ic_block;

    for (int iwork = 0; iwork < row_count + pf_depth - 1; iwork++) {
        pf_circ_buf_src[iwork % pf_depth] = {src, tr_src};

        if (iwork >= pf_depth - 1) {
            int old_idx = (iwork - pf_depth + 1) % pf_depth;
            auto ctx = jit_trans_src_t::ctx_t();
            ctx.src = pf_circ_buf_src[old_idx].src;
            ctx.tr_src = pf_circ_buf_src[old_idx].tr_src;
            ctx.src_prf = src;
            ctx.tr_src_prf = tr_src;
            (*trans_kernel_)(&ctx);
        }
        src += src_stride;
        tr_src += tr_src_stride;
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::trans_dst(
        diff_dst_data_t *tr_diff_dst, const diff_dst_data_t *diff_dst,
        int row_count) const {

    const jit_conv_conf_t jcp = this->kernel_->jcp;
    const int pf_depth = 2;
    struct {
        const diff_dst_data_t *diff_dst;
        diff_dst_data_t *tr_diff_dst;
    } pf_circ_buf_dst[pf_depth];

    assert(jcp.ic_block == 16 || jcp.is_1stconv);
    const int diff_dst_stride = jcp.ow * jcp.oc_block;
    const int tr_diff_dst_stride = jcp.tr_ow * jcp.oc_block;

    for (int iwork = 0; iwork < row_count + pf_depth - 1; iwork++) {
        pf_circ_buf_dst[iwork % pf_depth]
                = {(diff_dst_data_t *)diff_dst, tr_diff_dst};

        if (iwork >= pf_depth - 1) {
            int old_idx = (iwork - pf_depth + 1) % pf_depth;
            auto ctx = jit_trans_dst_t::ctx_t();
            ctx.src = pf_circ_buf_dst[old_idx].diff_dst;
            ctx.tr_src = pf_circ_buf_dst[old_idx].tr_diff_dst;
            ctx.src_prf = diff_dst;
            ctx.tr_src_prf = tr_diff_dst;
            (*trans_dst_kernel_)(&ctx);
        }
        diff_dst += diff_dst_stride;
        tr_diff_dst += tr_diff_dst_stride;
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::compute_diff_weights_2d(
        const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto &jcp = kernel_->jcp;
    const int wei_size
            = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw * jcp.kd;

    float *diff_wei;
    if (diff_weights_d.data_type() == data_type::bf16)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    auto tr_diff_dst_off = [&](int g, int oc, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        return tr_diff_dst_buf_number(ti, g, oc) * jcp.tr_diff_dst_buf_size
                + oj * tr_row_size;
    };

    int img {0}, oh_s {0};
    int start = ti->img_start;
    int end = ti->img_end;

    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);

    nd_iterator_init(start, img, jcp.mb, oh_s, jcp.oh);

    while (start < end) {
        auto p = jit_conv_call_s();
        int work_rem = end - start;
        const int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
        int ih_s = nstl::max(0, -jcp.t_pad + oh_s * jcp.stride_h);
        const int ih_e = nstl::min(
                jcp.ih, -jcp.t_pad + (oh_e - 1) * jcp.stride_h + ext_kh);

        auto tr_src_off = [&](int g, int ic, int ij) {
            const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
            // Aligned to buffer end to use guard elements
            return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size
                    + (jcp.ih - ih_e + ij) * tr_row_size;
        };

        if (dnnl_thr_syncable()) {
            using simple_barrier::barrier;
            // TODO: try to call local transpositions just before jit kernel
            /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
            if (jcp.transpose_src) {
                int j {0};
                const int work_amount
                        = ti->g_work * ti->ic_b_work * (ih_e - ih_s);
                int tr_start {0}, tr_end {0};
                balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, tr_start,
                        tr_end);

                int g {0}, ic_b {0};
                nd_iterator_init(tr_start, g, ti->g_work, ic_b, ti->ic_b_work,
                        j, ih_e - ih_s);

                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
                while (tr_start < tr_end) {
                    int g_ = g + ti->g_start;
                    int ic_b_ = ic_b + ti->ic_b_start;
                    int j_s = j + ih_s;
                    int j_e = j_s + nstl::min(tr_end - tr_start, ih_e - j_s);
                    const int ic = g_ * jcp.nb_ic + ic_b_;
                    const src_data_t *src
                            = &ti->src[src_d.blk_off(img, ic, j_s)];
                    src_data_t *tr_src
                            = &ti->tr_src[tr_src_off(g_, ic_b_, j_s)];

                    trans_src(tr_src, src, j_e - j_s);
                    j += j_e - j_s - 1;
                    tr_start += j_e - j_s - 1;
                    nd_iterator_jump(tr_start, tr_end, g, ti->g_work, ic_b,
                            ti->ic_b_work, j, ih_e - ih_s);
                }
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            }
            if (jcp.transpose_dst) {
                int j {0};
                const int work_amount
                        = ti->g_work * ti->oc_b_work * (oh_e - oh_s);
                int tr_start {0}, tr_end {0};
                balance211(work_amount, nthr_ic_b_, ti->ithr_ic_b, tr_start,
                        tr_end);

                int g {0}, oc_b {0};
                nd_iterator_init(tr_start, g, ti->g_work, oc_b, ti->oc_b_work,
                        j, oh_e - oh_s);

                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
                while (tr_start < tr_end) {
                    int g_ = g + ti->g_start;
                    int oc_b_ = oc_b + ti->oc_b_start;
                    int j_s = j + oh_s;
                    int j_e = j_s + nstl::min(tr_end - tr_start, oh_e - j_s);
                    const int oc = g_ * jcp.nb_oc + oc_b_;
                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(img, oc, j_s)];
                    diff_dst_data_t *tr_diff_dst
                            = &ti->tr_diff_dst[tr_diff_dst_off(g_, oc_b_, j_s)];

                    trans_dst(tr_diff_dst, diff_dst, j_e - j_s);
                    j += j_e - j_s - 1;
                    tr_start += j_e - j_s - 1;
                    nd_iterator_jump(tr_start, tr_end, g, ti->g_work, oc_b,
                            ti->oc_b_work, j, oh_e - oh_s);
                }
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            }
        }

        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
            const int _oc = g * jcp.nb_oc + oc_b;
            const int _ic = g * jcp.nb_ic + ic_b;
            if (jcp.transpose_src) {
                if (!dnnl_thr_syncable()) {
                    const src_data_t *src
                            = (src_data_t *)&ti
                                      ->src[src_d.blk_off(img, _ic, ih_s)];
                    src_data_t *tr_src = &ti->tr_src[tr_src_off(g, ic_b, ih_s)];
                    trans_src(tr_src, src, ih_e - ih_s);
                    p.src = tr_src;
                } else {
                    p.src = &ti->tr_src[tr_src_off(g, ic_b, ih_s)];
                }
            } else {
                p.src = &ti->src[src_d.blk_off(img, _ic, ih_s)];
            }

            if (jcp.transpose_dst) {
                if (!dnnl_thr_syncable()) {
                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(img, _oc, oh_s)];
                    diff_dst_data_t *tr_diff_dst
                            = &ti->tr_diff_dst[tr_diff_dst_off(0, 0, 0)];
                    trans_dst(tr_diff_dst, diff_dst, oh_e - oh_s);
                    p.dst = tr_diff_dst;
                } else {
                    p.dst = &ti->tr_diff_dst[tr_diff_dst_off(g, oc_b, oh_s)];
                }
            } else {
                p.dst = &ti->diff_dst[diff_dst_d.blk_off(img, _oc, oh_s)];
            }

            p.filt = diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = nullptr;
            p.channel = (start == ti->img_start);
            p.os_index_begin = oh_s;
            p.os_index_end = oh_e;
            assert(oh_e <= jcp.oh);
            kernel_->jit_ker(&p);
        }

        nd_iterator_jump(start, end, img, jcp.mb, oh_s, jcp.oh);
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::compute_diff_weights_3d(
        const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto &jcp = kernel_->jcp;
    const int wei_size
            = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw * jcp.kd;

    float *diff_wei;
    if (diff_weights_d.data_type() == data_type::bf16)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    auto tr_diff_dst_off_3d = [&](int g, int oc, int od) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        const size_t tr_3d_size = tr_row_size * jcp.oh;
        return tr_diff_dst_buf_number(ti, g, oc) * jcp.tr_diff_dst_buf_size
                + od * tr_3d_size;
    };
    int img {0}, od_s {0};
    int start = ti->img_start;
    int end = ti->img_end;

    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);

    nd_iterator_init(start, img, jcp.mb, od_s, jcp.od);

    while (start < end) {
        auto p = jit_conv_call_s();
        int work_rem = end - start;
        const int od_e = od_s + work_rem > jcp.od ? jcp.od : od_s + work_rem;
        int id_s = nstl::max(0, -jcp.f_pad + od_s * jcp.stride_d);
        const int id_e = nstl::min(
                jcp.id, -jcp.f_pad + (od_e - 1) * jcp.stride_d + ext_kd);
        const int kd_front_pad = nstl::max(0, jcp.f_pad - od_s * jcp.stride_d);
        // Assumes kd_back_pad = 0 when kernel is dilated
        const int kd_back_pad = nstl::max(
                0, od_s * jcp.stride_d + jcp.kd - jcp.f_pad - jcp.id);
        int kd_pad_off = nstl::min(jcp.kd - 1, kd_front_pad) * jcp.kh * jcp.kw
                * jcp.ic_block * jcp.oc_block * jcp.typesize_out;

        auto tr_src_off_3d = [&](int g, int ic, int id) {
            const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
            const size_t tr_3d_size = tr_row_size * jcp.ih;
            // Aligned to buffer end to use guard elements
            return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size
                    + (jcp.id - id_e + id) * tr_3d_size;
        };

        if (dnnl_thr_syncable()) {
            using simple_barrier::barrier;
            // TODO: try to call local transpositions just before jit kernel
            /* tr_src[nb_ic][id][16][~iw~] <- src[nb_ic][id][iw][16] */
            if (jcp.transpose_src) {
                int d {0};

                const int work_amount
                        = ti->g_work * ti->ic_b_work * (id_e - id_s);

                int tr_start {0}, tr_end {0};
                balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, tr_start,
                        tr_end);

                int g {0}, ic_b {0};

                nd_iterator_init(tr_start, g, ti->g_work, ic_b, ti->ic_b_work,
                        d, id_e - id_s);

                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
                while (tr_start < tr_end) {
                    int g_ = g + ti->g_start;
                    int ic_b_ = ic_b + ti->ic_b_start;
                    int d_s = d + id_s;
                    int d_e = d_s + nstl::min(tr_end - tr_start, id_e - d_s);

                    const int ic = g_ * jcp.nb_ic + ic_b_;
                    const src_data_t *src
                            = &ti->src[src_d.blk_off(img, ic, d_s)];
                    src_data_t *tr_src
                            = &ti->tr_src[tr_src_off_3d(g_, ic_b_, d_s)];
                    trans_src(tr_src, src, (d_e - d_s) * jcp.ih);
                    d += d_e - d_s - 1;
                    tr_start += d_e - d_s - 1;
                    nd_iterator_jump(tr_start, tr_end, g, ti->g_work, ic_b,
                            ti->ic_b_work, d, id_e - id_s);
                }
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            }
            if (jcp.transpose_dst) {
                int d {0};

                const int work_amount
                        = ti->g_work * ti->oc_b_work * (od_e - od_s);

                int tr_start {0}, tr_end {0};
                balance211(work_amount, nthr_ic_b_, ti->ithr_ic_b, tr_start,
                        tr_end);

                int g {0}, oc_b {0};

                nd_iterator_init(tr_start, g, ti->g_work, oc_b, ti->oc_b_work,
                        d, od_e - od_s);

                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
                while (tr_start < tr_end) {
                    int g_ = g + ti->g_start;
                    int oc_b_ = oc_b + ti->oc_b_start;
                    int d_s = d + od_s;
                    int d_e = d_s + nstl::min(tr_end - tr_start, od_e - d_s);

                    const int oc = g_ * jcp.nb_oc + oc_b_;
                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(img, oc, d_s)];
                    diff_dst_data_t *tr_diff_dst
                            = &ti->tr_diff_dst[tr_diff_dst_off_3d(
                                    g_, oc_b_, d_s)];

                    trans_dst(tr_diff_dst, diff_dst, (d_e - d_s) * jcp.oh);
                    d += d_e - d_s - 1;
                    tr_start += d_e - d_s - 1;
                    nd_iterator_jump(tr_start, tr_end, g, ti->g_work, oc_b,
                            ti->oc_b_work, d, od_e - od_s);
                }
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            }
        }

        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
            const int _oc = g * jcp.nb_oc + oc_b;
            const int _ic = g * jcp.nb_ic + ic_b;
            if (jcp.transpose_src) {
                if (!dnnl_thr_syncable()) {
                    const src_data_t *src
                            = (src_data_t *)&ti
                                      ->src[src_d.blk_off(img, _ic, id_s)];
                    src_data_t *tr_src
                            = &ti->tr_src[tr_src_off_3d(g, ic_b, id_s)];
                    trans_src(tr_src, src, (id_e - id_s) * jcp.ih);
                    p.src = tr_src;
                } else {
                    p.src = &ti->tr_src[tr_src_off_3d(g, ic_b, id_s)];
                }
            } else {
                p.src = &ti->src[src_d.blk_off(img, _ic, id_s)];
            }

            if (jcp.transpose_dst) {
                if (!dnnl_thr_syncable()) {
                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(img, _oc, od_s)];
                    diff_dst_data_t *tr_diff_dst
                            = &ti->tr_diff_dst[tr_diff_dst_off_3d(0, 0, 0)];
                    trans_dst(tr_diff_dst, diff_dst, (od_e - od_s) * jcp.oh);
                    p.dst = tr_diff_dst;
                } else {
                    p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(g, oc_b, od_s)];
                }
            } else {
                p.dst = &ti->diff_dst[diff_dst_d.blk_off(img, _oc, od_s)];
            }

            p.filt = diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = nullptr;
            p.channel = (start == ti->img_start);
            p.os_index_begin = od_s;
            p.os_index_end = od_e;
            p.kd_padding = jcp.kd - kd_front_pad - kd_back_pad;
            p.kd_offset = kd_pad_off;
            assert(od_e <= jcp.od);
            kernel_->jit_ker(&p);
        }

        nd_iterator_jump(start, end, img, jcp.mb, od_s, jcp.od);
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::compute_diff_weights(
        const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto &jcp = kernel_->jcp;
    const int wei_size
            = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw * jcp.kd;

    float *diff_wei;
    if (diff_weights_d.data_type() == data_type::bf16)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    auto tr_src_off = [&](int g, int ic, int ij) {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size
                + ij * tr_row_size;
    };

    auto tr_src_off_3d = [&](int g, int ic, int id, int ij) {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_3d_size = tr_row_size * jcp.ih;
        return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size
                + id * tr_3d_size + ij * tr_row_size;
    };

    auto tr_diff_dst_off = [&](int g, int oc, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        return tr_diff_dst_buf_number(ti, g, oc) * jcp.tr_diff_dst_buf_size
                + oj * tr_row_size;
    };

    auto tr_diff_dst_off_3d = [&](int g, int oc, int od, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        const size_t tr_3d_size = tr_row_size * jcp.oh;
        return tr_diff_dst_buf_number(ti, g, oc) * jcp.tr_diff_dst_buf_size
                + od * tr_3d_size + oj * tr_row_size;
    };

    auto uker_trans = [&](int img, int g = 0, int ic_b = 0) {
        int j {0}, d {0};
        int my_work = jcp.ih * jcp.id;
        int ic;
        if (dnnl_thr_syncable()) {
            const int work_amount
                    = ti->g_work * ti->ic_b_work * jcp.ih * jcp.id;

            int start {0}, end {0};
            balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, start, end);
            my_work = end - start;

            if (jcp.ndims == 5)
                nd_iterator_init(start, g, ti->g_work, ic_b, ti->ic_b_work, d,
                        jcp.id, j, jcp.ih);
            else
                nd_iterator_init(
                        start, g, ti->g_work, ic_b, ti->ic_b_work, j, jcp.ih);
            g += ti->g_start;
            ic_b += ti->ic_b_start;
            ic = g * jcp.nb_ic + ic_b;
        } else {
            ic = g * jcp.nb_ic + ic_b;
            g = 0;
            ic_b = 0;
        }

        src_data_t *src = (jcp.ndims == 5)
                ? (src_data_t *)&ti->src[src_d.blk_off(img, ic, d, j)]
                : (src_data_t *)&ti->src[src_d.blk_off(img, ic, j)];
        src_data_t *tr_src = (jcp.ndims == 5)
                ? &ti->tr_src[tr_src_off_3d(g, ic_b, d, j)]
                : &ti->tr_src[tr_src_off(g, ic_b, j)];

        trans_src(tr_src, src, my_work);
    };

    auto diff_dst_trans = [&](int img, int g = 0, int oc_b = 0) {
        int j {0}, d {0};
        int my_work = jcp.oh * jcp.od;
        int oc;

        if (dnnl_thr_syncable()) {
            const size_t work_amount
                    = ti->g_work * ti->oc_b_work * jcp.oh * jcp.od;

            size_t start {0}, end {0};
            balance211(work_amount, nthr_ic_b_, ti->ithr_ic_b, start, end);
            my_work = end - start;

            if (jcp.ndims == 5)
                nd_iterator_init(start, g, ti->g_work, oc_b, ti->oc_b_work, d,
                        jcp.od, j, jcp.oh);
            else
                nd_iterator_init(
                        start, g, ti->g_work, oc_b, ti->oc_b_work, j, jcp.oh);
            g += ti->g_start;
            oc_b += ti->oc_b_start;
            oc = g * jcp.nb_oc + oc_b;
        } else {
            oc = g * jcp.nb_oc + oc_b;
            g = 0;
            oc_b = 0;
        }

        const diff_dst_data_t *diff_dst = (jcp.ndims == 5)
                ? &ti->diff_dst[diff_dst_d.blk_off(img, oc, d, j)]
                : &ti->diff_dst[diff_dst_d.blk_off(img, oc, j)];
        diff_dst_data_t *tr_diff_dst = (jcp.ndims == 5)
                ? &ti->tr_diff_dst[tr_diff_dst_off_3d(g, oc_b, d, j)]
                : &ti->tr_diff_dst[tr_diff_dst_off(g, oc_b, j)];

        trans_dst(tr_diff_dst, diff_dst, my_work);
    };

    for (int img = ti->img_start; img < ti->img_end; ++img) {
        auto p = jit_conv_call_s();
        if (dnnl_thr_syncable()) {
            using simple_barrier::barrier;
            // TODO: try to call local transpositions just before jit kernel
            /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
            if (jcp.transpose_src) {
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
                uker_trans(img);
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            }
            if (jcp.transpose_dst) {
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
                diff_dst_trans(img);
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            }
        }

        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
            const int _oc = g * jcp.nb_oc + oc_b;
            const int _ic = g * jcp.nb_ic + ic_b;
            if (jcp.transpose_src) {
                if (!dnnl_thr_syncable()) {
                    uker_trans(img, g, ic_b);
                    if (jcp.ndims == 5) {
                        p.src = &ti->tr_src[tr_src_off_3d(g, ic_b, 0, 0)];
                    } else {
                        p.src = &ti->tr_src[tr_src_off(g, ic_b, 0)];
                    }
                } else {
                    if (jcp.ndims == 5) {
                        p.src = &ti->tr_src[tr_src_off_3d(g, ic_b, 0, 0)];
                    } else {
                        p.src = &ti->tr_src[tr_src_off(g, ic_b, 0)];
                    }
                }
            } else {
                p.src = &ti->src[src_d.blk_off(img, _ic)];
            }

            if (jcp.transpose_dst) {
                if (!dnnl_thr_syncable()) {
                    diff_dst_trans(img, g, oc_b);
                    if (jcp.ndims == 5) {
                        p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(
                                0, 0, 0, 0)];
                    } else {
                        p.dst = &ti->tr_diff_dst[tr_diff_dst_off(0, 0, 0)];
                    }
                } else {
                    if (jcp.ndims == 5) {
                        p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(
                                g, oc_b, 0, 0)];
                    } else {
                        p.dst = &ti->tr_diff_dst[tr_diff_dst_off(g, oc_b, 0)];
                    }
                }
            } else {
                p.dst = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)];
            }
            p.filt = diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = nullptr;
            p.channel = (img == ti->img_start);
            kernel_->jit_ker(&p);
        }
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::
        reduce_and_convert_diff_weights(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw
            * ((jcp.ndims == 5) ? jcp.kd : 1);

    const bool is_bf16_out = diff_weights_d.data_type() == data_type::bf16;
    if (nthr_mb_ == 1 && is_bf16_out) {
        // reduction is not required, only conversion
        for_(int g = ti->g_start; g < ti->g_end; g++)
        for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; oc_b++) {
            const size_t acc_size = (size_t)ti->ic_b_work * jcp.kh * jcp.kw
                    * ((jcp.ndims == 5) ? jcp.kd : 1) * jcp.ic_block
                    * jcp.oc_block;
            const size_t off
                    = wht_blk_off(diff_weights_d, g, oc_b, ti->ic_b_start);
            cvt_float_to_bfloat16((bfloat16_t *)(ti->diff_weights) + off,
                    (ti->wei_bia_reduction + off), acc_size);
        }
        return;
    }

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    if (dnnl_thr_syncable())
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const int ic_b_kh_work
            = ti->ic_b_work * ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
    const int work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int start {0}, end {0};
    balance211(work, nthr_mb_, ti->ithr_mb, start, end);
    if (start == end) return;

    const int _start_nthr_mb = 1;
    for (int thr_mb = _start_nthr_mb; thr_mb < nthr_mb_; ++thr_mb) {
        int w = start;
        int sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_kh_start {0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const int g = ti->g_start + sub_g_start;
            const int oc_b = ti->oc_b_start + sub_oc_b_start;
            const int ic_b = ti->ic_b_start
                    + sub_ic_b_kh_start / ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
            const int kX
                    = sub_ic_b_kh_start % ((jcp.ndims == 5) ? jcp.kd : jcp.kh);

            const size_t acc_size = (size_t)jcp.kw * jcp.ic_block * jcp.oc_block
                    * ((jcp.ndims == 5) ? jcp.kh : 1)
                    * nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start);

            const size_t off = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kX);

            float *wei_reduced = is_bf16_out
                    ? ti->wei_bia_reduction + off
                    : (float *)(ti->diff_weights) + off;

            int thr_mb_buffer_idx = is_bf16_out ? thr_mb : thr_mb - 1;
            float *wei_to_reduce = ti->wei_bia_reduction
                    + thr_mb_buffer_idx * wei_size + off;

            if (is_bf16_out && thr_mb == nthr_mb_ - 1)
                // the last iteration for bfloat16 requires conversion and
                // store to diff_weights array
                add_floats_and_cvt_to_bfloat16(
                        (bfloat16_t *)(ti->diff_weights) + off, wei_reduced,
                        wei_to_reduce, acc_size);
            else
                acc_ker_->accumulate(wei_reduced, wei_to_reduce, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::compute_diff_bias(
        const thread_info_t *ti, const exec_ctx_t &ctx) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    auto rb = this->reducer_bias_;
    assert(nthr_ == rb->balancer().nthr_);

    const auto reducer_bia_scratchpad
            = memory_tracking::grantor_t(ti->scratchpad, prefix_reducer_bia);

    auto scratchpad = ctx.get_scratchpad_grantor();
    auto diff_dst_cvt_wsp
            = scratchpad.template get<float>(key_conv_dst_bf16_convert_wsp);

    const auto &jcp = kernel_->jcp;

    const int batch_job_start = rb->balancer().ithr_job_off(ti->ithr);
    const int b_njobs = rb->balancer().ithr_njobs(ti->ithr);

    if (b_njobs == 0) return;

    /* reduction dimension */
    int img_start {0}, img_end {0};
    balance211(jcp.mb, rb->balancer().nthr_per_group_,
            rb->balancer().id_in_group(ti->ithr), img_start, img_end);

    /* jobs */
    int g_start {0}, ocb_start {0};
    nd_iterator_init(
            batch_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_oc);
    for (int img = img_start; img < img_end; ++img) {
        int g = g_start, ocb = ocb_start;
        for (int batch_job_loc = 0; batch_job_loc < b_njobs; ++batch_job_loc) {
            const size_t _oc = g * jcp.nb_oc + ocb;

            const diff_dst_data_t *diff_dst
                    = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)];
            float *d_bias = &rb->get_local_ptr(ti->ithr, ti->diff_bias,
                    reducer_bia_scratchpad)[batch_job_loc
                    * rb->balancer().job_size_];

            const size_t dst_nelems
                    = (size_t)jcp.oh * jcp.ow * jcp.od * jcp.oc_block;
            auto dd_wsp = diff_dst_cvt_wsp + dst_nelems * ti->ithr;
            cvt_bfloat16_to_float(dd_wsp, diff_dst, dst_nelems);

            if (img == img_start)
                for (int o = 0; o < 16; ++o)
                    d_bias[o] = 0;
            for (int hw = 0; hw < jcp.oh * jcp.ow * jcp.od; ++hw) {
                PRAGMA_OMP_SIMD()
                for (int o = 0; o < 16; ++o)
                    d_bias[o] += dd_wsp[o];
                dd_wsp += 16;
            }

            nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc);
        }
    }

    if (dnnl_thr_syncable())
        rb->reduce(ti->ithr, ti->diff_bias, reducer_bia_scratchpad);
}

void jit_avx512_core_bf16_convolution_bwd_weights_t::prepare_scratchpad_data(
        const exec_ctx_t &ctx) const {
    auto scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = pd()->jcp_;

    // TODO: remove initialization of scratchpad by zeroes for 3d case
    if (jcp.ndims == 5 && (jcp.nthr_mb > 1 || jcp.wei_dt == data_type::bf16)) {
        auto wei_bia_reduction
                = scratchpad.template get<float>(key_conv_wei_bia_reduction);
        const int num_wei_buffers
                = jcp.wei_dt == data_type::bf16 ? jcp.nthr_mb : jcp.nthr_mb - 1;
        const size_t wei_size
                = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw * jcp.kd;
        const size_t bia_size = jcp.ngroups * jcp.oc;

        const size_t b_wei_size = (wei_size + bia_size) * num_wei_buffers;
        utils::array_set(wei_bia_reduction, 0.f, b_wei_size);
    }
    if (jcp.transpose_src) {
        // XXX: See the comment about tr_iw and guarding elements in
        // jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf()
        auto tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
        // Zero out guard elements that cross a buffer boundary to prevent a
        // race condition due to buffer overflows from memory optimization where
        // buffers sharing padding
        for (size_t ithr = 1; ithr <= jcp.tr_src_buf_count; ++ithr) {
            src_data_t *ts = &tr_src[ithr * jcp.tr_src_buf_size];
            for (int i = 0; i < jcp.tr_src_num_guard_elems; ++i)
                ts[i] = 0;
        }

        if (dnnl_thr_syncable() && jcp.nthr_oc_b > 1) {
            const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
            auto tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_src_bctx);
            for (int i = 0; i < tr_src_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_src_bctx[i]);
        }
    }
    if (dnnl_thr_syncable() && jcp.transpose_dst) {
        if (jcp.nthr_ic_b > 1) {
            const int tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
            auto tr_diff_dst_bctx
                    = scratchpad.template get<simple_barrier::ctx_t>(
                            key_conv_tr_diff_dst_bctx);
            for (int i = 0; i < tr_diff_dst_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_diff_dst_bctx[i]);
        }
    }

    if (dnnl_thr_syncable()
            && (nthr_mb_ > 1
                    || pd()->diff_weights_md(0)->data_type
                            == data_type::bf16)) {
        // TODO: don't use barrier for case
        // diff_weights_type == data_type::bf16 && nthr_mb_ == 1
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx));
    }

    const auto reducer_bia_scratchpad
            = memory_tracking::grantor_t(scratchpad, prefix_reducer_bia);
    auto rb = this->reducer_bias_;
    rb->init(reducer_bia_scratchpad);
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    prepare_scratchpad_data(ctx);

    const int _start_nthr_mb
            = pd()->diff_weights_md(0)->data_type != data_type::bf16;
    parallel(nthr_, [&](const int ithr, const int nthr) {
        assert(nthr_ == nthr);
        assert(utils::one_of(pd()->ndims(), 3, 4, 5));

        thread_info_t thread_info(this, ctx, ithr);
        switch (pd()->jcp_.harness) {
            case harness_2d_reduction:
                compute_diff_weights_2d(&thread_info);
                if (dnnl_thr_syncable() && nthr_mb_ > _start_nthr_mb)
                    reduce_and_convert_diff_weights(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info, ctx);
                break;
            case harness_3d_reduction:
                compute_diff_weights_3d(&thread_info);
                if (dnnl_thr_syncable() && nthr_mb_ > _start_nthr_mb)
                    reduce_and_convert_diff_weights(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info, ctx);
                break;
            case harness_compute_full_spatial:
            case harness_mb_reduction:
                compute_diff_weights(&thread_info);
                if (dnnl_thr_syncable() && nthr_mb_ > _start_nthr_mb)
                    reduce_and_convert_diff_weights(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info, ctx);
                break;
            default: assert(!"Invalid harness type");
        }
    });

    if (!dnnl_thr_syncable()
            && (nthr_mb_ > _start_nthr_mb || pd()->with_bias())) {
        parallel(nthr_, [&](const int ithr, const int nthr) {
            assert(nthr_ == nthr);
            thread_info_t thread_info(this, ctx, ithr);
            if (nthr_mb_ > _start_nthr_mb)
                reduce_and_convert_diff_weights(&thread_info);
            if (pd()->with_bias()) {
                auto rb = this->reducer_bias_;
                assert(nthr == rb->balancer().nthr_);
                MAYBE_UNUSED(nthr);
                if (rb->balancer().ithr_njobs(thread_info.ithr) == 0) return;
                const auto reducer_bia_scratchpad = memory_tracking::grantor_t(
                        thread_info.scratchpad, prefix_reducer_bia);
                rb->reduce_nolock(thread_info.ithr, thread_info.diff_bias,
                        reducer_bia_scratchpad);
            }
        });
    }

    /* TODO: put that into compute_diff_bias() */
    if (pd()->jcp_.bia_dt == data_type::bf16) {
        auto diff_bias_f32 = ctx.get_scratchpad_grantor().template get<float>(
                key_conv_bias_bf16_convert_wsp);
        auto diff_bias_in = CTX_OUT_MEM(
                prec_traits<data_type::bf16>::type *, DNNL_ARG_DIFF_BIAS);
        cvt_float_to_bfloat16(diff_bias_in, diff_bias_f32,
                pd()->jcp_.oc_without_padding * pd()->jcp_.ngroups);

    } else if (pd()->wants_padded_bias()) {
        auto diff_bias = ctx.get_scratchpad_grantor().template get<const float>(
                key_conv_padded_bias);
        auto diff_bias_in = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);
        utils::array_copy(
                diff_bias_in, diff_bias, pd()->jcp_.oc_without_padding);
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
