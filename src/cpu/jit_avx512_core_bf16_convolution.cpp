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

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_bf16_convolution.hpp"
#include "bfloat16.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

using namespace nstl;

#define wht_blk_off(d, g, ...) \
        (pd()->with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

void jit_avx512_core_bf16_convolution_fwd_t::prepare_padded_bias(
        const char *&bias, const memory_tracking::grantor_t &scratchpad) const {
    if (!pd()->wants_padded_bias()) return;

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;
    auto padded_bias = scratchpad.template get<char>(
            memory_tracking::names::key_conv_padded_bias);
    utils::array_copy(padded_bias, bias, bia_dt_size * pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + bia_dt_size * pd()->jcp_.oc_without_padding,
            0.f, bia_dt_size * (pd()->jcp_.oc - pd()->jcp_.oc_without_padding));
    bias = padded_bias;
}

void jit_avx512_core_bf16_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, MKLDNN_ARG_DST);

    prepare_padded_bias(bias, this->scratchpad(ctx));

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = kernel_->jcp;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh * jcp.nb_ow;

    int nthr;
    if (jcp.aligned_threads)
        nthr = jcp.aligned_threads;
    else
        nthr = mkldnn_get_max_threads();

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();
        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        int n{0}, g{0}, occ{0}, oh_s{0}, owb{0};

        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow,
                g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb,
                occ, oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
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
            int ow_s =  owb * jcp.ow_block;
            int iw_s =  ow_s * jcp.stride_w;

            auto bias_w = bias ? bias + bia_dt_size * g_oc : nullptr;

            size_t dst_off, src_off;
            if (jcp.ndims == 3) {
                dst_off = dst_d.blk_off(n, g_ocb, ow_s);
                src_off = src_d.blk_off(n, g_icb, iw_s);
            } else {
                dst_off = dst_d.blk_off(n, g_ocb, oh_s, ow_s);
                src_off = src_d.blk_off(n, g_icb, ih_s, iw_s);
            }
            auto dst_w = dst + jcp.typesize_out * dst_off;
            auto src_w = src + src_off;
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0);
            for (int oj = oh_s, ij = ih_s; oj < oh_e;
                ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = div_up(max(0, -ij), dilate_h);
                int i_b_overflow = div_up(max(0, ij - jcp.ih
                    + (jcp.kh - 1) * dilate_h + 1), dilate_h);
                int kh_padding = nstl::max(
                        0, jcp.kh - i_t_overflow - i_b_overflow);

                auto aux_src = src_w
                        + i_t_overflow * dilate_h * src_h_stride;
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
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                    g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, occ,
                    oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    });
}

void jit_avx512_core_bf16_convolution_bwd_data_t
    ::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, MKLDNN_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(char *, MKLDNN_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = kernel_->jcp;

    parallel(0, [&](const int ithr, const int nthr) {
        int start{0}, end{0};
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_amount = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih;
        balance211(work_amount, nthr, ithr, start, end);

        auto par_conv = jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        bool is_fast_path = jcp.dilate_h == 0 && jcp.stride_h == 1;

        int n{0}, g{0}, icc{0}, ih_s{0};
        if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start,
                icc, ic_chunks, g, jcp.ngroups, n, jcp.mb, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_gnc)
            nd_iterator_init(start,
                g, jcp.ngroups, n, jcp.mb, icc, ic_chunks, ih_s, jcp.ih);
        else
            assert(!"unsupported loop order");

        while (start < end) {
            int icb = icc * jcp.nb_ic_blocking;
            int g_icb = g * jcp.nb_ic + icb;
            int g_ocb = g * jcp.nb_oc;

            int work_rem = end - start;
            int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;

            auto diff_src_w = diff_src + jcp.typesize_out * diff_src_d.blk_off(n, g_icb);
            auto diff_dst_w = diff_dst
                + diff_dst_d.blk_off(n, g_ocb);
            auto wht_w = weights + wht_blk_off(weights_d, g, 0, icb);

            for (int ij = ih_s; ij < ih_e; ++ij) {
                int oj, k_len, k_lo;
                if (is_fast_path) { // dilate == 0 && stride == 1
                    int i_t_overflow = max(0, jcp.kh - 1 - ij
                        - jcp.t_pad);
                    int i_b_overflow = max(0, jcp.kh - jcp.ih + ij
                        - jcp.b_pad);
                    k_len = jcp.kh - i_t_overflow - i_b_overflow;
                    k_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow;
                } else if (jcp.dilate_h != 0) { // stride == 1
                    int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    int i_t_overflow
                        = div_up(max(0, (jcp.kh - 1) * dilate_h
                                - ij - jcp.t_pad), dilate_h);
                    int i_b_overflow
                        = div_up(max(0, (jcp.kh - 1) * dilate_h + 1
                                - jcp.ih + ij - jcp.b_pad), dilate_h);
                    k_len = jcp.kh - i_t_overflow - i_b_overflow;
                    k_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                } else { // dilate == 0
                    int i_t_overflow = max(0, (jcp.kh - 1 - ij
                        - jcp.t_pad) / jcp.stride_h);
                    int i_b_overflow = max(0, (jcp.kh - jcp.ih + ij
                        - jcp.b_pad) / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1 - abs((jcp.ih - 1
                        + jcp.b_pad - ij) % jcp.stride_h);
                    int overflow_kh_lo = (ij + jcp.t_pad)
                        % jcp.stride_h;

                    k_len = (overflow_kh_hi - overflow_kh_lo)
                        / jcp.stride_h + 1 - i_t_overflow
                        - i_b_overflow;
                    k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                    oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                }
                assert(k_len >= 0);

                par_conv.src = diff_src_w + jcp.typesize_out * ij * diff_src_h_stride;
                par_conv.dst = diff_dst_w + oj * diff_dst_h_stride;
                par_conv.filt = wht_w + k_lo * wht_h_stride;
                par_conv.kh_padding = k_len;

                kernel_->jit_ker(&par_conv);
            }

            if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end,
                  icc, ic_chunks, g, jcp.ngroups, n, jcp.mb, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_jump(start, end,
                  g, jcp.ngroups, n, jcp.mb, icc, ic_chunks, ih_s, jcp.ih);
            else
                assert(!"unsupported loop order");
        }
    });
}

}
}
}
// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
