/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
#include "jit_avx2_convolution.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <bool with_relu>
void _jit_avx2_convolution_fwd_t<with_relu>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = jcp.mb * jcp.ngroups * ocb_work * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int icbb = 0;
        while (icbb < jcp.nb_ic) {
            int icb_step = jcp.nb_ic_blocking;
            int icb_step_rem = jcp.nb_ic - icbb;
            if (icb_step_rem < jcp.nb_ic_blocking_max)
                icb_step = icb_step_rem;

            size_t n{0}, g{0}, ocbb{0}, oh{0};
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work,
                             oh, jcp.oh);
            for (size_t iwork = start; iwork < end; ++iwork) {
                int ocb = ocbb * jcp.nb_oc_blocking;
                int ocb_num = jcp.nb_oc_blocking;

                for (int icb = icbb; icb < icbb + icb_step; ++icb) {
                    jit_conv_call_s par_conv = {};

                    const int ij = oh * jcp.stride_h;
                    const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
                    const int i_b_overflow = nstl::max(jcp.ih, ij
                        + (jcp.kh-1) * (jcp.dilate_h+1) - jcp.t_pad+1) - jcp.ih;

                    const size_t _oc = g * jcp.nb_oc + ocb;
                    const size_t _ic = g * jcp.nb_ic + icb;

                    const int ih = nstl::max(ij - jcp.t_pad
                        + div_up(i_t_overflow,
                                 (jcp.dilate_h+1)) * (jcp.dilate_h + 1), 0);
                    par_conv.src = &src[src_d.blk_off(n,
                        jcp.ic == 3 ? 0 : _ic, ih, 0)];

                    par_conv.dst = &dst[dst_d.blk_off(n, _oc, oh, 0)];

                    const int wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
                    par_conv.filt = &weights[conf_.with_groups()
                                        ? weights_d.blk_off(g, ocb,
                                            jcp.ic == 3 ? 0 : icb, wh, 0)
                                        : weights_d.blk_off(ocb,
                                            jcp.ic == 3 ? 0 : icb, wh, 0)];

                    if (icb == 0) {
                        if (bias)
                            par_conv.bias =
                                    &bias[bias_d.blk_off(_oc * jcp.oc_block)];
                        par_conv.flags |= FLAG_IC_FIRST;
                    }

                    if (jcp.with_relu && icb + 1 == jcp.nb_ic) {
                        par_conv.flags |= FLAG_IC_LAST;
                    }

                    par_conv.oc_blocks =
                            nstl::min(ocb + ocb_num, jcp.nb_oc) - ocb;

                    par_conv.kw_padding = 0;
                    const int kh_padding = jcp.kh
                        - div_up(i_t_overflow, (jcp.dilate_h + 1))
                        - div_up(i_b_overflow, (jcp.dilate_h + 1));
                    par_conv.kh_padding = nstl::max(0, kh_padding);
                    kernel_->jit_ker(&par_conv);
                }
                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work,
                                oh, jcp.oh);
            }
            icbb += icb_step;
        }
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template void _jit_avx2_convolution_fwd_t<true>::execute_forward();
template void _jit_avx2_convolution_fwd_t<false>::execute_forward();

void jit_avx2_convolution_bwd_data_t::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const auto &jcp = kernel_->jcp;

    int icb_work = jcp.nb_ic / jcp.nb_ic_blocking;
    const size_t work_amount = jcp.mb * jcp.ngroups * icb_work;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, g{0}, icbb{0};
        nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, icbb, icb_work);
        for (size_t iwork = start; iwork < end; ++iwork) {
            for (int oc = 0; oc < jcp.nb_oc; ++oc) {
                for (int ih = 0; ih < jcp.ih; ++ih) {
                    jit_conv_call_s par_conv = {};

                    const int i_t_overflow = nstl::max(0,
                                        jcp.kh - 1 - ih - jcp.t_pad);
                    const int b_pad = jcp.ihp - jcp.ih - jcp.t_pad;
                    const int i_b_overflow = nstl::max(0,
                                        jcp.kh - 1 - (jcp.ih - 1 - ih) - b_pad);
                    const int oh = ih + jcp.t_pad - i_b_overflow;

                    const int simd_w = 8;

                    par_conv.src = &diff_src[diff_src_d.blk_off(n,
                            /*jcp.ic == 3 ? 0 :*/
                            g * jcp.nb_ic + jcp.nb_ic_blocking * icbb, ih, 0)];
                    par_conv.dst = &diff_dst[diff_dst_d.blk_off(
                            n, g * jcp.nb_oc + oc, oh, 0)];
                    par_conv.filt = &weights[
                        conf_.with_groups() ? weights_d.blk_off(g, oc,
                                jcp.ic == 3 ? 0 : jcp.nb_ic_blocking * icbb,
                                i_b_overflow, 0)
                        : weights_d.blk_off(oc,
                                jcp.ic == 3 ? 0 : jcp.nb_ic_blocking * icbb,
                                i_b_overflow, 0)];
                    par_conv.src_prf = nullptr;
                    par_conv.dst_prf = nullptr;
                    par_conv.filt_prf = nullptr;
                    // TODO: move initialization into the kernel
                    if (oc == 0)
                    {
                        for (int iw = 0; iw < jcp.iw; iw++)
                        {
                            for (int b = 0; b < jcp.nb_ic_blocking; b++)
                            {
                                int current_ic =
                                    (jcp.ic == 3 ? 0 : g * jcp.nb_ic)
                                    + jcp.nb_ic_blocking * icbb + b;
                                int current_idx =
                                    diff_src_d.blk_off(n, current_ic, ih, iw);
                                for (int v = 0; v < simd_w; v++)
                                    diff_src[current_idx + v] = 0.0;
                            }
                        }
                    }

                    par_conv.kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
                    par_conv.kw_padding = 0;

                    kernel_->jit_ker(&par_conv);
                }
            }
            nd_iterator_step(n, jcp.mb, g, jcp.ngroups, icbb, icb_work);
        }
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

void jit_avx2_convolution_bwd_weights_t::execute_backward_weights() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper src_d(conf_.src_pd(0));
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;

    auto ker = [&](int ithr, int nthr) {
        auto rw = this->reducer_weights_;
        assert(nthr == rw->balancer_.nthr_);

        const int w_job_start = rw->balancer_.ithr_job_off(ithr);
        const int w_njobs = rw->balancer_.ithr_njobs(ithr);

        if (w_njobs == 0) return;

        /* reduction dimension */
        int img_start{0}, img_end{0};
        balance211(jcp.mb, rw->balancer_.nthr_per_group_,
                rw->balancer_.id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start{0}, ocb_start{0}, icb_start{0};
        nd_iterator_init(w_job_start, g_start, jcp.ngroups, ocb_start,
                jcp.nb_oc, icb_start, jcp.nb_ic);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start, icb = icb_start;
            for (int w_job_loc = 0; w_job_loc < w_njobs; ++w_job_loc) {
                const size_t _oc = g * jcp.nb_oc + ocb;
                const size_t _ic = g * jcp.nb_ic + icb;

                jit_conv_call_s par_conv = {};
                par_conv.src = &src[src_d.blk_off(img, _ic)];
                par_conv.dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                par_conv.filt = &rw->get_local_ptr(ithr, diff_weights)[
                    w_job_loc * rw->balancer_.job_size_];

                /* TODO: put dw <-- 0 in kernel */
                if (img == img_start)
                    array_set((data_t *)par_conv.filt, 0,
                            rw->balancer_.job_size_);
                kernel_->jit_ker(&par_conv);

                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc, icb,
                        jcp.nb_ic);
            }
        }
        rw->reduce(ithr, diff_weights);
    };

    auto ker_bias = [&](int ithr, int nthr) {
        auto rb = this->reducer_bias_;
        assert(nthr == rb->balancer_.nthr_);

        const int b_job_start = rb->balancer_.ithr_job_off(ithr);
        const int b_njobs = rb->balancer_.ithr_njobs(ithr);

        if (b_njobs == 0) return;

        /* reduction dimension */
        int img_start{0}, img_end{0};
        balance211(jcp.mb, rb->balancer_.nthr_per_group_,
                rb->balancer_.id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start{0}, ocb_start{0};
        nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start,
                jcp.nb_oc);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const size_t _oc = g * jcp.nb_oc + ocb;

                const data_t *d_dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                data_t *d_bias = &rb->get_local_ptr(ithr, diff_bias)[
                    b_job_loc * rb->balancer_.job_size_];

                if (img == img_start)
                    for (int o = 0; o < 8; ++o)
                        d_bias[o] = 0.;

                for (int hw = 0; hw < jcp.oh * jcp.ow; ++hw) {
#                   pragma omp simd
                    for (int o = 0; o < 8; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += 8;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc);
            }
        }
        rb->reduce(ithr, diff_bias);
    };

#   pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        ker(ithr, nthr);
        if (conf_.with_bias())
            ker_bias(ithr, nthr);
    }
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
