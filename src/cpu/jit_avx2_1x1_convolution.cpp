/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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
#include "jit_avx2_1x1_convolution.hpp"
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
void _jit_avx2_1x1_convolution_fwd_t<with_relu>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const auto &jcp = kernel_->jcp;

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto ker = [&](const int ithr, const int nthr) {
        // TODO (Roma): remove this restriction
        assert(jcp.stride_w == 1 && jcp.stride_h == 1);

        jit_1x1_conv_call_s par_conv = {};

        const int nb_oc = jcp.nb_load;
        const int nb_ic = jcp.nb_reduce;
        const int nb_ic_blocking = jcp.nb_reduce_blocking;
        const int os_block = jcp.bcast_block;

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int iwork = start;
        while (iwork < end) {
            int n{0}, g{0}, osb{0};
            nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
                    jcp.nb_bcast);

            const int bcast_step_rem = jcp.nb_bcast - osb;
            int bcast_step = bcast_step_rem <= jcp.nb_bcast_blocking_max
                ? bcast_step_rem : jcp.nb_bcast_blocking;
            bcast_step = nstl::min(bcast_step, end - iwork);

            const int os = osb * os_block;
            const int ow = os % jcp.ow;
            const int oh = os / jcp.ow;
            const int iw = nstl::max(ow * jcp.stride_w - jcp.l_pad, 0);
            const int ih = nstl::max(oh * jcp.stride_h - jcp.t_pad, 0);

            par_conv.bcast_dim = this_block_size(os, jcp.os,
                    bcast_step * os_block);

            int ocb = 0;
            while (ocb < jcp.nb_load) {
                const int load_step_rem = jcp.nb_load - ocb;
                const int load_step = load_step_rem < jcp.nb_load_blocking_max
                    ? load_step_rem : jcp.nb_load_blocking;

                const size_t _ocb = g * nb_oc + ocb;
                par_conv.load_dim = this_block_size(ocb * jcp.oc_block, jcp.oc,
                        load_step * jcp.oc_block);

                const size_t dst_off = dst_d.blk_off(n, _ocb, oh, ow);
                par_conv.output_data = &dst[dst_off];

                par_conv.bias_data = &bias[_ocb * jcp.oc_block];

                for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                    par_conv.reduce_pos_flag = 0
                        | (icb == 0)
                            * jit_avx2_1x1_conv_kernel_f32::REDUCE_FLAG_FIRST
                        | (icb + nb_ic_blocking >= nb_ic)
                            * jit_avx2_1x1_conv_kernel_f32::REDUCE_FLAG_LAST;

                    par_conv.reduce_dim = this_block_size(icb * jcp.ic_block,
                            jcp.ic, nb_ic_blocking * jcp.ic_block);

                    const size_t _icb = g * nb_ic + icb;
                    const size_t src_off = src_d.blk_off(n, _icb, ih, iw);
                    par_conv.bcast_data = &src[src_off];

                    par_conv.load_data = &weights[conf_.with_groups()
                        ? weights_d.blk_off(g, ocb, icb)
                        : weights_d.blk_off(ocb, icb)];

                    kernel_->jit_ker(&par_conv);
                }

                ocb += load_step;
            }

            iwork += bcast_step;
        }
    };

#   pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template void _jit_avx2_1x1_convolution_fwd_t<true>::execute_forward();
template void _jit_avx2_1x1_convolution_fwd_t<false>::execute_forward();

void jit_avx2_1x1_convolution_bwd_data_t::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());

    const auto &jcp = kernel_->jcp;

    const size_t work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int icb = 0;
        while (icb < jcp.nb_load) {
            int load_step_rem = jcp.nb_load - icb;
            int load_step = (load_step_rem < jcp.nb_load_blocking_max)
                                    ? load_step_rem
                                    : jcp.nb_load_blocking;

            size_t n{0}, g{0}, isb{0};
            int iwork = (int)start;
            while (iwork < (int)end) {
                nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, isb,
                                 jcp.nb_bcast);
                int bcast_step = jcp.nb_bcast_blocking;
                int bcast_step_rem = jcp.nb_bcast - isb;
                if (bcast_step_rem <= jcp.nb_bcast_blocking_max)
                    bcast_step = bcast_step_rem;
                if ((iwork + bcast_step) > (int)end)
                    bcast_step = (int)end - iwork;

                for (int ocb = 0; ocb < jcp.nb_reduce;
                     ocb += jcp.nb_reduce_blocking) {
                    jit_1x1_conv_call_s par_conv = {};

                    // TODO (Roma): remove this restriction
                    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

                    int nb_ic = jcp.nb_load, nb_oc = jcp.nb_reduce;
                    int is_block = jcp.bcast_block;
                    int nb_ic_blocking = load_step;
                    int nb_oc_blocking = jcp.nb_reduce_blocking;

                    int is = isb * is_block;
                    int nb_is_blocking = bcast_step;

                    int iw = is % jcp.iw;
                    int ih = is / jcp.iw;
                    size_t _icb = g * nb_ic + icb;
                    size_t diff_src_off = diff_src_d.blk_off(n, _icb, ih, iw);
                    par_conv.output_data = &diff_src[diff_src_off];

                    int ow = iw;
                    int oh = ih;
                    size_t _ocb = g * nb_oc + ocb;
                    size_t diff_dst_off = diff_dst_d.blk_off(n, _ocb, oh, ow);
                    par_conv.bcast_data = &diff_dst[diff_dst_off];

                    par_conv.load_data = &weights[conf_.with_groups()
                                             ? weights_d.blk_off(g, ocb, icb)
                                             : weights_d.blk_off(ocb, icb)];

                    par_conv.reduce_pos_flag =
                        (ocb == 0
                            ? jit_avx2_1x1_conv_kernel_f32::REDUCE_FLAG_FIRST
                            : 0)
                          | (ocb + nb_oc_blocking >= nb_oc
                            ? jit_avx2_1x1_conv_kernel_f32::REDUCE_FLAG_LAST
                            : 0);

                    par_conv.reduce_dim = this_block_size(
                        ocb * jcp.oc_block, jcp.oc,
                        nb_oc_blocking * jcp.oc_block);
                    par_conv.bcast_dim = this_block_size(
                            is, jcp.is, nb_is_blocking * is_block);
                    par_conv.load_dim = this_block_size(
                        icb * jcp.ic_block, jcp.ic,
                        nb_ic_blocking * jcp.ic_block);

                    kernel_->jit_ker(&par_conv);
                }
                iwork += bcast_step;
            }
            icb += load_step;
        }
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

void jit_avx2_1x1_convolution_bwd_weights_t::execute_backward_weights() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    const auto &jcp = kernel_->jcp;

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int nb_ic = jcp.nb_bcast;
    const int nb_ic_blocking = jcp.nb_bcast_blocking;
    const int bcast_work = div_up(nb_ic, nb_ic_blocking);

    const int nb_oc = jcp.nb_load;
    const int nb_oc_blocking = jcp.nb_load_blocking;
    const int load_work = div_up(nb_oc, nb_oc_blocking);

    const int sp_dim = jcp.reduce_dim;
    const int mb_sp_work = jcp.mb * sp_dim;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto oc_ic_sp_loop = [=](int sp_start, int sp_end, bool first_image,
            data_t *store_to, size_t store_to_ld, const data_t *diff_dst,
            const data_t *src) {
        jit_1x1_conv_call_s p = {};
        p.output_stride = store_to_ld * sizeof(float);
        const int sp_step_def = jcp.nb_reduce_blocking * jcp.reduce_block;

        int oc_b_step = 0;
        for (int oc_b = 0; oc_b < nb_oc_blocking; oc_b += oc_b_step) {
            oc_b_step = step(12, nb_oc_blocking - oc_b, 18);
            p.load_dim = oc_b_step * jcp.oc_block;

            int ic_b_step = 0;
            for (int ic_b = 0; ic_b < nb_ic_blocking; ic_b += ic_b_step) {
                ic_b_step = step(12, nb_ic_blocking - ic_b, 18);
                p.bcast_dim = ic_b_step * jcp.ic_block;

                p.output_data = store_to + oc_b * store_to_ld
                    + ic_b * jcp.ic_block * jcp.oc_block;

                /* spatial reduction */
                int sp_step = 0;
                for (int sp = sp_start; sp < sp_end; sp += sp_step) {
                    sp_step = step(sp_step_def, sp_end - sp, 192);
                    p.reduce_dim = sp_step;

                    p.reduce_pos_flag = (sp == sp_start && first_image)
                        * jit_avx2_1x1_conv_kernel_f32::REDUCE_FLAG_FIRST;

                    p.load_data = diff_dst
                        + (oc_b * jcp.reduce_dim + sp) * jcp.oc_block;
                    p.bcast_data = src
                        + (ic_b * jcp.reduce_dim + sp) * jcp.ic_block;

                    kernel_->jit_ker(&p);
                }
            }
        }
    };

    auto ker = [&](const int ithr, const int nthr) {
        auto rw = this->reducer_weights_;
        assert(nthr == rw->balancer_.nthr_);

        const int w_njobs = rw->balancer_.ithr_njobs(ithr);
        if (w_njobs == 0) return;

        /* setup: independent work (oc, ic) */
        const int w_job_start = rw->balancer_.ithr_job_off(ithr);
        int g{0}, load_i{0}, bcast_i{0};
        nd_iterator_init(w_job_start, g, jcp.ngroups, load_i, load_work,
                bcast_i, bcast_work);

        /* setup: reduction work (mb, sp) */
        int mb_sp_start{0}, mb_sp_end{0};
        balance211(mb_sp_work, rw->balancer_.nthr_per_group_,
                rw->balancer_.id_in_group(ithr), mb_sp_start, mb_sp_end);
        int img_start{0}, sp_start{0};
        nd_iterator_init(mb_sp_start, img_start, jcp.mb, sp_start, sp_dim);

        /* independent work */
        for (int iwork = 0; iwork < w_njobs; ++iwork) {
            const int oc_b = nb_oc_blocking * load_i;
            const int ic_b = nb_ic_blocking * bcast_i;

            const int _ic_b = g * nb_ic + ic_b;
            const int _oc_b = g * nb_oc + oc_b;

            data_t *store_to;
            size_t store_to_ld;

            if (rw->balancer_.nthr_per_group_ == 1 ||
                    (rw->balancer_.master(ithr) && rw->master_uses_dst_)) {
                const size_t off = conf_.with_groups()
                    ? diff_weights_d.blk_off(g, oc_b, ic_b)
                    : diff_weights_d.blk_off(oc_b, ic_b);
                store_to = &diff_weights[off];
                store_to_ld = jcp.ic * jcp.oc_block;
            } else {
                const size_t off = iwork * rw->balancer_.job_size_;
                store_to = &rw->get_local_ptr(ithr, nullptr)[off];
                store_to_ld = nb_ic_blocking * jcp.ic_block * jcp.oc_block;
            }

            /* reduction work */
            int img = img_start;
            int sp = sp_start;
            int sp_step = 0;
            for (int mb_sp = mb_sp_start; mb_sp < mb_sp_end; mb_sp += sp_step)
            {
                sp_step = nstl::min(sp_dim - sp, mb_sp_end - mb_sp);

                const bool first_image = img == img_start;
                oc_ic_sp_loop(sp, sp + sp_step, first_image, store_to,
                        store_to_ld, &diff_dst[diff_dst_d.blk_off(img, _oc_b)],
                        &src[src_d.blk_off(img, _ic_b)]);

                sp = 0;
                img += 1;
            }

            nd_iterator_step(g, jcp.ngroups, load_i, load_work, bcast_i,
                             bcast_work);
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
        nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start, nb_oc);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const size_t _oc = g * nb_oc + ocb;

                const data_t *d_dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                data_t *d_bias = &rb->get_local_ptr(ithr, diff_bias)[
                    b_job_loc * rb->balancer_.job_size_];

                if (img == img_start)
                    for (int o = 0; o < 8; ++o) d_bias[o] = 0.;
                for (int hw = 0; hw < jcp.oh * jcp.ow; ++hw) {
                    for (int o = 0; o < 8; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += 8;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, nb_oc);
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
