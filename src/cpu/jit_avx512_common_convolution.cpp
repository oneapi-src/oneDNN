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
#include "jit_avx512_common_convolution.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace nstl;

using jit_conv_ker_t = void (*)(jit_conv_call_s *);

inline void jit_conv_ker_pipeline(jit_conv_ker_t ker, jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding)
{
#define PIPELINE(field) \
    do { \
        p.field = p.field ## _prf; \
        p.field ## _prf = field; \
    } while (0)

    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    PIPELINE(kh_padding);

    if (p.src)
        ker(&p);
}

#define wht_blk_off(d, g, ...) \
        (conf_.with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

template <bool with_relu, data_type_t src_type, data_type_t wei_type,
          data_type_t dst_type>
void _jit_avx512_common_convolution_fwd_t
    <with_relu, src_type, wei_type, dst_type>::execute_forward()
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const dst_data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

#   pragma omp parallel
    {
        int ithr = omp_get_thread_num(), nthr = omp_get_num_threads();

        int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        int start, end;
        int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);

        int n{0}, g{0}, occ{0}, oh_s{0};
        if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start,
                    occ, oc_chunks, g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gnc)
            nd_iterator_init(start,
                    g, jcp.ngroups, n, jcp.mb, occ, oc_chunks, oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");

        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t src_c_stride = src_d.blk_off(0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        jit_conv_call_s par_conv = {0};
        while (start < end) {
            int ocb = occ * jcp.nb_oc_blocking;
            int g_ocb = g * jcp.nb_oc + ocb;
            int g_oc = g_ocb * jcp.oc_block;
            int g_icb = g * jcp.nb_ic;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

            auto bias_w = bias ? bias + bias_d.blk_off(g_oc) : 0;
            auto dst_w = dst + dst_d.blk_off(n, g_ocb, oh_s);
            auto src_w = src + src_d.blk_off(n, g_icb, ih_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb);

            for (int icb = 0; icb < jcp.nb_ic; ++icb) {
                auto src_c = src_w;
                auto dst_c = dst_w;
                for (int oj = oh_s, ij = ih_s;
                        oj < oh_e; ++oj, ij += jcp.stride_h)
                {
                    int i_t_overflow = -min(0, ij);
                    int i_b_overflow = max(jcp.ih, ij + jcp.kh) - jcp.ih;
                    int kh_padding
                        = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow);

                    jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                            src_c + i_t_overflow * src_h_stride,
                            dst_c,
                            wht_w + i_t_overflow * wht_h_stride,
                            bias_w,
                            icb, kh_padding);

                    src_c += src_h_stride * jcp.stride_h;
                    dst_c += dst_h_stride;
                }
                src_w += src_c_stride;
                wht_w += wht_ic_stride;
            }

            if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end,
                        occ, oc_chunks, g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_jump(start, end,
                        g, jcp.ngroups, n, jcp.mb, occ, oc_chunks, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }

        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                src, dst, weights, bias, 0, 0);
    }
}
template struct _jit_avx512_common_convolution_fwd_t<false, data_type::f32>;
template struct _jit_avx512_common_convolution_fwd_t<true, data_type::f32>;
template struct _jit_avx512_common_convolution_fwd_t<false, data_type::s16,
        data_type::s16, data_type::s32>;
template struct _jit_avx512_common_convolution_fwd_t<true, data_type::s16,
        data_type::s16, data_type::s32>;

template <data_type_t diff_dst_type, data_type_t wei_type,
          data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
          diff_src_type>::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>
                                                       (this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const auto &jcp = kernel_->jcp;

#   pragma omp parallel
    {
        int ithr = omp_get_thread_num(), nthr = omp_get_num_threads();

        int start, end;
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_amount = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih;
        balance211(work_amount, nthr, ithr, start, end);

        int n{0}, g{0}, icc{0}, ih_s{0};
        if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start,
                    icc, ic_chunks, g, jcp.ngroups, n, jcp.mb, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_gnc)
            nd_iterator_init(start,
                    g, jcp.ngroups, n, jcp.mb, icc, ic_chunks, ih_s, jcp.ih);
        else
            assert(!"unsupported loop order");

        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        jit_conv_call_s par_conv = {0};
        while (start < end) {
            int icb = icc * jcp.nb_ic_blocking;
            int g_icb = g * jcp.nb_ic + icb;
            int g_ocb = g * jcp.nb_oc;

            int work_rem = end - start;
            int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;

            auto diff_src_w = diff_src + diff_src_d.blk_off(n, g_icb);
            auto diff_dst_w = diff_dst + diff_dst_d.blk_off(n, g_ocb);
            auto wht_w = weights + wht_blk_off(weights_d, g, 0, icb);

            for (int ocb = 0; ocb < jcp.nb_oc; ++ocb) {
                for (int ij = ih_s; ij < ih_e; ++ij) {
                    int i_t_overflow = max(0, jcp.kh - 1 - ij - jcp.t_pad);
                    int i_b_overflow = max(0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                    int oj = ij + jcp.t_pad - i_b_overflow;

                    jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                            diff_src_w + ij * diff_src_h_stride,
                            diff_dst_w + oj * diff_dst_h_stride,
                            wht_w + i_b_overflow * wht_h_stride,
                            0, ocb, jcp.kh - i_t_overflow - i_b_overflow);
                }
                diff_dst_w += diff_dst_c_stride;
                wht_w += wht_oc_stride;
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

        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                diff_src, diff_dst, weights, 0, 0, 0);
    }
}

template struct jit_avx512_common_convolution_bwd_data_t<data_type::f32>;
template struct jit_avx512_common_convolution_bwd_data_t<data_type::s16,
    data_type::s16, data_type::s32>;

void jit_avx512_common_convolution_bwd_weights_t::execute_backward_weights() {
    auto src = reinterpret_cast<const data_t * > (this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t * > (this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper src_d(conf_.src_pd(0));
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;

    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw;
    const int wei_block_size = jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block;

    simple_barrier::ctx_t reduction_barrier;
    simple_barrier::ctx_init(&reduction_barrier);

    // TODO: use memory descriptor with the same fmt as src
    //       (or use a macro :))
    auto tr_src_off = [&](int img, int ic, int ij) {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_chn_size = tr_row_size * jcp.ih;
        const size_t tr_img_size = tr_chn_size * jcp.nb_ic * jcp.ngroups;

        return img * tr_img_size + ic * tr_chn_size + ij * tr_row_size;
    };

    auto uker_trans = [&](int ithr_mb, int img, int g_start, int g_work,
            int ic_b_start, int ic_b_work, int ithr, int nthr) {
        const int work_amount = g_work * ic_b_work * jcp.ih;

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int g{0}, ic_b{0}, j{0};
        nd_iterator_init(start, g, g_work, ic_b, ic_b_work, j, jcp.ih);
        g += g_start;
        ic_b += ic_b_start;

        const int _ic = g * jcp.nb_ic + ic_b;
        data_t *src1 = (data_t*)&src[src_d.blk_off(img, _ic, j)];
        data_t *tr_src1 = &tr_src_[tr_src_off(ithr_mb, _ic, j)];

        assert(jcp.ic_block == 16);
        const int src_stride = jcp.iw * jcp.ic_block;
        const int tr_src_stride = jcp.tr_iw * jcp.ic_block;

        const int pf_depth = 2;
        struct { data_t *src, *tr_src; } pf_circ_buf[pf_depth];
        const int my_work = end - start;

        for (int iwork = 0; iwork < my_work + pf_depth - 1; iwork++) {
            pf_circ_buf[iwork % pf_depth] = {src1, tr_src1};

            if (iwork >= pf_depth - 1) {
                int old_idx = (iwork - pf_depth + 1) % pf_depth;
                jit_src_transpose_s par_trans = {};
                par_trans.src = pf_circ_buf[old_idx].src;
                par_trans.tr_src = pf_circ_buf[old_idx].tr_src;
                par_trans.src_prf = src1;
                par_trans.tr_src_prf = tr_src1;
                trans_kernel_->jit_ker(&par_trans);
            }
            src1 += src_stride;
            tr_src1 += tr_src_stride;
        }
    };

    auto ker = [&](int ithr, int nthr) {
        const int ithr_ic_b = ithr % nthr_ic_b_;
        const int ithr_oc_b = ithr / nthr_ic_b_ % nthr_oc_b_;
        const int ithr_g = ithr / nthr_ic_b_ / nthr_oc_b_ % nthr_g_;
        const int ithr_mb = ithr / nthr_ic_b_ / nthr_oc_b_ / nthr_g_;

        const int ithr_but_oc = (ithr_mb * nthr_g_ + ithr_g) * nthr_ic_b_
            + ithr_ic_b;

        /* reduction dimension */
        int img_start{0}, img_end{0};
        balance211(jcp.mb, nthr_mb_, ithr_mb, img_start, img_end);

        /* independent dimensions */
        int g_start{0}, oc_b_start{0}, ic_b_start{0};
        int g_end{0}, oc_b_end{0}, ic_b_end{0};

        balance211(jcp.ngroups, nthr_g_, ithr_g, g_start, g_end);
        balance211(jcp.nb_oc, nthr_oc_b_, ithr_oc_b, oc_b_start, oc_b_end);
        balance211(jcp.nb_ic, nthr_ic_b_, ithr_ic_b, ic_b_start, ic_b_end);

        const int g_work = g_end - g_start;
        const int oc_b_work = oc_b_end - oc_b_start;
        const int ic_b_work = ic_b_end - ic_b_start;

        data_t *diff_wei = ithr_mb == 0
            ? diff_weights : ws_reduction_ + (ithr_mb - 1) * wei_size;

        for (int img = img_start; img < img_end; ++img) {
            if (jcp.transpose_src) {
                /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
                if (nthr_oc_b_ > 1)
                    simple_barrier::barrier(&bctx_[ithr_but_oc], nthr_oc_b_);
                uker_trans(ithr_mb, img, g_start, g_work, ic_b_start,
                        ic_b_work, ithr_oc_b, nthr_oc_b_);
                if (nthr_oc_b_ > 1)
                    simple_barrier::barrier(&bctx_[ithr_but_oc], nthr_oc_b_);
            }

            for (int g = g_start; g < g_end; ++g) {
            for (int oc_b = oc_b_start; oc_b < oc_b_end; ++oc_b) {
            for (int ic_b = ic_b_start; ic_b < ic_b_end; ++ic_b) {
                const int _oc = g * jcp.nb_oc + oc_b;
                const int _ic = g * jcp.nb_ic + ic_b;

                jit_conv_call_s par_conv = {};
                par_conv.src = jcp.transpose_src
                    ? &tr_src_[tr_src_off(ithr_mb, _ic, 0)]
                    : &src[src_d.blk_off(img, _ic)];
                par_conv.dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];

                const size_t off = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                par_conv.filt = diff_wei + off;

                /* TODO: put dw <-- 0 in kernel */
                if (img == img_start)
                    array_set((data_t *)par_conv.filt, 0, wei_block_size);

                kernel_->jit_ker(&par_conv);
            }
            }
            }
        }

        /* diff_weights[:] += sum(ws_reduction_[thr_mb][:]) */
        if (nthr_mb_ > 1) {
            simple_barrier::barrier(&reduction_barrier, nthr_);

            const int ic_b_kh_work = ic_b_work * jcp.kh;
            const int work = g_work * oc_b_work * ic_b_kh_work;

            int start, end;
            balance211(work, nthr_mb_, ithr_mb, start, end);
            if (start == end) return;

            for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
                int w = start;
                int sub_g_start, sub_oc_b_start, sub_ic_b_kh_start;
                nd_iterator_init(w, sub_g_start, g_work, sub_oc_b_start,
                        oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
                while (w < end) {
                    const int g = g_start + sub_g_start;
                    const int oc_b = oc_b_start + sub_oc_b_start;
                    const int ic_b = ic_b_start + sub_ic_b_kh_start / jcp.kh;
                    const int kh = sub_ic_b_kh_start % jcp.kh;

                    const int acc_size
                        = nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start)
                        * jcp.kw * jcp.ic_block * jcp.oc_block;

                    const size_t off
                        = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kh);
                    data_t *d = diff_weights + off;
                    data_t *s = ws_reduction_ + (thr_mb - 1) * wei_size + off;

                    acc_ker_->accumulate(d, s, acc_size);

                    nd_iterator_jump(w, end, sub_g_start, g_work,
                            sub_oc_b_start, oc_b_work, sub_ic_b_kh_start,
                            ic_b_kh_work);
                }
            }
        }
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
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] = 0.;

                for (int hw = 0; hw < jcp.oh * jcp.ow; ++hw) {
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += 16;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc);
            }
        }
        rb->reduce(ithr, diff_bias);
    };

#   pragma omp parallel num_threads(nthr_)
    {
        int ithr = omp_get_thread_num();
        assert(nthr_ == omp_get_num_threads());
        ker(ithr, nthr_);
        if (conf_.with_bias())
            ker_bias(ithr, nthr_);
    }
}

void jit_avx512_common_convolution_bwd_weights_t::balance() {
    const int max_threads = omp_get_max_threads();
    const auto &j = conf_.jcp_;

    if (max_threads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;
        return;
    }

    nthr_g_ = j.ngroups;
    const int nthr = max_threads / nthr_g_;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level optimizer
         * tries to minimize memory consumption. few notes:
         *  (n1) unclear why, but that essentially helps first convolution...
         *  (n2) assuming the reduction over minibatch is always there:
         *    - instead of 8 it should be 5 here (write ~= 2 read):
         *      kernel: temporal workspace 1 write
         *      reduction: 1 read from workspace and 1 write to the diff_wei
         *    - but experiments showed 8 works better than 5 or 6... */
        return 0
            + 1
            * div_up(j.mb, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_ic, nthr_ic_b) * j.ic_block * j.ih * j.iw
            / j.stride_h / j.stride_w /* (n1) */
            + 1
            * div_up(j.mb, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * j.oc_block * j.oh * j.ow
            + 8 /* (n2) */
            * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * div_up(j.nb_ic, nthr_ic_b)
            * j.kh * j.kw * j.ic_block * j.oc_block;
    };

    auto calc_comp_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        return 1
            * div_up(j.mb, nthr_mb)
            * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b)
            * div_up(j.nb_ic, nthr_ic_b);
    };

    int nthr_mb = 1,  nthr_oc_b = 1, nthr_ic_b = 1;
    int best_mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, j.mb);
    for (nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);
            int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

    /* step 2: search for a thread distribution with lower compute cost.
     * the constrains:
     *  - memory cost cannot exceed 110% of the best found in the step 1
     *  - unless compute cost is 133% lower than the current best case
     * note: both constants were found empirically */
    int best_comp_cost = calc_comp_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);
    for (nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);
            int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            int comp_cost = calc_comp_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

            const bool opt1 = comp_cost <= best_comp_cost
                && mem_cost < 1.1 * best_mem_cost;
            const bool opt2 = 4 * comp_cost <= 3 * best_comp_cost;

            if (opt1 || opt2) {
                best_comp_cost = comp_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

    if (nthr_mb_ > max_threads/2 && nthr_mb_ < max_threads)
        nthr_mb_ = min(j.mb, max_threads);

    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;
    assert(nthr_ <= max_threads);
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
