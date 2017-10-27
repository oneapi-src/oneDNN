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

#if (defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1600) || defined(_MSC_VER)
/* Excluding ICC 16.0 from adding simd because it results in accuracy issues.
 * MSC doesn't support simd in _pragma */
#    define pragma_simd
#else
#    define pragma_simd _Pragma("simd")
#endif

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
        int start, end, start_copy;
        int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        jit_conv_call_s par_conv = { 0 };
        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t src_c_stride = src_d.blk_off(0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0 ; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n{0}, g{0}, occ{0}, oh_s{0};

            if (jcp.loop_order == loop_cgn)
                nd_iterator_init(start,
                    occ, oc_chunks, g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_init(start,
                    g, jcp.ngroups, n, jcp.mb, occ, oc_chunks, oh_s, jcp.oh);
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

                auto bias_w = bias ? bias + bias_d.blk_off(g_oc) : 0;
                auto dst_w = dst + dst_d.blk_off(n, g_ocb, oh_s);
                auto src_w = src + src_d.blk_off(n, g_icb + icb_l2, ih_s);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb, icb_l2);

                for (int icb = icb_l2;
                     icb < min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2); ++icb) {
                    auto src_c = src_w;
                    auto dst_c = dst_w;
                    for (int oj = oh_s, ij = ih_s;
                            oj < oh_e; ++oj, ij += jcp.stride_h)
                    {
                        int i_t_overflow = -min(0, ij);
                        int i_b_overflow = max(jcp.ih, ij + jcp.kh) - jcp.ih;
                        int kh_padding = nstl::max(0,
                            jcp.kh - i_t_overflow - i_b_overflow);

                        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                            src_c + i_t_overflow * src_h_stride,
                            dst_c, wht_w + i_t_overflow * wht_h_stride,
                            bias_w, icb, kh_padding);

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

        int start, end, start_copy;
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_amount = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        jit_conv_call_s par_conv = {0};
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
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

                auto diff_src_w = diff_src + diff_src_d.blk_off(n, g_icb);
                auto diff_dst_w = diff_dst
                    + diff_dst_d.blk_off(n, g_ocb + ocb_l2);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb);

                for (int ocb = ocb_l2;
                      ocb < min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2); ++ocb) {
                    for (int ij = ih_s; ij < ih_e; ++ij) {
                        int oj, k_len, k_lo;
                        if (jcp.stride_h == 1) { // fast path
                            int i_t_overflow = max(0, jcp.kh - 1 - ij
                                - jcp.t_pad);
                            int i_b_overflow = max(0, jcp.kh - jcp.ih + ij
                                - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else {
                            int b_pad = jcp.stride_h * (jcp.oh - 1) + jcp.kh
                                - jcp.ih - jcp.t_pad;
                            int i_t_overflow = max(0, (jcp.kh - 1 - ij
                                - jcp.t_pad) / jcp.stride_h);
                            int i_b_overflow = max(0, (jcp.kh - jcp.ih + ij
                                - b_pad) / jcp.stride_h);
                            int overflow_kh_hi = jcp.kh - 1 - abs((jcp.ih - 1
                                + b_pad - ij) % jcp.stride_h);
                            int overflow_kh_lo = (ij + jcp.t_pad)
                                % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                / jcp.stride_h + 1 - i_t_overflow
                                - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }
                        assert(k_len >= 0);

                        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                                diff_src_w + ij * diff_src_h_stride,
                                diff_dst_w + oj * diff_dst_h_stride,
                                wht_w + k_lo * wht_h_stride,
                                0, ocb, k_len);
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
        }

        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                diff_src, diff_dst, weights, 0, 0, 1);
    }
}

template struct jit_avx512_common_convolution_bwd_data_t<data_type::f32>;
template struct jit_avx512_common_convolution_bwd_data_t<data_type::s16,
    data_type::s16, data_type::s32>;

jit_avx512_common_convolution_bwd_weights_t::
jit_avx512_common_convolution_bwd_weights_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), kernel_(nullptr)
    , trans_kernel_(nullptr), acc_ker_(nullptr), reducer_bias_(nullptr)
    , tr_src_(nullptr), ws_reduction_(nullptr), tr_src_bctx_(nullptr)
{
    const auto &j = conf_.jcp_;
    kernel_ = new jit_avx512_common_conv_bwd_weights_kernel_f32(j);

    balance();

    if (j.ver == ver_4fma) {
        trans_kernel_ = create_trans_src(&j);

        if (j.is_1stconv) {
            const int tr_src_size =
                nthr_ / nthr_oc_b_ * j.ih * j.stride_w * j.tr_ld;
            tr_src_ = (data_t *)malloc(tr_src_size * sizeof(data_t), 64);
        } else {
            // XXX: See the comment about tr_iw and guarding elements in
            // jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf()
            const int max_nthr = nthr_mb_ * j.ngroups * j.nb_ic;
            const int min_tr_src_size_per_thr = j.ih * j.ic_block * j.tr_iw;
            const int tr_src_size = max_nthr * min_tr_src_size_per_thr
                + j.tr_src_num_guard_elems;
            tr_src_ = (data_t *)malloc(tr_src_size * sizeof(data_t), 64);
            /* to avoid NaNs in computations we zero tail num_guard_elems for
             * each possible thread group */
            for (int ithr = 1; ithr <= max_nthr; ++ithr) {
                data_t *ts = &tr_src_[ithr * min_tr_src_size_per_thr];
                for (int i = 0; i < j.tr_src_num_guard_elems; ++i)
                    ts[i] = 0;
            }
        }

        /* prepare synchronization contexts */
        if (nthr_oc_b_ > 1) {
            const int tr_src_bctx_size = nthr_ / nthr_oc_b_;
            tr_src_bctx_ = (simple_barrier::ctx_t *)malloc(
                    tr_src_bctx_size * sizeof(simple_barrier::ctx_t), 64);
            for (int i = 0; i < tr_src_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_src_bctx_[i]);
        }
    }

    if (nthr_mb_ > 1) {
        const int wei_size = j.ngroups * j.oc * j.ic * j.kh * j.kw;
        const int bia_size = j.ngroups * j.oc;
        ws_reduction_ = (data_t *)malloc(
                (nthr_mb_ - 1) * (wei_size + bia_size) * sizeof(data_t), 64);
        acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();
        simple_barrier::ctx_init(&reduction_bctx_);
    }

    if (conf_.with_bias()) {
        const size_t max_buffer_size = nthr_ * 3 * 5 * 5 * 16 * 16;
        reducer_bias_ = new cpu_reducer_t<data_type::f32>(reduce_balancer_t(
                    nthr_, j.oc_block, j.ngroups * j.nb_oc, j.mb,
                    max_buffer_size));
    }
}

struct jit_avx512_common_convolution_bwd_weights_t::thread_info_t {
    const data_t *src, *diff_dst;
    data_t *diff_weights, *diff_bias;

    int ithr;
    int ithr_ic_b, ithr_oc_b, ithr_g, ithr_mb;
    int ithr_but_oc;

    int img_start, img_end, img_work;
    int g_start, g_end, g_work;
    int oc_b_start, oc_b_end, oc_b_work;
    int ic_b_start, ic_b_end, ic_b_work;

    thread_info_t(const jit_avx512_common_convolution_bwd_weights_t *self,
            int ithr): ithr(ithr) {
        src = reinterpret_cast<const data_t *>(self->input_memory(0));
        diff_dst = reinterpret_cast<const data_t *>(self->input_memory(1));
        diff_weights = reinterpret_cast<data_t *>(self->memory(0));
        diff_bias = reinterpret_cast<data_t *>(self->memory(1));

        ithr_ic_b = ithr % self->nthr_ic_b_;
        ithr_oc_b = ithr / self->nthr_ic_b_ % self->nthr_oc_b_;
        ithr_g = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ % self->nthr_g_;
        ithr_mb = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ / self->nthr_g_;

        ithr_but_oc = (ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_ic_b_
            + ithr_ic_b;

        const auto &jcp = self->kernel_->jcp;

        /* reduction dimension */
        balance211(jcp.mb, self->nthr_mb_, ithr_mb, img_start, img_end);
        img_work = img_end - img_start;

        /* independent dimensions */
        balance211(jcp.ngroups, self->nthr_g_, ithr_g, g_start, g_end);
        g_work = g_end - g_start;

        balance211(jcp.nb_oc, self->nthr_oc_b_, ithr_oc_b, oc_b_start,
                oc_b_end);
        oc_b_work = oc_b_end - oc_b_start;

        balance211(jcp.nb_ic, self->nthr_ic_b_, ithr_ic_b, ic_b_start,
                ic_b_end);
        ic_b_work = ic_b_end - ic_b_start;
    }
};

void jit_avx512_common_convolution_bwd_weights_t::compute_diff_weights(
        const thread_info_t *ti) {
    const memory_desc_wrapper src_d(conf_.src_pd(0));
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw;

    data_t *diff_wei = ti->ithr_mb == 0
        ? ti->diff_weights : ws_reduction_ + (ti->ithr_mb - 1) * wei_size;
    data_t *diff_bia = ti->ithr_mb == 0
        ? ti->diff_bias : ws_reduction_ + (nthr_mb_ - 1) * wei_size
                + (ti->ithr_mb - 1) * jcp.ngroups * jcp.oc;

    // TODO: use memory descriptor with the same fmt as src (or use a macro :))
    auto tr_src_off = [&](int ithr_mb, int ic, int ij) {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_chn_size = tr_row_size * jcp.ih;
        const size_t tr_img_size = tr_chn_size * jcp.nb_ic * jcp.ngroups;

        return ti->ithr_mb * tr_img_size + ic * tr_chn_size + ij * tr_row_size;
    };

    auto uker_trans = [&](int img) {
        const int work_amount = ti->g_work * ti->ic_b_work * jcp.ih;

        int start{0}, end{0};
        balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, start, end);
        const int my_work = end - start;

        int g{0}, ic_b{0}, j{0};
        nd_iterator_init(start, g, ti->g_work, ic_b, ti->ic_b_work, j, jcp.ih);
        g += ti->g_start;
        ic_b += ti->ic_b_start;

        const int _ic = g * jcp.nb_ic + ic_b;
        data_t *src1 = (data_t*)&ti->src[src_d.blk_off(img, _ic, j)];
        data_t *tr_src1 = &tr_src_[tr_src_off(ti->ithr_mb, _ic, j)];

        assert(jcp.ic_block == 16);
        const int src_stride = jcp.iw * jcp.ic_block;
        const int tr_src_stride = jcp.tr_iw * jcp.ic_block;

        const int pf_depth = 2;
        struct { data_t *src, *tr_src; } pf_circ_buf[pf_depth];

        for (int iwork = 0; iwork < my_work + pf_depth - 1; iwork++) {
            pf_circ_buf[iwork % pf_depth] = {src1, tr_src1};

            if (iwork >= pf_depth - 1) {
                int old_idx = (iwork - pf_depth + 1) % pf_depth;
                jit_trans_src_t::ctx_t ctx = {};
                ctx.src = pf_circ_buf[old_idx].src;
                ctx.tr_src = pf_circ_buf[old_idx].tr_src;
                ctx.src_prf = src1;
                ctx.tr_src_prf = tr_src1;
                (*trans_kernel_)(&ctx);
            }
            src1 += src_stride;
            tr_src1 += tr_src_stride;
        }
    };

    if (jcp.is_1stconv && jcp.ver == ver_4fma) {
        /* prepare contexts */
        jit_trans_src_t::ctx_t tr_ctx = {};
        tr_ctx.tr_src = tr_src_
            + ti->ithr_but_oc * jcp.ih * jcp.stride_w * jcp.tr_ld;

        tr_ctx.nthr_oc_b = nthr_oc_b_;
        int ih_start{0}, ih_end{0};
        balance211(jcp.ih, nthr_oc_b_, ti->ithr_oc_b, ih_start, ih_end);
        tr_ctx.tr_src_ih_start = ih_start;
        tr_ctx.tr_src_ih_end = ih_end;
        tr_ctx.tr_src_bctx = tr_src_bctx_ + ti->ithr_but_oc;

        jit_conv_call_s p = {};
        p.src = tr_ctx.tr_src;

        /* zero diff_bias if applicable */
        if (jcp.with_bias && ti->ithr_ic_b == 0) {
            assert(jcp.oc_block == 16);
            for (int oc_b = ti->ic_b_start; oc_b < ti->oc_b_end; ++oc_b) {
                data_t *db = &diff_bia[oc_b * 16];
                for (int o = 0; o < 16; ++o)
                    db[o] = 0;
            }
        }

        for (int img = ti->img_start; img < ti->img_end; ++img) {
            p.flags = (img == ti->img_start) * FLAG_MB_FIRST;

            for (int g = ti->g_start; g < ti->g_end; ++g) {
            for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
                const int _ic = g * jcp.nb_ic + ic_b;
                tr_ctx.src = &ti->src[src_d.blk_off(img, _ic)];

                (*trans_kernel_)(&tr_ctx);

                if (ic_b == 0)
                    p.flags |= FLAG_IC_FIRST;
                else
                    p.flags &= ~FLAG_IC_FIRST;

                for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b) {
                    const int _oc = g * jcp.nb_oc + oc_b;
                    p.dst = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)];

                    const size_t off =
                        wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                    p.filt = diff_wei + off;
                    p.bias = diff_bia + _oc * jcp.oc_block;

                    kernel_->jit_ker(&p);
                }
            }
            }
        }
    } else {
        for (int img = ti->img_start; img < ti->img_end; ++img) {
            jit_conv_call_s p = {0};

            if (jcp.ver == ver_4fma) {
                /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
                using simple_barrier::barrier;
                if (nthr_oc_b_ > 1)
                    barrier(&tr_src_bctx_[ti->ithr_but_oc], nthr_oc_b_);
                uker_trans(img);
                if (nthr_oc_b_ > 1)
                    barrier(&tr_src_bctx_[ti->ithr_but_oc], nthr_oc_b_);
            }

            for (int g = ti->g_start; g < ti->g_end; ++g) {
            for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b) {
            for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
                const int _oc = g * jcp.nb_oc + oc_b;
                const int _ic = g * jcp.nb_ic + ic_b;

                jit_conv_ker_pipeline(kernel_->jit_ker, p,
                        (jcp.ver == ver_4fma
                         ? &tr_src_[tr_src_off(ti->ithr_mb, _ic, 0)]
                         : &ti->src[src_d.blk_off(img, _ic)]),
                        &ti->diff_dst[diff_dst_d.blk_off(img, _oc)],
                        diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                        0, (img == ti->img_start), 0);

            }
            }
            }

            const int _oc = ti->g_start * jcp.nb_oc + ti->oc_b_start;
            const int _ic = ti->g_start * jcp.nb_ic + ti->ic_b_start;
            jit_conv_ker_pipeline(kernel_->jit_ker, p,
                    (jcp.ver == ver_4fma
                     ? &tr_src_[tr_src_off(ti->ithr_mb, _ic, 0)]
                     : &ti->src[src_d.blk_off(img + 1, _ic)]),
                    &ti->diff_dst[diff_dst_d.blk_off(img + 1, _oc)],
                    diff_wei + wht_blk_off(
                        diff_weights_d, ti->g_start,
                        ti->oc_b_start, ti->ic_b_start),
                    0, 0, 0);
        }
    }
}

void jit_avx512_common_convolution_bwd_weights_t::reduce_diff_weights(
        const thread_info_t *ti) {
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw;
    const int bia_size = jcp.ngroups * jcp.oc;
    const data_t *diff_bias_ws = ws_reduction_ + (nthr_mb_ - 1) * wei_size;

    /* diff_weights[:] += sum(ws_reduction_[thr_mb][:]) */
    simple_barrier::barrier(&reduction_bctx_, nthr_);

    const int ic_b_kh_work = ti->ic_b_work * jcp.kh;
    const int work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int start{0}, end{0};
    balance211(work, nthr_mb_, ti->ithr_mb, start, end);
    if (start == end) return;

    for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
        int w = start;
        int sub_g_start{0}, sub_oc_b_start{0}, sub_ic_b_kh_start{0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const int g = ti->g_start + sub_g_start;
            const int oc_b = ti->oc_b_start + sub_oc_b_start;
            const int ic_b = ti->ic_b_start + sub_ic_b_kh_start / jcp.kh;
            const int kh = sub_ic_b_kh_start % jcp.kh;

            const int acc_size
                = nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start)
                * jcp.kw * jcp.ic_block * jcp.oc_block;

            const size_t off
                = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kh);
            data_t *d = ti->diff_weights + off;
            data_t *s = ws_reduction_ + (thr_mb - 1) * wei_size + off;

            acc_ker_->accumulate(d, s, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }

        if (jcp.with_bias && jcp.is_1stconv && jcp.ver == ver_4fma) {
            if (ti->ithr == 0)
                acc_ker_->accumulate(ti->diff_bias, diff_bias_ws, bia_size);
            diff_bias_ws += bia_size;
        }
    }
}

void jit_avx512_common_convolution_bwd_weights_t::compute_diff_bias(
        const thread_info_t *ti) {
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());

    auto rb = this->reducer_bias_;
    assert(nthr_ == rb->balancer_.nthr_);

    const auto &jcp = kernel_->jcp;

    if (jcp.with_bias && jcp.is_1stconv && jcp.ver == ver_4fma) return;

    const int b_job_start = rb->balancer_.ithr_job_off(ti->ithr);
    const int b_njobs = rb->balancer_.ithr_njobs(ti->ithr);

    if (b_njobs == 0) return;

    /* reduction dimension */
    int img_start{0}, img_end{0};
    balance211(jcp.mb, rb->balancer_.nthr_per_group_,
            rb->balancer_.id_in_group(ti->ithr), img_start, img_end);

    /* jobs */
    int g_start{0}, ocb_start{0};
    nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_oc);

    for (int img = img_start; img < img_end; ++img) {
        int g = g_start, ocb = ocb_start;
        for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
            const size_t _oc = g * jcp.nb_oc + ocb;

            const data_t *d_dst = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)];
            data_t *d_bias = &rb->get_local_ptr(ti->ithr, ti->diff_bias)[
                b_job_loc * rb->balancer_.job_size_];

            if (img == img_start)
                for (int o = 0; o < 16; ++o)
                    d_bias[o] = 0.;
            for (int hw = 0; hw < jcp.oh * jcp.ow; ++hw) {
#               pragma omp simd
                for (int o = 0; o < 16; ++o)
                    d_bias[o] += d_dst[o];
                d_dst += 16;
            }

            nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc);
        }
    }

    rb->reduce(ti->ithr, ti->diff_bias);
}

void jit_avx512_common_convolution_bwd_weights_t::execute_backward_weights() {
#   pragma omp parallel num_threads(nthr_)
    {
        int ithr = omp_get_thread_num();
        assert(nthr_ == omp_get_num_threads());

        thread_info_t thread_info(this, ithr);

        compute_diff_weights(&thread_info);
        if (nthr_mb_ > 1)
            reduce_diff_weights(&thread_info);

        if (conf_.with_bias())
            compute_diff_bias(&thread_info);
    }
}

void jit_avx512_common_convolution_bwd_weights_t::balance() {
    const int max_threads = omp_get_max_threads();
    const auto &j = conf_.jcp_;

    nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

    if (max_threads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }

    if (j.ver == ver_4fma && j.is_1stconv) {
        nthr_g_ = 1;
        nthr_oc_b_ = 1;
        nthr_ic_b_ = nstl::min(j.nb_ic, max_threads);
        nthr_mb_ = max_threads / nthr_ic_b_;
        nthr_ = nthr_mb_ * nthr_oc_b_ * nthr_ic_b_ * nthr_g_;
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

        const int src_coef = j.ver == ver_4fma ? 4 : 1;
        const int dst_coef = 1;
        const int wei_coef = 8;

        return 0
            + src_coef
            * div_up(j.mb, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_ic, nthr_ic_b) * j.ic_block * j.ih * j.iw
            / j.stride_h / j.stride_w /* (n1) */
            + dst_coef
            * div_up(j.mb, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * j.oc_block * j.oh * j.ow
            + wei_coef /* (n2) */
            * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * div_up(j.nb_ic, nthr_ic_b)
            * j.kh * j.kw * j.ic_block * j.oc_block;
    };

    int best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, j.mb);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);
            int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

#if 0
    auto calc_comp_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        return 1
            * div_up(j.mb, nthr_mb)
            * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b)
            * div_up(j.nb_ic, nthr_ic_b);
    };

    /* step 2: search for a thread distribution with lower compute cost.
     * the constrains:
     *  - memory cost cannot exceed 110% of the best found in the step 1
     *  - unless compute cost is 133% lower than the current best case
     * note: both constants were found empirically */
    int best_comp_cost = calc_comp_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);
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
#endif

    if (nthr_mb_ > max_threads/2 && nthr_mb_ < max_threads)
        nthr_mb_ = min(j.mb, max_threads);

    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;
    assert(nthr_ <= max_threads);
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
