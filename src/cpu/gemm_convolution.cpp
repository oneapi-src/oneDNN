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
#include "gemm_convolution.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"

#include "os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <bool with_relu, bool run_jit, cpu_isa_t isa>
void _gemm_convolution_fwd_t<with_relu, run_jit, isa>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t*>(this->memory());

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;

    const int M = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * M;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int K = jcp.ic * jcp.ks;
    const int N = jcp.oc;
    const int m = jcp.os;
    const int LDA = jcp.need_im2col ? m : M;

    const auto &post_ops = conf_.attr()->post_ops_;

    float nslope = jcp.with_relu ? jcp.relu_negative_slope : 0.f;
    int entry_idx = -1;
    for (int idx = 0; idx < post_ops.len_; ++idx) {
        const auto &e = post_ops.entry_[idx];
        if (e.is_relu(true, false)) {
            entry_idx = idx;
            nslope = post_ops.entry_[entry_idx].eltwise.alpha;
            break;
        }
    }
    const bool do_relu = jcp.with_relu || entry_idx >= 0;

    const data_t one = 1.0;

    const size_t work_amount = jcp.ngroups * jcp.mb * jcp.od;
#   pragma omp parallel num_threads(this->nthr_)
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();

        data_t *_col = this->col_ + (size_t)ithr * jcp.ic * jcp.ks * jcp.os;

        int g{0}, n{0}, od{0};
        size_t start = 0, end = 0;

        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, od, jcp.od);

        for (size_t iwork = start; iwork < end; ++iwork) {
            const data_t *_src = src + (n * jcp.ngroups + g) * src_step;
            const data_t *_weights = weights + g * weights_g_size;
            data_t *_dst = dst + (n * jcp.ngroups + g) * dst_step;

            if (jcp.need_im2col)
            {
                if (jcp.id == 1)
                    jit_gemm_convolution_utils::im2col(jcp, _src, _col);
                else
                    jit_gemm_convolution_utils::im2col_3d(jcp, _src, _col, od);
            }

            if (run_jit) {
                sgemm_->sgemm("N", "N", &m, &N, &K, &one,
                    jcp.need_im2col ? _col : _src + od * m, &LDA, _weights, &K,
                    &this->beta_, _dst + od * m, &M);
            } else {
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, N,
                    K, one, jcp.need_im2col ? _col : _src + od * m, LDA,
                    _weights, K, this->beta_, _dst + od * m, M);
            }

            if (jcp.with_bias || do_relu) {
                data_t *d = _dst + od * m, b = 0.0;
                for (int oc = 0; oc < jcp.oc; ++oc) {
                    if(jcp.with_bias) b = bias[g * jcp.oc + oc];
                    for (int oS = 0; oS < m; ++oS) {
                        if (jcp.with_bias) d[oS] += b;
                        if (do_relu && d[oS] < 0)
                            d[oS] *= nslope;
                    }
                    d += M;
                }
            }
            nd_iterator_step(g, jcp.ngroups, n, jcp.mb, od, jcp.od);
        }
    }
}

template <bool run_jit, cpu_isa_t isa>
void _gemm_convolution_bwd_data_t<run_jit, isa>::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory());

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;

    const int M = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * M;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int m = jcp.os;
    const int K = jcp.oc;
    const int N = jcp.ic * jcp.ks;
    const int LDC = jcp.need_im2col ? m : M;
    const data_t zero = 0.0, one = 1.0;

    const size_t work_amount = jcp.ngroups * jcp.mb;
#pragma omp parallel num_threads(this->nthr_)
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();

        data_t *_col = this->col_ + (size_t)ithr * jcp.ic * jcp.ks * jcp.os;

        if (jcp.id > 1) {
        #pragma omp for
        for (size_t i = 0; i < jcp.ngroups*jcp.mb*src_step; ++i)
            diff_src[i] = 0.;
        }

        int g{0}, n{0};
        size_t start = 0, end = 0;
        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb);
        for (size_t iwork = start; iwork < end; ++iwork) {

            data_t *_diff_src = diff_src + (n * jcp.ngroups + g)*src_step;
            const data_t *_weights = weights + g * weights_g_size;
            for (int od = 0; od < jcp.od; ++od) {
                const data_t *_diff_dst = diff_dst + (n * jcp.ngroups + g)
                    *dst_step + od * m;

                if (run_jit) {
                    sgemm_->sgemm("N", "T", &m, &N, &K, &one, _diff_dst, &M,
                    _weights, &N, &zero,
                    jcp.need_im2col ? _col:_diff_src + od * m, &LDC);
                } else {
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, N,
                    K, one, _diff_dst, M, _weights, N, zero,
                    jcp.need_im2col ? _col : _diff_src + od * m, LDC);
                }

                if (jcp.need_im2col)
                {
                    if (jcp.id == 1)
                        jit_gemm_convolution_utils::col2im(jcp, _col,
                            _diff_src);
                    else
                        jit_gemm_convolution_utils::col2im_3d(jcp, _col,
                            _diff_src, od);
                }
            }
            nd_iterator_step(g, jcp.ngroups, n, jcp.mb);
        }
    }
}

template <bool run_jit, cpu_isa_t isa>
void _gemm_convolution_bwd_weights_t<run_jit, isa>::execute_backward_weights() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t*>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;
    const int K = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * K;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const int k = jcp.os;
    const int N = jcp.oc;
    const int M = jcp.ic * jcp.ks;
    const int LDA = jcp.need_im2col ? k : K;
    const data_t zero = 0.0, one = 1.0;
#pragma omp parallel num_threads(this->nthr_)
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();

        int ithr_g, nthr_g, ithr_mb, nthr_mb;
        size_t g_start{0}, g_end{0}, mb_start{0}, mb_end{0};

        jit_gemm_convolution_utils::bwd_weights_balance(ithr, nthr,
                jcp.ngroups, jcp.mb, ithr_g, nthr_g, ithr_mb, nthr_mb);

        const int need_reduction = nthr_mb != 1;

        if (ithr_g != -1 && ithr_mb != -1) {
            balance211((size_t)jcp.ngroups, nthr_g, ithr_g, g_start, g_end);
            balance211((size_t)jcp.mb, nthr_mb, ithr_mb, mb_start, mb_end);

            assert(implication((g_end - g_start) > 1, need_reduction == 0));

            data_t *_col = this->col_ + (size_t)ithr * jcp.ic * jcp.ks * jcp.os;
            data_t *weights_reduce_base = this->wei_reduction_
                    + ithr_g * nthr_mb * weights_g_size;
            data_t *weights_reduce = weights_reduce_base
                    + ithr_mb * weights_g_size;
            for (size_t g = g_start; g < g_end; ++g) {
                data_t *_diff_weights = need_reduction
                        ? weights_reduce : (diff_weights + g * weights_g_size);
                for (size_t mb = mb_start; mb < mb_end; ++mb) {
                    const data_t *_src = src + (mb*jcp.ngroups+g)*src_step;
                    for (int od = 0; od < jcp.od; ++od) {
                    const data_t *_diff_dst = diff_dst
                            + (mb*jcp.ngroups+g)*dst_step + od * k;

                    if (jcp.need_im2col)
                    {
                        if (jcp.id == 1)
                            jit_gemm_convolution_utils::im2col(jcp, _src, _col);
                        else
                            jit_gemm_convolution_utils::im2col_3d(jcp, _src,
                                _col, od);
                    }
                    if (run_jit) {
                        (mb == mb_start && od == 0 ? sgemm_0 :sgemm_1)->sgemm(
                            "T", "N", &M, &N, &K, &one,
                            jcp.need_im2col ? _col : _src + od * k,
                            &LDA, _diff_dst, &K,
                            mb == mb_start && od == 0 ? &zero : &one,
                            _diff_weights, &M);
                    } else {
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, M,
                        N, k, one, jcp.need_im2col ? _col : _src + od * k, LDA,
                        _diff_dst, K, mb == mb_start && od == 0 ? zero : one,
                        _diff_weights, M);
                    }
                    }
                }
            }
            if (need_reduction) {
                #pragma omp barrier
                data_t *weights_base = diff_weights + g_start * weights_g_size;
                jit_gemm_convolution_utils::bwd_weights_reduction_par(
                    ithr_mb, nthr_mb, jcp, weights_reduce_base, weights_base);
            }
        } else
            if (need_reduction) {
                #pragma omp barrier
            }
    }
    if (jcp.with_bias) {
        const size_t work_amount = jcp.ngroups * jcp.oc;
    #pragma omp parallel
        {
            const int ithr = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            int g{0}, oc{0};
            size_t start = 0, end = 0;
            balance211(work_amount, nthr, ithr, start, end);
            nd_iterator_init(start, g, jcp.ngroups, oc, jcp.oc);
            for (size_t iwork = start; iwork < end; ++iwork) {
                data_t db = 0;
                size_t offset_ = g*dst_step + oc * K;
                for (int mb = 0; mb < jcp.mb; ++mb)
                {
                    size_t offset = offset_ + mb*jcp.ngroups*dst_step;
                    for (int od = 0; od < jcp.od; ++od)
                    for (int oh = 0; oh < jcp.oh; ++oh)
#                   pragma omp simd reduction(+:db)
                    for (int ow = 0; ow < jcp.ow; ++ow)
                    {
                        db += diff_dst[offset];
                        offset ++;
                    }
                }
                diff_bias[g*jcp.oc+oc] = db;
                nd_iterator_step(g, jcp.ngroups, oc, jcp.oc);
            }
        }
    }
}

template struct _gemm_convolution_fwd_t<true, true, avx512_common>;
template struct _gemm_convolution_fwd_t<true, true, avx2>;
template struct _gemm_convolution_fwd_t<false, true, avx512_common>;
template struct _gemm_convolution_fwd_t<false, true, avx2>;
template struct _gemm_convolution_fwd_t<true, false, isa_any>;
template struct _gemm_convolution_fwd_t<false, false, isa_any>;

template struct _gemm_convolution_bwd_data_t<true, avx512_common>;
template struct _gemm_convolution_bwd_data_t<true, avx2>;
template struct _gemm_convolution_bwd_data_t<false, isa_any>;

template struct _gemm_convolution_bwd_weights_t<true, avx512_common>;
template struct _gemm_convolution_bwd_weights_t<true, avx2>;
template struct _gemm_convolution_bwd_weights_t<false, isa_any>;

}
}
}
