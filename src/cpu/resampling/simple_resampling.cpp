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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "math_utils.hpp"
#include "type_helpers.hpp"

#include "simple_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace format_tag;
using namespace resampling_utils;

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::nearest(const float *src, float *dst,
        dim_t stride_d, dim_t stride_h, dim_t stride_w, dim_t od, dim_t oh,
        dim_t ow) const {
    dim_t id = nearest_idx(od, pd()->FD()), ih = nearest_idx(oh, pd()->FH()),
          iw = nearest_idx(ow, pd()->FW());

    PRAGMA_OMP_SIMD()
    for (dim_t nsp1 = 0; nsp1 < nsp_inner_; nsp1++)
        dst[nsp1] = src[id * stride_d + ih * stride_h + iw * stride_w + nsp1];
}

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::linear(const float *src, float *dst,
        dim_t stride_d, dim_t stride_h, dim_t stride_w, dim_t od, dim_t oh,
        dim_t ow) const {
    linear_coeffs_t iw = linear_coeffs_[pd()->OD() + pd()->OH() + ow];

    PRAGMA_OMP_SIMD()
    for (dim_t nsp1 = 0; nsp1 < nsp_inner_; nsp1++) {
        float d = 0;
        for (int k = 0; k < 2; k++)
            d += src[iw.idx[k] * stride_w + nsp1] * iw.wei[k];
        dst[nsp1] = d;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::bilinear(const float *src, float *dst,
        dim_t stride_d, dim_t stride_h, dim_t stride_w, dim_t od, dim_t oh,
        dim_t ow) const {
    linear_coeffs_t ih = linear_coeffs_[pd()->OD() + oh],
                    iw = linear_coeffs_[pd()->OD() + pd()->OH() + ow];

    PRAGMA_OMP_SIMD()
    for (dim_t nsp1 = 0; nsp1 < nsp_inner_; nsp1++) {
        float d = 0;
        for_(int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
            d += src[ih.idx[j] * stride_h + iw.idx[k] * stride_w + nsp1]
                    * ih.wei[j] * iw.wei[k];
        dst[nsp1] = d;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::trilinear(const float *src, float *dst,
        dim_t stride_d, dim_t stride_h, dim_t stride_w, dim_t od, dim_t oh,
        dim_t ow) const {
    linear_coeffs_t id = linear_coeffs_[od],
                    ih = linear_coeffs_[pd()->OD() + oh],
                    iw = linear_coeffs_[pd()->OD() + pd()->OH() + ow];

    PRAGMA_OMP_SIMD()
    for (dim_t nsp1 = 0; nsp1 < nsp_inner_; nsp1++) {
        float d = 0;
        for_(int i = 0; i < 2; i++)
        for_(int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
            d += src[id.idx[i] * stride_d + ih.idx[j] * stride_h
                         + iw.idx[k] * stride_w + nsp1]
                    * id.wei[i] * ih.wei[j] * iw.wei[k];
        dst[nsp1] = d;
    }
}

template <impl::data_type_t data_type>
void simple_resampling_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    // input/output layout: (nsp0, d, h, w, nsp1)
    const memory_desc_wrapper src_d(pd()->src_md());
    const dim_t nsp_outer = src_d.nelems(true) / (ID * IH * IW * nsp_inner_);
    const dim_t stride_d = IH * IW * nsp_inner_;
    const dim_t stride_h = IW * nsp_inner_;
    const dim_t stride_w = nsp_inner_;

    parallel_nd(nsp_outer, OD, OH, OW,
            [&](dim_t nsp0, dim_t od, dim_t oh, dim_t ow) {
                dim_t src_off = nsp0 * ID * IH * IW * nsp_inner_;
                dim_t dst_off
                        = (nsp0 * OD * OH * OW + od * OH * OW + oh * OW + ow)
                        * nsp_inner_;
                (this->*(interpolate))(src + src_off, dst + dst_off, stride_d,
                        stride_h, stride_w, od, oh, ow);
            });
}

template struct simple_resampling_fwd_t<data_type::f32>;

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::nearest(float *diff_src,
        const float *diff_dst, dim_t stride_d, dim_t stride_h, dim_t stride_w,
        dim_t id, dim_t ih, dim_t iw) const {
    dim_t ow_start = ceil_idx(iw * pd()->FW() - 0.5f),
          oh_start = ceil_idx(ih * pd()->FH() - 0.5f),
          od_start = ceil_idx(id * pd()->FD() - 0.5f);
    dim_t ow_end = ceil_idx((iw + 1.f) * pd()->FW() - 0.5f),
          oh_end = ceil_idx((ih + 1.f) * pd()->FH() - 0.5f),
          od_end = ceil_idx((id + 1.f) * pd()->FD() - 0.5f);

    PRAGMA_OMP_SIMD()
    for (dim_t nsp1 = 0; nsp1 < nsp_inner_; nsp1++) {
        diff_src[nsp1] = 0;
        for_(dim_t od = od_start; od < od_end; od++)
        for_(dim_t oh = oh_start; oh < oh_end; oh++)
        for (dim_t ow = ow_start; ow < ow_end; ow++) {
            diff_src[nsp1] += diff_dst[od * stride_d + oh * stride_h
                    + ow * stride_w + nsp1];
        }
    }
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::linear(float *diff_src,
        const float *diff_dst, dim_t stride_d, dim_t stride_h, dim_t stride_w,
        dim_t id, dim_t ih, dim_t iw) const {

    // It's wrapped in lambdas to workaround an issue that appears in the
    // following case: Intel Compiler 19.0 + PRAGMA_OMP_SIMD +
    // generation multiple code paths (ax option) + SNB
    const auto idx_s = [&](int k) {
        return bwd_linear_coeffs_[pd()->ID() + pd()->IH() + iw].start[k];
    };

    const auto idx_e = [&](int k) {
        return bwd_linear_coeffs_[pd()->ID() + pd()->IH() + iw].end[k];
    };

    PRAGMA_OMP_SIMD()
    for (dim_t nsp1 = 0; nsp1 < nsp_inner_; nsp1++) {
        diff_src[nsp1] = 0;
        for_(int k = 0; k < 2; k++)
        for (dim_t ow = idx_s(k); ow < idx_e(k); ow++) {
            diff_src[nsp1] += diff_dst[ow * stride_w + nsp1]
                    * bwd_linear_weights_[2 * (pd()->OD() + pd()->OH() + ow)
                            + k];
        }
    }
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::bilinear(float *diff_src,
        const float *diff_dst, dim_t stride_d, dim_t stride_h, dim_t stride_w,
        dim_t id, dim_t ih, dim_t iw) const {
    bwd_linear_coeffs_t h = bwd_linear_coeffs_[pd()->ID() + ih],
                        w = bwd_linear_coeffs_[pd()->ID() + pd()->IH() + iw];

    PRAGMA_OMP_SIMD()
    for (dim_t nsp1 = 0; nsp1 < nsp_inner_; nsp1++) {
        diff_src[nsp1] = 0;
        for_(int j = 0; j < 2; j++)
        for_(int k = 0; k < 2; k++)
        for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
        for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
            diff_src[nsp1] += diff_dst[oh * stride_h + ow * stride_w + nsp1]
                    * bwd_linear_weights_[2 * (pd()->OD() + oh) + j]
                    * bwd_linear_weights_[2 * (pd()->OD() + pd()->OH() + ow)
                            + k];
        }
    }
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::trilinear(float *diff_src,
        const float *diff_dst, dim_t stride_d, dim_t stride_h, dim_t stride_w,
        dim_t id, dim_t ih, dim_t iw) const {
    bwd_linear_coeffs_t d = bwd_linear_coeffs_[id],
                        h = bwd_linear_coeffs_[pd()->ID() + ih],
                        w = bwd_linear_coeffs_[pd()->ID() + pd()->IH() + iw];

    PRAGMA_OMP_SIMD()
    for (dim_t nsp1 = 0; nsp1 < nsp_inner_; nsp1++) {
        diff_src[nsp1] = 0;
        for_(int i = 0; i < 2; i++)
        for_(int j = 0; j < 2; j++)
        for_(int k = 0; k < 2; k++)
        for_(dim_t od = d.start[i]; od < d.end[i]; od++)
        for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
        for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
            diff_src[nsp1] += diff_dst[od * stride_d + oh * stride_h
                                      + ow * stride_w + nsp1]
                    * bwd_linear_weights_[2 * od + i]
                    * bwd_linear_weights_[2 * (pd()->OD() + oh) + j]
                    * bwd_linear_weights_[2 * (pd()->OD() + pd()->OH() + ow)
                            + k];
        }
    }
}

template <impl::data_type_t data_type>
void simple_resampling_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    const auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    // input/output layout: (nsp0, d, h, w, nsp1)
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const dim_t nsp_outer
            = diff_src_d.nelems(true) / (ID * IH * IW * nsp_inner_);
    const dim_t stride_d = OH * OW * nsp_inner_;
    const dim_t stride_h = OW * nsp_inner_;
    const dim_t stride_w = nsp_inner_;
    parallel_nd(nsp_outer, ID, IH, IW,
            [&](dim_t nsp0, dim_t id, dim_t ih, dim_t iw) {
                dim_t diff_dst_off = nsp0 * OD * OH * OW * nsp_inner_;
                dim_t diff_src_off
                        = (nsp0 * ID * IH * IW + id * IH * IW + ih * IW + iw)
                        * nsp_inner_;
                (this->*(interpolate))(diff_src + diff_src_off,
                        diff_dst + diff_dst_off, stride_d, stride_h, stride_w,
                        id, ih, iw);
            });
}

template struct simple_resampling_bwd_t<data_type::f32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
