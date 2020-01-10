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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "math_utils.hpp"
#include "type_helpers.hpp"

#include "ref_resampling.hpp"
#include "resampling_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

static inline dim_t get_offset(
        const memory_desc_wrapper &data_d, int n, int c, int d, int h, int w) {
    if (data_d.ndims() == 5)
        return data_d.off(n, c, d, h, w);
    else if (data_d.ndims() == 4)
        return data_d.off(n, c, h, w);
    else
        return data_d.off(n, c, w);
}

using namespace resampling_utils;

template <impl::data_type_t data_type>
void ref_resampling_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    if (this->pd()->has_zero_dim_memory()) return;

    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto alg = pd()->desc()->alg_kind;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const float FD = pd()->FD();
    const float FH = pd()->FH();
    const float FW = pd()->FW();

    auto lin_interp = [&](float c0, float c1, float w) {
        return c0 * w + c1 * (1 - w);
    };
    auto bilin_interp = [&](float c00, float c01, float c10, float c11,
                                float w0, float w1) {
        return lin_interp(
                lin_interp(c00, c10, w0), lin_interp(c01, c11, w0), w1);
    };
    auto trilin_interp = [&](float c000, float c001, float c010, float c011,
                                 float c100, float c101, float c110, float c111,
                                 float w0, float w1, float w2) {
        return lin_interp(bilin_interp(c000, c010, c100, c110, w0, w1),
                bilin_interp(c001, c011, c101, c111, w0, w1), w2);
    };
    parallel_nd(MB, C, OD, OH, OW,
            [&](dim_t mb, dim_t ch, dim_t od, dim_t oh, dim_t ow) {
                if (alg == alg_kind::resampling_nearest) {
                    dim_t id = nearest_idx(od, FD), ih = nearest_idx(oh, FH),
                          iw = nearest_idx(ow, FW);
                    dst[get_offset(dst_d, mb, ch, od, oh, ow)]
                            = src[get_offset(src_d, mb, ch, id, ih, iw)];
                } else if (alg == alg_kind::resampling_linear) {
                    // Trilinear interpolation (linear interpolation on a 3D spatial
                    // tensor) can be expressed as linear interpolation along
                    // dimension x followed by interpolation along dimension y and z
                    //      C011--C11--C111
                    //     -          - |
                    //   -          -   |
                    //C001--C01--C111   |
                    // -     .C   -    C110
                    // -          -    -
                    // -          -  -
                    //C000--C00--C100
                    auto id = linear_coeffs_t(od, FD, ID);
                    auto iw = linear_coeffs_t(ow, FW, IW);
                    auto ih = linear_coeffs_t(oh, FH, IH);
                    dim_t src_l[8] = {0};
                    for_(int i = 0; i < 2; i++)
                    for_(int j = 0; j < 2; j++)
                    for (int k = 0; k < 2; k++) {
                        src_l[4 * i + 2 * j + k] = src[get_offset(src_d, mb, ch,
                                id.idx[i], ih.idx[j], iw.idx[k])];
                    }
                    dst[get_offset(dst_d, mb, ch, od, oh, ow)]
                            = trilin_interp(src_l[0], src_l[1], src_l[2],
                                    src_l[3], src_l[4], src_l[5], src_l[6],
                                    src_l[7], id.wei[0], ih.wei[0], iw.wei[0]);
                }
            });
}

template struct ref_resampling_fwd_t<data_type::f32>;
template struct ref_resampling_fwd_t<data_type::bf16>;

template <impl::data_type_t data_type>
void ref_resampling_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    if (this->pd()->has_zero_dim_memory()) return;

    const auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto alg = pd()->desc()->alg_kind;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const float FD = pd()->FD();
    const float FH = pd()->FH();
    const float FW = pd()->FW();

    parallel_nd(MB, C, ID, IH, IW,
            [&](dim_t mb, dim_t ch, dim_t id, dim_t ih, dim_t iw) {
                diff_src[get_offset(diff_src_d, mb, ch, id, ih, iw)] = 0.f;
            });
    parallel_nd(MB, C, [&](dim_t mb, dim_t ch) {
        for_(int od = 0; od < OD; ++od)
        for_(int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
            if (alg == alg_kind::resampling_nearest) {
                dim_t id = nearest_idx(od, FD), ih = nearest_idx(oh, FH),
                      iw = nearest_idx(ow, FW);
                diff_src[get_offset(diff_src_d, mb, ch, id, ih, iw)]
                        += diff_dst[get_offset(diff_dst_d, mb, ch, od, oh, ow)];
            } else if (alg == alg_kind::resampling_linear) {
                auto id = linear_coeffs_t(od, FD, ID);
                auto iw = linear_coeffs_t(ow, FW, IW);
                auto ih = linear_coeffs_t(oh, FH, IH);
                // accessor for source values on a cubic lattice
                data_t dd
                        = diff_dst[get_offset(diff_dst_d, mb, ch, od, oh, ow)];
                for_(int i = 0; i < 2; i++)
                for_(int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    auto off = get_offset(diff_src_d, mb, ch, id.idx[i],
                            ih.idx[j], iw.idx[k]);
                    diff_src[off] += dd * id.wei[i] * ih.wei[j] * iw.wei[k];
                }
            }
        }
    });
}

template struct ref_resampling_bwd_t<data_type::f32>;
template struct ref_resampling_bwd_t<data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
