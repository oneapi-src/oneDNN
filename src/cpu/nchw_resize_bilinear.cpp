/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"

#include "nchw_resize_bilinear.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void nchw_resize_bilinear_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace alg_kind;

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper ws_d(pd()->workspace_md());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const float rateH = pd()->RATE(IH, OH);
    const float rateW = pd()->RATE(IW, OW);
    std::vector<CachedInterpolation> ys(OH + 1);
    std::vector<CachedInterpolation> xs(OW + 1);
    pd()->interpolationWeights(IH, OH, rateH, ys.data());
    pd()->interpolationWeights(IW, OW, rateW, xs.data());

    auto ker = [&](int _b, int _c, int _oh, int _ow) {
        const float h_lower = ys[_oh].lower;
        const float h_upper = ys[_oh].upper;
        const float h_lerp = ys[_oh].lerp;
        const float w_lower = xs[_ow].lower;
        const float w_upper = xs[_ow].upper;
        const float w_lerp = xs[_ow].lerp;

        long top_left_id = _b*IH*IW*C + _c*IH*IW + h_lower*IW + w_lower;
        long top_right_id = _b*IH*IW*C + _c*IH*IW + h_lower*IW + w_upper;
        long bottom_left_id = _b*IH*IW*C + _c*IH*IW + h_upper*IW + w_lower;
        long bottom_right_id = _b*IH*IW*C + _c*IH*IW + h_upper*IW + w_upper;
        long dst_id = _b*OH*OW*C + _c*OH*OW + _oh*OW + _ow;

        const float top_left = src[top_left_id];
        const float top_right = src[top_right_id];
        const float bottom_left = src[bottom_left_id];
        const float bottom_right = src[bottom_right_id];

        float top = top_left + (top_right - top_left) * w_lerp;
        float bottom = bottom_left + (bottom_right - bottom_left) * w_lerp;
        dst[dst_id] = top + (bottom - top) * h_lerp;
    };

#   pragma omp parallel for collapse(4) schedule(static)
    for (int _b = 0; _b < MB; _b ++) {
        for (int _c = 0; _c < C; _c ++) {
            for ( int _oh = 0; _oh < OH; _oh ++ ) {
                for ( int _ow = 0; _ow < OW; _ow ++ ) {
                    ker(_b, _c, _oh, _ow);
                }
            }
        }
    }    
}

template struct nchw_resize_bilinear_fwd_t<data_type::f32>;
template struct nchw_resize_bilinear_fwd_t<data_type::u8>;
template struct nchw_resize_bilinear_fwd_t<data_type::s8>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
