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

#ifdef __INTEL_COMPILER
#include <immintrin.h>
#endif

#include <iostream>
#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"

#include "nhwc_resize_bilinear.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#define MEM_D(name) name##_d

template <impl::data_type_t data_type>
void nhwc_resize_bilinear_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace alg_kind;
    using namespace prop_kind;

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper MEM_D(src)(pd()->src_md());
    const memory_desc_wrapper MEM_D(dst)(pd()->dst_md());
    const memory_desc_wrapper MEM_D(ws)(pd()->workspace_md());

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

#ifdef __INTEL_COMPILER
    int C_end = C / 16;
    int C_rest = C % 16;

    auto ker_m512 = [&](int _b, int _oh, int _ow, int _c) {
        __m512 h_lerp  = _mm512_set1_ps(ys[_oh].lerp);
        __m512 w_lerp  = _mm512_set1_ps(xs[_ow].lerp);

        int top_left_id = src_d.off(_b, _c, ys[_oh].lower, xs[_ow].lower);
        int top_right_id = src_d.off(_b, _c, ys[_oh].lower, xs[_ow].upper);
        int bottom_left_id = src_d.off(_b, _c, ys[_oh].upper, xs[_ow].lower);
        int bottom_right_id = src_d.off(_b, _c, ys[_oh].upper, xs[_ow].upper);
        int dst_id = dst_d.off(_b, _c, _oh, _ow);

        __m512 top_left, top_right, bottom_left, bottom_right;
        for ( int i = 0; i < 16; i ++ ) {
            top_left[i] = src[top_left_id + i];
            top_right[i] = src[top_right_id + i];
            bottom_left[i] = src[bottom_left_id + i];
            bottom_right[i] = src[bottom_right_id + i];
        }

        __m512 top = _mm512_sub_ps(top_right, top_left);
        top = _mm512_mul_ps(top, w_lerp);
        top = _mm512_add_ps(top_left, top);

        __m512 bottom = _mm512_sub_ps(bottom_right, bottom_left);
        bottom = _mm512_mul_ps(bottom, w_lerp);
        bottom = _mm512_add_ps(bottom_left, bottom);

        __m512 val = _mm512_sub_ps(bottom, top);
        val = _mm512_mul_ps(val, h_lerp);
        val = _mm512_add_ps(top, val);

        for ( int i = 0; i < 16; i ++ ) dst[dst_id + i] = val[i];
    };
#endif
    auto ker_norm = [&](int _b, int _oh, int _ow, int _c) {
        const float h_lower = ys[_oh].lower;
        const float h_upper = ys[_oh].upper;
        const float h_lerp = ys[_oh].lerp;
        const float w_lower = xs[_ow].lower;
        const float w_upper = xs[_ow].upper;
        const float w_lerp = xs[_ow].lerp;

        long top_left_id     = _b*IH*IW*C + h_lower*IW*C + w_lower*C + _c;
        long top_right_id    = _b*IH*IW*C + h_lower*IW*C + w_upper*C + _c;
        long bottom_left_id  = _b*IH*IW*C + h_upper*IW*C + w_lower*C + _c;
        long bottom_right_id = _b*IH*IW*C + h_upper*IW*C + w_upper*C + _c;
        long dst_id = _b*OH*OW*C + _oh*OW*C + _ow*C + _c;

        const float top_left = src[top_left_id];
        const float top_right = src[top_right_id];
        const float bottom_left = src[bottom_left_id];
        const float bottom_right = src[bottom_right_id];

        float top = top_left + (top_right - top_left) * w_lerp;
        float bottom = bottom_left + (bottom_right - bottom_left) * w_lerp;
        dst[dst_id] = top + (bottom - top) * h_lerp;
    };

    for (int _b = 0; _b < MB; _b ++) {
#   pragma omp parallel for collapse(2) schedule(static)
        for ( int _oh = 0; _oh < OH; _oh ++ ) {
            for ( int _ow = 0; _ow < OW; _ow ++ ) {
#   pragma omp simd
#ifdef __INTEL_COMPILER
                for (int _c = 0; _c < C_end; _c ++) ker_m512(_b, _oh, _ow, _c*16);
                if ( C_rest > 0 ) {
                    for (int _c = C_end*16; _c < C; _c ++) ker_norm(_b, _oh, _ow, _c);
                }
#else
                for (int _c = 0; _c < C; _c ++) ker_norm(_b, _oh, _ow, _c);
#endif
            }
        }
    }    
}

template struct nhwc_resize_bilinear_fwd_t<data_type::f32>;
template struct nhwc_resize_bilinear_fwd_t<data_type::u8>;
template struct nhwc_resize_bilinear_fwd_t<data_type::s8>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
