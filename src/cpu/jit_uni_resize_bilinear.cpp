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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "nstl.hpp"
//#include "utils.hpp"

#include "jit_uni_resize_bilinear.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
void jit_uni_resize_bilinear_fwd_t<isa>::execute_forward(const data_t *src,
        data_t *dst, char *indices) const {
printf("avx512/jit_uni_resize_bilinear.cpp\n");
#if 0
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));
#endif
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());


    const auto &jpp = pd()->jpp_;
    const int MB = pd()->MB();
    const int IC = pd()->C();
    const int X = jpp.c_block;
    const int C = jpp.nb_c; // C of n'C'hwxc
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
    auto ker = [&](int _b, int _c, int _oh, int _ow) {
        __m512 h_lerp = _mm512_set1_ps(ys[_oh].lerp);
        __m512 w_lerp = _mm512_set1_ps(xs[_ow].lerp);

        long top_left_id     = _b*IH*IW*IC + _c*IH*IW*X + ys[_oh].lower*IW*X + xs[_ow].lower*X;
        long top_right_id    = _b*IH*IW*IC + _c*IH*IW*X + ys[_oh].lower*IW*X + xs[_ow].upper*X;
        long bottom_left_id  = _b*IH*IW*IC + _c*IH*IW*X + ys[_oh].upper*IW*X + xs[_ow].lower*X;
        long bottom_right_id = _b*IH*IW*IC + _c*IH*IW*X + ys[_oh].upper*IW*X + xs[_ow].upper*X;
        long dst_id          = _b*OH*OW*IC + _c*OH*OW*X + _oh*OW*X + _ow*X;

        __m512 top_left = _mm512_loadu_ps(&src[top_left_id]);
        __m512 top_right = _mm512_loadu_ps(&src[top_right_id]);
        __m512 bottom_left = _mm512_loadu_ps(&src[bottom_left_id]);
        __m512 bottom_right = _mm512_loadu_ps(&src[bottom_right_id]);
        data_t *d = &dst[dst_id];

        __m512 top = _mm512_sub_ps(top_right, top_left);
        top = _mm512_mul_ps(top, w_lerp);
        top = _mm512_add_ps(top_left, top);

        __m512 bottom = _mm512_sub_ps(bottom_right, bottom_left);
        bottom = _mm512_mul_ps(bottom, w_lerp);
        bottom = _mm512_add_ps(bottom_left, bottom);

        __m512 val = _mm512_sub_ps(bottom, top);
        val = _mm512_mul_ps(val, h_lerp);
        val = _mm512_add_ps(top, val);

        _mm512_storeu_ps(d, val);
    };
#else 
    auto ker = [&](int _b, int _c, int _oh, int _ow, int _x) {
#if 1
        const float h_lower = ys[_oh].lower;
        const float w_lower = xs[_ow].lower;
        const float h_upper = ys[_oh].upper;
        const float w_upper = xs[_ow].upper;
        const float h_lerp = ys[_oh].lerp;
        const float w_lerp = xs[_ow].lerp;

        long top_left_id     = _b*IH*IW*IC + _c*IH*IW*X + h_lower*IW*X + w_lower*X + _x;
        long top_right_id    = _b*IH*IW*IC + _c*IH*IW*X + h_lower*IW*X + w_upper*X + _x;
        long bottom_left_id  = _b*IH*IW*IC + _c*IH*IW*X + h_upper*IW*X + w_lower*X + _x;
        long bottom_right_id = _b*IH*IW*IC + _c*IH*IW*X + h_upper*IW*X + w_upper*X + _x;
        long dst_id          = _b*OH*OW*IC + _c*OH*OW*X + _oh*OW*X + _ow*X + _x;

        const float top_left = src[top_left_id];
        const float top_right = src[top_right_id];
        const float bottom_left = src[bottom_left_id];
        const float bottom_right = src[bottom_right_id];

        float top = top_left + (top_right - top_left) * w_lerp;
        float bottom = bottom_left + (bottom_right - bottom_left) * w_lerp;
        dst[dst_id] = top + (bottom - top) * h_lerp;
#else
        const float top_left = src[src_d.off(_b, _c, ys[_oh].lower, xs[_ow].lower) + _x];
        const float top_right = src[src_d.off(_b, _c, ys[_oh].lower, xs[_ow].upper) + _x];
        const float bottom_left = src[src_d.off(_b, _c, ys[_oh].upper, xs[_ow].lower) + _x];
        const float bottom_right = src[src_d.off(_b, _c, ys[_oh].upper, xs[_ow].upper) + _x];

        float top = top_left + (top_right - top_left) * xs[_ow].lerp;
        float bottom = bottom_left + (bottom_right - bottom_left) * xs[_ow].lerp;

        dst[dst_d.off(_b, _c, _oh, _ow) + _x] = top + (bottom - top) * ys[_oh].lerp;
#endif
    };
#endif


#ifdef __INTEL_COMPILER
#   pragma omp parallel for collapse(4) schedule(static)
#else
#   pragma omp parallel for collapse(5) schedule(static)
#endif
    for (int _b = 0; _b < MB; _b ++) {
        for (int _c = 0; _c < C; _c ++) {
            for ( int _oh = 0; _oh < OH; _oh ++ ) {
                for ( int _ow = 0; _ow < OW; _ow ++ ) {
#ifdef __INTEL_COMPILER
                    ker(_b, _c, _oh, _ow);
#else
                    for ( int _x = 0; _x < X; _x ++ ) ker(_b, _c, _oh, _ow, _x);
#endif
                }
            }
        }
    }
}

template struct jit_uni_resize_bilinear_fwd_t<sse41>;
template struct jit_uni_resize_bilinear_fwd_t<avx2>;
template struct jit_uni_resize_bilinear_fwd_t<avx512_common>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
