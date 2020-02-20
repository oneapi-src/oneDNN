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
#include <math.h>

#include "src/common/dnnl_thread.hpp"

#include "resampling/resampling.hpp"

namespace resampling {

float linear_map(const int64_t y, const float f) {
    volatile float s = (y + 0.5f)
            * (1.f / f); // prevent Intel Compiler optimizing operation for better accuracy
    return s - 0.5f;
}
int64_t left(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return MAX2(
            (int64_t)floorf(linear_map(y, (float)y_max / x_max)), (int64_t)0);
}
int64_t right(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return MIN2((int64_t)ceilf(linear_map(y, (float)y_max / x_max)), x_max - 1);
}
int64_t near(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return roundf(linear_map(y, (float)y_max / x_max));
}
float weight(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return fabs(linear_map(y, (float)y_max / x_max) - left(y, y_max, x_max));
}
void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src, dnn_mem_t &dst) {
    auto ker_nearest = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh,
                               int64_t ow) {
        const int64_t id = near(od, p->od, p->id), ih = near(oh, p->oh, p->ih),
                      iw = near(ow, p->ow, p->iw);
        const auto dst_off = dst_off_f(p, mb, ic, od, oh, ow);
        dst.set_elem(dst_off, src.get_elem(src_off_f(p, mb, ic, id, ih, iw)));
    };
    auto ker_linear = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh,
                              int64_t ow) {
        const int64_t id[2] = {left(od, p->od, p->id), right(od, p->od, p->id)},
                      ih[2] = {left(oh, p->oh, p->ih), right(oh, p->oh, p->ih)},
                      iw[2] = {left(ow, p->ow, p->iw), right(ow, p->ow, p->iw)};
        const float wd[2]
                = {1.f - weight(od, p->od, p->id), weight(od, p->od, p->id)},
                wh[2]
                = {1.f - weight(oh, p->oh, p->ih), weight(oh, p->oh, p->ih)},
                ww[2]
                = {1.f - weight(ow, p->ow, p->iw), weight(ow, p->ow, p->iw)};

        float cd[2][2] = {{0}};
        for_(int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            cd[i][j] = src.get_elem(src_off_f(p, mb, ic, id[0], ih[i], iw[j]))
                            * wd[0]
                    + src.get_elem(src_off_f(p, mb, ic, id[1], ih[i], iw[j]))
                            * wd[1];

        float ch[2] = {0};
        for (int i = 0; i < 2; i++)
            ch[i] = cd[0][i] * wh[0] + cd[1][i] * wh[1];

        float cw = ch[0] * ww[0] + ch[1] * ww[1];

        const auto dst_off = dst_off_f(p, mb, ic, od, oh, ow);
        dst.set_elem(dst_off, cw);
    };
    dnnl::impl::parallel_nd(p->mb, p->ic, p->od, p->oh, p->ow,
            [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
                if (p->alg == nearest) {
                    ker_nearest(mb, ic, od, oh, ow);
                } else {
                    ker_linear(mb, ic, od, oh, ow);
                }
            });
}
void compute_ref_bwd(
        const prb_t *p, dnn_mem_t &diff_src, const dnn_mem_t &diff_dst) {
    auto zero_diff_src = [&](int64_t mb, int64_t ic) {
        for (int64_t id = 0; id < p->id; ++id)
            for (int64_t ih = 0; ih < p->ih; ++ih)
                for (int64_t iw = 0; iw < p->iw; ++iw)
                    diff_src.set_elem(src_off_f(p, mb, ic, id, ih, iw), 0.);
    };
    auto ker_nearest = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh,
                               int64_t ow) {
        const auto diff_dst_off = dst_off_f(p, mb, ic, od, oh, ow);
        float diff_dst_val = diff_dst.get_elem(diff_dst_off);
        const int64_t id = near(od, p->od, p->id), ih = near(oh, p->oh, p->ih),
                      iw = near(ow, p->ow, p->iw);
        ((float *)diff_src)[src_off_f(p, mb, ic, id, ih, iw)] += diff_dst_val;
    };
    auto ker_linear = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh,
                              int64_t ow) {
        const auto diff_dst_off = dst_off_f(p, mb, ic, od, oh, ow);
        float diff_dst_val = diff_dst.get_elem(diff_dst_off);
        const int64_t id[2] = {left(od, p->od, p->id), right(od, p->od, p->id)},
                      ih[2] = {left(oh, p->oh, p->ih), right(oh, p->oh, p->ih)},
                      iw[2] = {left(ow, p->ow, p->iw), right(ow, p->ow, p->iw)};
        const float wd[2]
                = {1.f - weight(od, p->od, p->id), weight(od, p->od, p->id)},
                wh[2]
                = {1.f - weight(oh, p->oh, p->ih), weight(oh, p->oh, p->ih)},
                ww[2]
                = {1.f - weight(ow, p->ow, p->iw), weight(ow, p->ow, p->iw)};
        for_(int i = 0; i < 2; i++)
        for_(int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
            ((float *)diff_src)[src_off_f(p, mb, ic, id[i], ih[j], iw[k])]
                    += wd[i] * wh[j] * ww[k] * diff_dst_val;
        }
    };

    dnnl::impl::parallel_nd(p->mb, p->ic, [&](int64_t mb, int64_t ic) {
        zero_diff_src(mb, ic);
        for (int64_t od = 0; od < p->od; ++od)
            for (int64_t oh = 0; oh < p->oh; ++oh)
                for (int64_t ow = 0; ow < p->ow; ++ow)
                    if (p->alg == nearest) {
                        ker_nearest(mb, ic, od, oh, ow);
                    } else {
                        ker_linear(mb, ic, od, oh, ow);
                    }
    });
}

} // namespace resampling
