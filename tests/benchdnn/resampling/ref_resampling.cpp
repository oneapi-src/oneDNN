/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include "utils/parallel.hpp"

#include "resampling/resampling.hpp"

namespace resampling {

float linear_map(const int64_t y, const int64_t y_max, const int64_t x_max) {
    const float s = (y + 0.5f) * x_max / y_max;
    return s - 0.5f;
}
int64_t left(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return MAX2((int64_t)floorf(linear_map(y, y_max, x_max)), (int64_t)0);
}
int64_t right(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return MIN2((int64_t)ceilf(linear_map(y, y_max, x_max)), x_max - 1);
}
int64_t near(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return roundf(linear_map(y, y_max, x_max));
}
float weight(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return fabs(linear_map(y, y_max, x_max) - left(y, y_max, x_max));
}

void compute_ref_fwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);

    float *dst_ptr = (float *)dst;

    int64_t MB = prb->mb;
    int64_t IC = prb->ic;
    int64_t ID = prb->id;
    int64_t IH = prb->ih;
    int64_t IW = prb->iw;
    int64_t OD = prb->od;
    int64_t OH = prb->oh;
    int64_t OW = prb->ow;

    auto ker_nearest = [&](float &result, int64_t mb, int64_t ic, int64_t od,
                               int64_t oh, int64_t ow) {
        const int64_t id = near(od, OD, ID);
        const int64_t ih = near(oh, OH, IH);
        const int64_t iw = near(ow, OW, IW);
        result = src.get_elem(src_off_f(prb, mb, ic, id, ih, iw));
    };

    auto ker_linear = [&](float &result, int64_t mb, int64_t ic, int64_t od,
                              int64_t oh, int64_t ow) {
        const int64_t id[2] = {left(od, OD, ID), right(od, OD, ID)};
        const int64_t ih[2] = {left(oh, OH, IH), right(oh, OH, IH)};
        const int64_t iw[2] = {left(ow, OW, IW), right(ow, OW, IW)};
        const float wd[2] = {1.f - weight(od, OD, ID), weight(od, OD, ID)};
        const float wh[2] = {1.f - weight(oh, OH, IH), weight(oh, OH, IH)};
        const float ww[2] = {1.f - weight(ow, OW, IW), weight(ow, OW, IW)};

        float cd[2][2] = {{0}};
        for_(int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            cd[i][j] = src.get_elem(src_off_f(prb, mb, ic, id[0], ih[i], iw[j]))
                            * wd[0]
                    + src.get_elem(src_off_f(prb, mb, ic, id[1], ih[i], iw[j]))
                            * wd[1];

        float ch[2] = {0};
        for (int i = 0; i < 2; i++)
            ch[i] = cd[0][i] * wh[0] + cd[1][i] * wh[1];

        float cw = ch[0] * ww[0] + ch[1] * ww[1];

        result = cw;
    };

    auto v_po_masks = prb->attr.post_ops.get_po_masks();
    benchdnn_parallel_nd(MB, IC, OD, OH, OW,
            [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
                float result = 0.f;
                if (prb->alg == nearest) {
                    ker_nearest(result, mb, ic, od, oh, ow);
                } else {
                    ker_linear(result, mb, ic, od, oh, ow);
                }
                const auto dst_off = dst_off_f(prb, mb, ic, od, oh, ow);

                const auto v_po_vals
                        = prepare_po_vals(dst, args, v_po_masks, dst_off);

                maybe_post_ops(
                        prb->attr, result, dst.get_elem(dst_off), v_po_vals);
                dst_ptr[dst_off] = result;
            });
}

void compute_ref_bwd(const prb_t *prb, const args_t &args) {
    const dnn_mem_t &d_dst = args.find(DNNL_ARG_DIFF_DST);
    const dnn_mem_t &d_src = args.find(DNNL_ARG_DIFF_SRC);

    float *d_src_ptr = (float *)d_src;

    int64_t MB = prb->mb;
    int64_t IC = prb->ic;
    int64_t ID = prb->id;
    int64_t IH = prb->ih;
    int64_t IW = prb->iw;
    int64_t OD = prb->od;
    int64_t OH = prb->oh;
    int64_t OW = prb->ow;

    auto ker_nearest
            = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
                  const auto d_dst_off = dst_off_f(prb, mb, ic, od, oh, ow);
                  float d_dst_val = d_dst.get_elem(d_dst_off);
                  const int64_t id = near(od, OD, ID);
                  const int64_t ih = near(oh, OH, IH);
                  const int64_t iw = near(ow, OW, IW);
                  d_src_ptr[src_off_f(prb, mb, ic, id, ih, iw)] += d_dst_val;
              };

    auto ker_linear = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh,
                              int64_t ow) {
        const auto d_dst_off = dst_off_f(prb, mb, ic, od, oh, ow);
        float d_dst_val = d_dst.get_elem(d_dst_off);
        const int64_t id[2] = {left(od, OD, ID), right(od, OD, ID)};
        const int64_t ih[2] = {left(oh, OH, IH), right(oh, OH, IH)};
        const int64_t iw[2] = {left(ow, OW, IW), right(ow, OW, IW)};
        const float wd[2] = {1.f - weight(od, OD, ID), weight(od, OD, ID)};
        const float wh[2] = {1.f - weight(oh, OH, IH), weight(oh, OH, IH)};
        const float ww[2] = {1.f - weight(ow, OW, IW), weight(ow, OW, IW)};
        for_(int i = 0; i < 2; i++)
        for_(int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
            d_src_ptr[src_off_f(prb, mb, ic, id[i], ih[j], iw[k])]
                    += wd[i] * wh[j] * ww[k] * d_dst_val;
        }
    };

    // zeroing d_src for correct result
    benchdnn_parallel_nd(MB, IC, ID, IH, IW,
            [&](int64_t mb, int64_t ic, int64_t id, int64_t ih, int64_t iw) {
                d_src_ptr[src_off_f(prb, mb, ic, id, ih, iw)] = 0;
            });

    benchdnn_parallel_nd(MB, IC, [&](int64_t mb, int64_t ic) {
        for_(int64_t od = 0; od < OD; ++od)
        for_(int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow)
            if (prb->alg == nearest) {
                ker_nearest(mb, ic, od, oh, ow);
            } else {
                ker_linear(mb, ic, od, oh, ow);
            }
    });
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    if (prb->dir & FLAG_FWD)
        compute_ref_fwd(prb, args);
    else
        compute_ref_bwd(prb, args);
}

} // namespace resampling
