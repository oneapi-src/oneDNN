/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "src/common/mkldnn_thread.hpp"

#include "lrn/lrn.hpp"

namespace lrn {

void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src, dnn_mem_t &dst) {
    const int size = p->ls;
    const int half_size = (size - 1) / 2;
    const int summands = p->alg == ACROSS
        ? size
        : p->id > 1
            ? size * size * size
            : size * size;

    mkldnn::impl::parallel_nd(p->mb, p->ic, p->id, p->ih, p->iw,
            [&](int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) {
                float sum = 0;
                if (p->alg == ACROSS) {
                    const int64_t c_st = MAX2(c - half_size + 0, 0);
                    const int64_t c_en = MIN2(c + half_size + 1, p->ic);

                    for (int64_t cs = c_st; cs < c_en; ++cs) {
                        const auto off = data_off(p, mb, cs, d, h, w);
                        const float s = src.get_elem(off);
                        sum += s * s;
                    }
                } else if (p->alg == WITHIN) {
                    const int64_t d_st = MAX2(d - half_size + 0, 0);
                    const int64_t d_en = MIN2(d + half_size + 1, p->id);
                    const int64_t h_st = MAX2(h - half_size + 0, 0);
                    const int64_t h_en = MIN2(h + half_size + 1, p->ih);
                    const int64_t w_st = MAX2(w - half_size + 0, 0);
                    const int64_t w_en = MIN2(w + half_size + 1, p->iw);
                    for (int64_t ds = d_st; ds < d_en; ++ds)
                        for (int64_t hs = h_st; hs < h_en; ++hs)
                            for (int64_t ws = w_st; ws < w_en; ++ws) {
                                const auto off = data_off(p, mb, c, ds, hs, ws);
                                const float s = src.get_elem(off);
                                sum += s * s;
                            }
                }

                sum = p->k + p->alpha * sum / summands;
                const auto off = data_off(p, mb, c, d, h, w);
                dst.set_elem(off, src.get_elem(off) / powf(sum, p->beta));
            });
}

void compute_ref_bwd(const prb_t *p, const dnn_mem_t &src,
        const dnn_mem_t &d_dst, dnn_mem_t &d_src) {
    assert(p->alg == ACROSS); // WITHIN is not supported in the library

    const int size = p->ls;
    const int half_size = (size - 1) / 2;
    const int summands = size;

    mkldnn::impl::parallel_nd(p->mb, p->ic, p->id, p->ih, p->iw,
            [&](int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) {
                float A = 0, B = 0, sum_mid = 0;
                const int64_t c_st = MAX2(c - half_size + 0, 0);
                const int64_t c_en = MIN2(c + half_size + 1, p->ic);

                for (int64_t cs = c_st; cs < c_en; ++cs) {
                    float sum = 0;
                    const int64_t i_st = MAX2(cs - half_size, 0);
                    const int64_t i_en = MIN2(cs + size - half_size, p->ic);

                    for (int64_t i = i_st; i < i_en; ++i) {
                        const auto off = data_off(p, mb, i, d, h, w);
                        const float s = src.get_elem(off);
                        sum += s * s;
                    }

                    sum = p->k + p->alpha * sum / summands;
                    if (c == cs)
                        sum_mid = sum;

                    float sum_beta_exp = (p->beta == 0.75)
                        ? sum / sqrt(sqrt(sum)) : powf(sum, p->beta);
                    float tmp = sum * sum_beta_exp;
                    const auto off = data_off(p, mb, cs, d, h, w);
                    B += (src.get_elem(off) * d_dst.get_elem(off) / tmp);
                }

                const auto off = data_off(p, mb, c, d, h, w);
                float sum_beta_exp = (p->beta == 0.75)
                    ? sum_mid / sqrt(sqrt(sum_mid)) : powf(sum_mid, p->beta);
                A = d_dst.get_elem(off) / sum_beta_exp;
                B *= (src.get_elem(off) * 2 * p->alpha * p->beta / summands);
                d_src.set_elem(off, A - B);
            });
}

}
