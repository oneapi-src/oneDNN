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

#include <stdlib.h>

#include "src/common/dnnl_thread.hpp"

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"
#include "rnn/rnn_cells.hpp"

namespace rnn {
template <typename T1, typename T2>
void lbr_gru_fwd_postgemm_template(T1 func1, T2 func2, const prb_t &p,
        float *gates_, const float *src_iter_h_, const float *bias_,
        float *dst_iter_h_, float *ws_local_) {
    AOC<const float> src_iter_h(src_iter_h_, p.mb, p.wc);
    AOC<const float> bias(bias_, p.n_gates() + 1, p.dic);
    AOC<float> gates(gates_, p.mb, p.n_gates(), p.dic);
    AOC<float> h_dst(dst_iter_h_, p.mb, p.wc);
    AOC<float> tmp_ws(ws_local_, p.mb, p.n_gates(), p.dic);

    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t j = 0; j < p.n_gates() - 1; j++)
            for (int64_t k = 0; k < p.dic; k++) {
                gates(i, j, k) = func1(p.linear_scales[j],
                        gates(i, j, k) + tmp_ws(i, j, k) + bias(j, k));
            }

    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t k = 0; k < p.dic; k++) {
            gates(i, GRU_O, k) = func2(p.linear_scales[GRU_O],
                    gates(i, GRU_O, k)
                            + gates(i, GRU_R, k)
                                    * (tmp_ws(i, GRU_O, k)
                                            + bias(LBR_GRU_U_PRIME, k))
                            + bias(GRU_O, k));
        }

    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t k = 0; k < p.dic; k++) {
            h_dst(i, k) = gates(i, GRU_U, k) * src_iter_h(i, k)
                    + (1 - gates(i, GRU_U, k)) * gates(i, GRU_O, k);
        }
}

void lbr_gru_fwd_postgemm(const prb_t &p, float *gates_,
        const float *src_iter_h_, const float *bias_, float *dst_iter_h_,
        float *ws_local_) {
    if (p.skip_nonlinear)
        lbr_gru_fwd_postgemm_template(
                [](float scale, float a) { return scale * a; },
                [](float scale, float a) { return scale * a; }, p, gates_,
                src_iter_h_, bias_, dst_iter_h_, ws_local_);
    else
        lbr_gru_fwd_postgemm_template(
                [](float scale, float a) { return logistic(a); },
                [](float scale, float a) { return tanhf(a); }, p, gates_,
                src_iter_h_, bias_, dst_iter_h_, ws_local_);
}

void lbr_gru_fwd(const prb_t &p, float *dst_iter_h_, float *gates_,
        const float *weights_layer_, const float *weights_iter_h_,
        const float *bias_, const float *src_layer_, const float *src_iter_h_,
        float *ws_local_) {
    gemm("C", "N", "N", p.mb, p.n_gates() * p.dic, p.slc, 1.0, src_layer_, p.wc,
            weights_layer_, p.n_gates() * p.dic, 0.0, gates_,
            p.n_gates() * p.dic);

    gemm("C", "N", "N", p.mb, p.n_gates() * p.dic, p.sic, 1.0, src_iter_h_,
            p.wc, weights_iter_h_, p.n_gates() * p.dic, 0.0, ws_local_,
            p.n_gates() * p.dic);

    lbr_gru_fwd_postgemm(p, gates_, src_iter_h_, bias_, dst_iter_h_, ws_local_);
}

void lbr_gru_bwd_pregemm(const prb_t &p, const float *src_iter_,
        const float *diff_dst_layer_, const float *diff_dst_iter_h_,
        const float *gates_, float *diff_src_iter_, float *b_gates_,
        float *ws_local_) {
    float *Wh_b_ = ws_local_;
    float *b_gates_r_ = ws_local_ + p.dic * p.mb;

    AOC<const float> src_iter(src_iter_, p.mb, p.wc);
    AOC<const float> diff_dst_layer(diff_dst_layer_, p.mb, p.wc);
    AOC<const float> diff_dst_iter_h(diff_dst_iter_h_, p.mb, p.wc);
    AOC<const float> gates(gates_, p.mb, p.n_gates(), p.dic);

    AOC<float> diff_src_iter(diff_src_iter_, p.mb, p.wc);
    AOC<float> b_gates(b_gates_, p.mb, p.n_gates(), p.dic);
    AOC<float> b_gates_r(b_gates_r_, p.mb, p.n_gates(), p.dic);
    AOC<float> Wh_b(Wh_b_, p.mb, p.dic);

    // do = (1 - u) * dh; do^ = one_m_square(o) * do;
    // du = (h - o) * dh; du^ = x_m_square(u) * du;
    // dr = (Wh + b) * do^; dr^ = x_m_square(r) * dr;
    for (int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dic; ih++) {
            float h = src_iter(ib, ih);
            float dh = diff_dst_layer(ib, ih) + diff_dst_iter_h(ib, ih);
            float u = gates(ib, GRU_U, ih);
            float r = gates(ib, GRU_R, ih);
            float o = gates(ib, GRU_O, ih);
            float du = (h - o) * dh;
            float dO = (1.0f - u) * dh;

            b_gates(ib, GRU_U, ih) = x_m_square(u) * du;
            b_gates(ib, GRU_O, ih) = one_m_square(o) * dO;

            float dr = Wh_b(ib, ih) * b_gates(ib, GRU_O, ih);
            b_gates(ib, GRU_R, ih) = x_m_square(r) * dr;

            b_gates_r(ib, GRU_U, ih) = b_gates(ib, GRU_U, ih);
            b_gates_r(ib, GRU_R, ih) = b_gates(ib, GRU_R, ih);
            b_gates_r(ib, GRU_O, ih) = b_gates(ib, GRU_O, ih) * r;
            diff_src_iter(ib, ih) = dh * u;
        }
}

void lbr_gru_bwd(const prb_t &p, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_h_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *dst_iter_h_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_h_,
        float *ws_local_) {
    AOC<const float> weights_iter_h(weights_iter_h_, p.sic, p.n_gates(), p.dic);
    AOC<const float> bias(bias_, p.n_gates() + 1, p.dic);

    float *Wh_b_ = ws_local_;
    float *b_gates_r_ = ws_local_ + p.dic * p.mb;
    AOC<float> Wh_b(Wh_b_, p.mb, p.dic);
    AOC<float> b_gates_r(b_gates_r_, p.mb, p.n_gates(), p.dic);

    // TODO: save this this GEMM + bias in the fwd pass
    for (int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dic; ih++)
            Wh_b(ib, ih) = bias(LBR_GRU_U_PRIME, ih);

    gemm("C", "N", "N", p.mb, p.dic, p.sic, 1.0, src_iter_, p.wc,
            &weights_iter_h(0, GRU_O, 0), p.n_gates() * p.dic, 1.0, Wh_b_,
            p.dic);

    lbr_gru_bwd_pregemm(p, src_iter_, diff_dst_layer_, diff_dst_iter_h_, gates_,
            diff_src_iter_, b_gates_, ws_local_);

    gemm("C", "T", "N", p.sic, p.n_gates() * p.dic, p.mb, 1.0, src_iter_, p.wc,
            b_gates_r_, p.n_gates() * p.dic, 1.0, diff_weights_iter_h_,
            p.n_gates() * p.dic);
    gemm("C", "T", "N", p.slc, p.n_gates() * p.dic, p.mb, 1.0, src_layer_, p.wc,
            b_gates_, p.n_gates() * p.dic, 1.0, diff_weights_layer_,
            p.n_gates() * p.dic);

    gemm("C", "N", "T", p.mb, p.slc, p.n_gates() * p.dic, 1.0, b_gates_,
            p.n_gates() * p.dic, weights_layer_, p.n_gates() * p.dic, 0.0,
            diff_src_layer_, p.wc);
    gemm("C", "N", "T", p.mb, p.sic, p.n_gates() * p.dic, 1.0, b_gates_r_,
            p.n_gates() * p.dic, weights_iter_h_, p.n_gates() * p.dic, 1.0,
            diff_src_iter_, p.wc);

    gates_reduction(p, b_gates_, diff_bias_);
    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t k = 0; k < p.dic; k++)
            diff_bias_[LBR_GRU_U_PRIME * p.dic + k] += b_gates_r(i, GRU_O, k);
}

} // namespace rnn
