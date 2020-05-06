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

#include <stdlib.h>

#include "tests/test_thread.hpp"

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"

#include "rnn/cells.hpp"

namespace rnn {
template <typename T1, typename T2>
void lbr_gru_fwd_postgemm_template(T1 func1, T2 func2, const prb_t &p,
        float *gates_, const float *src_iter_, const float *bias_,
        float *dst_layer_, float *cell_scratchpad_) {
    AOC<const float> src_iter(src_iter_, p.mb, p.wc);
    AOC<const float> bias(bias_, p.n_gates() + 1, p.dhc);
    AOC<float> gates(gates_, p.mb, p.n_gates(), p.dhc);
    AOC<float> dst_layer(dst_layer_, p.mb, p.wc);
    AOC<float> cell_scratchpad(cell_scratchpad_, p.mb, p.n_gates(), p.dhc);

    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t j = 0; j < p.n_gates() - 1; j++)
            for (int64_t k = 0; k < p.dhc; k++) {
                gates(i, j, k) = func1(p.linear_scales[j],
                        gates(i, j, k) + cell_scratchpad(i, j, k) + bias(j, k));
            }

    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t k = 0; k < p.dhc; k++) {
            gates(i, GRU_O, k) = func2(p.linear_scales[GRU_O],
                    gates(i, GRU_O, k)
                            + gates(i, GRU_R, k)
                                    * (cell_scratchpad(i, GRU_O, k)
                                            + bias(LBR_GRU_U_PRIME, k))
                            + bias(GRU_O, k));
        }

    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t k = 0; k < p.dhc; k++) {
            dst_layer(i, k) = gates(i, GRU_U, k) * src_iter(i, k)
                    + (1 - gates(i, GRU_U, k)) * gates(i, GRU_O, k);
        }
}

void lbr_gru_fwd_postgemm(const prb_t &p, float *gates_, const float *src_iter_,
        const float *bias_, float *dst_layer_, float *cell_scratchpad_) {
    if (p.skip_nonlinear)
        lbr_gru_fwd_postgemm_template(
                [](float scale, float a) { return scale * a; },
                [](float scale, float a) { return scale * a; }, p, gates_,
                src_iter_, bias_, dst_layer_, cell_scratchpad_);
    else
        lbr_gru_fwd_postgemm_template(
                [](float scale, float a) { return logistic(a); },
                [](float scale, float a) { return tanhf(a); }, p, gates_,
                src_iter_, bias_, dst_layer_, cell_scratchpad_);
}

void lbr_gru_fwd(const prb_t &p, float *dst_layer_, float *gates_,
        const float *weights_layer_, const float *weights_iter_,
        const float *bias_, const float *src_layer_, const float *src_iter_,
        float *cell_scratchpad_) {
    gemm("C", "N", "N", p.mb, p.n_gates() * p.dhc, p.slc, 1.0, src_layer_, p.wc,
            weights_layer_, p.n_gates() * p.dhc, 0.0, gates_,
            p.n_gates() * p.dhc);

    gemm("C", "N", "N", p.mb, p.n_gates() * p.dhc, p.sic, 1.0, src_iter_, p.wc,
            weights_iter_, p.n_gates() * p.dhc, 0.0, cell_scratchpad_,
            p.n_gates() * p.dhc);

    lbr_gru_fwd_postgemm(
            p, gates_, src_iter_, bias_, dst_layer_, cell_scratchpad_);
}

void lbr_gru_bwd_pregemm(const prb_t &p, const float *src_iter_,
        const float *diff_dst_layer_, const float *diff_dst_iter_,
        const float *gates_, const float *Wh_b_, float *diff_src_iter_,
        float *b_gates_, float *b_gates_r_) {
    AOC<const float> src_iter(src_iter_, p.mb, p.wc);
    AOC<const float> diff_dst_layer(diff_dst_layer_, p.mb, p.wc);
    AOC<const float> diff_dst_iter(diff_dst_iter_, p.mb, p.wc);
    AOC<const float> gates(gates_, p.mb, p.n_gates(), p.dhc);
    AOC<const float> Wh_b(Wh_b_, p.mb, p.dhc);

    AOC<float> diff_src_iter(diff_src_iter_, p.mb, p.wc);
    AOC<float> b_gates(b_gates_, p.mb, p.n_gates(), p.dhc);
    AOC<float> b_gates_r(b_gates_r_, p.mb, p.n_gates(), p.dhc);

    // do = (1 - u) * dh; do^ = one_m_square(o) * do;
    // du = (h - o) * dh; du^ = x_m_square(u) * du;
    // dr = (Wh + b) * do^; dr^ = x_m_square(r) * dr;
    for (int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dhc; ih++) {
            float h = src_iter(ib, ih);
            float dh = diff_dst_layer(ib, ih) + diff_dst_iter(ib, ih);
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
        float *diff_weights_layer_, float *diff_weights_iter_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_, const float *bias_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_,
        float *cell_scratchpad_) {
    AOC<const float> weights_iter(weights_iter_, p.sic, p.n_gates(), p.dhc);
    AOC<const float> bias(bias_, p.n_gates() + 1, p.dhc);

    float *Wh_b_ = cell_scratchpad_;
    float *b_gates_r_ = cell_scratchpad_ + p.dhc * p.mb;
    AOC<float> Wh_b(Wh_b_, p.mb, p.dhc);
    AOC<float> b_gates_r(b_gates_r_, p.mb, p.n_gates(), p.dhc);

    // TODO: save this this GEMM + bias in the fwd pass
    for (int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dhc; ih++)
            Wh_b(ib, ih) = bias(LBR_GRU_U_PRIME, ih);

    gemm("C", "N", "N", p.mb, p.dhc, p.sic, 1.0, src_iter_, p.wc,
            &weights_iter(0, GRU_O, 0), p.n_gates() * p.dhc, 1.0, Wh_b_, p.dhc);

    lbr_gru_bwd_pregemm(p, src_iter_, diff_dst_layer_, diff_dst_iter_, gates_,
            Wh_b_, diff_src_iter_, b_gates_, b_gates_r_);

    gemm("C", "T", "N", p.sic, p.n_gates() * p.dhc, p.mb, 1.0, src_iter_, p.wc,
            b_gates_r_, p.n_gates() * p.dhc, 1.0, diff_weights_iter_,
            p.n_gates() * p.dhc);
    gemm("C", "T", "N", p.slc, p.n_gates() * p.dhc, p.mb, 1.0, src_layer_, p.wc,
            b_gates_, p.n_gates() * p.dhc, 1.0, diff_weights_layer_,
            p.n_gates() * p.dhc);

    gemm("C", "N", "T", p.mb, p.slc, p.n_gates() * p.dhc, 1.0, b_gates_,
            p.n_gates() * p.dhc, weights_layer_, p.n_gates() * p.dhc, 0.0,
            diff_src_layer_, p.wc);
    gemm("C", "N", "T", p.mb, p.sic, p.n_gates() * p.dhc, 1.0, b_gates_r_,
            p.n_gates() * p.dhc, weights_iter_, p.n_gates() * p.dhc, 1.0,
            diff_src_iter_, p.wc);

    gates_reduction(p, b_gates_, diff_bias_);
    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t k = 0; k < p.dhc; k++)
            diff_bias_[LBR_GRU_U_PRIME * p.dhc + k] += b_gates_r(i, GRU_O, k);
}

} // namespace rnn
