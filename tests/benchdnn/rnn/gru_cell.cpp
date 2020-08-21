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

template <typename T>
void gru_fwd_postgemm_part1_template(T func1, const prb_t &p, float *gates_,
        const float *src_iter_, const float *bias_, float *dst_layer_) {
    AOC<const float> bias(bias_, p.n_gates(), p.dhc);
    AOC<const float> src_iter(src_iter_, p.mb, p.wc);
    AOC<float> dst_layer(dst_layer_, p.mb, p.wc);
    AOC<float> gates(gates_, p.mb, p.n_gates(), p.dhc);

    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t k = 0; k < p.dhc; k++) {
            gates(i, GRU_U, k) = func1(p.linear_scales[GRU_U],
                    maybe_deq(p, gates(i, GRU_U, k), GRU_U * p.dhc + k)
                            + bias(GRU_U, k));
            gates(i, GRU_R, k) = func1(p.linear_scales[GRU_R],
                    maybe_deq(p, gates(i, GRU_R, k), GRU_R * p.dhc + k)
                            + bias(GRU_R, k));
            dst_layer(i, k) = maybe_q(
                    p, (maybe_deq(p, src_iter(i, k)) * gates(i, GRU_R, k)));
        }
}

void gru_fwd_postgemm_part1(const prb_t &p, float *gates_,
        const float *src_iter_, const float *bias_, float *dst_layer_) {
    if (p.skip_nonlinear)
        gru_fwd_postgemm_part1_template(
                [](float scale, float a) { return scale * a; }, p, gates_,
                src_iter_, bias_, dst_layer_);
    else
        gru_fwd_postgemm_part1_template(
                [](float scale, float a) { return logistic(a); }, p, gates_,
                src_iter_, bias_, dst_layer_);
}

template <typename T>
void gru_fwd_postgemm_part2_template(T func1, const prb_t &p, float *gates_,
        const float *src_iter_, const float *bias_, float *dst_layer_) {
    AOC<const float> bias(bias_, p.n_gates(), p.dhc);
    AOC<const float> src_iter(src_iter_, p.mb, p.wc);
    AOC<float> dst_layer(dst_layer_, p.mb, p.wc);
    AOC<float> gates(gates_, p.mb, p.n_gates(), p.dhc);
    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t k = 0; k < p.dhc; k++) {
            double U = gates(i, GRU_U, k);
            double O = func1(p.linear_scales[GRU_O],
                    maybe_deq(p, gates(i, GRU_O, k), GRU_O * p.dhc + k)
                            + bias(GRU_O, k));
            dst_layer(i, k) = maybe_q(p,
                    (float)(U * maybe_deq(p, src_iter(i, k)) + (1.0 - U) * O));

            gates(i, GRU_O, k) = O;
        }
}

void gru_fwd_postgemm_part2(const prb_t &p, float *gates_,
        const float *src_iter_, const float *bias_, float *dst_layer_) {
    if (p.skip_nonlinear)
        gru_fwd_postgemm_part2_template(
                [](float scale, float a) { return scale * a; }, p, gates_,
                src_iter_, bias_, dst_layer_);
    else
        gru_fwd_postgemm_part2_template(
                [](float scale, float a) { return tanhf(a); }, p, gates_,
                src_iter_, bias_, dst_layer_);
}

void gru_fwd(const prb_t &p, float *dst_layer_, float *gates_,
        const float *weights_layer_, const float *weights_iter_,
        const float *bias_, const float *src_layer_, const float *src_iter_) {
    AOC<const float> weights_iter(weights_iter_, p.sic, p.n_gates(), p.dhc);
    AOC<float> gates(gates_, p.mb, p.n_gates(), p.dhc);

    gemm("C", "N", "N", p.mb, p.n_gates() * p.dhc, p.slc, 1.0, src_layer_, p.wc,
            weights_layer_, p.n_gates() * p.dhc, 0.0, gates_,
            p.n_gates() * p.dhc);
    gemm("C", "N", "N", p.mb, (p.n_gates() - 1) * p.dhc, p.sic, 1.0, src_iter_,
            p.wc, weights_iter_, p.n_gates() * p.dhc, 1.0, gates_,
            p.n_gates() * p.dhc);

    gru_fwd_postgemm_part1(p, gates_, src_iter_, bias_, dst_layer_);

    gemm("C", "N", "N", p.mb, p.dhc, p.sic, 1.0, dst_layer_, p.wc,
            &(weights_iter(0, GRU_O, 0)), p.n_gates() * p.dhc, 1.0,
            &(gates(0, GRU_O, 0)), p.n_gates() * p.dhc);

    gru_fwd_postgemm_part2(p, gates_, src_iter_, bias_, dst_layer_);
}

void gru_bwd_pregemm_part1(const prb_t &p, const float *src_iter_,
        const float *diff_dst_layer_, const float *diff_dst_iter_,
        const float *gates_, float *diff_src_iter_, float *b_gates_) {
    AOC<const float> src_iter(src_iter_, p.mb, p.wc);
    AOC<const float> diff_dst_layer(diff_dst_layer_, p.mb, p.wc);
    AOC<const float> diff_dst_iter(diff_dst_iter_, p.mb, p.wc);
    AOC<const float> gates(gates_, p.mb, p.n_gates(), p.dhc);

    AOC<float> diff_src_iter(diff_src_iter_, p.mb, p.wc);
    AOC<float> b_gates(b_gates_, p.mb, p.n_gates(), p.dhc);

    // do = (1 - u) * dh; do^ = one_m_square(o) * do;
    // du = (h - u) * dh; du^ = x_m_square(u) * du;
    for (int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dhc; ih++) {
            float h = src_iter(ib, ih);
            float o = gates(ib, GRU_O, ih);
            float u = gates(ib, GRU_U, ih);
            float dh = diff_dst_layer(ib, ih) + diff_dst_iter(ib, ih);
            float du = (h - o) * dh;
            float dO = (1.0f - u) * dh;
            b_gates(ib, GRU_U, ih) = x_m_square(u) * du;
            b_gates(ib, GRU_O, ih) = one_m_square(o) * dO;
            diff_src_iter(ib, ih) = dh * u;
        }
}

void gru_bwd_pregemm_part2(const prb_t &p, const float *src_iter_,
        const float *gates_, const float *dhr_, float *diff_src_iter_,
        float *b_gates_, float *hr_) {
    AOC<const float> src_iter(src_iter_, p.mb, p.wc);
    AOC<const float> gates(gates_, p.mb, p.n_gates(), p.dhc);
    AOC<const float> dhr(dhr_, p.mb, p.dhc);
    AOC<float> diff_src_iter(diff_src_iter_, p.mb, p.wc);
    AOC<float> b_gates(b_gates_, p.mb, p.n_gates(), p.dhc);
    AOC<float> hr(hr_, p.mb, p.dhc);

    // dhr = Wo do^;
    // dr = h * dhr; dr^ = x_m_square(r) * dr;
    for (int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dhc; ih++) {
            float h = src_iter(ib, ih);
            float r = gates(ib, GRU_R, ih);
            float dr = h * dhr(ib, ih);
            hr(ib, ih) = h * r;
            diff_src_iter(ib, ih) += dhr(ib, ih) * r;
            b_gates(ib, GRU_R, ih) = x_m_square(r) * dr;
        }
}

void gru_bwd(const prb_t &p, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_, const float *bias_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_,
        float *cell_scratchpad_) {
    AOC<const float> weights_iter(weights_iter_, p.sic, p.n_gates(), p.dhc);

    AOC<float> diff_weights_iter(diff_weights_iter_, p.sic, p.n_gates(), p.dhc);
    AOC<float> b_gates(b_gates_, p.mb, p.n_gates(), p.dhc);

    assert(p.dhc == p.sic);
    float *dhr_ = cell_scratchpad_;
    float *hr_ = cell_scratchpad_ + p.mb * p.dhc;

    gru_bwd_pregemm_part1(p, src_iter_, diff_dst_layer_, diff_dst_iter_, gates_,
            diff_src_iter_, b_gates_);

    gemm("C", "N", "T", p.mb, p.sic, p.dhc, 1.0, &(b_gates(0, GRU_O, 0)),
            p.n_gates() * p.dhc, &(weights_iter(0, GRU_O, 0)),
            p.n_gates() * p.dhc, 0.0, dhr_, p.dhc);

    gru_bwd_pregemm_part2(
            p, src_iter_, gates_, dhr_, diff_src_iter_, b_gates_, hr_);

    // dWx += xdu^ | xdr^ | xdo^
    // dWh += hdu^ | ddr^ | (h * r)do^
    gemm("C", "T", "N", p.sic, (p.n_gates() - 1) * p.dhc, p.mb, 1.0, src_iter_,
            p.wc, b_gates_, p.n_gates() * p.dhc, 1.0, diff_weights_iter_,
            p.n_gates() * p.dhc);
    gemm("C", "T", "N", p.sic, p.dhc, p.mb, 1.0, hr_, p.dhc,
            &(b_gates(0, GRU_O, 0)), p.n_gates() * p.dhc, 1.0,
            &(diff_weights_iter(0, GRU_O, 0)), p.n_gates() * p.dhc);
    gemm("C", "T", "N", p.slc, p.n_gates() * p.dhc, p.mb, 1.0, src_layer_, p.wc,
            b_gates_, p.n_gates() * p.dhc, 1.0, diff_weights_layer_,
            p.n_gates() * p.dhc);

    // dx_next = Wxudu^ + Wxrdr^ + Wxodo^
    // dh_next = dh * u + Whudu^ + Whzdz^ + r * Whodo^
    gemm("C", "N", "T", p.mb, p.sic, (p.n_gates() - 1) * p.dhc, 1.0, b_gates_,
            p.n_gates() * p.dhc, weights_iter_, p.n_gates() * p.dhc, 1.0,
            diff_src_iter_, p.wc);
    gemm("C", "N", "T", p.mb, p.slc, p.n_gates() * p.dhc, 1.0, b_gates_,
            p.n_gates() * p.dhc, weights_layer_, p.n_gates() * p.dhc, 0.0,
            diff_src_layer_, p.wc);

    gates_reduction(p, b_gates_, diff_bias_);
}

} // namespace rnn
