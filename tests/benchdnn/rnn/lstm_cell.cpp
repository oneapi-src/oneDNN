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

#include "src/common/dnnl_thread.hpp"

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"

#include "rnn/cells.hpp"

namespace rnn {

template <typename T1, typename T2>
void lstm_fwd_postgemm_template(T1 func1, T2 func2, const prb_t &p,
        float *gates_, const float *weights_peephole_, const float *bias_,
        const float *src_iter_c_, float *dst_layer_, float *dst_iter_c_) {
    AOC<float> gates(gates_, p.mb, p.n_gates(), p.dhc);
    AOC<const float> weights_peephole(weights_peephole_, 3, p.dhc);
    AOC<const float> bias(bias_, p.n_gates(), p.dhc);
    AOC<const float> src_iter_c(src_iter_c_, p.mb, p.wc);
    AOC<float> dst_layer(dst_layer_, p.mb, p.wc);
    AOC<float> dst_iter_c(dst_iter_c_, p.mb, p.wc);

    auto maybe_deq_w = [&](float g, int64_t oc) {
        if (!p.cfg.is_int8()) return g;
        float scale = 1.;
        if (p.scale_policy == policy_t::PER_OC)
            scale = p.wei_oc_scales[oc];
        else if (p.scale_policy == policy_t::COMMON)
            scale = p.wei_scale;
        scale *= p.data_scale;
        return g * (1.f / scale);
    };

    auto maybe_q_d = [&](float h) {
        if (!p.cfg.is_int8()) return h;
        float fp = p.data_scale * h + p.data_shift;
        if (fp > p.cfg[SRC_LAYER].max) fp = p.cfg[SRC_LAYER].max;
        if (fp < p.cfg[SRC_LAYER].min) fp = p.cfg[SRC_LAYER].min;
        fp = mxcsr_round(fp);
        return fp;
    };

    // run the eltwise
    dnnl::impl::parallel_nd(p.mb, [&](int64_t ib) {
        for (int64_t ih = 0; ih < p.dhc; ih++) {
            float peephole_extra_i = 0, peephole_extra_f = 0;
            if (p.is_lstm_peephole()) {
                peephole_extra_i = weights_peephole(0, ih) * src_iter_c(ib, ih);
                peephole_extra_f = weights_peephole(1, ih) * src_iter_c(ib, ih);
            }

            gates(ib, LSTM_I, ih) = func1(p.linear_scales[LSTM_I],
                    maybe_deq_w(gates(ib, LSTM_I, ih), LSTM_I * p.dhc + ih)
                            + peephole_extra_i + bias(LSTM_I, ih));
            gates(ib, LSTM_F, ih) = func1(p.linear_scales[LSTM_F],
                    maybe_deq_w(gates(ib, LSTM_F, ih), LSTM_F * p.dhc + ih)
                            + peephole_extra_f + bias(LSTM_F, ih));

            gates(ib, LSTM_C, ih) = func2(p.linear_scales[LSTM_C],
                    maybe_deq_w(gates(ib, LSTM_C, ih), LSTM_C * p.dhc + ih)
                            + bias(LSTM_C, ih));

            // compute C_t_l and H_t_l
            float tmp = gates(ib, LSTM_F, ih) * src_iter_c(ib, ih)
                    + gates(ib, LSTM_I, ih) * gates(ib, LSTM_C, ih);
            dst_iter_c(ib, ih) = tmp;

            float peephole_extra_o = 0;
            if (p.is_lstm_peephole())
                peephole_extra_o = weights_peephole(2, ih) * tmp;

            gates(ib, LSTM_O, ih) = func1(p.linear_scales[LSTM_O],
                    maybe_deq_w(gates(ib, LSTM_O, ih), LSTM_O * p.dhc + ih)
                            + peephole_extra_o + bias(LSTM_O, ih));

            dst_layer(ib, ih) = maybe_q_d(
                    gates(ib, LSTM_O, ih) * func2(p.linear_cscale, tmp));

            for (int64_t ig = 0; ig < 4; ig++) {
                BENCHDNN_PRINT(80,
                        "activation 1 a[" IFMT "][" IFMT "][" IFMT "] = %.7f\n",
                        ib, ig, ih, gates(ib, ig, ih));
            }
            BENCHDNN_PRINT(80, "recomp tmp(%a) cin(%a) ht(%a)\n", tmp,
                    src_iter_c(ib, ih), dst_layer(ib, ih));
        }
    });
}

void lstm_fwd_postgemm(const prb_t &p, float *gates_,
        const float *weights_peephole_, const float *bias_,
        const float *src_iter_c_, float *dst_layer_, float *dst_iter_c_) {
    if (p.skip_nonlinear)
        lstm_fwd_postgemm_template(
                [](float scale, float val) { return scale * val; },
                [](float scale, float val) { return scale * val; }, p, gates_,
                weights_peephole_, bias_, src_iter_c_, dst_layer_, dst_iter_c_);
    else
        lstm_fwd_postgemm_template(
                [](float scale, float val) { return logistic(val); },
                [](float scale, float val) { return tanhf(val); }, p, gates_,
                weights_peephole_, bias_, src_iter_c_, dst_layer_, dst_iter_c_);
}

void lstm_fwd(const prb_t &p, float *dst_layer_, float *dst_iter_,
        float *dst_iter_c_, float *gates_, const float *weights_layer_,
        const float *weights_iter_, const float *weights_peephole_,
        const float *weights_projection_, const float *bias_,
        const float *src_layer_, const float *src_iter_,
        const float *src_iter_c_) {

    gemm("C", "N", "N", p.mb, p.n_gates() * p.dhc, p.slc, 1.0, src_layer_, p.wc,
            weights_layer_, p.n_gates() * p.dhc, 0.0, gates_,
            p.n_gates() * p.dhc);
    gemm("C", "N", "N", p.mb, p.n_gates() * p.dhc, p.sic, 1.0, src_iter_, p.wc,
            weights_iter_, p.n_gates() * p.dhc, 1.0, gates_,
            p.n_gates() * p.dhc);

    lstm_fwd_postgemm(p, gates_, weights_peephole_, bias_, src_iter_c_,
            dst_layer_, dst_iter_c_);

    if (p.is_lstm_projection()) {
        assert(dst_layer_ != dst_iter_);
        gemm("C", "N", "N", p.mb, p.dic, p.dhc, 1.0, dst_layer_, p.wc,
                weights_projection_, p.dic, 0.0, dst_iter_, p.wc);
    } else {
        assert(p.dic == p.dhc);
        assert(dst_layer_ == dst_iter_);
    }
}

template <typename T1>
void lstm_bwd_pregemm_template(T1 func1, const prb_t &p,
        const float *src_iter_c_, const float *dst_iter_c_,
        const float *weights_peephole_, const float *diff_hidden_state_,
        const float *diff_dst_iter_c_, const float *gates_,
        float *diff_src_iter_c_, float *b_gates_) {
    AOC<const float> src_iter_c(src_iter_c_, p.mb, p.wc);
    AOC<const float> dst_iter_c(dst_iter_c_, p.mb, p.wc);
    AOC<const float> weights_peephole(weights_peephole_, 3, p.dhc);
    AOC<const float> diff_hidden_state(diff_hidden_state_, p.mb, p.dhc);
    AOC<const float> diff_dst_iter_c(diff_dst_iter_c_, p.mb, p.wc);
    AOC<const float> gates(gates_, p.mb, p.n_gates(), p.dhc);
    AOC<float> diff_src_iter_c(diff_src_iter_c_, p.mb, p.wc);
    AOC<float> b_gates(b_gates_, p.mb, p.n_gates(), p.dhc);

    for (int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dhc; ih++) {
            BENCHDNN_PRINT(80, "rnn_single_bwd: ib = " IFMT " ih = " IFMT "\n",
                    ib, ih);
            float hi = gates(ib, LSTM_I, ih);
            float hf = gates(ib, LSTM_F, ih);
            float hc = gates(ib, LSTM_C, ih);
            float ho = gates(ib, LSTM_O, ih);

            float dh = diff_hidden_state(ib, ih);

            float tanhC = func1(p.linear_cscale, dst_iter_c(ib, ih));
            float dho = tanhC * dh;
            b_gates(ib, LSTM_O, ih) = x_m_square(ho) * dho;

            float dc = diff_dst_iter_c(ib, ih);
            dc += ho * dh * one_m_square(tanhC);

            if (p.is_lstm_peephole())
                dc += b_gates(ib, LSTM_O, ih) * weights_peephole(2, ih);

            float dc_tm1 = hf * dc;

            float c_old = src_iter_c(ib, ih);
            float dhf = c_old * dc;
            b_gates(ib, LSTM_F, ih) = x_m_square(hf) * dhf;

            float dhi = hc * dc;
            b_gates(ib, LSTM_I, ih) = x_m_square(hi) * dhi;

            float dhc = hi * dc;
            b_gates(ib, LSTM_C, ih) = one_m_square(hc) * dhc;

            if (p.is_lstm_peephole()) {
                dc_tm1 += b_gates(ib, LSTM_F, ih) * weights_peephole(1, ih);
                dc_tm1 += b_gates(ib, LSTM_I, ih) * weights_peephole(0, ih);
            }

            diff_src_iter_c(ib, ih) = dc_tm1;
        }
}

void lstm_bwd_pregemm(const prb_t &p, const float *src_iter_c_,
        const float *dst_iter_c_, const float *weights_peephole_,
        const float *diff_hidden_state_, const float *diff_dst_iter_c_,
        const float *gates_, float *diff_src_iter_c_, float *b_gates_) {
    if (p.skip_nonlinear)
        lstm_bwd_pregemm_template(
                [](float scale, float val) { return scale * val; }, p,
                src_iter_c_, dst_iter_c_, weights_peephole_, diff_hidden_state_,
                diff_dst_iter_c_, gates_, diff_src_iter_c_, b_gates_);

    else
        lstm_bwd_pregemm_template(
                [](float scale, float val) { return tanhf(val); }, p,
                src_iter_c_, dst_iter_c_, weights_peephole_, diff_hidden_state_,
                diff_dst_iter_c_, gates_, diff_src_iter_c_, b_gates_);
}

void lstm_bwd_weights_peephole(const prb_t &p, const float *src_iter_c_,
        const float *dst_iter_c_, const float *b_gates_,
        float *diff_weights_peephole_) {
    AOC<const float> src_iter_c(src_iter_c_, p.mb, p.wc);
    AOC<const float> dst_iter_c(dst_iter_c_, p.mb, p.wc);
    AOC<const float> b_gates(b_gates_, p.mb, p.n_gates(), p.dhc);
    AOC<float> diff_weights_peephole(diff_weights_peephole_, 3, p.dhc);

    for_(int64_t ib = 0; ib < p.mb; ++ib)
    for (int64_t ih = 0; ih < p.dhc; ++ih)
        diff_weights_peephole(2, ih)
                += b_gates(ib, LSTM_O, ih) * dst_iter_c(ib, ih);

    for_(int64_t ib = 0; ib < p.mb; ++ib)
    for_(int64_t j = 0; j < 2; ++j)
    for (int64_t ih = 0; ih < p.dhc; ++ih)
        diff_weights_peephole(j, ih) += b_gates(ib, j, ih) * src_iter_c(ib, ih);
}

void lstm_bwd(const prb_t &p, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_src_iter_c_, float *diff_weights_layer_,
        float *diff_weights_iter_, float *diff_weights_peephole_,
        float *diff_weights_projection_, float *diff_bias_, float *b_gates_,
        const float *src_layer_, const float *src_iter_,
        const float *src_iter_c_, const float *weights_layer_,
        const float *weights_iter_, const float *weights_peephole_,
        const float *weights_projection_, const float *bias_,
        const float *dst_layer_, const float *dst_iter_c_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_,
        const float *diff_dst_iter_c_, float *cell_scratchpad_) {
    float *diff_hidden_state_ = cell_scratchpad_;

    AOC<float> diff_hidden_state(diff_hidden_state_, p.mb, p.dhc);
    AOC<const float> diff_dst_layer(diff_dst_layer_, p.mb, p.wc);
    AOC<const float> diff_dst_iter(diff_dst_iter_, p.mb, p.wc);

    if (p.is_lstm_projection()) {
        gemm("C", "T", "N", p.dhc, p.dic, p.mb, 1.0, dst_layer_, p.wc,
                diff_dst_iter_, p.wc, 1.0, diff_weights_projection_, p.dic);
        gemm("C", "N", "T", p.mb, p.dhc, p.dic, 1.0, diff_dst_iter_, p.wc,
                weights_projection_, p.dic, 0.0, diff_hidden_state_, p.dhc);

        for_(int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dhc; ih++)
            diff_hidden_state(ib, ih) += diff_dst_layer(ib, ih);
    } else {
        for_(int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dhc; ih++)
            diff_hidden_state(ib, ih)
                    = diff_dst_layer(ib, ih) + diff_dst_iter(ib, ih);
    }

    lstm_bwd_pregemm(p, src_iter_c_, dst_iter_c_, weights_peephole_,
            diff_hidden_state_, diff_dst_iter_c_, gates_, diff_src_iter_c_,
            b_gates_);

    gemm("C", "T", "N", p.sic, p.n_gates() * p.dhc, p.mb, 1.0, src_iter_, p.wc,
            b_gates_, p.n_gates() * p.dhc, 1.0, diff_weights_iter_,
            p.n_gates() * p.dhc);
    gemm("C", "T", "N", p.slc, p.n_gates() * p.dhc, p.mb, 1.0, src_layer_, p.wc,
            b_gates_, p.n_gates() * p.dhc, 1.0, diff_weights_layer_,
            p.n_gates() * p.dhc);

    gemm("C", "N", "T", p.mb, p.sic, p.n_gates() * p.dhc, 1.0, b_gates_,
            p.n_gates() * p.dhc, weights_iter_, p.n_gates() * p.dhc, 0.0,
            diff_src_iter_, p.wc);
    gemm("C", "N", "T", p.mb, p.slc, p.n_gates() * p.dhc, 1.0, b_gates_,
            p.n_gates() * p.dhc, weights_layer_, p.n_gates() * p.dhc, 0.0,
            diff_src_layer_, p.wc);

    if (p.is_lstm_peephole())
        lstm_bwd_weights_peephole(
                p, src_iter_c_, dst_iter_c_, b_gates_, diff_weights_peephole_);

    gates_reduction(p, b_gates_, diff_bias_);
}

} // namespace rnn
