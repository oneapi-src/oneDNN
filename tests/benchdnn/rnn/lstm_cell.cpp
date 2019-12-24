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
void lstm_fwd_postgemm_template(T1 func1, T2 func2, const prb_t &p,
        float *gates_, const float *bias_, const float *src_iter_c_,
        float *dst_iter_h_, float *dst_iter_c_) {
    AOC<float> gates(gates_, p.mb, p.n_gates(), p.dic);
    AOC<const float> bias(bias_, p.n_gates(), p.dic);
    AOC<const float> src_iter_c(src_iter_c_, p.mb, p.wc);
    AOC<float> h_dst(dst_iter_h_, p.mb, p.wc);
    AOC<float> dst_iter_c(dst_iter_c_, p.mb, p.wc);

    auto maybe_deq_w = [&](float g, int64_t oc) {
        if (!is_cfg_u8(p.cfg)) return g;
        float scale = 1.;
        if (p.scale_policy == policy_t::PER_OC)
            scale = p.wei_oc_scales[oc];
        else if (p.scale_policy == policy_t::COMMON)
            scale = p.wei_scale;
        scale *= p.data_scale;
        return g * (1.f / scale);
    };

    auto maybe_q_d = [&](float h) {
        if (!is_cfg_u8(p.cfg)) return h;
        float fp = p.data_scale * h + p.data_shift;
        if (fp > p.cfg[input].max) fp = p.cfg[input].max;
        if (fp < p.cfg[input].min) fp = p.cfg[input].min;
        fp = mxcsr_round(fp);
        return fp;
    };

    // run the eltwise
    dnnl::impl::parallel_nd(p.mb, [&](int64_t ib) {
        for (int64_t ih = 0; ih < p.dic; ih++) {
            gates(ib, LSTM_I, ih) = func1(p.linear_scales[LSTM_I],
                    maybe_deq_w(gates(ib, LSTM_I, ih), LSTM_I * p.dic + ih)
                            + bias(LSTM_I, ih));
            gates(ib, LSTM_F, ih) = func1(p.linear_scales[LSTM_F],
                    maybe_deq_w(gates(ib, LSTM_F, ih), LSTM_F * p.dic + ih)
                            + bias(LSTM_F, ih));
            gates(ib, LSTM_C, ih) = func2(p.linear_scales[LSTM_C],
                    maybe_deq_w(gates(ib, LSTM_C, ih), LSTM_C * p.dic + ih)
                            + bias(LSTM_C, ih));
            gates(ib, LSTM_O, ih) = func1(p.linear_scales[LSTM_O],
                    maybe_deq_w(gates(ib, LSTM_O, ih), LSTM_O * p.dic + ih)
                            + bias(LSTM_O, ih));
            for (int64_t ig = 0; ig < 4; ig++) {
                BENCHDNN_PRINT(80,
                        "activation 1 a[" IFMT "][" IFMT "][" IFMT "] = %.7f\n",
                        ib, ig, ih, gates(ib, ig, ih));
            }

            // compute C_t_l and H_t_l
            float tmp = gates(ib, LSTM_F, ih) * src_iter_c(ib, ih)
                    + gates(ib, LSTM_I, ih) * gates(ib, LSTM_C, ih);
            dst_iter_c(ib, ih) = tmp;
            h_dst(ib, ih) = maybe_q_d(
                    gates(ib, LSTM_O, ih) * func2(p.linear_cscale, tmp));
            BENCHDNN_PRINT(80, "recomp tmp(%a) cin(%a) ht(%a)\n", tmp,
                    src_iter_c(ib, ih), h_dst(ib, ih));
        }
    });
}

void lstm_fwd_postgemm(const prb_t &p, float *gates_, const float *bias_,
        const float *src_iter_c_, float *dst_iter_h_, float *dst_iter_c_) {
    if (p.skip_nonlinear)
        lstm_fwd_postgemm_template(
                [](float scale, float val) { return scale * val; },
                [](float scale, float val) { return scale * val; }, p, gates_,
                bias_, src_iter_c_, dst_iter_h_, dst_iter_c_);
    else
        lstm_fwd_postgemm_template(
                [](float scale, float val) { return logistic(val); },
                [](float scale, float val) { return tanhf(val); }, p, gates_,
                bias_, src_iter_c_, dst_iter_h_, dst_iter_c_);
}

void lstm_fwd(const prb_t &p, float *dst_iter_h_, float *dst_iter_c_,
        float *gates_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *src_layer_, const float *src_iter_h_,
        const float *src_iter_c_) {

    gemm("C", "N", "N", p.mb, p.n_gates() * p.dic, p.slc, 1.0, src_layer_, p.wc,
            weights_layer_, p.n_gates() * p.dic, 0.0, gates_,
            p.n_gates() * p.dic);
    gemm("C", "N", "N", p.mb, p.n_gates() * p.dic, p.sic, 1.0, src_iter_h_,
            p.wc, weights_iter_h_, p.n_gates() * p.dic, 1.0, gates_,
            p.n_gates() * p.dic);

    lstm_fwd_postgemm(p, gates_, bias_, src_iter_c_, dst_iter_h_, dst_iter_c_);
}

template <typename T1>
void lstm_bwd_pregemm_template(T1 func1, const prb_t &p,
        const float *src_iter_c_, const float *dst_iter_c_,
        const float *diff_dst_layer_, const float *diff_dst_iter_h_,
        const float *diff_dst_iter_c_, const float *gates_,
        float *diff_src_iter_c_, float *b_gates_) {
    AOC<const float> src_iter_c(src_iter_c_, p.mb, p.wc);
    AOC<const float> dst_iter_c(dst_iter_c_, p.mb, p.wc);
    AOC<const float> diff_dst_layer(diff_dst_layer_, p.mb, p.wc);
    AOC<const float> diff_dst_iter_h(diff_dst_iter_h_, p.mb, p.wc);
    AOC<const float> diff_dst_iter_c(diff_dst_iter_c_, p.mb, p.wc);
    AOC<const float> gates(gates_, p.mb, p.n_gates(), p.dic);
    AOC<float> diff_src_iter_c(diff_src_iter_c_, p.mb, p.wc);
    AOC<float> b_gates(b_gates_, p.mb, p.n_gates(), p.dic);

    for (int64_t ib = 0; ib < p.mb; ib++)
        for (int64_t ih = 0; ih < p.dic; ih++) {
            BENCHDNN_PRINT(80, "rnn_single_bwd: ib = " IFMT " ih = " IFMT "\n",
                    ib, ih);
            float ho = gates(ib, LSTM_O, ih);
            float hf = gates(ib, LSTM_F, ih);
            float hc = gates(ib, LSTM_C, ih);
            float hi = gates(ib, LSTM_I, ih);
            float dh = diff_dst_layer(ib, ih) + diff_dst_iter_h(ib, ih);
            float c = dst_iter_c(ib, ih);
            float tanhC = func1(p.linear_cscale, c);
            float dho = tanhC * dh;
            b_gates(ib, LSTM_O, ih) = x_m_square(ho) * dho;

            float dc_next = diff_dst_iter_c(ib, ih);
            float dc = ho * dh * one_m_square(tanhC) + dc_next;
            diff_src_iter_c(ib, ih) = hf * dc;

            float c_old = src_iter_c(ib, ih);
            float dhf = c_old * dc;
            b_gates(ib, LSTM_F, ih) = x_m_square(hf) * dhf;

            float dhi = hc * dc;
            b_gates(ib, LSTM_I, ih) = x_m_square(hi) * dhi;

            float dhc = hi * dc;
            b_gates(ib, LSTM_C, ih) = one_m_square(hc) * dhc;
        }
}

void lstm_bwd_pregemm(const prb_t &p, const float *src_iter_c_,
        const float *dst_iter_c_, const float *diff_dst_layer_,
        const float *diff_dst_iter_h_, const float *diff_dst_iter_c_,
        const float *gates_, float *diff_src_iter_c_, float *b_gates_) {
    if (p.skip_nonlinear)
        lstm_bwd_pregemm_template(
                [](float scale, float val) { return scale * val; }, p,
                src_iter_c_, dst_iter_c_, diff_dst_layer_, diff_dst_iter_h_,
                diff_dst_iter_c_, gates_, diff_src_iter_c_, b_gates_);

    else
        lstm_bwd_pregemm_template(
                [](float scale, float val) { return tanhf(val); }, p,
                src_iter_c_, dst_iter_c_, diff_dst_layer_, diff_dst_iter_h_,
                diff_dst_iter_c_, gates_, diff_src_iter_c_, b_gates_);
}

void lstm_bwd(const prb_t &p, float *diff_src_layer_, float *diff_src_iter_h_,
        float *diff_src_iter_c_, float *diff_weights_layer_,
        float *diff_weights_iter_h_, float *diff_bias_, float *b_gates_,
        const float *src_layer_, const float *src_iter_h_,
        const float *src_iter_c_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *dst_iter_h_, const float *dst_iter_c_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_h_,
        const float *diff_dst_iter_c_) {
    lstm_bwd_pregemm(p, src_iter_c_, dst_iter_c_, diff_dst_layer_,
            diff_dst_iter_h_, diff_dst_iter_c_, gates_, diff_src_iter_c_,
            b_gates_);

    gemm("C", "T", "N", p.sic, p.n_gates() * p.dic, p.mb, 1.0, src_iter_h_,
            p.wc, b_gates_, p.n_gates() * p.dic, 1.0, diff_weights_iter_h_,
            p.n_gates() * p.dic);
    gemm("C", "T", "N", p.slc, p.n_gates() * p.dic, p.mb, 1.0, src_layer_, p.wc,
            b_gates_, p.n_gates() * p.dic, 1.0, diff_weights_layer_,
            p.n_gates() * p.dic);

    gemm("C", "N", "T", p.mb, p.sic, p.n_gates() * p.dic, 1.0, b_gates_,
            p.n_gates() * p.dic, weights_iter_h_, p.n_gates() * p.dic, 0.0,
            diff_src_iter_h_, p.wc);
    gemm("C", "N", "T", p.mb, p.slc, p.n_gates() * p.dic, 1.0, b_gates_,
            p.n_gates() * p.dic, weights_layer_, p.n_gates() * p.dic, 0.0,
            diff_src_layer_, p.wc);

    gates_reduction(p, b_gates_, diff_bias_);
}

} // namespace rnn
