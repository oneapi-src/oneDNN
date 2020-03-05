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
#include "rnn/rnn_cells.hpp"

namespace rnn {

float activation(const prb_t &p, float x, bool is_fwd = true) {
    float result = 0.0f;
    if (p.skip_nonlinear)
        result = p.linear_scales[0] * x;
    else
        switch (p.activation) {
            case RELU:
                result = is_fwd ? relu(x, p.alpha) : drelu(x, p.alpha);
                break;
            case LOGISTIC: result = is_fwd ? logistic(x) : x_m_square(x); break;
            case TANH: result = is_fwd ? tanhf(x) : one_m_square(x); break;
            default: assert(!"unknown activation");
        }
    return result;
}

void rnn_fwd_postgemm(
        const prb_t &p, const float *bias_, float *gates_, float *dst_iter_h_) {
    AOC<float> dst_iter_h(dst_iter_h_, p.mb, p.n_gates(), p.wc);
    AOC<const float> bias(bias_, p.n_gates(), p.dhc);
    AOC<float> gates(gates_, p.mb, p.n_gates(), p.dhc);

    for (int64_t i = 0; i < p.mb; i++)
        for (int64_t j = 0; j < p.n_gates(); j++)
            for (int64_t k = 0; k < p.dhc; k++) {
                const auto tmp = activation(p, gates(i, j, k) + bias(j, k));
                gates(i, j, k) = tmp;
                dst_iter_h(i, j, k) = tmp;
            }
}

void rnn_fwd(const prb_t &p, float *dst_iter_h_, float *gates_,
        const float *weights_layer_, const float *weights_iter_h_,
        const float *bias_, const float *src_layer_, const float *src_iter_h_) {
    gemm("C", "N", "N", p.mb, p.n_gates() * p.dhc, p.slc, 1.0, src_layer_, p.wc,
            weights_layer_, p.n_gates() * p.dhc, 0.0, gates_,
            p.n_gates() * p.dhc);
    gemm("C", "N", "N", p.mb, p.n_gates() * p.dhc, p.sic, 1.0, src_iter_h_,
            p.wc, weights_iter_h_, p.n_gates() * p.dhc, 1.0, gates_,
            p.n_gates() * p.dhc);
    rnn_fwd_postgemm(p, bias_, gates_, dst_iter_h_);
}

void rnn_bwd_pregemm(const prb_t &p, const float *diff_dst_layer_,
        const float *diff_dst_iter_h_, const float *gates_, float *b_gates_) {
    AOC<const float> diff_dst_layer(diff_dst_layer_, p.mb, p.wc);
    AOC<const float> diff_dst_iter_h(diff_dst_iter_h_, p.mb, p.wc);
    AOC<const float> gates(gates_, p.mb, p.n_gates(), p.dhc);
    AOC<float> b_gates(b_gates_, p.mb, p.n_gates(), p.dhc);

    for (int64_t b = 0; b < p.mb; ++b)
        for (int64_t h = 0; h < p.dhc; ++h) {
            const float g = gates(b, 0, h);
            const float dd = diff_dst_layer(b, h) + diff_dst_iter_h(b, h);
            b_gates(b, 0, h) = activation(p, g, false) * dd;
        }
}

void rnn_bwd(const prb_t &p, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_h_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *dst_iter_h_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_h_) {
    AOC<float> b_gates(b_gates_, p.mb, p.n_gates(), p.dhc);

    rnn_bwd_pregemm(p, diff_dst_layer_, diff_dst_iter_h_, gates_, b_gates_);

    gemm("C", "T", "N", p.sic, p.n_gates() * p.dhc, p.mb, 1.0, src_iter_, p.wc,
            b_gates_, p.n_gates() * p.dhc, 1.0, diff_weights_iter_h_,
            p.n_gates() * p.dhc);
    gemm("C", "T", "N", p.slc, p.n_gates() * p.dhc, p.mb, 1.0, src_layer_, p.wc,
            b_gates_, p.n_gates() * p.dhc, 1.0, diff_weights_layer_,
            p.n_gates() * p.dhc);
    for (int64_t b = 0; b < p.mb; ++b)
        copy(p.n_gates(), p.dhc, p.dhc, p.dhc, &b_gates(b, 0, 0), diff_bias_,
                action_sum);

    gemm("C", "N", "T", p.mb, p.slc, p.n_gates() * p.dhc, 1.0, b_gates_,
            p.n_gates() * p.dhc, weights_layer_, p.n_gates() * p.dhc, 0.0,
            diff_src_layer_, p.wc);
    gemm("C", "N", "T", p.mb, p.sic, p.n_gates() * p.dhc, 1.0, b_gates_,
            p.n_gates() * p.dhc, weights_iter_h_, p.n_gates() * p.dhc, 0.0,
            diff_src_iter_, p.wc);
}

} // namespace rnn
