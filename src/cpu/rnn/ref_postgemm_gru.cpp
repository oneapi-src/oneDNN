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

/*
 * Cell execution LSTM
 */

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "jit_uni_rnn_common_postgemm_dispatcher.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
using namespace rnn_utils;
#define AOC array_offset_calculator

template <typename T1, typename acc_data_t, typename src_data_t>
void gru_fwd_part1_postgemm_template(T1 func1, const float *scales,
        const rnn_utils::rnn_conf_t &rnn, acc_data_t *ws_gates_,
        src_data_t *states_t_l_, src_data_t *states_tm1_l_, float *bias_) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            ws_gates(i, 0, j) // default func1 is sigmoid
                    = func1(scales, ws_gates(i, 0, j) + bias(0, j));
            ws_gates(i, 1, j) // default func1 is sigmoid
                    = func1(scales + 1, ws_gates(i, 1, j) + bias(1, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 1, j);
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::gru_part1_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part1_postgemm_template(logistic_f, scales, rnn, ws_gates_,
                states_t_l_, states_tm1_l_, bias_);
    else
        gru_fwd_part1_postgemm_template(linear_f, scales, rnn, ws_gates_,
                states_t_l_, states_tm1_l_, bias_);
}

template <typename T1, typename acc_data_t, typename src_data_t>
void gru_fwd_part2_postgemm_template(T1 func1, const float *scales,
        const rnn_utils::rnn_conf_t &rnn, acc_data_t *ws_gates_,
        src_data_t *states_t_l_, src_data_t *states_tm1_l_, float *bias_) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            ws_gates(i, 2, j) // default func1 is tanh
                    = func1(scales + 2, ws_gates(i, 2, j) + bias(2, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 0, j)
                    + (1.0f - ws_gates(i, 0, j)) * ws_gates(i, 2, j);
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::gru_part2_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part2_postgemm_template(tanh_f, scales, rnn, ws_gates_,
                states_t_l_, states_tm1_l_, bias_);
    else
        gru_fwd_part2_postgemm_template(linear_f, scales, rnn, ws_gates_,
                states_t_l_, states_tm1_l_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::gru_part1_postgemm) {
    assert(!"GRU int8 is not supported");
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::gru_part2_postgemm) {
    assert(!"GRU int8 is not supported");
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::gru_part1_postgemm) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);
    ws_diff_states_aoc_t diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc_t diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc_t diff_states_t_lp1(rnn, diff_states_t_lp1_);

    // dG2^ = dh * (1 - G0) * (1 - G2^2)
    // dG0^ = dh * (ht-1 - G2) * u * (1 - G0)
    // dht-1 (part) = dh * G0
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float h = states_tm1_l(i, j);
            float dHt = diff_states_tp1_l(0, i, j)
                    + diff_states_t_lp1(rnn.n_states, i, j);
            float dG2 = (1.0f - ws_gates(i, 0, j)) * dHt
                    * one_m_square(ws_gates(i, 2, j));
            float dG0 = (h - ws_gates(i, 2, j)) * dHt
                    * x_m_square(ws_gates(i, 0, j));

            diff_states_t_l(0, i, j) = dHt * ws_gates(i, 0, j);
            ws_gates(i, 0, j) = dG0;
            ws_gates(i, 2, j) = dG2;
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::gru_part2_postgemm) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);
    ws_diff_states_aoc_t diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc_t diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc_t diff_states_t_lp1(rnn, diff_states_t_lp1_);

    float *dhG1_ = &(diff_states_t_l(rnn.n_states, 0, 0));
    float *hG1_ = dhG1_;
    AOC<float, 2> dhG1(dhG1_, rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 2> hG1(hG1_, rnn.states_nld, rnn.states_ws_ld);

    // dG1^ = d(hG1) * h * G1 * (1 - G1)
    // dht-1 (part) += d(hG1) * G1
    // h * G1 (required for dWh)
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float h = states_tm1_l(i, j);
            float G1 = ws_gates(i, 1, j);
            diff_states_t_l(0, i, j) += dhG1(i, j) * G1;
            ws_gates(i, 1, j) = dhG1(i, j) * h * x_m_square(G1);
            hG1(i, j) = G1 * h;
        }
    });
}

#undef AOC
} // namespace cpu
} // namespace impl
} // namespace mkldnn
