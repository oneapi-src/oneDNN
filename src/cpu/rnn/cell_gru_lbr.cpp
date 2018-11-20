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
 * Cell execution GRU with linear before reset
 */

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
#define AOC array_offset_calculator

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::forward>::gru_lbr_elemwise) {
    AOC<float, 2> ws_gates(ws_gates_, rnn.gates_nld, rnn.gates_ws_ld);
    AOC<float, 2> ws_Wh_b(ws_grid_, rnn.mb, rnn.dic);
    AOC<const float, 2> bias(bias_, rnn.n_bias, rnn.dic);
    AOC<float, 2> states_t_l(states_t_l_, rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 2> states_tm1_l(states_tm1_l_, rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 3> ws_gemm_state(ws_cell_, rnn.gates_nld, rnn.gates_ws_ld);
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float Wh_b = ws_gemm_state(i, 2 * rnn.dic + j) + bias(3, j);
            ws_gates(i, 0 * rnn.dic + j)
                    = logistic_fwd(ws_gates(i, 0 * rnn.dic + j)
                            + ws_gemm_state(i, j) + bias(0, j));
            ws_gates(i, 1 * rnn.dic + j)
                    = logistic_fwd(ws_gates(i, 1 * rnn.dic + j)
                            + ws_gemm_state(i, rnn.dic + j) + bias(1, j));
            ws_gates(i, 2 * rnn.dic + j) = tanh_fwd(ws_gates(i, 2 * rnn.dic + j)
                    + ws_gates(i, 1 * rnn.dic + j) * Wh_b + bias(2, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 0 * rnn.dic + j)
                    + (1.0f - ws_gates(i, 0 * rnn.dic + j))
                            * ws_gates(i, 2 * rnn.dic + j);
            if (rnn.is_training)
                ws_Wh_b(i, j) = Wh_b;
        }
    });
}

template <>
cell_execution_sig(
        _ref_rnn_common_t<prop_kind::forward>::cell_execution_gru_lbr) {
    if (!rnn.merge_gemm_layer) {
        (this->*gemm_input_func)('N', 'N', rnn.n_gates * rnn.dic, rnn.mb,
                rnn.slc, 1.0, w_input_[0], rnn.weights_layer_ws_ld, states_t_lm1_,
                rnn.states_ws_ld, 0.0, ws_gates_, rnn.gates_ws_ld);
    }
    (this->*gemm_state_func)('N', 'N', rnn.n_gates * rnn.dic, rnn.mb, rnn.sic,
            1.0, w_state_[0], rnn.weights_iter_ws_ld, states_tm1_l_,
            rnn.states_ws_ld, 0.0, ws_cell_, rnn.gates_ws_ld);
    (this->*elemwise_func)(rnn, iter_stride, ws_gates_, states_t_l_,
            states_t_lm1_, states_tm1_l_, diff_states_t_l_, diff_states_t_lp1_,
            diff_states_tp1_l_, bias_, ws_grid_, ws_cell_);
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::backward>::gru_lbr_elemwise) {
    AOC<float, 2> ws_gates(ws_gates_, rnn.gates_nld, rnn.gates_ws_ld);
    AOC<const float, 2> states_tm1_l(
            states_tm1_l_, rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 4> diff_states_t_l(diff_states_t_l_, rnn.n_states + 1,
            iter_stride, rnn.states_nld, rnn.states_ws_ld); // dht-1 dxt
    AOC<float, 4> diff_states_tp1_l(diff_states_tp1_l_, rnn.n_states + 1,
            iter_stride, rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 4> diff_states_t_lp1(diff_states_t_lp1_, rnn.n_states + 1,
            iter_stride, rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 3> ws_gates_r(ws_cell_, rnn.gates_nld, rnn.gates_ws_ld);
    AOC<float, 2> ws_Wh_b(ws_grid_, rnn.mb, rnn.dic);

    // 1. calculate dG1 dG2 dG3
    // dG0 = (dht - G2) * dht * (1 - G0) * G0
    // dG1 = (W*h + b) * dG2 * (1 - G1) * G1
    // dG2 = (1 - G0) * dht * (1 - G2*G2)
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float h = states_tm1_l(i, j);
            float dHt = diff_states_tp1_l(0, 0, i, j)
                    + diff_states_t_lp1(rnn.n_states, 0, i, j);
            float dG0 = (h - ws_gates(i, 2 * rnn.dic + j)) * dHt
                    * x_m_square(ws_gates(i, 0 * rnn.dic + j));
            float dG2 = (1.0f - ws_gates(i, 0 * rnn.dic + j))
                    * one_m_square(ws_gates(i, 2 * rnn.dic + j)) * dHt;
            float dG1 = ws_Wh_b(i, j) * dG2
                    * x_m_square(ws_gates(i, 1 * rnn.dic + j));

            diff_states_t_l(0, 0, i, j) = dHt * ws_gates(i, 0 * rnn.dic + j);
            ws_gates(i, 2 * rnn.dic + j) = dG2;
            ws_gates_r(i, 2 * rnn.dic + j) = dG2 * ws_gates(i, 1 * rnn.dic + j);
            ws_gates(i, 0 * rnn.dic + j) = ws_gates_r(i, 0 * rnn.dic + j) = dG0;
            ws_gates(i, 1 * rnn.dic + j) = ws_gates_r(i, 1 * rnn.dic + j) = dG1;
        }
    });
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::backward>::cell_execution_gru_lbr) {
    AOC<float, 2> diff_bias(diff_bias_, rnn.n_bias, rnn.dic);
    AOC<float, 3> ws_gates_r(ws_cell_, rnn.gates_nld, rnn.gates_ws_ld);

    (this->*elemwise_func)(rnn, iter_stride, ws_gates_, states_t_l_,
            states_t_lm1_, states_tm1_l_, diff_states_t_l_, diff_states_t_lp1_,
            diff_states_tp1_l_, bias_, ws_grid_, ws_cell_);

    if (!rnn.merge_gemm_layer) {
        //  dx = dG * Wx^t
        (this->*gemm_input_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dic, 1.0, w_input_[0], rnn.weights_layer_ws_ld,
                ws_gates_, rnn.gates_ws_ld, 0.0, diff_states_t_l_
                        + rnn.n_states * iter_stride
                                * (rnn.mb * rnn.states_ws_ld),
                rnn.states_ws_ld);
        // dWx +=  dG^t * x
        gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.slc, rnn.mb, 1.0, ws_gates_,
                rnn.gates_ws_ld, states_t_lm1_, rnn.states_ws_ld, 1.0,
                diff_w_input_, rnn.diff_weights_layer_ws_ld);
    }
    // dh +=  dGr * Wh^t
    (this->*gemm_state_func)('N', 'N', rnn.sic, rnn.mb, rnn.n_gates * rnn.dic,
            1.0, w_state_[0], rnn.weights_iter_ws_ld, ws_cell_, rnn.gates_ws_ld,
            1.0, diff_states_t_l_, rnn.states_ws_ld);

    // dWh += dGr^t * h
    gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.sic, rnn.mb, 1.0, ws_cell_,
            rnn.gates_ws_ld, states_tm1_l_, rnn.states_ws_ld, 1.0,
            diff_w_state_, rnn.diff_weights_layer_ws_ld);

    // db1-3 += e * dG
    // db4 += e * (r * dG2)
    gates_reduction(rnn, ws_gates_, diff_bias_);

    parallel_nd(rnn.dic, [&](int j) {
        for (int i = 0; i < rnn.mb; i++) {
            diff_bias_[3 * rnn.dic + j] += ws_gates_r(i, 2 * rnn.dic + j);
        }
    });
}

#undef AOC

}
}
}
