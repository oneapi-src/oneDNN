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
    bool is_training = conf_.is_training();
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<float, 2> ws_Wh_b(ws_grid_, batch, dic);
    AOC<const float, 2> bias(bias_, n_gates + 1, dic);
    AOC<float, 2> states_t_l(states_t_l_, batch, wic);
    AOC<float, 2> states_tm1_l(states_tm1_l_, batch, wic);
    AOC<float, 3> ws_gemm_state(ws_cell_, batch, conf_.GC());
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            float Wh_b = ws_gemm_state(i, 2 * dic + j) + bias(3, j);
            ws_gates(i, 0 * dic + j) = logistic_fwd(ws_gates(i, 0 * dic + j)
                    + ws_gemm_state(i, j) + bias(0, j));
            ws_gates(i, 1 * dic + j) = logistic_fwd(ws_gates(i, 1 * dic + j)
                    + ws_gemm_state(i, dic + j) + bias(1, j));
            ws_gates(i, 2 * dic + j) = tanh_fwd(ws_gates(i, 2 * dic + j)
                    + ws_gates(i, 1 * dic + j) * Wh_b + bias(2, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 0 * dic + j)
                    + (1.0f - ws_gates(i, 0 * dic + j))
                            * ws_gates(i, 2 * dic + j);
            if (is_training)
                ws_Wh_b(i, j) = Wh_b;
        }
    });
}

template <>
cell_execution_sig(
        _ref_rnn_common_t<prop_kind::forward>::cell_execution_gru_lbr) {
    if (!merge_gemm_layer) {
        (this->*gemm_input_func)(n_gates * dic, batch, slc, conf_.WL_GLD(), slc,
                batch, wic, conf_.GC(), batch, w_input_[0], states_t_lm1_,
                ws_gates_, false, 0.0f);
    }
    (this->*gemm_state_func)(n_gates * dic, batch, sic, conf_.WI_GLD(), sic,
            batch, wic, conf_.GC(), batch, w_state_[0], states_tm1_l_, ws_cell_,
            false, 0.0f);
    (this->*elemwise_func)(dic, wic, batch, n_states, iter_stride, n_gates,
            ws_gates_, states_t_l_, states_t_lm1_, states_tm1_l_,
            diff_states_t_l_, diff_states_t_lp1_, diff_states_tp1_l_, bias_,
            ws_grid_, ws_cell_);
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::backward>::gru_lbr_elemwise) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> states_tm1_l(states_tm1_l_, batch, wic);
    AOC<float, 4> diff_states_t_l(diff_states_t_l_, n_states + 1, iter_stride,
            batch, wic); // dht-1 dxt
    AOC<float, 4> diff_states_tp1_l(
            diff_states_tp1_l_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 4> diff_states_t_lp1(
            diff_states_t_lp1_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 3> ws_gates_r(ws_cell_, batch, conf_.GC());
    AOC<float, 2> ws_Wh_b(ws_grid_, batch, dic);

    // 1. calculate dG1 dG2 dG3
    // dG0 = (dht - G2) * dht * (1 - G0) * G0
    // dG1 = (W*h + b) * dG2 * (1 - G1) * G1
    // dG2 = (1 - G0) * dht * (1 - G2*G2)
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            float h = states_tm1_l(i, j);
            float dHt = diff_states_tp1_l(0, 0, i, j)
                    + diff_states_t_lp1(n_states, 0, i, j);
            float dG0 = (h - ws_gates(i, 2 * dic + j)) * dHt
                    * x_m_square(ws_gates(i, 0 * dic + j));
            float dG2 = (1.0f - ws_gates(i, 0 * dic + j))
                    * one_m_square(ws_gates(i, 2 * dic + j)) * dHt;
            float dG1 = ws_Wh_b(i, j) * dG2
                    * x_m_square(ws_gates(i, 1 * dic + j));

            diff_states_t_l(0, 0, i, j) = dHt * ws_gates(i, 0 * dic + j);
            ws_gates(i, 2 * dic + j) = dG2;
            ws_gates_r(i, 2 * dic + j) = dG2 * ws_gates(i, 1 * dic + j);
            ws_gates(i, 0 * dic + j) = ws_gates_r(i, 0 * dic + j) = dG0;
            ws_gates(i, 1 * dic + j) = ws_gates_r(i, 1 * dic + j) = dG1;
        }
    });
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::backward>::cell_execution_gru_lbr) {
    AOC<float, 2> diff_bias(diff_bias_, n_gates + 1, dic);
    AOC<float, 3> ws_gates_r(ws_cell_, batch, conf_.GC());

    (this->*elemwise_func)(dic, wic, batch, n_states, iter_stride, n_gates,
            ws_gates_, states_t_l_, states_t_lm1_, states_tm1_l_,
            diff_states_t_l_, diff_states_t_lp1_, diff_states_tp1_l_, bias_,
            ws_grid_, ws_cell_);

    if (!merge_gemm_layer) {
        //  dx = dG * Wx^t
        (this->*gemm_input_func)(slc, batch, n_gates * dic, conf_.WL_GLD(),
                n_gates * dic, batch, conf_.GC(), wic, batch, w_input_[0],
                ws_gates_,
                diff_states_t_l_ + n_states * iter_stride * (batch * wic),
                false, 0.0f);
        // dWx +=  dG^t * x
        gemm(n_gates * dic, slc, batch, conf_.GC(), batch, wic, batch,
                conf_.DWL_GLD(), slc, ws_gates_, states_t_lm1_, diff_w_input_,
                true, 1.0f);
    }
    // dh +=  dGr * Wh^t
    (this->*gemm_state_func)(sic, batch, n_gates * dic, conf_.WI_GLD(),
            n_gates * dic, batch, conf_.GC(), wic, batch, w_state_[0], ws_cell_,
            diff_states_t_l_, false, 1.0f);

    // dWh += dGr^t * h
    gemm(n_gates * dic, sic, batch, conf_.GC(), batch, wic, batch,
            conf_.DWL_GLD(), sic, ws_cell_, states_tm1_l_, diff_w_state_, true,
            1.0f);

    // db1-3 += e * dG
    // db4 += e * (r * dG2)
    gates_reduction(n_gates, dic, wic, batch, ws_gates_, diff_bias_);

    parallel_nd(dic, [&](int j) {
        for (int i = 0; i < batch; i++) {
            diff_bias_[3 * dic + j] += ws_gates_r(i, 2 * dic + j);
        }
    });
}

#undef AOC

}
}
}
