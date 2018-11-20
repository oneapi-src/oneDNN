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

#include "ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
#define AOC array_offset_calculator

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::forward>::lstm_elemwise) {
    AOC<float, 2> ws_gates(ws_gates_, rnn.gates_nld, rnn.gates_ws_ld);
    AOC<const float, 2> bias(bias_, rnn.n_bias, rnn.dic);
    AOC<float, 4> states_t_l(states_t_l_, rnn.n_states, iter_stride,
            rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 4> states_tm1_l(states_tm1_l_, rnn.n_states, iter_stride,
            rnn.states_nld, rnn.states_ws_ld);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            ws_gates(i, 0 * rnn.dic + j)
                    = logistic_fwd(ws_gates(i, 0 * rnn.dic + j) + bias(0, j));
            ws_gates(i, 1 * rnn.dic + j)
                    = logistic_fwd(ws_gates(i, 1 * rnn.dic + j) + bias(1, j));
            ws_gates(i, 2 * rnn.dic + j)
                    = tanh_fwd(ws_gates(i, 2 * rnn.dic + j) + bias(2, j));
            ws_gates(i, 3 * rnn.dic + j)
                    = logistic_fwd(ws_gates(i, 3 * rnn.dic + j) + bias(3, j));

            float tmp = ws_gates(i, 1 * rnn.dic + j) * states_tm1_l(1, 0, i, j)
                    + ws_gates(i, 0 * rnn.dic + j)
                            * ws_gates(i, 2 * rnn.dic + j);
            states_t_l(0, 0, i, j)
                    = ws_gates(i, 3 * rnn.dic + j) * tanh_fwd(tmp);
            states_t_l(1, 0, i, j) = tmp;
        }
    });
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::backward>::lstm_elemwise) {
    AOC<float, 2> ws_gates(ws_gates_, rnn.gates_nld, rnn.gates_ws_ld);
    AOC<const float, 2> bias(bias_, rnn.n_gates, rnn.dic);
    AOC<float, 4> states_t_l(states_t_l_, rnn.n_states, iter_stride,
            rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 4> states_tm1_l(states_tm1_l_, rnn.n_states, iter_stride,
            rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 4> diff_states_t_l(diff_states_t_l_, rnn.n_states + 1,
            iter_stride, rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 4> diff_states_tp1_l(diff_states_tp1_l_, rnn.n_states + 1,
            iter_stride, rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 4> diff_states_t_lp1(diff_states_t_lp1_, rnn.n_states + 1,
            iter_stride, rnn.states_nld, rnn.states_ws_ld);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float Ct = states_t_l(1, 0, i, j);
            /// @todo save it in the workspace in fwd pass or recompute it to
            /// save bw
            float tanhCt = tanh_fwd(Ct);
            // we have 2 incoming diffs on Ht
            float dHt = diff_states_tp1_l(0, 0, i, j)
                    + diff_states_t_lp1(rnn.n_states, 0, i, j);
            float dCt = diff_states_tp1_l(1, 0, i, j)
                    + one_m_square(tanhCt) * ws_gates(i, 3 * rnn.dic + j) * dHt;

            float dG1 = states_tm1_l(1, 0, i, j) * dCt
                    * x_m_square(ws_gates(i, 1 * rnn.dic + j));
            float dG0 = ws_gates(i, 2 * rnn.dic + j) * dCt
                    * x_m_square(ws_gates(i, 0 * rnn.dic + j));
            float dG3 = tanhCt * dHt * x_m_square(ws_gates(i, 3 * rnn.dic + j));
            float dG2 = ws_gates(i, 0 * rnn.dic + j) * dCt
                    * one_m_square(ws_gates(i, 2 * rnn.dic + j));

            diff_states_t_l(1, 0, i, j) = dCt * ws_gates(i, 1 * rnn.dic + j);

            ws_gates(i, 0 * rnn.dic + j) = dG0;
            ws_gates(i, 1 * rnn.dic + j) = dG1;
            ws_gates(i, 2 * rnn.dic + j) = dG2;
            ws_gates(i, 3 * rnn.dic + j) = dG3;
        }
    });
}
#undef AOC

}
}
}
