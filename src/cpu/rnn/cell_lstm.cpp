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
using namespace rnn_utils;

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::forward>::lstm_elemwise) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            ws_gates(i, 0, j) = logistic_fwd(ws_gates(i, 0, j) + bias(0, j));
            ws_gates(i, 1, j) = logistic_fwd(ws_gates(i, 1, j) + bias(1, j));
            ws_gates(i, 2, j) = tanh_fwd(ws_gates(i, 2, j) + bias(2, j));
            ws_gates(i, 3, j) = logistic_fwd(ws_gates(i, 3, j) + bias(3, j));

            float tmp = ws_gates(i, 1, j) * states_tm1_l(1, i, j)
                    + ws_gates(i, 0, j) * ws_gates(i, 2, j);
            states_t_l(0, i, j) = ws_gates(i, 3, j) * tanh_fwd(tmp);
            states_t_l(1, i, j) = tmp;
        }
    });
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::backward>::lstm_elemwise) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);
    ws_diff_states_aoc_t diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc_t diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc_t diff_states_t_lp1(rnn, diff_states_t_lp1_);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float Ct = states_t_l(1, i, j);
            /// @todo save it in the workspace in fwd pass or recompute it to
            /// save bw
            float tanhCt = tanh_fwd(Ct);
            // we have 2 incoming diffs on Ht
            float dHt = diff_states_tp1_l(0, i, j)
                    + diff_states_t_lp1(rnn.n_states, i, j);
            float dCt = diff_states_tp1_l(1, i, j)
                    + one_m_square(tanhCt) * ws_gates(i, 3, j) * dHt;

            float dG1 = states_tm1_l(1, i, j) * dCt
                    * x_m_square(ws_gates(i, 1, j));
            float dG0 = ws_gates(i, 2, j) * dCt * x_m_square(ws_gates(i, 0, j));
            float dG3 = tanhCt * dHt * x_m_square(ws_gates(i, 3, j));
            float dG2
                    = ws_gates(i, 0, j) * dCt * one_m_square(ws_gates(i, 2, j));

            diff_states_t_l(1, i, j) = dCt * ws_gates(i, 1, j);

            ws_gates(i, 0, j) = dG0;
            ws_gates(i, 1, j) = dG1;
            ws_gates(i, 2, j) = dG2;
            ws_gates(i, 3, j) = dG3;
        }
    });
}

}
}
}
