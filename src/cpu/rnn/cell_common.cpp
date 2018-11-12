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
 * Common for RNN and LSTM cell execution
 */
#include "ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::forward>::cell_execution) {
    if (!merge_gemm_layer) {
        (this->*gemm_input_func)(n_gates * dic, batch, slc, conf_.WL_GLD(), slc,
                batch, wic, conf_.GC(), batch, w_input_[0], states_t_lm1_,
                ws_gates_, false, 0.0f);
    }
    (this->*gemm_state_func)(n_gates * dic, batch, sic, conf_.WI_GLD(), sic,
            batch, wic, conf_.GC(), batch, w_state_[0], states_tm1_l_,
            ws_gates_, false, 1.0f);
    (this->*elemwise_func)(dic, wic, batch, n_states, iter_stride, n_gates,
            ws_gates_, states_t_l_, states_t_lm1_, states_tm1_l_,
            diff_states_t_l_, diff_states_t_lp1_, diff_states_tp1_l_, bias_,
            ws_grid_, ws_cell_);
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::backward>::cell_execution) {
    (this->*elemwise_func)(dic, wic, batch, n_states, iter_stride, n_gates,
            ws_gates_, states_t_l_, states_t_lm1_, states_tm1_l_,
            diff_states_t_l_, diff_states_t_lp1_, diff_states_tp1_l_, bias_,
            ws_grid_, ws_cell_);

    /// bwd by data on the cell
    (this->*gemm_state_func)(sic, batch, n_gates * dic, conf_.WI_GLD(),
            n_gates * dic, batch, conf_.GC(), wic, batch, w_state_[0],
            ws_gates_, diff_states_t_l_, false, 0.0f);

    if (!merge_gemm_layer) {
        (this->*gemm_input_func)(slc, batch, n_gates * dic, conf_.WL_GLD(),
                n_gates * dic, batch, conf_.GC(), wic, batch, w_input_[0],
                ws_gates_,
                diff_states_t_l_ + n_states * iter_stride * (batch * wic),
                false, 0.0f);

        /// bwd by weights on the cell
        gemm(n_gates * dic, slc, batch, conf_.GC(), batch, wic, batch,
                conf_.DWL_GLD(), slc, ws_gates_, states_t_lm1_, diff_w_input_,
                true, 1.0f);
    }

    if (!merge_gemm_iter)
        gemm(n_gates * dic, sic, batch, conf_.GC(), batch, wic, batch,
                conf_.DWI_GLD(), sic, ws_gates_, states_tm1_l_, diff_w_state_,
                true, 1.0f);
    /// bwd by bias we just accumulate diffs from the gates
    gates_reduction(n_gates, dic, wic, batch, ws_gates_, diff_bias_);
}

}
}
}
