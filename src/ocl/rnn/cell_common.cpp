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
/*
 * Common for RNN and LSTM cell execution
 */
#include "ocl/rnn/ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace mkldnn::impl::utils;
using namespace rnn_utils;

#define AOC array_offset_calculator

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
cell_execution_sig(
        (_ref_rnn_common_t<aprop, src_type, weights_type>::cell_execution)) {
    const rnn_conf_t &rnn_conf = this->pd()->rnn_conf_;

    if (aprop == prop_kind::forward) {

        AOC<size_t, 3> off_weights_i(
                weights_input, n_layer, n_dir, n_parts_weights_layer);
        AOC<size_t, 3> off_weights_st(
                weights_states, n_layer, n_dir, n_parts_weights_iter);

        cl_ulong offset_w_input = (cl_ulong)(off_weights_i(lay, dir, 0));
        cl_ulong offset_w_state = (cl_ulong)(off_weights_st(lay, dir, 0));

        cl_ulong offset_states = (cl_ulong)(ws_states_offset_
                + OFF4(lay + 1, n_layer + 1, dir, n_dir, iter, n_iter + 1, 0,
                        batch * rnn_conf.states_ws_ld));
        cl_ulong offset_input = (cl_ulong)(ws_states_offset_
                + OFF4(lay, n_layer + 1, dir, n_dir, iter + 1, n_iter + 1, 0,
                        batch * rnn_conf.states_ws_ld));
        cl_ulong offset_gates = (cl_ulong)(ws_gates_offset_
                + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                        batch * rnn_conf.gates_ws_ld));

        gemm_primitive(ctx, n_gates * dic, batch, slc, n_gates * dic, slc,
                batch, wic, n_gates * dic, batch, w_input, offset_w_input,
                workspace, offset_input, workspace, offset_gates, false, 0.0f,
                gemm_layer);
        gemm_primitive(ctx, n_gates * dic, batch, sic, n_gates * dic, sic,
                batch, wic, n_gates * dic, batch, w_state, offset_w_state,
                workspace, offset_states, workspace, offset_gates, false, 1.0f,
                gemm_iter);
        (this->*elemwise_func)(
                ctx, dir, lay, iter, dic, wic, batch, workspace, bias);

    } else { // backward

        AOC<size_t, 3> off_weights_i(
                weights_input, n_layer, n_dir, n_parts_weights_layer);
        AOC<size_t, 3> off_weights_st(
                weights_states, n_layer, n_dir, n_parts_weights_iter);

        (this->*elemwise_func)(
                ctx, dir, lay, iter, dic, wic, batch, workspace, bias);

        cl_ulong offset_w_state = (cl_ulong)(off_weights_st(lay, dir, 0));
        cl_ulong offset_w_input = (cl_ulong)(off_weights_i(lay, dir, 0));

        gemm_primitive(ctx, sic, batch, n_gates * dic, sic, n_gates * dic,
                batch, n_gates * dic, wic, batch, w_state, offset_w_state,
                workspace,
                ws_gates_offset_
                        + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                                batch * rnn_conf.gates_ws_ld),
                workspace,
                ws_diff_states_offset_
                        + OFF4(lay, n_layer + 1, dir, n_dir, iter, n_iter + 1,
                                0,
                                (n_states + 1) * batch * rnn_conf.states_ws_ld),
                false, 0.0f, gemm_iter);
        gemm_primitive(ctx, sic, batch, n_gates * dic, slc, n_gates * dic,
                batch, n_gates * dic, wic, batch, w_input, offset_w_input,
                workspace,
                ws_gates_offset_
                        + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                                batch * rnn_conf.gates_ws_ld),
                workspace,
                ws_diff_states_offset_
                        + OFF4(lay, n_layer + 1, dir, n_dir, iter, n_iter + 1,
                                0,
                                (n_states + 1) * batch * rnn_conf.states_ws_ld)
                        + n_states * batch * rnn_conf.states_ws_ld,
                false, 0.0f, gemm_layer);
        gemm_primitive(ctx, n_gates * dic, slc, batch, n_gates * dic, batch,
                wic, batch, n_gates * dic, slc, workspace,
                ws_gates_offset_
                        + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                                batch * rnn_conf.gates_ws_ld),
                workspace,
                ws_states_offset_
                        + OFF4(lay, n_layer + 1, dir, n_dir, iter + 1,
                                n_iter + 1, 0, batch * rnn_conf.states_ws_ld),
                diff_weights_layer,
                OFF3(lay, n_layer, dir, n_dir, 0,
                        rnn_conf.diff_weights_layer_nld
                                * rnn_conf.diff_weights_layer_ld),
                true, 1.0f, gemm_diff_wei_layer);
        gemm_primitive(ctx, n_gates * dic, sic, batch, n_gates * dic, batch,
                wic, batch, n_gates * dic, sic, workspace,
                ws_gates_offset_
                        + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                                batch * rnn_conf.gates_ws_ld),
                workspace,
                ws_states_offset_
                        + OFF4(lay + 1, n_layer + 1, dir, n_dir, iter,
                                n_iter + 1, 0, batch * rnn_conf.states_ws_ld),
                diff_weights_iter,
                OFF3(lay, n_layer, dir, n_dir, 0,
                        rnn_conf.diff_weights_iter_nld
                                * rnn_conf.diff_weights_iter_ld),
                true, 1.0f, gemm_diff_wei_iter);

        gates_reduction(
                ctx, dir, lay, iter, n_gates, dic, batch, workspace, diff_bias);
    }
}
template cell_execution_sig(ref_rnn_fwd_f16_t::cell_execution);
template cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution);
template cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution);

} // namespace ocl
} // namespace impl
} // namespace mkldnn
