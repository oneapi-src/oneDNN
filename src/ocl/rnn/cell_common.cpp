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

namespace dnnl {
namespace impl {
namespace ocl {

using namespace dnnl::impl::utils;
using namespace rnn_utils;

#define AOC array_offset_calculator

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
cell_execution_sig(
        (_ref_rnn_common_t<aprop, src_type, weights_type>::cell_execution)) {
    using src_t = typename prec_traits<src_type>::type;
    using wei_t = typename prec_traits<weights_type>::type;
    const rnn_conf_t &rnn = this->pd()->rnn_conf_;

    if (aprop == prop_kind::forward) {

        // offsets for gemm by bytes
        cl_ulong offset_states = (cl_ulong)(ws_states_offset_
                + OFF4(lay + 1, n_layer + 1, dir, n_dir, iter, n_iter + 1, 0,
                          batch * rnn.states_ws_ld)
                        * sizeof(src_t));
        cl_ulong offset_input = (cl_ulong)(ws_states_offset_
                + OFF4(lay, n_layer + 1, dir, n_dir, iter + 1, n_iter + 1, 0,
                          batch * rnn.states_ws_ld)
                        * sizeof(src_t));
        cl_ulong offset_gates = (cl_ulong)(ws_gates_offset_
                + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                          batch * rnn.gates_ws_ld)
                        * rnn.acc_data_type_elsz);

        if (!rnn.merge_gemm_layer)
            gemm_primitive(ctx, w_input,
                    OFF3(lay, n_layer, dir, n_dir, 0,
                            rnn.weights_layer_nld * rnn.weights_layer_ld)
                            * sizeof(wei_t),
                    workspace, offset_input, workspace, offset_gates,
                    gemm_layer);
        gemm_primitive(ctx, w_state,
                OFF3(lay, n_layer, dir, n_dir, 0,
                        rnn.weights_iter_nld * rnn.weights_iter_ld)
                        * sizeof(wei_t),
                workspace, offset_states, workspace, offset_gates, gemm_iter);

        (this->*elemwise_func)(ctx, dir, lay, iter, dic, wic, batch, workspace,
                scales, bias, tm_scales);

    } else { // backward

        AOC<size_t, 3> off_weights_i(
                weights_input, n_layer, n_dir, n_parts_weights_layer);
        AOC<size_t, 3> off_weights_st(
                weights_states, n_layer, n_dir, n_parts_weights_iter);

        (this->*elemwise_func)(ctx, dir, lay, iter, dic, wic, batch, workspace,
                scales, bias, tm_scales);

        cl_ulong offset_w_state
                = (cl_ulong)(off_weights_st(lay, dir, 0)) * sizeof(wei_t);
        cl_ulong offset_w_input
                = (cl_ulong)(off_weights_i(lay, dir, 0)) * sizeof(wei_t);

        gemm_primitive(ctx, w_state, offset_w_state, workspace,
                ws_gates_offset_
                        + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                                  batch * rnn.gates_ws_ld)
                                * rnn.acc_data_type_elsz,
                workspace,
                ws_diff_states_offset_
                        + OFF5(lay, n_layer + 1, dir, n_dir, 0, n_states + 1,
                                  iter, n_iter + 1, 0,
                                  rnn.states_nld * rnn.states_ws_ld)
                                * sizeof(src_t),
                gemm_iter);
        if (!rnn.merge_gemm_layer) {
            gemm_primitive(ctx, w_input, offset_w_input, workspace,
                    ws_gates_offset_
                            + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                                      batch * rnn.gates_ws_ld)
                                    * rnn.acc_data_type_elsz,
                    workspace,
                    ws_diff_states_offset_
                            + OFF5(lay, n_layer + 1, dir, n_dir, n_states,
                                      n_states + 1, iter, n_iter + 1, 0,
                                      rnn.states_nld * rnn.states_ws_ld)
                                    * sizeof(src_t),
                    gemm_layer);
            gemm_primitive(ctx, workspace,
                    ws_gates_offset_
                            + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                                      batch * rnn.gates_ws_ld)
                                    * rnn.acc_data_type_elsz,
                    workspace,
                    ws_states_offset_
                            + OFF4(lay, n_layer + 1, dir, n_dir, iter + 1,
                                      n_iter + 1, 0, batch * rnn.states_ws_ld)
                                    * sizeof(src_t),
                    diff_weights_layer,
                    OFF3(lay, n_layer, dir, n_dir, 0,
                            rnn.diff_weights_layer_nld
                                    * rnn.diff_weights_layer_ld)
                            * sizeof(wei_t),
                    gemm_diff_wei_layer);
        }
        if (!rnn.merge_gemm_iter)
            gemm_primitive(ctx, workspace,
                    ws_gates_offset_
                            + OFF4(lay, n_layer, dir, n_dir, iter, n_iter, 0,
                                      batch * rnn.gates_ws_ld)
                                    * rnn.acc_data_type_elsz,
                    workspace,
                    ws_states_offset_
                            + OFF4(lay + 1, n_layer + 1, dir, n_dir, iter,
                                      n_iter + 1, 0, batch * rnn.states_ws_ld)
                                    * sizeof(src_t),
                    diff_weights_iter,
                    OFF3(lay, n_layer, dir, n_dir, 0,
                            rnn.diff_weights_iter_nld
                                    * rnn.diff_weights_iter_ld)
                            * sizeof(wei_t),
                    gemm_diff_wei_iter);

        gates_reduction(
                ctx, dir, lay, iter, n_gates, dic, batch, workspace, diff_bias);
    }
}
template cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution);
template cell_execution_sig(ref_rnn_fwd_f16_t::cell_execution);
template cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution);
template cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution);
template cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution);
template cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution);

} // namespace ocl
} // namespace impl
} // namespace dnnl
