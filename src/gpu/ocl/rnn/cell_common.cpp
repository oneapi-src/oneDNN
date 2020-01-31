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
#include "gpu/ocl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::utils;
using namespace rnn_utils;

#define AOC array_offset_calculator

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution)) {
    data_type_t src_t = this->pd()->src_type;
    data_type_t wei_t = this->pd()->weights_type;

    const rnn_conf_t &rnn = this->pd()->rnn_conf_;

    cl_ulong offset_scratch_gates
            = (rnn.merge_gemm_iter || rnn.merge_gemm_layer)
            ? (cl_ulong)(
                    OFF2(iter, n_iter, 0, rnn.gates_nld * rnn.scratch_gates_ld)
                    * rnn.scratch_gates_elsz)
            : (size_t)0;
    if (aprop == prop_kind::forward) {

        // offsets for gemm by bytes
        cl_ulong offset_states = (cl_ulong)(ws_states_offset_
                + OFF4(lay + 1, n_layer + 1, dir, n_dir, iter, n_iter + 1, 0,
                          batch * rnn.states_ws_ld)
                        * types::data_type_size(src_t));
        cl_ulong offset_input = (cl_ulong)(ws_states_offset_
                + OFF4(lay, n_layer + 1, dir, n_dir, iter + 1, n_iter + 1, 0,
                          batch * rnn.states_ws_ld)
                        * types::data_type_size(src_t));
        cl_ulong offset_w_input
                = OFF3(lay, n_layer, dir, n_dir, 0,
                          rnn.weights_layer_nld * rnn.weights_layer_ld)
                * types::data_type_size(wei_t);
        cl_ulong offset_w_state
                = OFF3(lay, n_layer, dir, n_dir, 0,
                          rnn.weights_iter_nld * rnn.weights_iter_ld)
                * types::data_type_size(wei_t);
        if (!rnn.merge_gemm_layer)
            gemm_primitive(ctx, w_input, offset_w_input, workspace,
                    offset_input, scratch_gates, offset_scratch_gates,
                    gemm_layer_fwd);

        gemm_primitive(ctx, w_state, offset_w_state, workspace, offset_states,
                scratch_gates, offset_scratch_gates, gemm_iter_fwd);

        (this->*elemwise_func)(ctx, dir, lay, iter, dic, wic, batch, workspace,
                scratch_gates, scales, bias, tm_scales);

    } else { // backward

        AOC<size_t, 3> off_weights_i(
                weights_input, n_layer, n_dir, n_parts_weights_layer);
        AOC<size_t, 3> off_weights_st(
                weights_states, n_layer, n_dir, n_parts_weights_iter);

        (this->*elemwise_func)(ctx, dir, lay, iter, dic, wic, batch, workspace,
                scratch_gates, scales, bias, tm_scales);

        cl_ulong offset_w_state = (cl_ulong)(off_weights_st(lay, dir, 0))
                * types::data_type_size(wei_t);
        cl_ulong offset_w_input = (cl_ulong)(off_weights_i(lay, dir, 0))
                * types::data_type_size(src_t);
        cl_ulong offset_workspace_common = ws_diff_states_offset_
                + OFF5(lay, n_layer + 1, dir, n_dir, 0, n_states + 1, iter,
                          n_iter + 1, 0, rnn.states_nld * rnn.diff_states_ws_ld)
                        * sizeof(float);
        cl_ulong offset_diff_states = ws_diff_states_offset_
                + OFF5(lay, n_layer + 1, dir, n_dir, n_states, n_states + 1,
                          iter, n_iter + 1, 0,
                          rnn.states_nld * rnn.diff_states_ws_ld)
                        * sizeof(float);
        cl_ulong offset_workspace_layer = ws_states_offset_
                + OFF4(lay, n_layer + 1, dir, n_dir, iter + 1, n_iter + 1, 0,
                          batch * rnn.states_ws_ld)
                        * types::data_type_size(src_t);
        cl_ulong offset_diff_weights_layer
                = OFF3(lay, n_layer, dir, n_dir, 0,
                          rnn.diff_weights_layer_nld
                                  * rnn.diff_weights_layer_ld)
                * sizeof(float);

        gemm_primitive(ctx, w_state, offset_w_state, scratch_gates,
                offset_scratch_gates, workspace, offset_workspace_common,
                gemm_iter_bwd);

        if (!rnn.merge_gemm_layer) {

            gemm_primitive(ctx, w_input, offset_w_input, scratch_gates,
                    offset_scratch_gates, workspace, offset_diff_states,
                    gemm_layer_bwd);

            gemm_primitive(ctx, scratch_gates, offset_scratch_gates, workspace,
                    offset_workspace_layer, diff_weights_layer,
                    offset_diff_weights_layer, gemm_diff_wei_layer);
        }
        if (!rnn.merge_gemm_iter) {

            cl_ulong offset_workspace_iter = ws_states_offset_
                    + OFF4(lay + 1, n_layer + 1, dir, n_dir, iter, n_iter + 1,
                              0, batch * rnn.states_ws_ld)
                            * types::data_type_size(src_t);
            cl_ulong offset_weights_iter
                    = OFF3(lay, n_layer, dir, n_dir, 0,
                              rnn.diff_weights_iter_nld
                                      * rnn.diff_weights_iter_ld)
                    * sizeof(float);

            gemm_primitive(ctx, scratch_gates, offset_scratch_gates, workspace,
                    offset_workspace_iter, diff_weights_iter,
                    offset_weights_iter, gemm_diff_wei_iter);
        }
        gates_reduction(ctx, dir, lay, iter, n_gates, dic, batch, scratch_gates,
                diff_bias);
    }
}
template cell_execution_sig(ref_rnn_fwd_t::cell_execution);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution);
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
