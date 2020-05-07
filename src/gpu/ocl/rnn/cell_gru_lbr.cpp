/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include "gpu/ocl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
using namespace dnnl::impl::utils;
using namespace rnn_utils;

#define AOC array_offset_calculator

template cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru_lbr);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru_lbr);

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution_gru_lbr)) {
    data_type_t src_t = this->pd()->src_type;
    data_type_t wei_t = this->pd()->weights_type;

    const conf_t &rnn = this->pd()->rnn_conf;

    cl_ulong offset_scratch_memory
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

        // call made when cell execution is enabled
        if (!rnn.merge_gemm_layer)
            gemm_primitive(engine, ctx, w_input, offset_w_input, workspace,
                    offset_input, scratch_gates, offset_scratch_memory,
                    gemm_layer_fwd);

        // call to gemm iter
        gemm_primitive(engine, ctx, w_state, offset_w_state, workspace,
                offset_states, scratch_cell, 0, gemm_iter_fwd);

        (this->*elemwise_func)(ctx, dir, lay, iter, dhc, wic, batch, workspace,
                scratch_gates, scratch_cell, scales, bias, tm_scales);

    } else {
        assert(!"backward is unimplemented");
    }
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
