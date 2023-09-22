/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

template cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru_lbr);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru_lbr);

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution_gru_lbr)) {
    const conf_t &rnn = this->pd()->rnn_conf;
    const ocl_conf_t &ocl_conf = this->pd()->ocl_conf;
    data_type_t src_t = this->pd()->src_type;

    dim_t cell_scratch_offset, cell_ws_iter_offset, cell_ws_lay_offset,
            cell_wei_iter_offset;

    set_offsets_fwd_gemm(rnn, iter, dir, lay, src_t, wei_iter_offsets,
            ws_states_offset_, cell_ws_iter_offset, cell_ws_lay_offset,
            cell_scratch_offset, cell_wei_iter_offset);

    if (aprop == prop_kind::forward) {
        // call made when cell execution is enabled
        if (!rnn.merge_gemm_layer)
            CHECK(gemm_primitive(engine, ctx, wei_layer, wei_layer_offset,
                    workspace.ws(), cell_ws_lay_offset, scratch_gates,
                    cell_scratch_offset, gemm_layer_fwd));

        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                workspace.ws(), cell_ws_iter_offset, scratch_cell, 0,
                gemm_iter_fwd));

        CHECK((this->*elemwise_gru_lbr)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, 1,
                workspace, scratch_gates, scratch_cell, scratch_diff_states,
                bias, tm_scales, diff_bias));

    } else {
        dim_t cell_diff_wei_iter_off, cell_diff_wei_lay_off,
                cell_scr_diff_iter_off, cell_scr_diff_lay_off;

        set_offsets_bwd_gemm(rnn, iter, dir, lay, cell_diff_wei_iter_off,
                cell_diff_wei_lay_off, cell_scr_diff_lay_off,
                cell_scr_diff_iter_off);

        CHECK((this->*elemwise_gru_lbr)(ctx, dir, lay, iter, rnn.dhc, rnn.mb,
                ocl_conf.elemwise_bwd_batch_block, workspace, scratch_gates,
                scratch_cell, scratch_diff_states, bias, tm_scales, diff_bias));

        if (!rnn.merge_gemm_layer) {
            CHECK(gemm_primitive(engine, ctx, scratch_gates,
                    cell_scratch_offset, workspace.ws(), cell_ws_lay_offset,
                    diff_weights_layer, cell_diff_wei_lay_off,
                    gemm_diff_wei_layer));

            CHECK(gemm_primitive(engine, ctx, wei_layer, wei_layer_offset,
                    scratch_gates, cell_scratch_offset, scratch_diff_states,
                    cell_scr_diff_lay_off, gemm_layer_bwd));
        }

        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                scratch_cell, 0, scratch_diff_states, cell_scr_diff_iter_off,
                gemm_iter_bwd));

        CHECK(gemm_primitive(engine, ctx, scratch_cell, 0, workspace.ws(),
                cell_ws_iter_offset, diff_weights_iter, cell_diff_wei_iter_off,
                gemm_diff_wei_iter));
    }
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
