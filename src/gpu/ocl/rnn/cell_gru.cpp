/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gpu/ocl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
#define PART_ONE 1
#define PART_TWO 2

using namespace dnnl::impl::utils;
using namespace rnn_utils;

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution_gru)) {
    const conf_t &rnn = this->pd()->rnn_conf;
    data_type_t src_t = this->pd()->src_type;

    cl_ulong cell_scratch_offset, cell_ws_iter_offset, cell_ws_lay_offset,
            cell_wei_iter_offset;

    set_offsets_fwd_gemm(rnn, iter, dir, lay, src_t, wei_iter_offset_ptr,
            ws_states_offset_, cell_ws_iter_offset, cell_ws_lay_offset,
            cell_scratch_offset, cell_wei_iter_offset);

    if (aprop == prop_kind::forward) {
        // 1. gemm Wx[0-2],x
        if (!rnn.merge_gemm_layer)
            gemm_primitive(engine, ctx, wei_layer, wei_layer_offset[0],
                    workspace, cell_ws_lay_offset, scratch_gates,
                    cell_scratch_offset, gemm_layer_fwd);

        // 2. gemm Wh[0-1],h
        gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset, workspace,
                cell_ws_iter_offset, scratch_gates, cell_scratch_offset,
                gemm_iter_fwd);

        // 3. activation zt and rt + elemwise multiplication rt,ht-1
        (this->*elemwise_func)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, workspace,
                scratch_gates, scratch_cell, scales, bias, tm_scales, PART_ONE);

        update_gru_offsets(rnn, iter, dir, lay, src_t, wei_iter_offset_ptr,
                ws_states_offset_, cell_wei_iter_offset, cell_scratch_offset,
                cell_ws_iter_offset);

        // 4. gemm Wh[2],h~t
        gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset, workspace,
                cell_ws_iter_offset, scratch_gates, cell_scratch_offset,
                gemm_iter_fwd_2);

        // 5. activation h~t + calculate ht
        (this->*elemwise_func)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, workspace,
                scratch_gates, scratch_cell, scales, bias, tm_scales, PART_TWO);
    } else {
        assert("Bwd Vanilla GRU is unimplemented");
    }
}

template cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru);
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
