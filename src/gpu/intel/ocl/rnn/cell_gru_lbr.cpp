/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "gpu/intel/ocl/rnn/rnn_grid.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace dnnl::impl::utils;
using namespace rnn_utils;

template cell_execution_sig(simple_rnn_fwd_t::cell_execution_gru_lbr);
template cell_execution_sig(simple_rnn_bwd_t::cell_execution_gru_lbr);

template <prop_kind_t aprop>
cell_execution_sig((simple_rnn_common_t<aprop>::cell_execution_gru_lbr)) {
    const conf_t &rnn = this->pd()->rnn_conf;
    const ocl_conf_t &ocl_conf = this->pd()->ocl_conf;
    const rnn_offsets_t &offsets = this->pd()->off;

    auto cell_layer = !rnn.copy_src_layer && lay == 0
            ? user_data.src_layer(dir, iter)
            : workspace.states(lay - 1, dir, iter);
    auto gemm_cell_layer_fwd = !rnn.copy_src_layer && lay == 0
            ? gemm_layer_fwd_src
            : gemm_layer_fwd;
    auto gemm_diff_wei_cell_layer = !rnn.copy_src_layer && lay == 0
            ? gemm_diff_wei_layer_src
            : gemm_diff_wei_layer;
    auto cell_iter = workspace.states(lay, dir, iter - 1);

    auto scratch_gates = scratch.gates(iter);
    auto &scratch_cell = scratch.cell() ? *scratch.cell()
                                        : memory_storage_t::empty_storage();

    auto wei_layer = user_data.wei_layer(lay, dir);
    auto wei_iter = user_data.wei_iter(lay, dir);

    if (aprop == prop_kind::forward) {
        // call made when cell execution is enabled
        if (!rnn.merge_gemm_layer)
            CHECK(gemm_primitive(engine, ctx, wei_layer, cell_layer,
                    scratch_gates, gemm_cell_layer_fwd));

        CHECK(gemm_primitive(
                engine, ctx, wei_iter, cell_iter, scratch_cell, gemm_iter_fwd));

        CHECK((this->*elemwise_gru_lbr)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, 1,
                user_data, workspace, scratch_gates, {}, scratch_cell, {}, {},
                {}, 0, tm_scales, diff_bias));

    } else {
        auto diff_states_iter = scratch.diff_states(lay, dir, 0, iter + 1);
        auto diff_states_layer
                = !rnn.copy_diff_dst_layer && lay + 1 == rnn.n_layer
                ? user_data.diff_dst_layer(dir, iter)
                : scratch.diff_states(lay + 1, dir, rnn.n_states, iter);
        auto diff_states_layer_ld
                = !rnn.copy_diff_dst_layer && lay + 1 == rnn.n_layer
                ? offsets.diff_dst_layer[1]
                : rnn.scratch_diff_states_ld;

        auto diff_states = scratch.diff_states(lay, dir, 0, iter);
        auto diff_states1 = !rnn.copy_diff_src_layer && lay == 0
                ? user_data.diff_src_layer(dir, iter)
                : scratch.diff_states(lay, dir, rnn.n_states, iter);

        auto diff_gates = scratch.diff_gates(iter);

        CHECK((this->*elemwise_gru_lbr)(ctx, dir, lay, iter, rnn.dhc, rnn.mb,
                ocl_conf.elemwise_bwd_batch_block, user_data, workspace,
                scratch_gates, diff_gates, scratch_cell, diff_states,
                diff_states_iter, diff_states_layer, diff_states_layer_ld,
                tm_scales, diff_bias));

        if (!rnn.merge_gemm_layer) {
            CHECK(gemm_primitive(engine, ctx, diff_gates, cell_layer,
                    user_data.diff_wei_layer(lay, dir),
                    gemm_diff_wei_cell_layer));

            auto gemm_layer_cell_bwd = !rnn.copy_diff_src_layer && lay == 0
                    ? gemm_layer_bwd_src
                    : gemm_layer_bwd;
            CHECK(gemm_primitive(engine, ctx, wei_layer, diff_gates,
                    diff_states1, gemm_layer_cell_bwd));
        }

        CHECK(gemm_primitive(engine, ctx, wei_iter, scratch_cell, diff_states,
                gemm_iter_bwd));

        CHECK(gemm_primitive(engine, ctx, scratch_cell, cell_iter,
                user_data.diff_wei_iter(lay, dir), gemm_diff_wei_iter));
    }
    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
