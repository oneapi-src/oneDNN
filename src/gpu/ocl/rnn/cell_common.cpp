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

// Common for RNN and LSTM cell execution

#include "gpu/ocl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::utils;
using namespace rnn_utils;

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution)) {
    const conf_t &rnn = this->pd()->rnn_conf;
    const ocl_conf_t &ocl_conf = this->pd()->ocl_conf;
    const rnn_offsets_t &offsets = this->pd()->off;

    dim_t cell_wei_iter_offset;

    set_offsets_fwd_gemm(
            rnn, iter, dir, lay, wei_iter_offsets, cell_wei_iter_offset);

    auto cell_layer = !rnn.copy_src_layer && lay == 0
            ? user_data.src_layer(dir, iter)
            : workspace.states(lay - 1, dir, iter);
    auto cell_iter = workspace.states(lay, dir, iter - 1);
    auto scratch_gates_owner = scratch.gates(iter);
    auto scratch_gates = scratch_gates_owner.get();

    if (aprop == prop_kind::forward || rnn.recompute_gates) {
        if (!rnn.merge_gemm_layer) {
            auto gemm_cell_layer_fwd = !rnn.copy_src_layer && lay == 0
                    ? gemm_layer_fwd_src
                    : gemm_layer_fwd;
            CHECK(gemm_primitive(engine, ctx, wei_layer, wei_layer_offset,
                    *cell_layer, 0, *scratch_gates, 0, gemm_cell_layer_fwd));
        }

        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                *cell_iter, 0, *scratch_gates, 0, gemm_iter_fwd));
    }

    if (aprop == prop_kind::forward) {
        CHECK((this->*elemwise_common)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, 1,
                workspace, scratch_gates, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, 0, scales, bias, tm_scales, diff_bias));

    } else { // backward
        dim_t cell_diff_wei_iter_off, cell_diff_wei_lay_off;

        set_offsets_bwd_gemm(rnn, iter, dir, lay, cell_diff_wei_iter_off,
                cell_diff_wei_lay_off);

        auto diff_states_iter = scratch.diff_states(lay, dir, 0, iter + 1);
        auto diff_states_iter_s1 = rnn.n_states == 2
                ? scratch.diff_states(lay, dir, 1, iter + 1)
                : nullptr;
        auto diff_states_layer
                = !rnn.copy_diff_dst_layer && lay + 1 == rnn.n_layer
                ? user_data.diff_dst_layer(dir, iter)
                : scratch.diff_states(lay + 1, dir, rnn.n_states, iter);
        auto diff_states_layer_ld
                = !rnn.copy_diff_dst_layer && lay + 1 == rnn.n_layer
                ? offsets.diff_dst_layer[1]
                : rnn.scratch_diff_states_ld;

        auto diff_states = scratch.diff_states(lay, dir, 0, iter);
        auto diff_states_s1 = rnn.n_states == 2
                ? scratch.diff_states(lay, dir, 1, iter)
                : nullptr;
        auto diff_states1 = scratch.diff_states(lay, dir, rnn.n_states, iter);
        auto diff_gates = scratch.diff_gates(iter);

        CHECK((this->*elemwise_common)(ctx, dir, lay, iter, rnn.dhc, rnn.mb,
                ocl_conf.elemwise_bwd_batch_block, workspace, scratch_gates,
                diff_gates.get(), diff_states.get(), diff_states_s1.get(),
                diff_states_iter.get(), diff_states_iter_s1.get(),
                diff_states_layer.get(), diff_states_layer_ld, scales, bias,
                tm_scales, diff_bias));

        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                *diff_gates, 0, *diff_states, 0, gemm_iter_bwd));

        if (!rnn.merge_gemm_layer) {
            CHECK(gemm_primitive(engine, ctx, wei_layer, wei_layer_offset,
                    *diff_gates, 0, *diff_states1, 0, gemm_layer_bwd));

            auto gemm_diff_wei_cell_layer = !rnn.copy_src_layer && lay == 0
                    ? gemm_diff_wei_layer_src
                    : gemm_diff_wei_layer;

            CHECK(gemm_primitive(engine, ctx, *diff_gates, 0, *cell_layer, 0,
                    diff_weights_layer, cell_diff_wei_lay_off,
                    gemm_diff_wei_cell_layer));
        }

        if (!rnn.merge_gemm_iter) {
            CHECK(gemm_primitive(engine, ctx, *diff_gates, 0, *cell_iter, 0,
                    diff_weights_iter, cell_diff_wei_iter_off,
                    gemm_diff_wei_iter));
        }
    }
    return status::success;
}
template cell_execution_sig(ref_rnn_fwd_t::cell_execution);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution);
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
