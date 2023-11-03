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
    const ocl_conf_t &ocl_conf = this->pd()->ocl_conf;
    const rnn_offsets_t &offsets = this->pd()->off;

    dim_t cell_wei_iter_offset, cell_wei_iter_offset2, cell_scratch_offset2;

    set_offsets_fwd_gemm(
            rnn, iter, dir, lay, wei_iter_offsets, cell_wei_iter_offset);

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
    auto cell_iter2 = workspace.states(lay, dir, iter);

    set_gru_offsets_part2(rnn, iter, dir, lay, wei_iter_offsets,
            cell_wei_iter_offset2, cell_scratch_offset2);

    auto scratch_gates_owner = scratch.gates(iter);
    auto scratch_gates = scratch_gates_owner.get();

    if (aprop == prop_kind::forward) {
        // 1. gemm Wx[0-2],x
        if (!rnn.merge_gemm_layer)
            CHECK(gemm_primitive(engine, ctx, wei_layer, wei_layer_offset,
                    *cell_layer, 0, *scratch_gates, 0, gemm_cell_layer_fwd));

        // 2. gemm Wh[0-1],h
        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                *cell_iter, 0, *scratch_gates, 0, gemm_iter_fwd));

        // 3. activation zt and rt + elemwise multiplication rt,ht-1
        CHECK((this->*elemwise_gru)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, 1,
                workspace, scratch_gates, nullptr, scratch.cell(), nullptr,
                nullptr, nullptr, 0, nullptr, bias, tm_scales, diff_bias,
                PART_ONE));

        // 4. gemm Wh[2],h~t
        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset2,
                *cell_iter2, 0, *scratch_gates, cell_scratch_offset2,
                gemm_iter_fwd_2));

        // 5. activation h~t + calculate ht
        CHECK((this->*elemwise_gru)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, 1,
                workspace, scratch_gates, nullptr, scratch.cell(), nullptr,
                nullptr, nullptr, 0, nullptr, bias, tm_scales, diff_bias,
                PART_TWO));

    } else {
        dim_t cell_diff_wei_iter_off, cell_diff_wei_lay_off,
                cell_diff_wei_iter_off2;

        set_offsets_bwd_gemm(rnn, iter, dir, lay, cell_diff_wei_iter_off,
                cell_diff_wei_lay_off, cell_diff_wei_iter_off2);

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
        dim_t cell_scratch_diff_off2
                = 2 * rnn.dhc * rnn.scratch_diff_gates_elsz;

        // 1. calculate dG2, dG1, and part of dht-1
        CHECK((this->*elemwise_gru)(ctx, dir, lay, iter, rnn.dhc, rnn.mb,
                ocl_conf.elemwise_bwd_batch_block, workspace, scratch_gates,
                diff_gates.get(), scratch.cell(), diff_states.get(),
                diff_states_iter.get(), diff_states_layer.get(),
                diff_states_layer_ld, scratch.diff_ht(), bias, tm_scales,
                diff_bias, PART_ONE));

        // 2. calculate intermediate d(hG1)
        // d(hG1) = dG2 * W2h^t
        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset2,
                *diff_gates, cell_scratch_diff_off2, *scratch.diff_ht(), 0,
                gemm_iter_bwd_2));

        // 3. calculate dG1^ and part of dht-1
        // hg1 needs to be bf16 as it is used as gemm output
        CHECK((this->*elemwise_gru)(ctx, dir, lay, iter, rnn.dhc, rnn.mb,
                ocl_conf.elemwise_bwd_batch_block, workspace, scratch_gates,
                diff_gates.get(), scratch.cell(), diff_states.get(),
                diff_states_iter.get(), diff_states_layer.get(),
                diff_states_layer_ld, scratch.diff_ht(), bias, tm_scales,
                diff_bias, PART_TWO));

        // 4. calculate diff weights
        // dWh1 += dG1 * h, dWh2 += dG2 * h, dWh3 += dG3 * (G1(*)h)
        CHECK(gemm_primitive(engine, ctx, *diff_gates, 0, *cell_iter, 0,
                diff_weights_iter, cell_diff_wei_iter_off, gemm_diff_wei_iter));

        CHECK(gemm_primitive(engine, ctx, *diff_gates, cell_scratch_diff_off2,
                *scratch.cell(), 0, diff_weights_iter, cell_diff_wei_iter_off2,
                gemm_diff_wei_iter_2));

        // 5. calculate diff states
        // dht-1 += dG1 * W1h + dG0 * W0h
        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                *diff_gates, 0, *diff_states, 0, gemm_iter_bwd));

        if (!rnn.merge_gemm_layer) {
            // dWx += [dG0 dG1 dG2] * [x]
            CHECK(gemm_primitive(engine, ctx, *diff_gates, 0, *cell_layer, 0,
                    diff_weights_layer, cell_diff_wei_lay_off,
                    gemm_diff_wei_cell_layer));

            // dx = dG2 * W2x + dG1 * W1x + dG0 * W0x
            CHECK(gemm_primitive(engine, ctx, wei_layer, wei_layer_offset,
                    *diff_gates, 0, *diff_states1, 0, gemm_layer_bwd));
        }
    }
    return status::success;
}
template cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru);
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
