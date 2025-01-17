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

// Common for RNN and LSTM cell execution

#include "gpu/intel/ocl/rnn/rnn_grid.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace dnnl::impl::utils;
using namespace rnn_utils;

template <size_t out_ndims, size_t in_ndims>
strides_t<out_ndims> inner(const strides_t<in_ndims> &s) {
    static_assert(in_ndims >= out_ndims,
            "The output strides are expected to be smaller than the input "
            "strides");
    strides_t<out_ndims> ret;
    for (size_t i = 0; i < out_ndims; i++) {
        ret[i] = s[i + in_ndims - out_ndims];
    }
    return ret;
}

status_t compute_cell_fwd(const exec_ctx_t &ctx,
        const compute::kernel_t &kernel, dim_t lay, dim_t dir, dim_t iter,
        const workspace_t &workspace, const user_data_t user_data,
        const sub_buffer_t &weights_layer, const sub_buffer_t &weights_iter,
        const sub_buffer_t &cell_layer, const strides_t<4> &cell_layer_strides,
        const sub_buffer_t &cell_iter, const strides_t<4> &cell_iter_strides,
        const sub_buffer_t &scratch_gates,
        const strides_t<2> &scratch_gates_strides, float alpha,
        const memory_storage_t *tm_scales, const conf_t &conf,
        const ocl_conf_t &ocl_conf, const rnn_offsets_t &offsets) {

    auto &cell_conf = ocl_conf.cell_comp;
    const size_t dhc = conf.dhc;
    const size_t dhc_thr = cell_conf.dhc_thr;
    const size_t dhc_tg = cell_conf.dhc_tg;
    const size_t dhc_loop = utils::rnd_up(conf.dhc_loop, dhc_thr * dhc_tg);

    gpu_assert(dhc_tg % ocl_conf.subgroup_size == 0);

    const size_t mb = conf.mb;
    const size_t batch_tg = cell_conf.mb_tg;
    const size_t batch_thr = cell_conf.mb_thr;
    const size_t batch_local = batch_thr * batch_tg;
    compute::nd_range_t nd_range {
            {utils::div_up(dhc, dhc_loop) * dhc_tg,
                    utils::div_up(mb, batch_local) * batch_tg},
            {dhc_tg, batch_tg}};

    auto gates = workspace.gates(lay, dir, iter);
    auto gates_strides = workspace.gates_strides();
    auto states = workspace.states(lay, dir, iter);
    auto states_strides = workspace.states_strides();
    auto bias = user_data.bias(lay, dir);
    auto c_states_t_l = ocl_conf.cell_kind == alg_kind::vanilla_lstm
            ? workspace.c_states(lay, dir, iter)
            : sub_buffer_t();
    auto c_states_tm1_l = ocl_conf.cell_kind == alg_kind::vanilla_lstm
            ? workspace.c_states(lay, dir, iter - 1)
            : sub_buffer_t();

    arg_list_t arg_list;
    arg_list.append(weights_layer, ocl_conf.wei_dt);
    arg_list.append(offsets.weights_layer);
    arg_list.append(weights_iter, ocl_conf.wei_dt);
    arg_list.append(offsets.weights_iter);
    arg_list.append(cell_layer, ocl_conf.ws_state_dt);
    arg_list.append(inner<2>(cell_layer_strides));
    arg_list.append(cell_iter, ocl_conf.ws_state_dt);
    arg_list.append(inner<2>(cell_iter_strides));
    arg_list.append(gates, ocl_conf.aux_dt);
    arg_list.append(inner<2>(gates_strides));
    arg_list.append(states, ocl_conf.ws_state_dt);
    arg_list.append(inner<2>(states_strides));

    if (ocl_conf.cell_kind == alg_kind::vanilla_lstm) {
        arg_list.append(c_states_t_l, ocl_conf.aux_dt);
        arg_list.append(c_states_tm1_l, ocl_conf.aux_dt);
        arg_list.append(conf.tm_cscale);
    }

    if (!(cell_conf.compute_gemm_layer && cell_conf.compute_gemm_iter)) {
        arg_list.append(scratch_gates, ocl_conf.aux_dt);
        arg_list.append(scratch_gates_strides);
    }

    if (cell_conf.enable_iter_block) { arg_list.append(conf.iter_loop); }

    arg_list.append(bias, ocl_conf.bia_dt);
    arg_list.append(alpha);
    arg_list.append(get_storage(tm_scales));
    arg_list.append(conf.mb);
    arg_list.append(conf.dhc);
    arg_list.append(conf.slc);
    arg_list.append(conf.sic);

    arg_list.append(into<dim_t>(dhc_loop));

    return gpu_primitive_t::parallel_for(ctx, nd_range, kernel, arg_list.args);
}

template <prop_kind_t aprop>
cell_execution_sig((simple_rnn_common_t<aprop>::cell_execution)) {
    const conf_t &rnn = this->pd()->rnn_conf;
    const ocl_conf_t &ocl_conf = this->pd()->ocl_conf;
    const rnn_offsets_t &offsets = this->pd()->off;

    const bool use_cell = ocl_conf.cell_comp.is_enabled;

    strides_t<4> user_layer_strides {[&]() {
        auto s = user_data.src_layer_strides(dir);
        return strides_t<4> {0, 0, s[0], s[1]};
    }()};

    auto cell_layer = !rnn.copy_src_layer && lay == 0
            ? user_data.src_layer(dir, iter)
            : workspace.states(lay - 1, dir, iter);
    auto &cell_layer_strides = !rnn.copy_src_layer && lay == 0
            ? user_layer_strides
            : workspace.states_strides();
    auto cell_iter = workspace.states(lay, dir, iter - 1);
    auto &cell_iter_strides = workspace.states_strides();
    auto scratch_gates = scratch.gates(iter);
    strides_t<2> scratch_gates_strides
            = {scratch.calc_off_gates(1), rnn.scratch_gates_ld};

    auto wei_layer = user_data.wei_layer(lay, dir);
    auto wei_iter = user_data.wei_iter(lay, dir);

    if ((aprop == prop_kind::forward) || rnn.recompute_gates) {
        if (!rnn.merge_gemm_layer && !rnn.cell_fusion.gemm_layer) {
            auto gemm_cell_layer_fwd = !rnn.copy_src_layer && lay == 0
                    ? gemm_layer_fwd_src
                    : gemm_layer_fwd;
            CHECK(gemm_primitive(engine, ctx, wei_layer, cell_layer,
                    scratch_gates, gemm_cell_layer_fwd));
        }

        if (!rnn.cell_fusion.gemm_iter)
            CHECK(gemm_primitive(engine, ctx, wei_iter, cell_iter,
                    scratch_gates, gemm_iter_fwd));
    }

    if (aprop == prop_kind::forward) {
        if (!use_cell) {
            CHECK((this->*elemwise_common)(ctx, dir, lay, iter, rnn.dhc, rnn.mb,
                    1, user_data, workspace, scratch_gates, {}, {}, {}, {}, {},
                    {}, 0, scales, tm_scales, diff_bias));
        } else {
            CHECK(compute_cell_fwd(ctx, kernels_[kernel_id::cell_fwd], lay, dir,
                    iter, workspace, user_data, wei_layer, wei_iter, cell_layer,
                    cell_layer_strides, cell_iter, cell_iter_strides,
                    scratch_gates, scratch_gates_strides, pd()->desc()->alpha,
                    tm_scales, rnn, ocl_conf, offsets));
        }

    } else { // backward
        auto diff_states_iter = scratch.diff_states(lay, dir, 0, iter + 1);
        auto diff_states_iter_s1 = rnn.n_states == 2
                ? scratch.diff_states(lay, dir, 1, iter + 1)
                : sub_buffer_t();
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
                : sub_buffer_t();
        auto diff_states1 = !rnn.copy_diff_src_layer && lay == 0
                ? user_data.diff_src_layer(dir, iter)
                : scratch.diff_states(lay, dir, rnn.n_states, iter);
        auto diff_gates = scratch.diff_gates(iter);

        CHECK((this->*elemwise_common)(ctx, dir, lay, iter, rnn.dhc, rnn.mb,
                ocl_conf.elemwise_bwd_batch_block, user_data, workspace,
                scratch_gates, diff_gates, diff_states, diff_states_s1,
                diff_states_iter, diff_states_iter_s1, diff_states_layer,
                diff_states_layer_ld, scales, tm_scales, diff_bias));

        CHECK(gemm_primitive(
                engine, ctx, wei_iter, diff_gates, diff_states, gemm_iter_bwd));

        if (!rnn.merge_gemm_layer) {

            auto gemm_layer_cell_bwd = !rnn.copy_diff_src_layer && lay == 0
                    ? gemm_layer_bwd_src
                    : gemm_layer_bwd;
            CHECK(gemm_primitive(engine, ctx, wei_layer, diff_gates,
                    diff_states1, gemm_layer_cell_bwd));

            auto gemm_diff_wei_cell_layer = !rnn.copy_src_layer && lay == 0
                    ? gemm_diff_wei_layer_src
                    : gemm_diff_wei_layer;

            CHECK(gemm_primitive(engine, ctx, diff_gates, cell_layer,
                    user_data.diff_wei_layer(lay, dir),
                    gemm_diff_wei_cell_layer));
        }

        if (!rnn.merge_gemm_iter) {
            CHECK(gemm_primitive(engine, ctx, diff_gates, cell_iter,
                    user_data.diff_wei_iter(lay, dir), gemm_diff_wei_iter));
        }
    }
    return status::success;
}
template cell_execution_sig(simple_rnn_fwd_t::cell_execution);
template cell_execution_sig(simple_rnn_bwd_t::cell_execution);
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
