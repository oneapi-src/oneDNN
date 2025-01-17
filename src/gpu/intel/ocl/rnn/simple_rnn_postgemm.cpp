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

#include "gpu/intel/ocl/rnn/rnn_grid.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace dnnl::impl::gpu::intel::gpu_utils;
using namespace rnn_utils;

template <prop_kind_t aprop>
elemwise_sig((simple_rnn_common_t<aprop>::rnn_elemwise)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? kernels_[kernel_id::elemwise_fwd]
            : kernels_[kernel_id::elemwise_bwd];

    arg_list_t arg_list;
    if (aprop == prop_kind::backward) {
        arg_list.append(into<int32_t>(dir));
        arg_list.append(into<int32_t>(lay));
        arg_list.append(into<int32_t>(iter));
    }
    if (aprop == prop_kind::forward) {
        arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
    } else {
        arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
        arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
                pd()->ocl_conf.acc_dt);
    }
    auto bias = user_data.bias(lay, dir);
    arg_list.append(bias, pd()->ocl_conf.bia_dt);
    arg_list.append(pd()->desc()->alpha);
    // for test mode
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    data_type_t ws_dt = pd()->ocl_conf.src_dt;
    auto states_t_l = workspace.states(lay, dir, iter);
    arg_list.append(states_t_l, ws_dt);

    auto c_states_t_l = workspace.c_states(lay, dir, iter);
    auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
    arg_list.append(c_states_t_l, pd()->ocl_conf.aux_dt);
    arg_list.append(c_states_tm1_l, pd()->ocl_conf.aux_dt);

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(gates, pd()->ocl_conf.aux_dt);

    auto ws_grid = workspace.grid_comp(lay, dir, iter);
    arg_list.append(ws_grid, pd()->ocl_conf.aux_dt);

    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    if (aprop == prop_kind::forward)
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    else {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    }

    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    if (aprop == dnnl_backward) {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
        arg_list.append(into<int32_t>(diff_states_layer_ld));
    }

    arg_list.append(pd()->rnn_conf.tm_cscale);
    if (aprop != dnnl_forward) {
        auto diff_dt = pd()->ocl_conf.diff_dt;
        arg_list.append(scratch_diff_states, diff_dt);
        arg_list.append(scratch_diff_states_iter, diff_dt);
        arg_list.append(scratch_diff_states_layer, diff_dt);
        arg_list.append(diff_bias);
        arg_list.append(pd()->off.diff_bias);
    }
    return parallel_for(ctx, nd_range, kernel, arg_list.args);
}
template elemwise_sig(simple_rnn_fwd_t::rnn_elemwise);
template elemwise_sig(simple_rnn_bwd_t::rnn_elemwise);

template <prop_kind_t aprop>
elemwise_sig((simple_rnn_common_t<aprop>::lstm_elemwise)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? kernels_[kernel_id::elemwise_fwd]
            : kernels_[kernel_id::elemwise_bwd];

    arg_list_t arg_list;
    if (aprop == prop_kind::backward) {
        arg_list.append(into<int32_t>(dir));
        arg_list.append(into<int32_t>(lay));
        arg_list.append(into<int32_t>(iter));
    }
    if (aprop == prop_kind::forward) {
        arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
    } else {
        arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
        arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
                pd()->ocl_conf.acc_dt);
    }
    auto bias = user_data.bias(lay, dir);
    arg_list.append(bias, pd()->ocl_conf.bia_dt);
    arg_list.append(pd()->desc()->alpha);
    // for test mode
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    data_type_t ws_dt = pd()->ocl_conf.src_dt;
    auto states_t_l = workspace.states(lay, dir, iter);
    arg_list.append(states_t_l, ws_dt);

    auto c_states_t_l = workspace.c_states(lay, dir, iter);
    auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
    arg_list.append(c_states_t_l, pd()->ocl_conf.aux_dt);
    arg_list.append(c_states_tm1_l, pd()->ocl_conf.aux_dt);

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(gates, pd()->ocl_conf.aux_dt);

    auto ws_grid = workspace.grid_comp(lay, dir, iter);
    arg_list.append(ws_grid, pd()->ocl_conf.aux_dt);

    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    if (aprop == prop_kind::forward) {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    } else {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    }
    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    if (aprop == dnnl_backward) {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
        arg_list.append(into<int32_t>(diff_states_layer_ld));
    }

    arg_list.append(pd()->rnn_conf.tm_cscale);
    if (aprop != dnnl_forward) {
        auto diff_dt = pd()->ocl_conf.diff_dt;
        arg_list.append(scratch_diff_states, diff_dt);
        arg_list.append(scratch_diff_states_iter, diff_dt);
        arg_list.append(scratch_diff_states_layer, diff_dt);
        arg_list.append(scratch_diff_states_s1, diff_dt);
        arg_list.append(scratch_diff_states_iter_s1, diff_dt);
        arg_list.append(diff_bias);
        arg_list.append(pd()->off.diff_bias);
    }
    return parallel_for(ctx, nd_range, kernel, arg_list.args);
}
template elemwise_sig(simple_rnn_fwd_t::lstm_elemwise);
template elemwise_sig(simple_rnn_bwd_t::lstm_elemwise);

template <prop_kind_t aprop>
elemwise_sig((simple_rnn_common_t<aprop>::lstm_elemwise_u8s8)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    arg_list_t arg_list;
    arg_list.append(into<int32_t>(dir));
    arg_list.append(into<int32_t>(lay));
    arg_list.append(into<int32_t>(iter));
    if (aprop == prop_kind::forward) {
        arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
    } else {
        arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
        arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
                pd()->ocl_conf.acc_dt);
    }
    arg_list.append(scales ? *scales : memory_storage_t::empty_storage());
    arg_list.append(pd()->desc()->alpha);
    arg_list.append(data_shift);
    arg_list.append(data_scale);
    // for test mode
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    data_type_t ws_dt = pd()->ocl_conf.src_dt;
    auto states_t1_l = workspace.states(lay, dir, iter);
    arg_list.append(states_t1_l, ws_dt);

    auto c_states_t_l = workspace.c_states(lay, dir, iter);
    auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
    arg_list.append(c_states_t_l, data_type::f32);
    arg_list.append(c_states_tm1_l, data_type::f32);

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(gates, pd()->ocl_conf.aux_dt);

    arg_list.append(workspace.bias());

    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    if (aprop == prop_kind::forward) {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    } else {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    }
    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_layer));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_dir));
    arg_list.append(pd()->rnn_conf.tm_cscale);
    return parallel_for(
            ctx, nd_range, kernels_[kernel_id::elemwise_fwd], arg_list.args);
}
template elemwise_sig(simple_rnn_fwd_t::lstm_elemwise_u8s8);
template elemwise_sig(simple_rnn_bwd_t::lstm_elemwise_u8s8);

template <prop_kind_t aprop>
elemwise_sig_gru_lbr((simple_rnn_common_t<aprop>::gru_lbr_elemwise)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? kernels_[kernel_id::elemwise_fwd]
            : kernels_[kernel_id::elemwise_bwd];

    arg_list_t arg_list;
    if (aprop == prop_kind::backward) {
        arg_list.append(into<int32_t>(dir));
        arg_list.append(into<int32_t>(lay));
        arg_list.append(into<int32_t>(iter));
    }
    if (aprop == prop_kind::forward) {
        arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
    } else {
        arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
        arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
                pd()->ocl_conf.acc_dt);
    }
    auto bias = user_data.bias(lay, dir);
    arg_list.append(bias, pd()->ocl_conf.bia_dt);
    arg_list.append(pd()->desc()->alpha);
    // for test mode
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    data_type_t ws_dt = pd()->ocl_conf.src_dt;
    auto states_t1_l = workspace.states(lay, dir, iter);
    auto states_tm1_l = workspace.states(lay, dir, iter - 1);
    arg_list.append(
            aprop == prop_kind::forward ? states_t1_l : states_tm1_l, ws_dt);

    auto c_states_t_l = workspace.c_states(lay, dir, iter);
    auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
    arg_list.append(c_states_t_l, pd()->ocl_conf.aux_dt);
    arg_list.append(c_states_tm1_l, pd()->ocl_conf.aux_dt);

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(gates, pd()->ocl_conf.aux_dt);

    auto ws_grid = workspace.grid_comp(lay, dir, iter);
    arg_list.append(ws_grid, pd()->ocl_conf.aux_dt);

    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    if (aprop == prop_kind::forward) {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    } else {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    }
    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    if (aprop == dnnl_backward) {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
        arg_list.append(into<int32_t>(diff_states_layer_ld));
    }

    if (aprop == dnnl_forward) { arg_list.append(states_tm1_l, ws_dt); }
    arg_list.append(scratch_cell);
    if (aprop != dnnl_forward) {
        auto diff_dt = pd()->ocl_conf.diff_dt;
        arg_list.append(scratch_diff_states, diff_dt);
        arg_list.append(scratch_diff_states_iter, diff_dt);
        arg_list.append(scratch_diff_states_layer, diff_dt);
        arg_list.append(diff_bias);
        arg_list.append(pd()->off.diff_bias);
    }
    return parallel_for(ctx, nd_range, kernel, arg_list.args);
}
template elemwise_sig_gru_lbr(simple_rnn_fwd_t::gru_lbr_elemwise);
template elemwise_sig_gru_lbr(simple_rnn_bwd_t::gru_lbr_elemwise);

template <prop_kind_t aprop>
elemwise_sig_gru((simple_rnn_common_t<aprop>::gru_elemwise)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? kernels_[kernel_id::elemwise_fwd]
            : kernels_[kernel_id::elemwise_bwd];

    arg_list_t arg_list;
    if (aprop == prop_kind::backward) {
        arg_list.append(into<int32_t>(dir));
        arg_list.append(into<int32_t>(lay));
        arg_list.append(into<int32_t>(iter));
    }
    if (aprop == prop_kind::forward) {
        arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
    } else {
        arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
        arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
                pd()->ocl_conf.acc_dt);
    }
    auto bias = user_data.bias(lay, dir);
    arg_list.append(bias, pd()->ocl_conf.bia_dt);
    arg_list.append(pd()->desc()->alpha);
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    data_type_t ws_dt = pd()->ocl_conf.src_dt;
    auto states_t1_l = workspace.states(lay, dir, iter);
    auto states_tm1_l = workspace.states(lay, dir, iter - 1);
    arg_list.append(
            aprop == prop_kind::forward ? states_t1_l : states_tm1_l, ws_dt);

    auto c_states_t_l = workspace.c_states(lay, dir, iter);
    auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
    arg_list.append(c_states_t_l, pd()->ocl_conf.aux_dt);
    arg_list.append(c_states_tm1_l, pd()->ocl_conf.aux_dt);

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(gates, pd()->ocl_conf.aux_dt);

    auto ws_grid = workspace.grid_comp(lay, dir, iter);
    arg_list.append(ws_grid, pd()->ocl_conf.aux_dt);

    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    if (aprop == prop_kind::forward) {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    } else {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    }
    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    if (aprop == dnnl_backward) {
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
        arg_list.append(into<int32_t>(diff_states_layer_ld));
    }

    if (aprop == dnnl_forward) { arg_list.append(states_tm1_l, ws_dt); }
    arg_list.append(part);
    if (aprop != dnnl_forward) {
        auto diff_dt = pd()->ocl_conf.diff_dt;
        arg_list.append(scratch_cell);
        arg_list.append(scratch_dhG1, diff_dt);
        arg_list.append(scratch_diff_states, diff_dt);
        arg_list.append(scratch_diff_states_iter, diff_dt);
        arg_list.append(scratch_diff_states_layer, diff_dt);
        arg_list.append(diff_bias);
        arg_list.append(pd()->off.diff_bias);
    }
    return parallel_for(ctx, nd_range, kernel, arg_list.args);
}
template elemwise_sig_gru(simple_rnn_fwd_t::gru_elemwise);
template elemwise_sig_gru(simple_rnn_bwd_t::gru_elemwise);
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
