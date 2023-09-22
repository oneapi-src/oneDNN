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
#include "gpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::gpu::gpu_utils;

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::rnn_elemwise)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;

    compute::kernel_arg_list_t arg_list;
    arg_list.append(into<int32_t>(dir));
    arg_list.append(into<int32_t>(lay));
    arg_list.append(into<int32_t>(iter));
    arg_list.append(scratch_gates);
    arg_list.append(bias);
    arg_list.append(pd()->desc()->alpha);
    // for test mode
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    auto states_t_l = workspace.states(lay + 1, dir, iter + 1);
    arg_list.append(rnn_utils::get_storage(states_t_l));

    auto c_states_t_l = workspace.c_states(lay + 1, dir, iter + 1);
    auto c_states_tm1_l = workspace.c_states(lay + 1, dir, iter);
    arg_list.append(rnn_utils::get_storage(c_states_t_l));
    arg_list.append(rnn_utils::get_storage(c_states_tm1_l));

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(rnn_utils::get_storage(gates));

    auto ws_grid = workspace.grid_comp(lay, dir, iter);
    arg_list.append(rnn_utils::get_storage(ws_grid));

    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));

    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_layer));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_gates));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_iter_scratch_gates));
    if (aprop == dnnl_forward) {
        rnn_utils::append_strides(arg_list, pd()->off.bias, 4);
    } else {
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_dir));
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_states));
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_iter));
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
    }

    arg_list.append(pd()->rnn_conf.tm_cscale);
    if (aprop != dnnl_forward) {
        arg_list.append(scratch_diff_states);
        arg_list.append(diff_bias);
        rnn_utils::append_strides(arg_list, pd()->off.diff_bias, 4);
    }
    return parallel_for(ctx, nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_t::rnn_elemwise);
template elemwise_sig(ref_rnn_bwd_t::rnn_elemwise);

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::lstm_elemwise)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;

    compute::kernel_arg_list_t arg_list;
    arg_list.append(into<int32_t>(dir));
    arg_list.append(into<int32_t>(lay));
    arg_list.append(into<int32_t>(iter));
    arg_list.append(scratch_gates);
    arg_list.append(bias);
    arg_list.append(pd()->desc()->alpha);
    // for test mode
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    auto states_t_l = workspace.states(lay + 1, dir, iter + 1);
    arg_list.append(rnn_utils::get_storage(states_t_l));

    auto c_states_t_l = workspace.c_states(lay + 1, dir, iter + 1);
    auto c_states_tm1_l = workspace.c_states(lay + 1, dir, iter);
    arg_list.append(rnn_utils::get_storage(c_states_t_l));
    arg_list.append(rnn_utils::get_storage(c_states_tm1_l));

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(rnn_utils::get_storage(gates));

    auto ws_grid = workspace.grid_comp(lay, dir, iter);
    arg_list.append(rnn_utils::get_storage(ws_grid));

    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_layer));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_gates));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_iter_scratch_gates));
    if (aprop == dnnl_forward) {
        rnn_utils::append_strides(arg_list, pd()->off.bias, 4);
    } else {
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_dir));
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_states));
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_iter));
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
    }

    arg_list.append(pd()->rnn_conf.tm_cscale);
    if (aprop != dnnl_forward) {
        arg_list.append(scratch_diff_states);
        arg_list.append(diff_bias);
        rnn_utils::append_strides(arg_list, pd()->off.diff_bias, 4);
    }
    return parallel_for(ctx, nd_range, kernel, arg_list);
}
template elemwise_sig(ref_rnn_fwd_t::lstm_elemwise);
template elemwise_sig(ref_rnn_bwd_t::lstm_elemwise);

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::lstm_elemwise_u8s8)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    compute::kernel_arg_list_t arg_list;
    arg_list.append(into<int32_t>(dir));
    arg_list.append(into<int32_t>(lay));
    arg_list.append(into<int32_t>(iter));
    arg_list.append(scratch_gates);
    arg_list.append(scales ? *scales : memory_storage_t::empty_storage());
    arg_list.append(bias);
    arg_list.append(pd()->desc()->alpha);
    arg_list.append(data_shift);
    arg_list.append(data_scale);
    // for test mode
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    auto states_t1_l = workspace.states(lay + 1, dir, iter + 1);
    arg_list.append(rnn_utils::get_storage(states_t1_l));

    auto c_states_t_l = workspace.c_states(lay + 1, dir, iter + 1);
    auto c_states_tm1_l = workspace.c_states(lay + 1, dir, iter);
    arg_list.append(rnn_utils::get_storage(c_states_t_l));
    arg_list.append(rnn_utils::get_storage(c_states_tm1_l));

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(rnn_utils::get_storage(gates));

    arg_list.append(workspace.bias());

    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_layer));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_dir));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_bias));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_iter_scratch_gates));
    arg_list.append(pd()->rnn_conf.tm_cscale);
    return parallel_for(ctx, nd_range, elemwise_fwd_kernel_, arg_list);
}
template elemwise_sig(ref_rnn_fwd_t::lstm_elemwise_u8s8);
template elemwise_sig(ref_rnn_bwd_t::lstm_elemwise_u8s8);

template <prop_kind_t aprop>
elemwise_sig_gru_lbr((_ref_rnn_common_t<aprop>::gru_lbr_elemwise)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;

    compute::kernel_arg_list_t arg_list;
    arg_list.append(into<int32_t>(dir));
    arg_list.append(into<int32_t>(lay));
    arg_list.append(into<int32_t>(iter));
    arg_list.append(scratch_gates);
    arg_list.append(bias);
    arg_list.append(pd()->desc()->alpha);
    // for test mode
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    auto states_t1_l = workspace.states(lay + 1, dir, iter + 1);
    auto states_tm1_l = workspace.states(lay + 1, dir, iter);
    arg_list.append(rnn_utils::get_storage(
            aprop == prop_kind::forward ? states_t1_l : states_tm1_l));

    auto c_states_t_l = workspace.c_states(lay + 1, dir, iter + 1);
    auto c_states_tm1_l = workspace.c_states(lay + 1, dir, iter);
    arg_list.append(rnn_utils::get_storage(c_states_t_l));
    arg_list.append(rnn_utils::get_storage(c_states_tm1_l));

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(rnn_utils::get_storage(gates));

    auto ws_grid = workspace.grid_comp(lay, dir, iter);

    arg_list.append(rnn_utils::get_storage(ws_grid));
    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_layer));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_gates));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_iter_scratch_gates));
    if (aprop == dnnl_forward) {
        rnn_utils::append_strides(arg_list, pd()->off.bias, 4);
    } else {
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_dir));
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_states));
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_iter));
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
    }

    if (aprop == dnnl_forward) {
        arg_list.append(rnn_utils::get_storage(states_tm1_l));
    }
    arg_list.append(scratch_cell);
    if (aprop != dnnl_forward) {
        arg_list.append(scratch_diff_states);
        arg_list.append(diff_bias);
        rnn_utils::append_strides(arg_list, pd()->off.diff_bias, 4);
    }
    return parallel_for(ctx, nd_range, kernel, arg_list);
}
template elemwise_sig_gru_lbr(ref_rnn_fwd_t::gru_lbr_elemwise);
template elemwise_sig_gru_lbr(ref_rnn_bwd_t::gru_lbr_elemwise);

template <prop_kind_t aprop>
elemwise_sig_gru((_ref_rnn_common_t<aprop>::gru_elemwise)) {
    auto nd_range = get_nd_range({dhc,
            utils::div_up(
                    batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    const compute::kernel_t &kernel = (aprop == prop_kind::forward)
            ? elemwise_fwd_kernel_
            : elemwise_bwd_kernel_;

    compute::kernel_arg_list_t arg_list;
    arg_list.append(into<int32_t>(dir));
    arg_list.append(into<int32_t>(lay));
    arg_list.append(into<int32_t>(iter));
    arg_list.append(scratch_gates);
    arg_list.append(bias);
    arg_list.append(pd()->desc()->alpha);
    arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    auto states_t1_l = workspace.states(lay + 1, dir, iter + 1);
    auto states_tm1_l = workspace.states(lay + 1, dir, iter);
    arg_list.append(rnn_utils::get_storage(
            aprop == prop_kind::forward ? states_t1_l : states_tm1_l));

    auto c_states_t_l = workspace.c_states(lay + 1, dir, iter + 1);
    auto c_states_tm1_l = workspace.c_states(lay + 1, dir, iter);
    arg_list.append(rnn_utils::get_storage(c_states_t_l));
    arg_list.append(rnn_utils::get_storage(c_states_tm1_l));

    auto gates = workspace.gates(lay, dir, iter);
    arg_list.append(rnn_utils::get_storage(gates));

    auto ws_grid = workspace.grid_comp(lay, dir, iter);

    arg_list.append(rnn_utils::get_storage(ws_grid));
    arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    arg_list.append(into<int32_t>(batch));
    arg_list.append(into<int32_t>(dhc));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_layer));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_gates));
    arg_list.append(into<int32_t>(pd()->rnn_conf.n_iter_scratch_gates));
    if (aprop == dnnl_forward) {
        rnn_utils::append_strides(arg_list, pd()->off.bias, 4);
    } else {
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_dir));
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_states));
        arg_list.append(into<int32_t>(pd()->rnn_conf.n_iter));
        arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
    }

    if (aprop == dnnl_forward) {
        arg_list.append(rnn_utils::get_storage(states_tm1_l));
    }
    arg_list.append(part);
    if (aprop != dnnl_forward) {
        arg_list.append(scratch_cell);
        arg_list.append(scratch_dhG1);
        arg_list.append(scratch_diff_states);
        arg_list.append(diff_bias);
        rnn_utils::append_strides(arg_list, pd()->off.diff_bias, 4);
    }
    return parallel_for(ctx, nd_range, kernel, arg_list);
}
template elemwise_sig_gru(ref_rnn_fwd_t::gru_elemwise);
template elemwise_sig_gru(ref_rnn_bwd_t::gru_elemwise);
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
