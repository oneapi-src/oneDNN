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

#include "c_types_map.hpp"

#include "ref_rnn.hpp"
#include "rnn_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace mkldnn::impl::utils;
using namespace prop_kind;
using namespace data_type;

bool rnn_utils::is_ldigo(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked)
        return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[4] == 1
            && str[3] == dims[4] && str[1] == str[2] * dims[2]
            && str[0] == str[1] * dims[1];
};

bool rnn_utils::is_ldgoi(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked)
        return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[2] == 1
            && str[3] == dims[4] * str[4] && str[1] == str[3] * dims[3]
            && str[0] == str[1] * dims[1];
};

int rnn_utils::get_good_ld(int dim, int sizeof_dt) {
    // we want matrices leading dimentions to be 64-byte aligned,
    // and not divisible by 256 to avoid 4K aliasing effects
    int ld = rnd_up(dim, 64 / sizeof_dt);
    return (ld % 256 == 0) ? ld + 64 / sizeof_dt : ld;
}

void rnn_utils::set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_states_offset, size_t &ws_c_states_offset,
        size_t &ws_diff_states_offset, size_t &ws_grid_comp_offset,
        size_t &ws_cell_comp_offset, size_t &ws_bias_offset,
        size_t &scratchpad_size, size_t &workspace_size) {
    const size_t page_size = 4096; // 2097152;
    size_t current_offset;
    size_t dt_size = rnn.dt_conf == all_f32 ? sizeof(cl_float) : sizeof(cl_half);

    // Mandatory workspaces: go to workspace if use_workspace, scratchpad
    // otherwise
    current_offset = 0; // assumes the workspace base pointer is page aligned
    ws_gates_offset = current_offset;
    current_offset += rnn.ws_gates_size * dt_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_states_offset = current_offset;
    current_offset += rnn.ws_states_size * dt_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_c_states_offset = current_offset;
    current_offset += rnn.ws_c_states_size * dt_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_diff_states_offset = current_offset;
    current_offset += rnn.ws_diff_states_size * dt_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_grid_comp_offset = current_offset;
    current_offset += rnn.ws_grid_comp_size * dt_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_cell_comp_offset = current_offset;
    current_offset += rnn.ws_cell_comp_size * dt_size;

    workspace_size = rnn.use_workspace ? current_offset : 0;

    // Optional scratchpads
    // Assumes the scratchpad base pointer is page aligned.
    // If use_workspace, the following goes to scratchpad alone,
    // otherwise, all goes to scratchpad and continue incrementing offset
    current_offset = rnn.use_workspace ? 0 : current_offset;

    if (rnn.copy_bias) {
        current_offset = utils::rnd_up(current_offset, page_size);
        ws_bias_offset = current_offset;
        current_offset += rnn.ws_bias_size * dt_size;
    }

    scratchpad_size = current_offset;
}

void rnn_utils::init_rnn_conf(rnn_conf_t &rnn_conf, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d) {
    rnn_conf.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    rnn_conf.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    rnn_conf.is_lbr = rd.cell_kind == mkldnn_lbr_gru;

    switch (rd.direction) {
    case mkldnn_unidirectional_left2right: rnn_conf.exec_dir = b2t_l2r; break;
    case mkldnn_unidirectional_right2left: rnn_conf.exec_dir = b2t_r2l; break;
    case mkldnn_bidirectional_concat: rnn_conf.exec_dir = b2t_bi_concat; break;
    case mkldnn_bidirectional_sum: rnn_conf.exec_dir = b2t_bi_sum; break;
    default: break;
    }

    if (everyone_is(f32, src_layer_d.data_type(), dst_layer_d.data_type(),
                weights_layer_d.data_type()))
        rnn_conf.dt_conf = all_f32;
    else if (everyone_is(f16, src_layer_d.data_type(), dst_layer_d.data_type(),
                weights_layer_d.data_type()))
        rnn_conf.dt_conf = all_f16;
    else
        assert(!"unsuppoted data type");

    rnn_conf.n_layer = weights_layer_d.dims()[0];
    rnn_conf.n_iter = src_layer_d.dims()[0];
    rnn_conf.n_dir = weights_layer_d.dims()[1];
    rnn_conf.n_gates = weights_layer_d.dims()[3];
    rnn_conf.n_states = rd.cell_kind == mkldnn_vanilla_lstm ? 2 : 1;
    rnn_conf.n_bias = rnn_conf.n_gates + rnn_conf.is_lbr;
    rnn_conf.mb = src_layer_d.dims()[1];

    rnn_conf.sic = weights_iter_d.dims()[2];
    rnn_conf.slc = weights_layer_d.dims()[2];
    rnn_conf.dic = weights_layer_d.dims()[4];
    rnn_conf.dlc = dst_layer_d.dims()[2];

    rnn_conf.gates_ld = rnn_conf.dic * rnn_conf.n_gates;
    rnn_conf.gates_nld = rnn_conf.mb;
    rnn_conf.states_nld = rnn_conf.mb;

    // Set the correct number of weights parts
    bool is_orig_gru = rd.cell_kind == alg_kind::vanilla_gru;
    rnn_conf.n_parts_weights_layer = 1;
    rnn_conf.parts_weights_layer[0] = rnn_conf.n_gates;
    rnn_conf.parts_weights_layer[1] = 0;

    rnn_conf.n_parts_weights_iter = is_orig_gru ? 2 : 1;
    rnn_conf.parts_weights_iter[0] = is_orig_gru ? 2 : rnn_conf.n_gates;
    rnn_conf.parts_weights_iter[1] = is_orig_gru ? 1 : 0;

    rnn_conf.n_parts_bias = 1;
    rnn_conf.parts_bias[0] = rnn_conf.n_bias;
    rnn_conf.parts_bias[1] = 0;

    rnn_conf.use_workspace = rnn_conf.is_training;

    int sizeof_states_dt
        = rnn_conf.dt_conf == all_f32 ? sizeof(cl_float) : sizeof(cl_half);
    rnn_conf.states_ws_ld = get_good_ld(nstl::max(rnn_conf.slc,
            nstl::max(rnn_conf.sic, rnn_conf.dic)), sizeof_states_dt);
}

void rnn_utils::set_rnn_conf(rnn_conf_t &rnn_conf, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d) {

    //Set leading dimensions for input weights arrays depending on input format
    auto set_dims = [&](const memory_desc_wrapper &md, int &ld, int &nld) {
        ld = 0; nld = 0;
        if (md.is_blocking_desc()) {
            if (is_ldigo(md)) {
                ld = (int)md.blocking_desc().strides[2];
                nld = md.dims()[2];
            } else if (is_ldgoi(md)) {
                ld = (int)md.blocking_desc().strides[4];
                nld = md.dims()[3] * md.dims()[4];
            } else
                assert(!"unsupported weights format");
        }
    };
    set_dims(weights_layer_d, rnn_conf.weights_layer_ld,
        rnn_conf.weights_layer_nld);
    set_dims(weights_iter_d, rnn_conf.weights_iter_ld,
        rnn_conf.weights_iter_nld);
    if (!rnn_conf.is_fwd) {
        set_dims(diff_weights_layer_d, rnn_conf.diff_weights_layer_ld,
                rnn_conf.diff_weights_layer_nld);
        set_dims(diff_weights_iter_d, rnn_conf.diff_weights_iter_ld,
                rnn_conf.diff_weights_iter_nld);
    }

    rnn_conf.gates_ws_ld = get_good_ld(rnn_conf.gates_ld,
        rnn_conf.dt_conf == all_f32 ? sizeof(cl_float) : sizeof(cl_half));

    // Set workspace sizes to store:
    // states to copmute a pass
    // diff states to copmute bwd pass (training only)
    // intermediate results from the gates

    rnn_conf.use_workspace = rnn_conf.is_training;
    rnn_conf.ws_states_size = (size_t)(rnn_conf.n_layer + 1) * rnn_conf.n_dir
        * (rnn_conf.n_iter + 1) * rnn_conf.n_states * rnn_conf.mb
        * rnn_conf.states_ws_ld;
    rnn_conf.ws_c_states_size = 0;
    rnn_conf.ws_diff_states_size = rnn_conf.is_training
        ? (size_t)(rnn_conf.n_layer + 1) * rnn_conf.n_dir * (rnn_conf.n_iter + 1)
          * (rnn_conf.n_states + 1) * rnn_conf.mb * rnn_conf.states_ws_ld
        : (size_t)0;
    rnn_conf.ws_gates_size = (size_t)rnn_conf.n_layer * rnn_conf.n_dir
        * rnn_conf.n_iter * rnn_conf.mb * rnn_conf.gates_ws_ld;

    // set other sizes
    rnn_conf.ws_per_cell = (size_t)rnn_conf.is_lbr * rnn_conf.mb * rnn_conf.dic;
    rnn_conf.ws_cell_comp_size
        = rnn_conf.is_lbr || rnn_conf.dt_conf != all_f32
            ? (size_t) rnn_conf.gates_nld * rnn_conf.gates_ws_ld : 0;
    rnn_conf.ws_grid_comp_size = (size_t)rnn_conf.is_lbr * rnn_conf.is_training
        * rnn_conf.n_layer * rnn_conf.n_dir * rnn_conf.n_iter
        * rnn_conf.ws_per_cell;
    rnn_conf.ws_bias_size = (size_t)rnn_conf.n_layer * rnn_conf.n_dir
        * rnn_conf.n_bias * rnn_conf.dic;
}

void rnn_utils::get_scratchpad_and_workspace_sizes(const rnn_conf_t &rnn,
        size_t &scratchpad_size, size_t &workspace_size) {
    size_t ws_gates_offset, ws_states_offset, ws_c_states_offset,
            ws_diff_states_offset, ws_grid_comp_offset, ws_cell_comp_offset,
            ws_bias_offset;
    set_offsets(rnn, ws_gates_offset, ws_states_offset, ws_diff_states_offset,
            ws_c_states_offset, ws_grid_comp_offset, ws_cell_comp_offset,
            ws_bias_offset, scratchpad_size, workspace_size);
}

status_t rnn_utils::set_good_strides(
        memory_desc_t &weights_md, format_tag_t tag) {
    auto &strides = weights_md.format_desc.blocking.strides;
    auto dims = weights_md.dims;
    using namespace format_tag;

    if (tag == ldigo) {
        strides[2] = rnn_utils::get_good_ld((int)strides[2],
                (int)types::data_type_size(weights_md.data_type));
        strides[1] = dims[2] * strides[2];
        strides[0] = dims[1] * strides[1];
    } else if (tag == ldgoi) {
        strides[4] = rnn_utils::get_good_ld((int)strides[4],
                (int)types::data_type_size(weights_md.data_type));
        strides[3] = dims[4] * strides[4];
        strides[1] = dims[3] * strides[3];
        strides[0] = dims[1] * strides[1];
    } else
        return status::unimplemented;

    return status::success;
}

status_t rnn_utils::set_expected_desc(rnn_conf_t &rnn,
        memory_desc_t &weights_md, bool is_iter) {
    using namespace format_tag;
    bool use_packed_gemm = false;
    if (use_packed_gemm) {
        // TBD
    } else {
        CHECK(memory_desc_init_by_tag(weights_md, rnn.is_fwd ? ldigo : ldgoi));
        // Adjust strides for good leading dimension in GEMM
        CHECK(set_good_strides(weights_md, rnn.is_fwd ? ldigo : ldgoi));
    }
    return status::success;
}

}
}
}
