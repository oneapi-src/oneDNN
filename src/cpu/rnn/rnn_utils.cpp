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

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "math_utils.hpp"

#include "gemm/gemm_pack.hpp"
#include "ref_rnn.hpp"
#include "rnn.hpp"
#include "rnn_utils.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace rnn_utils;
using namespace format_tag;
using namespace rnn_packed_format;
using namespace data_type;

bool rnn_utils::is_ldigo(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked) return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[4] == 1
            && str[3] == dims[4] && str[1] == str[2] * dims[2]
            && str[0] == str[1] * dims[1];
};

bool rnn_utils::is_ldgoi(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked) return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[2] == 1
            && str[3] == dims[4] * str[4] && str[1] == str[3] * dims[3]
            && str[0] == str[1] * dims[1];
};

bool rnn_utils::init_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &src_iter_c_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &dst_iter_c_d) {
    rnn.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    rnn.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    rnn.is_lbr = rd.cell_kind == dnnl_lbr_gru;
    rnn.is_lstm_peephole = rd.cell_kind == dnnl_vanilla_lstm
            && !memory_desc_wrapper(rd.weights_peephole_desc).is_zero();

    switch (rd.direction) {
        case dnnl_unidirectional_left2right: rnn.exec_dir = l2r; break;
        case dnnl_unidirectional_right2left: rnn.exec_dir = r2l; break;
        case dnnl_bidirectional_concat: rnn.exec_dir = bi_concat; break;
        case dnnl_bidirectional_sum: rnn.exec_dir = bi_sum; break;
        default: break;
    }

    if (everyone_is(f32, src_layer_d.data_type(), dst_layer_d.data_type(),
                weights_layer_d.data_type()))
        rnn.dt_conf = all_f32;
    else if (everyone_is(bf16, src_layer_d.data_type(), dst_layer_d.data_type(),
                     weights_layer_d.data_type())) {
        if (!mayiuse(avx512_core)) return false;
        rnn.dt_conf = all_bf16;
    } else if (dst_layer_d.data_type() == u8) {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            rnn.dt_conf = u8u8u8u8;
        else
            rnn.dt_conf = f32u8f32u8;
    } else {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            rnn.dt_conf = u8u8u8f32;
        else
            rnn.dt_conf = f32u8f32f32;
    }

    // Set problem members defining problem sizes
    rnn.n_layer = weights_layer_d.dims()[0];
    rnn.n_iter = src_layer_d.dims()[0];
    rnn.n_dir = weights_layer_d.dims()[1];
    rnn.n_gates = weights_layer_d.dims()[3];
    rnn.n_states = rd.cell_kind == dnnl_vanilla_lstm ? 2 : 1;
    rnn.n_bias = rnn.n_gates + rnn.is_lbr;
    rnn.mb = src_layer_d.dims()[1];
    rnn.sic = weights_iter_d.dims()[2];
    rnn.slc = weights_layer_d.dims()[2];
    rnn.dic = weights_layer_d.dims()[4];
    rnn.dlc = dst_layer_d.dims()[2];

    // set workspace (not)leading dimensions
    rnn.gates_ld = rnn.dic * rnn.n_gates;
    rnn.gates_nld = rnn.mb;
    rnn.states_nld = rnn.mb;

    // set members with user memories leading dimensions
    // Assumption: weights datatype size is the same as state datatype size
    int sizeof_states_dt = types::data_type_size(weights_layer_d.data_type());
    rnn.states_ws_ld = get_good_ld(
            nstl::max(rnn.slc, nstl::max(rnn.sic, rnn.dic)), sizeof_states_dt);
    // Assumption: {src,dst}_layer has tnc layout, {src,dst}_iter has ldnc,
    rnn.src_layer_ld_ = src_layer_d.blocking_desc().strides[1];
    rnn.dst_layer_ld_ = dst_layer_d.blocking_desc().strides[1];
    rnn.src_iter_ld_ = types::is_zero_md(src_iter_d.md_)
            ? 0
            : src_iter_d.blocking_desc().strides[2];
    rnn.dst_iter_ld_ = types::is_zero_md(dst_iter_d.md_)
            ? 0
            : dst_iter_d.blocking_desc().strides[2];
    rnn.src_iter_c_ld_ = types::is_zero_md(src_iter_c_d.md_)
            ? 0
            : src_iter_c_d.blocking_desc().strides[2];
    rnn.dst_iter_c_ld_ = types::is_zero_md(dst_iter_c_d.md_)
            ? 0
            : dst_iter_c_d.blocking_desc().strides[2];

    /* Set the correct number of weights parts */
    bool is_orig_gru = rd.cell_kind == alg_kind::vanilla_gru;
    rnn.n_parts_weights_layer = 1;
    rnn.parts_weights_layer[0] = rnn.n_gates;
    rnn.parts_weights_layer[1] = 0;

    rnn.n_parts_weights_iter = is_orig_gru ? 2 : 1;
    rnn.parts_weights_iter[0] = is_orig_gru ? 2 : rnn.n_gates;
    rnn.parts_weights_iter[1] = is_orig_gru ? 1 : 0;

    rnn.n_parts_bias = 1;
    rnn.parts_bias[0] = rnn.n_bias;
    rnn.parts_bias[1] = 0;

    /* Decide wich gemm implementation to use: packed/nonpacked jit/cblas
     * and if to mergre gemm across iterations */
    bool is_f32 = rnn.dt_conf == all_f32, is_bf16 = rnn.dt_conf == all_bf16;
    bool is_gru = utils::one_of(
            rd.cell_kind, alg_kind::vanilla_gru, alg_kind::lbr_gru);
    bool is_inference = !rnn.is_training;

    // To be able to merge the GEMM on the layer input when not
    // copying, we need to have a trivial stride for the T dimension
    auto src_layer_is_trivial_stride = src_layer_d.blocking_desc().strides[0]
            == (rnn.src_layer_ld_ * rnn.mb);
    auto dst_layer_is_trivial_stride = dst_layer_d.blocking_desc().strides[0]
            == (rnn.dst_layer_ld_ * rnn.mb);

    rnn.merge_gemm_layer = ((rnn.is_fwd && src_layer_is_trivial_stride)
                                   || ((rd.prop_kind == prop_kind::backward)
                                           && dst_layer_is_trivial_stride))
            && (((rnn.is_fwd && rnn.mb < 128) || !rnn.is_fwd) || rnn.is_int8());
    rnn.merge_gemm_iter
            = dst_layer_is_trivial_stride && !(rnn.is_fwd || is_gru);
    rnn.force_nocopy = !mayiuse(avx512_mic) && mayiuse(avx)
            && ((is_inference && (rnn.n_layer > 1 || rnn.mb < 100))
                    || (rnn.is_training && rnn.dic < 500));

    /* Decide to copy bias */
    rnn.copy_bias = rnn.is_int8();

    rnn.use_layer_packed_gemm
            = utils::one_of(weights_layer_d.format_kind(), format_kind::any,
                      format_kind::rnn_packed)
            && is_inference
            && ((is_f32 && pack_sgemm_supported() && rnn.n_iter == 1)
                    || rnn.is_int8() || is_bf16);
    rnn.use_iter_packed_gemm
            = utils::one_of(weights_iter_d.format_kind(), format_kind::any,
                      format_kind::rnn_packed)
            && is_inference
            && ((is_f32 && pack_sgemm_supported() && rnn.mb >= 16)
                    || rnn.is_int8() || is_bf16);

    /* Set packed gemm sizes */
    /* TODO: investigate the benefit of mixing packed and non-packed weights parts */
    auto set_pack_sizes
            = [&](bool merge, bool &do_pack, size_t &weights_pack_size,
                      int &n_parts, int *parts, size_t *parts_pack_size,
                      size_t &comp_offset, int feature_size) -> bool {
        bool pack = true;
        weights_pack_size = 0;
        for (int p = 0; p < n_parts; p++) {
            int m_p = rnn.is_fwd ? (parts[p] * rnn.dic) : feature_size;
            int k_p = rnn.is_fwd ? feature_size : (parts[p] * rnn.dic);
            int n_p = merge ? rnn.mb * rnn.n_iter : rnn.mb;
            bool pack_part = true;

            dnnl_status_t st = dnnl_success;
            switch (rnn.dt_conf) {
                case all_f32:
                    st = sgemm_pack_get_size("A", "N", "N", &m_p, &n_p, &k_p,
                            &m_p, &rnn.states_ws_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                case u8u8u8f32:
                case f32u8f32f32:
                case u8u8u8u8:
                case f32u8f32u8:
                    st = gemm_s8u8s32_pack_get_size("A", "N", "N", &m_p, &n_p,
                            &k_p, &m_p, &rnn.states_ws_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                case all_bf16:
                    st = gemm_bf16bf16f32_pack_get_size("A", "N", "N", &m_p,
                            &n_p, &k_p, &m_p, &rnn.states_ws_ld,
                            &parts_pack_size[p], &pack_part);
                    break;
                default: assert(!"Unsupported configuration");
            }
            if (st != dnnl_success) return false;

            pack = pack && pack_part;
            weights_pack_size += rnn.n_layer * rnn.n_dir * parts_pack_size[p];
        }

        // NOTE: pack is updated only for f32. We force pack for int8
        do_pack = (rnn.dt_conf == all_f32) ? pack : true;
        comp_offset = weights_pack_size;
        const bool need_compensation = rnn.is_int8();
        weights_pack_size += (need_compensation ? rnn.n_layer * rnn.n_dir : 0)
                * rnn.n_gates * rnn.dlc * sizeof(float);

        return true;
    };

    if (rnn.use_layer_packed_gemm) {
        bool ok = set_pack_sizes(rnn.merge_gemm_layer,
                rnn.use_layer_packed_gemm, rnn.weights_layer_pack_size,
                rnn.n_parts_weights_layer, rnn.parts_weights_layer,
                rnn.part_weights_layer_pack_size, rnn.weights_layer_comp_offset,
                rnn.slc);
        if (!ok) return false;
    }

    if (rnn.use_iter_packed_gemm) {
        bool ok = set_pack_sizes(rnn.merge_gemm_iter, rnn.use_iter_packed_gemm,
                rnn.weights_iter_pack_size, rnn.n_parts_weights_iter,
                rnn.parts_weights_iter, rnn.part_weights_iter_pack_size,
                rnn.weights_iter_comp_offset, rnn.sic);
        if (!ok) return false;
    }

    return true;
}

void rnn_utils::set_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d) {

    /* Set leading dimensions for input weights arrays depending on input format
     */
    auto set_dims = [&](const memory_desc_wrapper &md, int &ld, int &nld) {
        ld = 0;
        nld = 0;
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
    set_dims(weights_layer_d, rnn.weights_layer_ld, rnn.weights_layer_nld);
    set_dims(weights_iter_d, rnn.weights_iter_ld, rnn.weights_iter_nld);
    if (!rnn.is_fwd) {
        set_dims(diff_weights_layer_d, rnn.diff_weights_layer_ld,
                rnn.diff_weights_layer_nld);
        set_dims(diff_weights_iter_d, rnn.diff_weights_iter_ld,
                rnn.diff_weights_iter_nld);
    }

    assert(weights_layer_d.data_type() == weights_iter_d.data_type());
    assert(IMPLICATION(diff_weights_layer_d.ndims() != 0,
            (diff_weights_layer_d.data_type()
                    == diff_weights_iter_d.data_type())));
    // Here we assume that the weights type size is the same as the input type size
    int sizeof_states_dt = weights_layer_d.data_type_size();
    // Here we assume that we always use 32 bits for accumulation
    int sizeof_acc_dt = sizeof(float);
    // bounded by size of float, TODO: use the proper data_type
    int sizeof_scratch_dt = sizeof(float);

    rnn.gates_ws_ld = get_good_ld(rnn.gates_ld, sizeof_states_dt);

    /* Set workspace sizes to store:
     * states to copmute a pass
     * diff states to copmute bwd pass (training only)
     * intermediate results from the gates
     */
    rnn.use_workspace = rnn.is_training;
    rnn.ws_states_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.mb * rnn.states_ws_ld * sizeof_states_dt;
    bool is_lstm = rd.cell_kind == dnnl_vanilla_lstm;
    rnn.ws_c_states_size = is_lstm
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.states_ws_ld * sizeof(float)
            : 0;
    rnn.ws_diff_states_size = rnn.is_training ? (size_t)(rnn.n_layer + 1)
                    * rnn.n_dir * (rnn.n_iter + 1) * (rnn.n_states + 1) * rnn.mb
                    * rnn.states_ws_ld * sizeof_acc_dt
                                              : (size_t)0;
    rnn.ws_gates_size = rnn.is_training ? (size_t)rnn.n_layer * rnn.n_dir
                    * rnn.n_iter * rnn.mb * rnn.gates_ws_ld * sizeof_states_dt
                                        : (size_t)0;
    rnn.n_iter_scratch_gates
            = (rnn.merge_gemm_layer || rnn.merge_gemm_iter) ? rnn.n_iter : 1;
    rnn.scratch_gates_size = rnn.n_iter_scratch_gates * rnn.gates_nld
            * rnn.gates_ws_ld * sizeof_scratch_dt;

    /* set other sizes */
    /// scratchpad buffer for each cell to hold intermediate data in gru/lbr_gru
    rnn.scratch_cell_size = rnn.is_lbr
            ? (size_t)rnn.gates_nld * rnn.gates_ws_ld * sizeof_acc_dt
            : (rd.cell_kind == alg_kind::vanilla_gru ? (size_t)rnn.states_nld
                                    * rnn.states_ws_ld * sizeof_acc_dt
                                                     : 0);
    /// workspace needed for lbr GRU
    rnn.ws_per_cell = (size_t)rnn.is_lbr * rnn.mb * rnn.dic * sizeof_acc_dt;
    rnn.ws_grid_comp_size = (size_t)rnn.is_lbr * rnn.is_training * rnn.n_layer
            * rnn.n_dir * rnn.n_iter * rnn.ws_per_cell * sizeof(float);
    /// bias ws needed to add compensation in int8
    rnn.ws_bias_size = (size_t)rnn.n_layer * rnn.n_dir * rnn.n_bias * rnn.dic
            * sizeof(float);
}

int rnn_utils::get_good_ld(int dim, int sizeof_dt) {
    // we want matrices leading dimentions to be 64-byte aligned,
    // and not divisible by 256 to avoid 4K aliasing effects
    int ld = rnd_up(dim, 64 / sizeof_dt);
    return (ld % 256 == 0) ? ld + 64 / sizeof_dt : ld;
}

void rnn_utils::set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_states_offset, size_t &ws_c_states_offset,
        size_t &ws_diff_states_offset, size_t &ws_grid_comp_offset,
        size_t &ws_bias_offset, size_t &scratch_gates_offset,
        size_t &scratch_cell_offset, size_t &scratchpad_size,
        size_t &workspace_size) {

    const size_t page_size = 4096; // 2097152;
    size_t current_offset;
    /* Mandatory workspaces: go to workspace if use_workspace, scratchpad
     * otherwise */
    current_offset = 0; // assumes the workspace base pointer is page aligned
    ws_gates_offset = current_offset;
    current_offset += rnn.ws_gates_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_states_offset = current_offset;
    current_offset += rnn.ws_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_c_states_offset = current_offset;
    current_offset += rnn.ws_c_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_diff_states_offset = current_offset;
    current_offset += rnn.ws_diff_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_grid_comp_offset = current_offset;
    current_offset += rnn.ws_grid_comp_size;

    workspace_size = rnn.use_workspace ? current_offset : 0;

    /* Optional scratchpads */
    // Assumes the scratchpad base pointer is page aligned.
    // If use_workspace, the following goes to scratchpad alone,
    // otherwise, all goes to scratchpad and continue incrementing offset
    current_offset = rnn.use_workspace ? 0 : current_offset;

    current_offset = utils::rnd_up(current_offset, page_size);
    scratch_gates_offset = current_offset;
    current_offset += rnn.scratch_gates_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    scratch_cell_offset = current_offset;
    current_offset += rnn.scratch_cell_size;

    if (rnn.copy_bias) {
        current_offset = utils::rnd_up(current_offset, page_size);
        ws_bias_offset = current_offset;
        current_offset += rnn.ws_bias_size;
    }

    scratchpad_size = current_offset;
}

void rnn_utils::get_scratchpad_and_workspace_sizes(const rnn_conf_t &rnn,
        size_t &scratchpad_size, size_t &workspace_size) {
    size_t ws_gates_offset, ws_states_offset, ws_c_states_offset,
            ws_diff_states_offset, ws_grid_comp_offset, scratch_gates_offset,
            scratch_cell_offset, ws_bias_offset;
    set_offsets(rnn, ws_gates_offset, ws_states_offset, ws_diff_states_offset,
            ws_c_states_offset, ws_grid_comp_offset, ws_bias_offset,
            scratch_gates_offset, scratch_cell_offset, scratchpad_size,
            workspace_size);
}

status_t rnn_utils::set_good_strides(
        memory_desc_t &weights_md, format_tag_t tag) {
    auto &strides = weights_md.format_desc.blocking.strides;
    auto dims = weights_md.dims;

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

status_t rnn_utils::set_expected_desc(
        rnn_conf_t &rnn, memory_desc_t &weights_md, bool is_iter) {
    using namespace format_tag;
    bool use_packed_gemm
            = is_iter ? rnn.use_iter_packed_gemm : rnn.use_layer_packed_gemm;
    if (use_packed_gemm) {
        weights_md.format_kind = format_kind::rnn_packed;
        rnn_packed_desc_t &rnn_pdata = weights_md.format_desc.rnn_packed_desc;
        rnn_pdata.format = rnn.is_fwd ? dnnl_ldigo_p : dnnl_ldgoi_p;
        rnn_pdata.ldb = rnn.states_ws_ld;
        if (is_iter) {
            rnn_pdata.n = rnn.mb;
            rnn_pdata.n_parts = rnn.n_parts_weights_iter;
            array_copy(rnn_pdata.parts, rnn.parts_weights_iter,
                    DNNL_RNN_MAX_N_PARTS);
            array_copy(rnn_pdata.part_pack_size,
                    rnn.part_weights_iter_pack_size, DNNL_RNN_MAX_N_PARTS);
            rnn_pdata.offset_compensation = rnn.weights_iter_comp_offset;
            rnn_pdata.size = rnn.weights_iter_pack_size;
        } else {
            rnn_pdata.n = rnn.merge_gemm_layer ? rnn.n_iter * rnn.mb : rnn.mb;
            rnn_pdata.n_parts = rnn.n_parts_weights_layer;
            array_copy(rnn_pdata.parts, rnn.parts_weights_layer,
                    DNNL_RNN_MAX_N_PARTS);
            array_copy(rnn_pdata.part_pack_size,
                    rnn.part_weights_layer_pack_size, DNNL_RNN_MAX_N_PARTS);
            rnn_pdata.offset_compensation = rnn.weights_layer_comp_offset;
            rnn_pdata.size = rnn.weights_layer_pack_size;
        }
    } else {
        CHECK(memory_desc_init_by_tag(weights_md, rnn.is_fwd ? ldigo : ldgoi));
        // Adjust strides for good leading dimension in GEMM
        CHECK(set_good_strides(weights_md, rnn.is_fwd ? ldigo : ldgoi));
    }
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
