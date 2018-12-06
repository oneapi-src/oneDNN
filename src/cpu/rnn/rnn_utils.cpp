/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "ref_rnn.hpp"
#include "rnn_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace rnn_utils;
using namespace memory_format;

void rnn_utils::init_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d,
        const memory_desc_wrapper &dst_layer_d) {
    rnn.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    rnn.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    rnn.is_lbr = rd.cell_desc.cell_kind == mkldnn_gru_linear_before_reset;

    switch (rd.direction) {
    case mkldnn_unidirectional_left2right: rnn.exec_dir = l2r; break;
    case mkldnn_unidirectional_right2left: rnn.exec_dir = r2l; break;
    case mkldnn_bidirectional_concat: rnn.exec_dir = bi_concat; break;
    case mkldnn_bidirectional_sum: rnn.exec_dir = bi_sum; break;
    default: break;
    }

    rnn.n_layer = weights_layer_d.dims()[0];
    rnn.n_iter = src_layer_d.dims()[0];
    rnn.n_dir = weights_layer_d.dims()[1];
    rnn.n_gates = weights_layer_d.dims()[3];
    rnn.n_states = mkldnn_rnn_cell_get_states_count(&rd.cell_desc);
    rnn.n_bias = rnn.n_gates + rnn.is_lbr;
    rnn.mb = src_layer_d.dims()[1];
    rnn.sic = weights_iter_d.dims()[2];
    rnn.slc = weights_layer_d.dims()[2];
    rnn.dic = weights_layer_d.dims()[4];
    rnn.dlc = dst_layer_d.dims()[2];

    rnn.gates_ld = rnn.dic * rnn.n_gates;
    rnn.gates_nld = rnn.mb;
    rnn.states_nld = rnn.mb;

    /* Set leading dimensions for input weights arrays depending on input format
     */
    auto set_weights_dims = [&](bool is_igo, int ic, int &ld, int &nld) {
        if (is_igo) {
            ld = rnn.dic * rnn.n_gates;
            nld = ic;
        } else {
            ld = ic;
            nld = rnn.dic * rnn.n_gates;
        }
    };
    rnn.weights_layer_fmt = weights_layer_d.format();
    rnn.weights_iter_fmt = weights_iter_d.format();
    set_weights_dims(one_of(rnn.weights_layer_fmt, ldigo, ldigo_p), rnn.slc,
            rnn.weights_layer_ld, rnn.weights_layer_nld);
    set_weights_dims(one_of(rnn.weights_iter_fmt, ldigo, ldigo_p), rnn.sic,
            rnn.weights_iter_ld, rnn.weights_iter_nld);
    if (!rnn.is_fwd) {
        rnn.diff_weights_layer_fmt = diff_weights_layer_d.format();
        rnn.diff_weights_iter_fmt = diff_weights_iter_d.format();
        set_weights_dims(one_of(rnn.diff_weights_layer_fmt, ldigo, ldigo_p), rnn.slc,
                rnn.diff_weights_layer_ld, rnn.diff_weights_layer_nld);
        set_weights_dims(one_of(rnn.diff_weights_iter_fmt, ldigo, ldigo_p), rnn.sic,
                rnn.diff_weights_iter_ld, rnn.diff_weights_iter_nld);
    }

    /* Set the correct number of weights parts */
    bool is_orig_gru = rd.cell_desc.cell_kind == alg_kind::vanilla_gru;
    rnn.n_parts_weights_layer = 1;
    rnn.parts_weights_layer[0] = rnn.n_gates;
    rnn.parts_weights_layer[1] = 0;

    rnn.n_parts_weights_iter = is_orig_gru ? 2 : 1;
    rnn.parts_weights_iter[0] = is_orig_gru ? 2 : rnn.n_gates;
    rnn.parts_weights_iter[1] = is_orig_gru ? 1 : 0;

    rnn.n_parts_bias = 1;
    rnn.parts_bias[0] = rnn.n_bias;
    rnn.parts_bias[1] = 0;
    /* Decide to copy weights to workspace with padded leading dimension */
    auto decide_to_copy
            = [](bool copy_enabled, int ld, int &wld, bool &copy) {
                  if (copy_enabled && ld != get_good_ld(ld)) {
                      copy = true;
                      wld = get_good_ld(ld);
                  } else {
                      copy = false;
                      wld = ld;
                  }
              };
    bool weights_copy_enabled = rnn.n_iter > 1;
    decide_to_copy(weights_copy_enabled, rnn.weights_layer_ld,
            rnn.weights_layer_ws_ld, rnn.copy_weights_layer);
    decide_to_copy(weights_copy_enabled, rnn.weights_iter_ld,
            rnn.weights_iter_ws_ld, rnn.copy_weights_iter);
    decide_to_copy(weights_copy_enabled && !rnn.is_fwd, rnn.diff_weights_iter_ld,
            rnn.diff_weights_iter_ws_ld, rnn.copy_diff_weights_iter);
    decide_to_copy(weights_copy_enabled && !rnn.is_fwd, rnn.diff_weights_layer_ld,
            rnn.diff_weights_layer_ws_ld, rnn.copy_diff_weights_layer);
    rnn.states_ws_ld
            = get_good_ld(nstl::max(rnn.slc, nstl::max(rnn.sic, rnn.dic)));
    rnn.gates_ws_ld = get_good_ld(rnn.gates_ld);

    /* Decide wich gemm implementation to use: packed/nonpacked jit/cblas
     * and if to mergre gemm across iterations */
    rnn.merge_gemm_layer = (rnn.is_fwd && rnn.mb < 128) || !rnn.is_fwd;
    bool is_gru = utils::one_of(rd.cell_desc.cell_kind, alg_kind::vanilla_gru,
            alg_kind::gru_linear_before_reset);
    rnn.merge_gemm_iter = !(rnn.is_fwd || is_gru);
    bool is_inference = !rnn.is_training;
    rnn.use_jit_gemm = !mayiuse(avx512_mic)
            && ((is_inference && (rnn.n_layer > 1 || rnn.mb < 100))
                || (rnn.is_training && rnn.dic < 500));
#ifdef USE_MKL_PACKED_GEMM
    rnn.use_packed_gemm = weights_copy_enabled && (rnn.mb == 32) && (rnn.sic == 512)
            && (rnn.slc == 512) && (rnn.dic == 512);
#else
    rnn.use_packed_gemm = false;
#endif

    /* Set workspace sizes to store:
     * states to copmute a pass
     * diff states to copmute bwd pass (training only)
     * intermediate results from the gates
     * weights (only if copy to padded leading dimension is enables and
     * required)
     */
    rnn.use_workspace = rnn.is_training;
    rnn.ws_states_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.n_states * rnn.mb * rnn.states_ws_ld;
    rnn.ws_diff_states_size = rnn.is_training ?
            (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1)
                    * (rnn.n_states + 1) * rnn.mb * rnn.states_ws_ld :
            (size_t)0;
    rnn.ws_gates_size = (size_t)rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.mb
            * rnn.gates_ws_ld;

    /* Set the sizes in case we pack the weights */
    /* Set the sizes in case we pack the weights layer */
    size_t weights_layer_gld_size = (size_t)rnn.n_layer * rnn.n_dir
        * rnn.weights_layer_nld * rnn.weights_layer_ws_ld;
    size_t weights_layer_pack_size = 0;
    {
    bool is_igo = one_of(rnn.weights_layer_fmt, ldigo, ldigo_p);
    for(int p=0; p < rnn.n_parts_weights_layer; p++) {
        int m_p = is_igo ? (rnn.parts_weights_layer[p] * rnn.dic) : rnn.slc;
        int k_p = is_igo ? rnn.slc : (rnn.parts_weights_layer[p] * rnn.dic);
        int n_p = rnn.mb;

#if USE_MKL_PACKED_GEMM
        rnn.part_weights_layer_pack_size[p] = (size_t) rnn.n_layer * rnn.n_dir *
            cblas_sgemm_pack_get_size(CblasAMatrix, m_p, n_p, k_p);
#else
        UNUSED(m_p);
        UNUSED(k_p);
        UNUSED(n_p);
        rnn.part_weights_layer_pack_size[p] = 0;
#endif
        weights_layer_pack_size += rnn.part_weights_layer_pack_size[p];
    }
    }
    rnn.ws_weights_layer_size = rnn.use_packed_gemm && rnn.copy_weights_layer
        ? weights_layer_pack_size : weights_layer_gld_size;

    size_t weights_iter_gld_size = (size_t)rnn.n_iter * rnn.n_dir
        * rnn.weights_iter_nld * rnn.weights_iter_ws_ld;

    int weights_iter_pack_size = 0;
    {
    bool is_igo = one_of(rnn.weights_iter_fmt, ldigo, ldigo_p);
    for(int p=0; p < rnn.n_parts_weights_iter; p++) {
        int m_p = is_igo ? (rnn.parts_weights_iter[p] * rnn.dic) : rnn.sic;
        int k_p = is_igo ? rnn.sic : (rnn.parts_weights_iter[p] * rnn.dic);
        int n_p = rnn.mb;

#if USE_MKL_PACKED_GEMM
        rnn.part_weights_iter_pack_size[p] = (size_t) rnn.n_iter * rnn.n_dir *
            cblas_sgemm_pack_get_size(CblasAMatrix, m_p, n_p, k_p);
#else
        UNUSED(m_p);
        UNUSED(k_p);
        UNUSED(n_p);
        rnn.part_weights_iter_pack_size[p] = 0;
#endif
        weights_iter_pack_size += rnn.part_weights_iter_pack_size[p];
    }
    }
    rnn.ws_weights_iter_size = rnn.use_packed_gemm && rnn.copy_weights_iter
        ? weights_iter_pack_size : weights_iter_gld_size;

    /* set other sizes */
    rnn.ws_diff_weights_layer_size
            = rnn.is_fwd ? (size_t)0 : (size_t)rnn.n_layer * rnn.n_dir
                    * rnn.diff_weights_layer_nld * rnn.diff_weights_layer_ws_ld;
    rnn.ws_diff_weights_iter_size = rnn.is_fwd ? (size_t)0 : (size_t)rnn.n_layer
                    * rnn.n_dir * rnn.diff_weights_iter_nld * rnn.diff_weights_iter_ws_ld;
    rnn.ws_per_cell = (size_t)rnn.is_lbr * rnn.mb * rnn.dic;
    rnn.ws_cell_comp_size
            = (size_t)rnn.is_lbr * rnn.gates_nld * rnn.gates_ws_ld;
    rnn.ws_grid_comp_size = (size_t)rnn.is_lbr * rnn.is_training * rnn.n_layer
            * rnn.n_dir * rnn.n_iter * rnn.ws_per_cell;
}

int rnn_utils::get_good_ld(int dim) {
    // we want matrices leading dimentions to be 64-byte aligned,
    // and not divisible by 256 to avoid 4K aliasing effects
    int ld = rnd_up(dim, (int)(64 / sizeof(float)));
    return (ld % 256 == 0) ? ld + 64 / sizeof(float) : ld;
}

void rnn_utils::set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_states_offset, size_t &ws_diff_states_offset,
        size_t &ws_grid_comp_offset, size_t &ws_cell_comp_offset,
        size_t &ws_weights_layer_offset, size_t &ws_weights_iter_offset,
        size_t &ws_bias_offset, size_t &ws_diff_weights_layer_offset,
        size_t &ws_diff_weights_iter_offset, size_t &scratchpad_size,
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
    ws_diff_states_offset = current_offset;
    current_offset += rnn.ws_diff_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_grid_comp_offset = current_offset;
    current_offset += rnn.ws_grid_comp_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_cell_comp_offset = current_offset;
    current_offset += rnn.ws_cell_comp_size;

    workspace_size = rnn.use_workspace ? current_offset : 0;

    /* Optional scratchpads */
    // Assumes the scratchpad base pointer is page aligned.
    // If use_workspace, the following goes to scratchpad alone,
    // otherwise, all goes to scratchpad and continue incrementing offset
    current_offset = rnn.use_workspace ? 0 : current_offset;

    if (rnn.copy_weights_layer) {
        current_offset = utils::rnd_up(current_offset, page_size);
        ws_weights_layer_offset = current_offset;
        current_offset += rnn.ws_weights_layer_size;
    }

    if (rnn.copy_weights_iter) {
        current_offset = utils::rnd_up(current_offset, page_size);
        ws_weights_iter_offset = current_offset;
        current_offset += rnn.ws_weights_iter_size;
    }

    if (rnn.copy_diff_weights_layer) {
        current_offset = utils::rnd_up(current_offset, page_size);
        ws_diff_weights_layer_offset = current_offset;
        current_offset += rnn.ws_diff_weights_layer_size;
    }

    if (rnn.copy_diff_weights_iter) {
        current_offset = utils::rnd_up(current_offset, page_size);
        ws_diff_weights_iter_offset = current_offset;
        current_offset += rnn.ws_diff_weights_iter_size;
    }
    scratchpad_size = current_offset;
}

void rnn_utils::get_scratchpad_and_workspace_sizes(const rnn_conf_t &rnn,
        size_t &scratchpad_size, size_t &workspace_size) {
    size_t ws_gates_offset, ws_states_offset, ws_diff_states_offset,
            ws_grid_comp_offset, ws_cell_comp_offset, ws_weights_layer_offset,
            ws_weights_iter_offset, ws_bias_offset, ws_diff_weights_layer_offset,
            ws_diff_weights_iter_offset;
    set_offsets(rnn, ws_gates_offset, ws_states_offset, ws_diff_states_offset,
            ws_grid_comp_offset, ws_cell_comp_offset, ws_weights_layer_offset,
            ws_weights_iter_offset, ws_bias_offset, ws_diff_weights_layer_offset,
            ws_diff_weights_iter_offset, scratchpad_size, workspace_size);
}

}
}
}
