/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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
 * Common for RNN and LSTM cell execution
 */
#include "ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
using namespace rnn_utils;

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::cell_execution)) {
    auto src_layer_ld = rnn.src_layer_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);

    if (rnn.need_gemm_layer(cell_position)) {
        (this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dic, rnn.mb,
                rnn.slc, 1.0, w_layer_[0], rnn.weights_layer_ld, states_t_lm1_,
                src_layer_ld, 0.0, scratch_gates_, rnn.gates_ws_ld);
    }
    (this->*gemm_iter_func)('N', 'N', rnn.n_gates * rnn.dic, rnn.mb, rnn.sic,
            1.0, w_iter_[0], rnn.weights_iter_ld, states_tm1_l_, src_iter_ld,
            1.0, scratch_gates_, rnn.gates_ws_ld);

    rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            states_t_l_, c_states_t_l_, states_tm1_l_, c_states_tm1_l_,
            diff_states_t_l_, diff_states_t_lp1_, diff_states_tp1_l_,
            weights_peephole_, bias_[0], ws_grid_, scratch_cell_,
            states_t_l_copy_);
}
template rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution);
template rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution);
template rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution);

template <typename scratch_data_t, typename acc_data_t>
void lstm_bwd_weights_peephole_and_bias(const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, const float *c_states_tm1_l_,
        const float *c_states_t_l_, const scratch_data_t *scratch_gates_,
        float *diff_weights_peephole_, acc_data_t *diff_bias_) {
    auto dst_iter_c_ld = rnn.dst_iter_c_ld(cell_position);
    auto src_iter_c_ld = rnn.src_iter_c_ld(cell_position);

    ws_states_aoc<const float> c_states_t_l(rnn, c_states_t_l_, dst_iter_c_ld);
    ws_states_aoc<const float> c_states_tm1_l(
            rnn, c_states_tm1_l_, src_iter_c_ld);
    ws_gates_aoc<const scratch_data_t> scratch_gates(rnn, scratch_gates_);
    weights_peephole_aoc_t<float> diff_weights_peephole(
            rnn, diff_weights_peephole_);

    parallel(0, [&](int ithr, int nthr) {
        int g_dic_start {}, g_dic_stop {};
        const int gates_to_process = 5; // 3 -- weights peephole +
                // 2 -- bias (process a pair at once)
        balance211(gates_to_process * rnn.dic, nthr, ithr, g_dic_start,
                g_dic_stop);
        int g = g_dic_start / rnn.dic;
        int dic = g_dic_start % rnn.dic;
        while (g_dic_start++ < g_dic_stop) {
            if (g < 3) {
                // weights peephole
                auto &c_states = g < 2 ? c_states_tm1_l : c_states_t_l;
                const int scratch_g = g < 2 ? g : 3;
                for (int mb = 0; mb < rnn.mb; ++mb) {
                    diff_weights_peephole(g, dic) += c_states(mb, dic)
                            * scratch_gates(mb, scratch_g, dic);
                }
            } else {
                // bias
                const int bias_g_start = 2 * (g - 3);
                const int bias_g_end = bias_g_start + 2;
                for (int bias_g = bias_g_start; bias_g < bias_g_end; ++bias_g) {
                    for (int mb = 0; mb < rnn.mb; ++mb)
                        diff_bias_[bias_g * rnn.dic + dic]
                                += scratch_gates(mb, bias_g, dic);
                }
            }
            if (++dic == rnn.dic) {
                dic = 0;
                g++;
            }
        }
    });
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename weights_data_t, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void common_bwd_cell_exec_template(T1 gemm_layer_f, T2 gemm_iter_f,
        T3 gemm_weights_layer_f, T4 gemm_weights_iter_f, T5 rnn_postgemm,
        const rnn_utils::rnn_conf_t &rnn, const cell_position_t cell_position,
        src_data_t *states_t_l_, float *c_states_t_l_,
        acc_data_t *diff_states_t_l_, weights_data_t **w_layer_,
        weights_data_t **w_iter_, const float *weights_peephole_, float **bias_,
        const src_data_t *states_t_lm1_, const src_data_t *states_tm1_l_,
        const float *c_states_tm1_l_, acc_data_t *diff_states_t_lp1_,
        acc_data_t *diff_states_tp1_l_, acc_data_t *diff_w_layer_,
        acc_data_t *diff_w_iter_, float *diff_weights_peephole_,
        acc_data_t *diff_bias_, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *ws_grid_,
        scratch_data_t *scratch_cell_, src_data_t *states_t_l_copy_) {
    ws_diff_states_aoc<float> diff_states_t_l(rnn, diff_states_t_l_);
    rnn_postgemm->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            states_t_l_, c_states_t_l_, states_tm1_l_, c_states_tm1_l_,
            diff_states_t_l_, diff_states_t_lp1_, diff_states_tp1_l_,
            weights_peephole_, bias_[0], ws_grid_, scratch_cell_,
            states_t_l_copy_);

    /// bwd by data on the cell
    gemm_iter_f(w_iter_[0], scratch_gates_, diff_states_t_l_);

    if (!rnn.merge_gemm_layer) {
        gemm_layer_f(w_layer_[0], scratch_gates_,
                &diff_states_t_l(rnn.n_states, 0, 0));

        /// bwd by weights on the cell
        gemm_weights_layer_f(scratch_gates_, states_t_lm1_, diff_w_layer_);
    }

    if (!rnn.merge_gemm_iter)
        gemm_weights_iter_f(scratch_gates_, states_tm1_l_, diff_w_iter_);

    if (rnn.is_lstm_peephole) {
        /// bwd by weights peephole and bias
        lstm_bwd_weights_peephole_and_bias(rnn, cell_position, c_states_tm1_l_,
                c_states_t_l_, scratch_gates_, diff_weights_peephole_,
                diff_bias_);
    } else {
        /// bwd by bias we just accumulate diffs from the gates
        gates_reduction(rnn, scratch_gates_, diff_bias_);
    }
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution) {
    auto gemm_layer = [&](const float *A, const float *B, float *C) {
        (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dic, 1.0, A, rnn.weights_layer_ld, B,
                rnn.gates_ws_ld, 0.0, C, rnn.states_ws_ld);
    };
    auto gemm_iter = [&](const float *A, const float *B, float *C) {
        (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
                rnn.n_gates * rnn.dic, 1.0, A, rnn.weights_iter_ld, B,
                rnn.gates_ws_ld, 0.0, C, rnn.states_ws_ld);
    };
    auto gemm_weights_layer = [&](const float *A, const float *B, float *C) {
        auto src_layer_ld = rnn.src_layer_ld(cell_position);
        gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.slc, rnn.mb, 1.0, A,
                rnn.gates_ws_ld, B, src_layer_ld, 1.0, C,
                rnn.diff_weights_layer_ld);
    };
    auto gemm_weights_iter = [&](const float *A, const float *B, float *C) {
        auto src_iter_ld = rnn.src_iter_ld(cell_position);
        gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.sic, rnn.mb, 1.0, A,
                rnn.gates_ws_ld, B, src_iter_ld, 1.0, C,
                rnn.diff_weights_iter_ld);
    };
    common_bwd_cell_exec_template(gemm_layer, gemm_iter, gemm_weights_layer,
            gemm_weights_iter, rnn_postgemm_, rnn, cell_position, states_t_l_,
            c_states_t_l_, diff_states_t_l_, w_layer_, w_iter_,
            weights_peephole_, bias_, states_t_lm1_, states_tm1_l_,
            c_states_tm1_l_, diff_states_t_lp1_, diff_states_tp1_l_,
            diff_w_layer_, diff_w_iter_, diff_weights_peephole_, diff_bias_,
            ws_gates_, scratch_gates_, ws_grid_, scratch_cell_,
            states_t_l_copy_);
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution) {
    auto gemm_layer = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
        (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dic, 1.0, A, rnn.weights_layer_ld, B,
                rnn.gates_ws_ld, 0.0, C, rnn.states_ws_ld);
    };
    auto gemm_iter = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
        (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
                rnn.n_gates * rnn.dic, 1.0, A, rnn.weights_iter_ld, B,
                rnn.gates_ws_ld, 0.0, C, rnn.states_ws_ld);
    };
    auto gemm_weights_layer
            = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
                  auto src_layer_ld = rnn.src_layer_ld(cell_position);
                  gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.slc, rnn.mb, 1.0, A,
                          rnn.gates_ws_ld, B, src_layer_ld, 1.0, C,
                          rnn.diff_weights_layer_ld);
              };
    auto gemm_weights_iter
            = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
                  auto src_iter_ld = rnn.src_iter_ld(cell_position);
                  gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.sic, rnn.mb, 1.0, A,
                          rnn.gates_ws_ld, B, src_iter_ld, 1.0, C,
                          rnn.diff_weights_iter_ld);
              };
    common_bwd_cell_exec_template(gemm_layer, gemm_iter, gemm_weights_layer,
            gemm_weights_iter, rnn_postgemm_, rnn, cell_position, states_t_l_,
            c_states_t_l_, diff_states_t_l_, w_layer_, w_iter_,
            weights_peephole_, bias_, states_t_lm1_, states_tm1_l_,
            c_states_tm1_l_, diff_states_t_lp1_, diff_states_tp1_l_,
            diff_w_layer_, diff_w_iter_, diff_weights_peephole_, diff_bias_,
            ws_gates_, scratch_gates_, ws_grid_, scratch_cell_,
            states_t_l_copy_);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
