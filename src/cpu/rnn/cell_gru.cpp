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

/*
 * Cell execution GRU
 */

#include "dnnl_thread.hpp"
#include "math_utils.hpp"

#include "ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;

#define AOC array_offset_calculator
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::cell_execution_gru)) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    bias_aoc_t bias(rnn, bias_[0]);
    ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_);
    ws_states_aoc<src_data_t> states_tm1_l(rnn, states_tm1_l_);

    // 1. gemm Wx[0-2],x
    if (!rnn.merge_gemm_layer) {
        (this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dic, rnn.mb,
                rnn.slc, 1.0, w_layer_[0], rnn.weights_layer_ld, states_t_lm1_,
                rnn.states_ws_ld, 0.0f, scratch_gates_, rnn.gates_ws_ld);
    }

    // 2. gemm Wh[0-1],h
    (this->*gemm_iter_func)('N', 'N', (rnn.n_gates - 1) * rnn.dic, rnn.mb,
            rnn.sic, 1.0, w_iter_[0], rnn.weights_iter_ld, states_tm1_l_,
            rnn.states_ws_ld, 1.0f, scratch_gates_, rnn.gates_ws_ld);

    // 3. activation zt and rt + elemwise multiplication rt,ht-1
    rnn_postgemm_->execute(rnn, ws_gates_, scratch_gates_, states_t_l_,
            c_states_t_l_, states_tm1_l_, c_states_tm1_l_, diff_states_t_l_,
            diff_states_t_lp1_, diff_states_tp1_l_, bias_[0], nullptr, nullptr);

    // 4. gemm Wh[2],h~t
    (this->*gemm_iter_func)('N', 'N', rnn.dic, rnn.mb, rnn.sic, 1.0, w_iter_[1],
            rnn.weights_iter_ld, states_t_l_, rnn.states_ws_ld, 1.0,
            &(scratch_gates(0, 2, 0)), rnn.gates_ws_ld);

    // 5. activation h~t + calculate ht
    rnn_postgemm_->execute_part2(rnn, ws_gates_, scratch_gates_, states_t_l_,
            c_states_t_l_, states_tm1_l_, c_states_tm1_l_, diff_states_t_l_,
            diff_states_t_lp1_, diff_states_tp1_l_, bias_[0], nullptr, nullptr);
}

template rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru);
template rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru) {
    assert(!"GRU int8 is not supported");
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename weights_data_t, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void gru_bwd_cell_exec_template(T1 gemm_layer_f, T2 gemm_iter_f,
        T3 gemm_weights_layer_f, T4 gemm_weights_iter_f, T5 rnn_postgemm_,
        const rnn_utils::rnn_conf_t &rnn, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *states_t_l_,
        src_data_t *states_tm1_l_, src_data_t *states_t_lm1_,
        weights_data_t **w_layer_, weights_data_t **w_iter_,
        acc_data_t *diff_w_layer_, acc_data_t *diff_w_iter_,
        acc_data_t *diff_states_t_l_, acc_data_t *diff_states_tp1_l_,
        acc_data_t *diff_states_t_lp1_, acc_data_t *diff_bias_,
        scratch_data_t *scratch_cell_) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_);
    ws_states_aoc<src_data_t> states_tm1_l(rnn, states_tm1_l_);
    ws_diff_w_iter_aoc_t diff_w_iter(rnn, diff_w_iter_);
    ws_diff_states_aoc<float> diff_states_t_l(rnn, diff_states_t_l_);

    // use state memory for intermediate computations
    // TODO: use cell ws for that
    float *dhG1_ = &(diff_states_t_l(rnn.n_states, 0, 0));
    AOC<acc_data_t, 2> dhG1(dhG1_, rnn.states_nld, rnn.states_ws_ld);
    // hg1 needs to be bf16 as it is used as gemm output
    // hence it cannot alias to dhG1, and should use scratch_cell
    AOC<scratch_data_t, 2> hG1(scratch_cell_, rnn.states_nld, rnn.states_ws_ld);

    // 1. calculate dG2, dG1, and part of dht-1
    rnn_postgemm_->execute(rnn, ws_gates_, scratch_gates_, states_t_l_, nullptr,
            states_tm1_l_, nullptr, diff_states_t_l_, diff_states_t_lp1_,
            diff_states_tp1_l_, nullptr, nullptr, scratch_cell_);

    // 2. calculate intermediate d(hG1)
    // d(hG1) = dG2 * W2h^t
    gemm_iter_f(rnn.sic, rnn.mb, rnn.dic, w_iter_[1], &(scratch_gates(0, 2, 0)),
            0.0f, dhG1_);

    // 3. calculate dG1^ and part of dht-1
    rnn_postgemm_->execute_part2(rnn, ws_gates_, scratch_gates_, states_t_l_,
            nullptr, states_tm1_l_, nullptr, diff_states_t_l_,
            diff_states_t_lp1_, diff_states_tp1_l_, nullptr, nullptr,
            scratch_cell_);

    // 4. calculate diff weights
    // dWh1 += dG1 * h, dWh2 += dG2 * h, dWh3 += dG3 * (G1(*)h)
    gemm_weights_iter_f((rnn.n_gates - 1) * rnn.dic, rnn.sic, rnn.mb,
            scratch_gates_, states_tm1_l_, 1.0f, diff_w_iter_);
    gemm_weights_iter_f(rnn.dic, rnn.sic, rnn.mb, &(scratch_gates(0, 2, 0)),
            scratch_cell_, 1.0f, &(diff_w_iter(0, 2, 0)));

    // 5. calculate diff states
    // dht-1 += dG1 * W1h + dG0 * W0h
    gemm_iter_f(rnn.sic, rnn.mb, (rnn.n_gates - 1) * rnn.dic, w_iter_[0],
            scratch_gates_, 1.0f, diff_states_t_l_);

    if (!rnn.merge_gemm_layer) {
        // dWx += [dG0 dG1 dG2] * [x]
        gemm_weights_layer_f(scratch_gates_, states_t_lm1_, diff_w_layer_);
        // dx = dG2 * W2x + dG1 * W1x + dG0 * W0x
        gemm_layer_f(w_layer_[0], scratch_gates_,
                &(diff_states_t_l(rnn.n_states, 0, 0)));
    }

    // 6. calculate diff bias
    gates_reduction(rnn, scratch_gates_, diff_bias_);
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru) {
    auto gemm_iter_f = [&](int m, int n, int k, weights_data_t *A,
                               src_data_t *B, float beta, acc_data_t *C) {
        (this->*gemm_iter_func)('N', 'N', m, n, k, 1.0f, A, rnn.weights_iter_ld,
                B, rnn.gates_ws_ld, beta, C, rnn.states_ws_ld);
    };
    auto gemm_layer_f = [&](weights_data_t *A, src_data_t *B, acc_data_t *C) {
        (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dic, 1.0, A, rnn.weights_layer_ld, B,
                rnn.gates_ws_ld, 0.0, C, rnn.states_ws_ld);
    };
    auto gemm_weights_layer_f
            = [&](src_data_t *A, weights_data_t *B, acc_data_t *C) {
                  gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.slc, rnn.mb, 1.0, A,
                          rnn.gates_ws_ld, B, rnn.states_ws_ld, 1.0, C,
                          rnn.diff_weights_layer_ld);
              };
    auto gemm_weights_iter_f
            = [&](int m, int n, int k, weights_data_t *A, src_data_t *B,
                      float beta, acc_data_t *C) {
                  gemm('N', 'T', m, n, k, 1.0f, A, rnn.gates_ws_ld, B,
                          rnn.states_ws_ld, 1.0f, C, rnn.diff_weights_iter_ld);
              };

    gru_bwd_cell_exec_template(gemm_layer_f, gemm_iter_f, gemm_weights_layer_f,
            gemm_weights_iter_f, this->rnn_postgemm_, rnn, ws_gates_,
            scratch_gates_, states_t_l_, states_tm1_l_, states_t_lm1_, w_layer_,
            w_iter_, diff_w_layer_, diff_w_iter_, diff_states_t_l_,
            diff_states_tp1_l_, diff_states_t_lp1_, diff_bias_, scratch_cell_);
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_gru) {
    auto gemm_iter_f = [&](int m, int n, int k, weights_data_t *A,
                               src_data_t *B, float beta, acc_data_t *C) {
        (this->*gemm_iter_func)('N', 'N', m, n, k, 1.0f, A, rnn.weights_iter_ld,
                B, rnn.gates_ws_ld, beta, C, rnn.states_ws_ld);
    };
    auto gemm_layer_f = [&](weights_data_t *A, src_data_t *B, acc_data_t *C) {
        (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dic, 1.0, A, rnn.weights_layer_ld, B,
                rnn.gates_ws_ld, 0.0, C, rnn.states_ws_ld);
    };
    auto gemm_weights_layer_f
            = [&](src_data_t *A, weights_data_t *B, acc_data_t *C) {
                  gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.slc, rnn.mb, 1.0, A,
                          rnn.gates_ws_ld, B, rnn.states_ws_ld, 1.0, C,
                          rnn.diff_weights_layer_ld);
              };
    auto gemm_weights_iter_f
            = [&](int m, int n, int k, weights_data_t *A, src_data_t *B,
                      float beta, acc_data_t *C) {
                  gemm('N', 'T', m, n, k, 1.0f, A, rnn.gates_ws_ld, B,
                          rnn.states_ws_ld, 1.0f, C, rnn.diff_weights_iter_ld);
              };

    gru_bwd_cell_exec_template(gemm_layer_f, gemm_iter_f, gemm_weights_layer_f,
            gemm_weights_iter_f, this->rnn_postgemm_, rnn, ws_gates_,
            scratch_gates_, states_t_l_, states_tm1_l_, states_t_lm1_, w_layer_,
            w_iter_, diff_w_layer_, diff_w_iter_, diff_states_t_l_,
            diff_states_tp1_l_, diff_states_t_lp1_, diff_bias_, scratch_cell_);
}

#undef AOC
} // namespace cpu
} // namespace impl
} // namespace dnnl
