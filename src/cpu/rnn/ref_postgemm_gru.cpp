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
 * Cell execution LSTM
 */

#include "dnnl_thread.hpp"
#include "math_utils.hpp"

#include "jit_uni_rnn_common_postgemm_dispatcher.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;
#define AOC array_offset_calculator

template <typename T1, typename T2, typename src_data_t,
        typename scratch_data_t>
void gru_fwd_part1_postgemm_template(T1 func1, T2 to_src, const float *scales,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *states_t_l_,
        const src_data_t *states_tm1_l_, float *bias_) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    bias_aoc_t bias(rnn, bias_);

    // auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
    // auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);

    auto dst_ld = rnn.dst_ld(cell_position);
    ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_, dst_ld);
    ws_states_aoc<const src_data_t> states_tm1_l(
            rnn, states_tm1_l_, src_iter_ld);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            auto G0 // default func1 is sigmoid
                    = func1(scales, scratch_gates(i, 0, j) + bias(0, j));
            auto G1 // default func1 is sigmoid
                    = func1(scales + 1, scratch_gates(i, 1, j) + bias(1, j));
            /* TODO: Can be optimized for fwd_training by using ws_gates instead of scratch_gates in p2 */
            scratch_gates(i, 0, j) = to_src(G0);
            scratch_gates(i, 1, j) = to_src(G1);
            auto t = to_src(states_tm1_l(i, j) * G1);
            states_t_l(i, j) = t;

            if (rnn.is_training) {
                ws_gates(i, 0, j) = to_src(G0);
                ws_gates(i, 1, j) = to_src(G1);
            }
        }
    });
}

template <typename T1, typename T2, typename src_data_t,
        typename scratch_data_t>
void gru_fwd_part2_postgemm_template(T1 func1, T2 to_src, const float *scales,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *states_t_l_,
        src_data_t *states_t_l_copy_, const src_data_t *states_tm1_l_,
        float *bias_) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    bias_aoc_t bias(rnn, bias_);

    auto dst_ld = rnn.dst_ld(cell_position);
    auto dst_copy_ld = rnn.dst_copy_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);
    ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_, dst_ld);
    ws_states_aoc<src_data_t> states_t_l_copy(
            rnn, states_t_l_copy_, dst_copy_ld);
    ws_states_aoc<const src_data_t> states_tm1_l(
            rnn, states_tm1_l_, src_iter_ld);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            auto G0 = scratch_gates(i, 0, j);
            auto G2 // default func1 is tanh
                    = func1(scales + 2, scratch_gates(i, 2, j) + bias(2, j));

            auto tmp = to_src(states_tm1_l(i, j) * G0 + (1.0f - G0) * G2);
            states_t_l(i, j) = tmp;
            if (states_t_l_copy_ != nullptr) states_t_l_copy(i, j) = tmp;

            if (rnn.is_training) { ws_gates(i, 2, j) = to_src(G2); }
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::gru_part1_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    auto to_src = [](float a) { return a; };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part1_postgemm_template(logistic_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, states_t_l_,
                states_tm1_l_, bias_);
    else
        gru_fwd_part1_postgemm_template(linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, states_t_l_,
                states_tm1_l_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::gru_part2_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };
    auto to_src = [](float a) { return a; };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part2_postgemm_template(tanh_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, states_t_l_,
                states_t_l_copy_, states_tm1_l_, bias_);
    else
        gru_fwd_part2_postgemm_template(linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, states_t_l_,
                states_t_l_copy_, states_tm1_l_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_bf16_t::gru_part1_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto to_src = [](float a) { return bfloat16_t(a); };
    auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part1_postgemm_template(logistic_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, states_t_l_,
                states_tm1_l_, bias_);
    else
        gru_fwd_part1_postgemm_template(linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, states_t_l_,
                states_tm1_l_, bias_);
}
template <>
rnn_postgemm_sig(rnn_postgemm_fwd_bf16_t::gru_part2_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };
    auto to_src = [](float a) { return bfloat16_t(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        gru_fwd_part2_postgemm_template(tanh_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, states_t_l_,
                states_t_l_copy_, states_tm1_l_, bias_);
    else
        gru_fwd_part2_postgemm_template(linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, states_t_l_,
                states_t_l_copy_, states_tm1_l_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::gru_part1_postgemm) {
    assert(!"GRU int8 is not supported");
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::gru_part2_postgemm) {
    assert(!"GRU int8 is not supported");
}

template <typename T, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void gru_bwd_part1_postgemm_template(T to_src, const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *states_t_l_,
        const src_data_t *states_tm1_l_, acc_data_t *diff_states_t_l_,
        acc_data_t *diff_states_tp1_l_, acc_data_t *diff_states_t_lp1_) {
    auto src_iter_ld = rnn.src_iter_ld(cell_position);
    auto dst_ld = rnn.dst_ld(cell_position);
    ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_, dst_ld);
    ws_states_aoc<const src_data_t> states_tm1_l(
            rnn, states_tm1_l_, src_iter_ld);
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    ws_diff_states_aoc<acc_data_t> diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc<acc_data_t> diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc<acc_data_t> diff_states_t_lp1(rnn, diff_states_t_lp1_);

    // dG2^ = dh * (1 - G0) * (1 - G2^2)
    // dG0^ = dh * (ht-1 - G2) * u * (1 - G0)
    // dht-1 (part) = dh * G0
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float h = states_tm1_l(i, j);
            float dHt = diff_states_tp1_l(0, i, j)
                    + diff_states_t_lp1(rnn.n_states, i, j);
            float dG2 = (1.0f - ws_gates(i, 0, j)) * dHt
                    * one_m_square(ws_gates(i, 2, j));
            float dG0 = (h - ws_gates(i, 2, j)) * dHt
                    * x_m_square(ws_gates(i, 0, j));

            diff_states_t_l(0, i, j) = dHt * ws_gates(i, 0, j);
            scratch_gates(i, 0, j) = to_src(dG0);
            scratch_gates(i, 2, j) = to_src(dG2);
        }
    });
}

template <typename T, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void gru_bwd_part2_postgemm_template(T to_src, const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *states_t_l_,
        const src_data_t *states_tm1_l_, acc_data_t *diff_states_t_l_,
        acc_data_t *diff_states_tp1_l_, acc_data_t *diff_states_t_lp1_,
        scratch_data_t *scratch_cell_) {
    auto src_iter_ld = rnn.src_iter_ld(cell_position);
    auto dst_ld = rnn.dst_ld(cell_position);
    ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_, dst_ld);
    ws_states_aoc<const src_data_t> states_tm1_l(
            rnn, states_tm1_l_, src_iter_ld);
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    ws_diff_states_aoc<acc_data_t> diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc<acc_data_t> diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc<acc_data_t> diff_states_t_lp1(rnn, diff_states_t_lp1_);

    AOC<acc_data_t, 2> dhG1(&(diff_states_t_l(rnn.n_states, 0, 0)),
            rnn.states_nld, rnn.states_ws_ld);
    AOC<scratch_data_t, 2> hG1(scratch_cell_, rnn.states_nld, rnn.states_ws_ld);

    // dG1^ = d(hG1) * h * G1 * (1 - G1)
    // dht-1 (part) += d(hG1) * G1
    // h * G1 (required for dWh)
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float h = states_tm1_l(i, j);
            float G1 = ws_gates(i, 1, j);
            diff_states_t_l(0, i, j) += dhG1(i, j) * G1;
            scratch_gates(i, 1, j) = to_src(dhG1(i, j) * h * x_m_square(G1));
            hG1(i, j) = to_src(G1 * h);
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::gru_part1_postgemm) {
    auto to_src = [](float a) { return a; };

    gru_bwd_part1_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, states_t_l_, states_tm1_l_, diff_states_t_l_,
            diff_states_tp1_l_, diff_states_t_lp1_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::gru_part2_postgemm) {
    auto to_src = [](float a) { return a; };

    gru_bwd_part2_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, states_t_l_, states_tm1_l_, diff_states_t_l_,
            diff_states_tp1_l_, diff_states_t_lp1_,
            (scratch_data_t *)scratch_cell_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::gru_part1_postgemm) {
    auto to_src = [](float a) { return bfloat16_t(a); };

    gru_bwd_part1_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, states_t_l_, states_tm1_l_, diff_states_t_l_,
            diff_states_tp1_l_, diff_states_t_lp1_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::gru_part2_postgemm) {
    auto to_src = [](float a) { return bfloat16_t(a); };

    gru_bwd_part2_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, states_t_l_, states_tm1_l_, diff_states_t_l_,
            diff_states_tp1_l_, diff_states_t_lp1_,
            (scratch_data_t *)scratch_cell_);
}

#undef AOC
} // namespace cpu
} // namespace impl
} // namespace dnnl
