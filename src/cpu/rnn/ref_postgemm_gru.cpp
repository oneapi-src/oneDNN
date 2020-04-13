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

/*
 * Cell execution LSTM
 */

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"

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
        scratch_data_t *scratch_gates_, src_data_t *dst_layer_,
        src_data_t *dst_iter_, const src_data_t *src_iter_, float *bias_) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    bias_aoc_t bias(rnn, bias_);

    auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
    auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);

    ws_states_layer_aoc<src_data_t> dst_layer(rnn, dst_layer_, dst_layer_ld);
    ws_states_iter_aoc<src_data_t> dst_iter(rnn, dst_iter_, dst_iter_ld);
    ws_states_iter_aoc<const src_data_t> src_iter(rnn, src_iter_, src_iter_ld);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dhc; j++) {
            auto G0 // default func1 is sigmoid
                    = func1(scales, scratch_gates(i, 0, j) + bias(0, j));
            auto G1 // default func1 is sigmoid
                    = func1(scales + 1, scratch_gates(i, 1, j) + bias(1, j));
            /* TODO: Can be optimized for fwd_training by using ws_gates instead of scratch_gates in p2 */
            scratch_gates(i, 0, j) = to_src(G0);
            scratch_gates(i, 1, j) = to_src(G1);
            auto t = to_src(src_iter(i, j) * G1);
            if (dst_layer_) dst_layer(i, j) = t;
            if (dst_iter_) dst_iter(i, j) = t;

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
        scratch_data_t *scratch_gates_, src_data_t *dst_layer_,
        src_data_t *dst_iter_, const src_data_t *src_iter_, float *bias_) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    bias_aoc_t bias(rnn, bias_);

    auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
    auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);
    ws_states_layer_aoc<src_data_t> dst_layer(rnn, dst_layer_, dst_layer_ld);
    ws_states_iter_aoc<src_data_t> dst_iter(rnn, dst_iter_, dst_iter_ld);
    ws_states_iter_aoc<const src_data_t> src_iter(rnn, src_iter_, src_iter_ld);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dhc; j++) {
            auto G0 = scratch_gates(i, 0, j);
            auto G2 // default func1 is tanh
                    = func1(scales + 2, scratch_gates(i, 2, j) + bias(2, j));

            auto tmp = to_src(src_iter(i, j) * G0 + (1.0f - G0) * G2);
            if (dst_layer_ != nullptr) dst_layer(i, j) = tmp;
            if (dst_iter_ != nullptr) dst_iter(i, j) = tmp;

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
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_);
    else
        gru_fwd_part1_postgemm_template(linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_);
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
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_);
    else
        gru_fwd_part2_postgemm_template(linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_);
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
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_);
    else
        gru_fwd_part1_postgemm_template(linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_);
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
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_);
    else
        gru_fwd_part2_postgemm_template(linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_);
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
        scratch_data_t *scratch_gates_, src_data_t *dst_layer_,
        const src_data_t *src_iter_, acc_data_t *diff_src_iter_,
        acc_data_t *diff_dst_iter_, acc_data_t *diff_dst_layer_) {
    auto src_iter_ld = rnn.src_iter_ld(cell_position);

    ws_states_iter_aoc<const src_data_t> src_iter(rnn, src_iter_, src_iter_ld);
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    ws_diff_states_iter_aoc<acc_data_t> diff_src_iter(rnn, diff_src_iter_);
    ws_diff_states_iter_aoc<acc_data_t> diff_dst_iter(rnn, diff_dst_iter_);
    ws_diff_states_layer_aoc<acc_data_t> diff_dst_layer(rnn, diff_dst_layer_);

    // dG2^ = dh * (1 - G0) * (1 - G2^2)
    // dG0^ = dh * (ht-1 - G2) * u * (1 - G0)
    // dht-1 (part) = dh * G0
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dhc; j++) {
            float h = src_iter(i, j);
            float dHt = diff_dst_iter(i, j) + diff_dst_layer(i, j);
            float dG2 = (1.0f - ws_gates(i, 0, j)) * dHt
                    * one_m_square(ws_gates(i, 2, j));
            float dG0 = (h - ws_gates(i, 2, j)) * dHt
                    * x_m_square(ws_gates(i, 0, j));

            diff_src_iter(i, j) = dHt * ws_gates(i, 0, j);
            scratch_gates(i, 0, j) = to_src(dG0);
            scratch_gates(i, 2, j) = to_src(dG2);
        }
    });
}

template <typename T, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void gru_bwd_part2_postgemm_template(T to_src, const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *dst_layer_,
        const src_data_t *src_iter_, acc_data_t *diff_src_layer_,
        acc_data_t *diff_src_iter_, acc_data_t *diff_dst_iter_,
        acc_data_t *diff_dst_layer_, scratch_data_t *scratch_cell_) {
    auto src_iter_ld = rnn.src_iter_ld(cell_position);
    // auto dst_ld = rnn.dst_ld(cell_position);
    // ws_states_layer_aoc<src_data_t> dst_layer(rnn, dst_layer_, dst_ld);
    ws_states_iter_aoc<const src_data_t> src_iter(rnn, src_iter_, src_iter_ld);
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    ws_diff_states_layer_aoc<acc_data_t> diff_dst_layer(rnn, diff_dst_layer_);
    ws_diff_states_iter_aoc<acc_data_t> diff_dst_iter(rnn, diff_dst_iter_);

    ws_diff_states_layer_aoc<acc_data_t> dhG1(rnn, diff_src_layer_);
    ws_diff_states_iter_aoc<acc_data_t> diff_src_iter(rnn, diff_src_iter_);
    AOC<scratch_data_t, 2> hG1(
            scratch_cell_, rnn.ws_states_layer_nld, rnn.ws_states_layer_ld);

    // dG1^ = d(hG1) * h * G1 * (1 - G1)
    // dht-1 (part) += d(hG1) * G1
    // h * G1 (required for dWh)
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dhc; j++) {
            float h = src_iter(i, j);
            float G1 = ws_gates(i, 1, j);
            diff_src_iter(i, j) += dhG1(i, j) * G1;
            scratch_gates(i, 1, j) = to_src(dhG1(i, j) * h * x_m_square(G1));
            hG1(i, j) = to_src(G1 * h);
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::gru_part1_postgemm) {
    auto to_src = [](float a) { return a; };

    gru_bwd_part1_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, dst_layer_, src_iter_, diff_src_iter_,
            diff_dst_iter_, diff_dst_layer_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::gru_part2_postgemm) {
    auto to_src = [](float a) { return a; };

    gru_bwd_part2_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, dst_layer_, src_iter_, diff_src_layer_,
            diff_src_iter_, diff_dst_iter_, diff_dst_layer_, scratch_cell_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::gru_part1_postgemm) {
    auto to_src = [](float a) { return bfloat16_t(a); };

    gru_bwd_part1_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, dst_layer_, src_iter_, diff_src_iter_,
            diff_dst_iter_, diff_dst_layer_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::gru_part2_postgemm) {
    auto to_src = [](float a) { return bfloat16_t(a); };

    gru_bwd_part2_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, dst_layer_, src_iter_, diff_src_layer_,
            diff_src_iter_, diff_dst_iter_, diff_dst_layer_, scratch_cell_);
}

#undef AOC
} // namespace cpu
} // namespace impl
} // namespace dnnl
