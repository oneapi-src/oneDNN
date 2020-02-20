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
 * Cell execution of Vanilla RNN
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

template <>
float activation<alg_kind::eltwise_relu, prop_kind::forward>(
        float s, float alpha, float cliping) {
    return relu_fwd<float>(s, alpha);
}

template <>
float activation<alg_kind::eltwise_relu, prop_kind::backward>(
        float s, float alpha, float cliping) {
    return relu_bwd<float>(s, alpha);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::forward>(
        float s, float alpha, float cliping) {
    return tanh_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::backward>(
        float s, float alpha, float cliping) {
    return one_m_square<float>(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::forward>(
        float s, float alpha, float cliping) {
    return logistic_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::backward>(
        float s, float alpha, float cliping) {
    return x_m_square<float>(s);
}

float linear(float s, float alpha, float clipping) {
    return alpha * s;
}

template <typename T, typename src_data_t, typename scratch_data_t>
void rnn_fwd_postgemm_template(T func1, const float *scales, float alpha,
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
    ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_, dst_ld);
    ws_states_aoc<src_data_t> states_t_l_copy(
            rnn, states_t_l_copy_, dst_copy_ld);

    if (scales != nullptr) alpha = scales[0];

    parallel_nd(rnn.mb, [&](int i) {
        for (int j = 0; j < rnn.dic; j++) {
            const float h
                    = func1(scratch_gates(i, 0, j) + bias(0, j), alpha, 0);
            states_t_l(i, j) = h;
            if (states_t_l_copy_ != nullptr) states_t_l_copy(i, j) = h;
            if (rnn.is_training) ws_gates(i, 0, j) = h;
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::rnn_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto act_f = [this](float a, float alpha, float clipping) {
        return this->activation_func(a, alpha, clipping);
    };
    auto linear_f = [](float a, float alpha, float clipping) {
        return linear(a, alpha, clipping);
    };
    auto alpha = pd_->desc()->alpha;
    if (!pd_->attr()->rnn_tparams_.test_mode_)
        rnn_fwd_postgemm_template(act_f, nullptr, alpha, rnn, cell_position,
                ws_gates_, scratch_gates_, states_t_l_, states_t_l_copy_,
                states_tm1_l_, bias_);
    else
        rnn_fwd_postgemm_template(linear_f, scales, alpha, rnn, cell_position,
                ws_gates_, scratch_gates_, states_t_l_, states_t_l_copy_,
                states_tm1_l_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_bf16_t::rnn_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto act_f = [this](float a, float alpha, float clipping) {
        return bfloat16_t(this->activation_func(a, alpha, clipping));
    };
    auto linear_f = [](float a, float alpha, float clipping) {
        return bfloat16_t(linear(a, alpha, clipping));
    };
    auto alpha = pd_->desc()->alpha;
    if (!pd_->attr()->rnn_tparams_.test_mode_)
        rnn_fwd_postgemm_template(act_f, nullptr, alpha, rnn, cell_position,
                ws_gates_, scratch_gates_, states_t_l_, states_t_l_copy_,
                states_tm1_l_, bias_);
    else
        rnn_fwd_postgemm_template(linear_f, scales, alpha, rnn, cell_position,
                ws_gates_, scratch_gates_, states_t_l_, states_t_l_copy_,
                states_tm1_l_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::rnn_postgemm) {
    assert(!"VANILLA RNN int8 is not supported");
}

template <typename T1, typename T2, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void rnn_bwd_postgemm_template(T1 func1, T2 to_src, const float *scales,
        float alpha, const rnn_utils::rnn_conf_t &rnn, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, acc_data_t *diff_states_tp1_l_,
        acc_data_t *diff_states_t_lp1_) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    ws_diff_states_aoc<acc_data_t> diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc<acc_data_t> diff_states_t_lp1(rnn, diff_states_t_lp1_);
    if (scales != nullptr) alpha = scales[0];

    parallel_nd(rnn.mb, [&](int i) {
        for (int j = 0; j < rnn.dic; ++j) {
            const float dH = diff_states_t_lp1(rnn.n_states, i, j)
                    + diff_states_tp1_l(0, i, j);
            auto g = (float)ws_gates(i, 0, j);
            float res = dH * func1(g, alpha, 0);
            src_data_t res_converted = to_src(res);
            scratch_gates(i, 0, j) = res_converted;
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::rnn_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto act_f = [this](float a, float alpha, float clipping) {
        return this->activation_func(a, alpha, 0);
    };
    auto linear_f = [](float a, float alpha, float clipping) {
        return linear(a, alpha, 0);
    };
    auto to_src = [&](float a) { return a; };
    auto alpha = pd_->desc()->alpha;
    if (!pd_->attr()->rnn_tparams_.test_mode_)
        rnn_bwd_postgemm_template(act_f, to_src, nullptr, alpha, rnn, ws_gates_,
                scratch_gates_, diff_states_tp1_l_, diff_states_t_lp1_);
    else
        rnn_bwd_postgemm_template(linear_f, to_src, scales, alpha, rnn,
                ws_gates_, scratch_gates_, diff_states_tp1_l_,
                diff_states_t_lp1_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::rnn_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    auto act_f = [this](float a, float alpha, float clipping) {
        return this->activation_func(a, alpha, 0);
    };
    auto linear_f = [](float a, float alpha, float clipping) {
        return linear(a, alpha, 0);
    };
    auto to_src = [&](float a) { return bfloat16_t(a); };
    auto alpha = pd_->desc()->alpha;
    if (!pd_->attr()->rnn_tparams_.test_mode_)
        rnn_bwd_postgemm_template(act_f, to_src, nullptr, alpha, rnn, ws_gates_,
                scratch_gates_, diff_states_tp1_l_, diff_states_t_lp1_);
    else
        rnn_bwd_postgemm_template(linear_f, to_src, scales, alpha, rnn,
                ws_gates_, scratch_gates_, diff_states_tp1_l_,
                diff_states_t_lp1_);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
