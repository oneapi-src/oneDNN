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
 * Cell execution of Vanilla RNN
 */

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
#define AOC array_offset_calculator

template <>
float activation<alg_kind::eltwise_relu, prop_kind::forward>(
        float dd, float s, float alpha, float cliping) {
    return relu_fwd<float>(s, alpha);
}

template <>
float activation<alg_kind::eltwise_relu, prop_kind::backward>(
        float dd, float s, float alpha, float cliping) {
    return relu_bwd<float>(dd, s, alpha);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::forward>(
        float dd, float s, float alpha, float cliping) {
    return tanh_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::backward>(
        float dd, float s, float alpha, float cliping) {
    return dd * one_m_square<float>(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::forward>(
        float dd, float s, float alpha, float cliping) {
    return logistic_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::backward>(
        float dd, float s, float alpha, float cliping) {
    return dd * x_m_square<float>(s);
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::forward>::rnn_elemwise) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> bias(bias_, n_gates, dic);
    AOC<float, 4> states_t_l(states_t_l_, n_states, iter_stride, batch, wic);
    parallel_nd(batch, [&](int i) {
        for (int j = 0; j < dic; j++) {
            const float h
                    = activation_func(0, ws_gates(i, j) + bias(0, j), 0, 0);
            ws_gates(i, j) = states_t_l(0, 0, i, j) = h;
        }
    });
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::backward>::rnn_elemwise) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<float, 4> diff_states_tp1_l(
            diff_states_tp1_l_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 4> diff_states_t_lp1(
            diff_states_t_lp1_, n_states + 1, iter_stride, batch, wic);
    parallel_nd(batch, [&](int i) {
        for (int j = 0; j < dic; ++j) {
            const float dH = diff_states_t_lp1(n_states, 0, i, j)
                    + diff_states_tp1_l(0, 0, i, j);
            auto g = ws_gates(i, j);
            ws_gates(i, j) = activation_func(dH, g, 0, 0);
        }
    });
}

#undef AOC

}
}
}
