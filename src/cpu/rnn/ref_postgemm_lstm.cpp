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

#include "../simple_q10n.hpp"
#include "jit_uni_rnn_common_postgemm_dispatcher.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;

template <typename T1, typename T2, typename T3, typename T4,
        typename src_data_t, typename scratch_data_t>
void lstm_fwd_postgemm_template(T1 func1, T2 func2, T3 to_src_dt, T4 to_float,
        const float *scales, const float *cscale,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *states_t_l_,
        src_data_t *states_t_l_copy_, float *c_states_t_l_,
        const src_data_t *states_tm1_l_, const float *c_states_tm1_l_,
        float *bias_) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    bias_aoc_t bias(rnn, bias_);

    auto dst_iter_c_ld = rnn.dst_iter_c_ld(cell_position);
    auto src_iter_c_ld = rnn.src_iter_c_ld(cell_position);

    auto dst_ld = rnn.dst_ld(cell_position);
    auto dst_copy_ld = rnn.dst_copy_ld(cell_position);

    ws_states_aoc<src_data_t> states_t_l(rnn, states_t_l_, dst_ld);
    ws_states_aoc<src_data_t> states_t_l_copy(
            rnn, states_t_l_copy_, dst_copy_ld);
    ws_states_aoc<float> c_states_t_l(rnn, c_states_t_l_, dst_iter_c_ld);
    ws_states_aoc<const float> c_states_tm1_l(
            rnn, c_states_tm1_l_, src_iter_c_ld);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float G0 = func1( // default func1 is sigmoid
                    scales,
                    to_float(scratch_gates(i, 0, j), 0, j) + bias(0, j));
            float G1 = func1( // default func1 is sigmoid
                    scales + 1,
                    to_float(scratch_gates(i, 1, j), 1, j) + bias(1, j));
            float G2 = func2( // default func2 is tanh
                    scales + 2,
                    to_float(scratch_gates(i, 2, j), 2, j) + bias(2, j));
            float G3 = func1( // default func1 is sigmoid
                    scales + 3,
                    to_float(scratch_gates(i, 3, j), 3, j) + bias(3, j));
            float tmp = G1 * c_states_tm1_l(i, j) + G0 * G2;
            states_t_l(i, j) = to_src_dt(G3 * func2(cscale, tmp));
            if (states_t_l_copy_ != nullptr)
                states_t_l_copy(i, j) = states_t_l(i, j);
            c_states_t_l(i, j) = tmp;

            // write gates back to memory for training
            // we to_src_dt them as as they are GEMM inputs in BWD
            if (rnn.is_training) {
                ws_gates(i, 0, j) = to_src_dt(G0);
                ws_gates(i, 1, j) = to_src_dt(G1);
                ws_gates(i, 2, j) = to_src_dt(G2);
                ws_gates(i, 3, j) = to_src_dt(G3);
            }
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::lstm_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);

    auto q_id = [&](float f) { return f; };
    auto deq_id = [&](float f, int i, int j) { return f; };

    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_fwd_postgemm_template(logistic_f, tanh_f, q_id, deq_id, scales,
                cscale, rnn, cell_position, ws_gates_, scratch_gates_,
                states_t_l_, states_t_l_copy_, c_states_t_l_, states_tm1_l_,
                c_states_tm1_l_, bias_);
    else
        lstm_fwd_postgemm_template(linear_f, linear_f, q_id, deq_id, scales,
                cscale, rnn, cell_position, ws_gates_, scratch_gates_,
                states_t_l_, states_t_l_copy_, c_states_t_l_, states_tm1_l_,
                c_states_tm1_l_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_bf16_t::lstm_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);
    auto round_f32_bf16 = [&](float f) { return bfloat16_t(f); };
    auto deq_id = [&](float f, int i, int j) { return f; };

    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_fwd_postgemm_template(logistic_f, tanh_f, round_f32_bf16, deq_id,
                scales, cscale, rnn, cell_position, ws_gates_, scratch_gates_,
                states_t_l_, states_t_l_copy_, c_states_t_l_, states_tm1_l_,
                c_states_tm1_l_, bias_);
    else
        lstm_fwd_postgemm_template(linear_f, linear_f, round_f32_bf16, deq_id,
                scales, cscale, rnn, cell_position, ws_gates_, scratch_gates_,
                states_t_l_, states_t_l_copy_, c_states_t_l_, states_tm1_l_,
                c_states_tm1_l_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::lstm_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);

    float *weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;
    float data_shift = pd_->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd_->attr()->rnn_data_qparams_.scale_;

    auto quantize_f32_u8 = [&](float f) {
        float qf = f * data_scale + data_shift;
        return qz_a1b0<float, src_data_t>()(qf);
    };

    auto dequantize_s32_f32 = [&](acc_data_t s, int gate, int j) {
        return pd_->attr()->rnn_weights_qparams_.mask_ == 0
                ? saturate<float>(s) * (1.f / (weights_scales[0] * data_scale))
                : saturate<float>(s)
                        * (1.f
                                / (weights_scales[gate * rnn.dic + j]
                                        * data_scale));
    };

    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_fwd_postgemm_template(logistic_f, tanh_f, quantize_f32_u8,
                dequantize_s32_f32, scales, cscale, rnn, cell_position,
                ws_gates_, scratch_gates_, states_t_l_, states_t_l_copy_,
                c_states_t_l_, states_tm1_l_, c_states_tm1_l_, bias_);
    else
        lstm_fwd_postgemm_template(linear_f, linear_f, quantize_f32_u8,
                dequantize_s32_f32, scales, cscale, rnn, cell_position,
                ws_gates_, scratch_gates_, states_t_l_, states_t_l_copy_,
                c_states_t_l_, states_tm1_l_, c_states_tm1_l_, bias_);
}

template <typename T1, typename T2, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void lstm_bwd_postgemm_template(T1 func1, T2 to_src_dt, const float *cscale,
        const rnn_utils::rnn_conf_t &rnn, const cell_position_t cell_position,
        src_data_t *ws_gates_, scratch_data_t *scratch_gates_,
        float *c_states_t_l_, const float *c_states_tm1_l_,
        acc_data_t *diff_states_t_l_, acc_data_t *diff_states_t_lp1_,
        acc_data_t *diff_states_tp1_l_, float *bias_) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    bias_aoc_t bias(rnn, bias_);
    auto dst_iter_c_ld = rnn.dst_iter_c_ld(cell_position);
    auto src_iter_c_ld = rnn.src_iter_c_ld(cell_position);
    ws_states_aoc<float> c_states_t_l(rnn, c_states_t_l_, dst_iter_c_ld);
    ws_states_aoc<const float> c_states_tm1_l(
            rnn, c_states_tm1_l_, src_iter_c_ld);
    ws_diff_states_aoc<acc_data_t> diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc<acc_data_t> diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc<acc_data_t> diff_states_t_lp1(rnn, diff_states_t_lp1_);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float Ct = c_states_t_l(i, j);
            /// @todo save it in the workspace in fwd pass or recompute it to
            /// save bw
            float tanhCt = func1(cscale, Ct);
            // we have 2 incoming diffs on Ht
            float dHt = diff_states_tp1_l(0, i, j)
                    + diff_states_t_lp1(rnn.n_states, i, j);
            float dCt = diff_states_tp1_l(1, i, j)
                    + one_m_square(tanhCt) * ws_gates(i, 3, j) * dHt;

            float dG1 = c_states_tm1_l(i, j) * dCt
                    * x_m_square(ws_gates(i, 1, j));
            float dG0 = ws_gates(i, 2, j) * dCt * x_m_square(ws_gates(i, 0, j));
            float dG3 = tanhCt * dHt * x_m_square(ws_gates(i, 3, j));
            float dG2
                    = ws_gates(i, 0, j) * dCt * one_m_square(ws_gates(i, 2, j));

            diff_states_t_l(1, i, j) = dCt * ws_gates(i, 1, j);

            scratch_gates(i, 0, j) = to_src_dt(dG0);
            scratch_gates(i, 1, j) = to_src_dt(dG1);
            scratch_gates(i, 2, j) = to_src_dt(dG2);
            scratch_gates(i, 3, j) = to_src_dt(dG3);
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::lstm_postgemm) {
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);
    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };
    auto to_src_dt = [](float a) { return a; };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_bwd_postgemm_template(tanh_f, to_src_dt, cscale, rnn,
                cell_position, ws_gates_, scratch_gates_, c_states_t_l_,
                c_states_tm1_l_, diff_states_t_l_, diff_states_t_lp1_,
                diff_states_tp1_l_, bias_);
    else
        lstm_bwd_postgemm_template(linear_f, to_src_dt, cscale, rnn,
                cell_position, ws_gates_, scratch_gates_, c_states_t_l_,
                c_states_tm1_l_, diff_states_t_l_, diff_states_t_lp1_,
                diff_states_tp1_l_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::lstm_postgemm) {
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);
    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };
    auto to_src_dt = [](float a) { return bfloat16_t(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_bwd_postgemm_template(tanh_f, to_src_dt, cscale, rnn,
                cell_position, ws_gates_, scratch_gates_, c_states_t_l_,
                c_states_tm1_l_, diff_states_t_l_, diff_states_t_lp1_,
                diff_states_tp1_l_, bias_);
    else
        lstm_bwd_postgemm_template(linear_f, to_src_dt, cscale, rnn,
                cell_position, ws_gates_, scratch_gates_, c_states_t_l_,
                c_states_tm1_l_, diff_states_t_l_, diff_states_t_lp1_,
                diff_states_tp1_l_, bias_);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
