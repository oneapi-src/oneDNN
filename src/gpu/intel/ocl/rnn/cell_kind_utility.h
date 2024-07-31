/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_RNN_CELL_KIND_UTILITY_H
#define GPU_INTEL_OCL_RNN_CELL_KIND_UTILITY_H

inline float one_m_square(float a) {
    return 1.0f - a * a;
}
inline float x_m_square(float a) {
    return (1.0f - a) * a;
}
inline float relu_fwd(float s, float alpha) {
    return s > 0 ? s : s * alpha;
}
inline float tanh_fwd(float s) {
    return tanh(s);
}
inline float logistic_fwd(float s) {
    return 1 / (1 + exp((float)-s));
}
inline float logistic_bwd(float s) {
    return x_m_square(s);
}
inline float relu_bwd(float s, float alpha) {
    return s > 0 ? 1.f : alpha;
}
inline float tanh_bwd(float s) {
    return (1 - s) * (1 + s);
}
inline float linear(float s, float alpha) {
    return alpha * s;
}

inline float relu_fwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return relu_fwd(s, alpha);
#else
    return linear(s, alpha);
#endif
}
inline float tanh_fwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return tanh(s);
#else
    return linear(s, alpha);
#endif
}
inline float logistic_fwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return logistic_fwd(s);
#else
    return linear(s, alpha);
#endif
}

inline float relu_bwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return relu_bwd(s, alpha);
#else
    return linear(s, alpha);
#endif
}
inline float tanh_bwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return tanh_bwd(s);
#else
    return linear(s, alpha);
#endif
}
inline float logistic_bwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return logistic_bwd(s);
#else
    return linear(s, alpha);
#endif
}

inline float activation_fwd(float s, float alpha, float cliping) {
#if CELL_KIND == VANILLA_RNN
#if ACTIVATION_KIND == ELTWISE_RELU
    return relu_fwd_tm(s, alpha);
#elif ACTIVATION_KIND == ELTWISE_TANH
    return tanh_fwd_tm(s, alpha);
#elif ACTIVATION_KIND == ELTWISE_LOGISTIC
    return logistic_fwd_tm(s, alpha);
#else
#error "Unsupported activation_kind"
#endif
#else
    return 0.0f;
#endif
}
inline float activation_bwd(float s, float alpha, float cliping) {
#if CELL_KIND == VANILLA_RNN
#if ACTIVATION_KIND == ELTWISE_RELU
    return relu_bwd_tm(s, alpha);
#elif ACTIVATION_KIND == ELTWISE_TANH
    return tanh_bwd_tm(s, alpha);
#elif ACTIVATION_KIND == ELTWISE_LOGISTIC
    return logistic_bwd_tm(s, alpha);
#else
#error "Unsupported activation_kind"
#endif
#else
    return 0.0f;
#endif
}

float vanilla_rnn_compute_gates(float G0, float B0, float alpha,
        const __global float *restrict tm_scales) {
    float G = activation_fwd(G0 + B0,
#if IS_TESTMODE
            tm_scales[0], 0);
#else
            alpha, 0);
#endif
    return G;
}

void store_vanilla_rnn(__global AUX_DATA_T *ws_gates, int gates_ws_ld,
        __global WS_STATE_DATA_T *h_states_t_l, int states_ws_ld, int dhc,
        int n, int c, float g) {
    if (!RECOMPUTE_GATES && IS_TRAINING) {
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, n, 0, c)] = g;
    }
    h_states_t_l[cell_ws_state(states_ws_ld, n, c)] = TO_WS_STATE(g);
}

typedef struct vanilla_lstm_gates_t {
    float G[vanilla_lstm_n_gates];
} vanilla_lstm_gates_t;

struct vanilla_lstm_gates_t vanilla_lstm_compute_gates(
        const float G[vanilla_lstm_n_gates],
        const float B[vanilla_lstm_n_gates],
        const __global float *restrict tm_scales) {

    vanilla_lstm_gates_t ret;
    ret.G[0] = logistic_fwd_tm(G[0] + B[0], tm_scales[0]);
    ret.G[1] = logistic_fwd_tm(G[1] + B[1], tm_scales[1]);
    ret.G[2] = tanh_fwd_tm(G[2] + B[2], tm_scales[2]);
    ret.G[3] = logistic_fwd_tm(G[3] + B[3], tm_scales[3]);
    return ret;
}

void vanilla_lstm_store(__global AUX_DATA_T *ws_gates, int gates_ws_ld,
        __global WS_STATE_DATA_T *h_states_t_l,
        __global AUX_DATA_T *c_states_t_l,
        const __global AUX_DATA_T *c_states_tm1_l, int states_ws_ld, int dhc,
        int n, int c, float tm_cscale, vanilla_lstm_gates_t gates) {
    float g_i = gates.G[0];
    float g_f = gates.G[1];
    float g_z = gates.G[2];
    float g_o = gates.G[3];

    if (!RECOMPUTE_GATES && IS_TRAINING) {
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, n, 0, c)] = g_i;
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, n, 1, c)] = g_f;
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, n, 2, c)] = g_z;
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, n, 3, c)] = g_o;
    }

    float Ct = g_f * c_states_tm1_l[cell_ws_state(states_ws_ld, n, c)]
            + g_i * g_z;
    float Ht = g_o * tanh_fwd_tm(Ct, tm_cscale);

    h_states_t_l[cell_ws_state(states_ws_ld, n, c)] = TO_WS_STATE(Ht);
    c_states_t_l[cell_ws_state(states_ws_ld, n, c)] = Ct;
}

#if IS_INT8 && CELL_KIND == VANILLA_LSTM
inline WS_STATE_DATA_T q_d(float f, float data_scale, float data_shift) {
    return TO_WS_STATE(f * data_scale + data_shift);
}
inline float deq_w(ACC_DATA_T s, int gate, int j, __global float *scales,
        float data_scale, int dhc) {
#if WEI_QPARAM_MASK
    float wei_scale = scales[gate * dhc + j];
#else
    float wei_scale = scales[0];
#endif
    return (float)(s) / (wei_scale * data_scale);
}
#endif // IS_INT8

#if CELL_KIND == LBR_GRU
typedef struct lbr_gru_gates_t {
    float Wh_b;
    float G[3];
} lbr_gru_gates_t;

struct lbr_gru_gates_t compute_gates_lbr_gru(
        const __global ACC_DATA_T *restrict scratch_gates,
        const __global AUX_DATA_T *restrict scratch_cell,
        const __global BIAS_DATA_T *restrict bias,
        const __global float *restrict tm_scales, int scratch_gates_ld, int dhc,
        int mb, int c) {
    float G[n_gates];
    float B[n_bias];
    float C[n_gates];
    for (int i = 0; i < n_gates; i++) {
        G[i] = convert_float(scratch_gates[cell_scratch_mem(
                scratch_gates_ld, dhc, mb, i, c)]);
        C[i] = convert_float(scratch_cell[cell_scratch_mem(
                scratch_gates_ld, dhc, mb, i, c)]);
    }
    for (int i = 0; i < n_bias; i++) {
        B[i] = convert_float(bias[off_ker_bias(dhc, i, c)]);
    }

    lbr_gru_gates_t ret;
    ret.Wh_b = C[2] + B[3];
    ret.G[0] = logistic_fwd_tm(G[0] + C[0] + B[0], tm_scales[0]);
    ret.G[1] = logistic_fwd_tm(G[1] + C[1] + B[1], tm_scales[1]);
    ret.G[2] = tanh_fwd_tm(G[2] + ret.G[1] * ret.Wh_b + B[2], tm_scales[2]);
    return ret;
}
#endif

#endif
