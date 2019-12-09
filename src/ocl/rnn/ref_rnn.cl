/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "ocl/ocl_types.h"

#ifdef OUTPUT_DATA_T
#if OUTPUT_DT_BF16
#define TO_OUTPUT(x) convert_f32_to_bf16(x)
#elif OUTPUT_DT_U8
#define TO_OUTPUT(x) convert_uchar_sat_rte(x)
#elif OUTPUT_DT_S8
#define TO_OUTPUT(x) convert_char_sat_rte(x)
#elif OUTPUT_DT_S32
#define TO_OUTPUT(x) convert_int_sat_rte(x)
#else
#define TO_OUTPUT(x) (x)
#endif
#endif

#if DT_F16 && !IS_FWD
#error "FP16 is not supported for BWD"
#endif

#define OFFTYPE ulong

#define OFF6(i0, D0, i1, D1, i2, D2, i3, D3, i4, D4, i5, D5) \
    ((((((i0) * (D1) + (i1)) * (D2) + (i2)) * (D3) + (i3)) * (D4) + (i4)) \
                    * (D5) \
            + (i5))
#define OFF5(i0, D0, i1, D1, i2, D2, i3, D3, i4, D4) \
    (((((i0) * (D1) + (i1)) * (D2) + (i2)) * (D3) + (i3)) * (D4) + (i4))
#define OFF4(i0, D0, i1, D1, i2, D2, i3, D3) \
    ((((i0) * (D1) + (i1)) * (D2) + (i2)) * (D3) + (i3))
#define OFF3(i0, D0, i1, D1, i2, D2) (((i0) * (D1) + (i1)) * (D2) + (i2))
#define OFF2(i0, D0, i1, D1) ((i0) * (D1) + (i1))

// used for the both H- and C-states
#define OFF_WS_STATE(i0, i1, i2, i3, i4) \
    OFF5((i0), N_LAYER + 1, (i1), N_DIR, (i2), N_ITER + 1, (i3), BATCH, (i4), \
            STATES_WS_LD)

#define OFF_WS_DIFF_STATES(i0, i1, i2, i3, i4, i5) \
    OFF6((i0), N_LAYER + 1, (i1), N_DIR, (i2), N_STATES + 1, (i3), N_ITER + 1, \
            (i4), BATCH, (i5), STATES_WS_LD)

// cannot be presented by OFF6 due to leading dimension across two dims
#define OFF_WS_GATES(i0, i1, i2, i3, i4, i5) \
    (i0) * N_DIR *N_ITER *BATCH *GATES_WS_LD + (i1)*N_ITER *BATCH *GATES_WS_LD \
            + (i2)*BATCH *GATES_WS_LD + (i3)*GATES_WS_LD + (i4)*DIC + (i5)

#define OFF_WS_BIAS(i0, i1, i2, i3) \
    OFF4((i0), N_LAYER, (i1), N_DIR, (i2), N_BIAS, (i3), DIC)

// for cell - shorter forms

#define CELL_WS_GATES(i3, i4, i5) OFF_WS_GATES(0, 0, 0, i3, i4, i5)
#define CELL_WS_STATE(i4, i5) OFF_WS_STATE(0, 0, 0, i4, i5)
#define CELL_WS_DIFF_STATES(i2, i4, i5) OFF_WS_DIFF_STATES(0, 0, i2, 0, i4, i5)

#define OFF_KER_BIAS(i0, i1) OFF2((i0), N_GATES, (i1), DIC)

#define SRC_L_OFF(x0, x1, x2) \
    (((x0) % SRC_L_B0) * SRC_L_SB0 + ((x0) / SRC_L_B0) * SRC_L_S0 \
            + ((x1) % SRC_L_B1) * SRC_L_SB1 + ((x1) / SRC_L_B1) * SRC_L_S1 \
            + ((x2) % SRC_L_B2) * SRC_L_SB2 + ((x2) / SRC_L_B2) * SRC_L_S2)
#define SRC_I_OFF(x0, x1, x2, x3) \
    (((x0) % SRC_I_B0) * SRC_I_SB0 + ((x0) / SRC_I_B0) * SRC_I_S0 \
            + ((x1) % SRC_I_B1) * SRC_I_SB1 + ((x1) / SRC_I_B1) * SRC_I_S1 \
            + ((x2) % SRC_I_B2) * SRC_I_SB2 + ((x2) / SRC_I_B2) * SRC_I_S2 \
            + ((x3) % SRC_I_B3) * SRC_I_SB3 + ((x3) / SRC_I_B3) * SRC_I_S3)
#define SRC_I_C_OFF(x0, x1, x2, x3) \
    (((x0) % SRC_I_C_B0) * SRC_I_C_SB0 + ((x0) / SRC_I_C_B0) * SRC_I_C_S0 \
            + ((x1) % SRC_I_C_B1) * SRC_I_C_SB1 \
            + ((x1) / SRC_I_C_B1) * SRC_I_C_S1 \
            + ((x2) % SRC_I_C_B2) * SRC_I_C_SB2 \
            + ((x2) / SRC_I_C_B2) * SRC_I_C_S2 \
            + ((x3) % SRC_I_C_B3) * SRC_I_C_SB3 \
            + ((x3) / SRC_I_C_B3) * SRC_I_C_S3)
#define DST_L_OFF(x0, x1, x2) \
    (((x0) % DST_L_B0) * DST_L_SB0 + ((x0) / DST_L_B0) * DST_L_S0 \
            + ((x1) % DST_L_B1) * DST_L_SB1 + ((x1) / DST_L_B1) * DST_L_S1 \
            + ((x2) % DST_L_B2) * DST_L_SB2 + ((x2) / DST_L_B2) * DST_L_S2)
#define DST_I_OFF(x0, x1, x2, x3) \
    (((x0) % DST_I_B0) * DST_I_SB0 + ((x0) / DST_I_B0) * DST_I_S0 \
            + ((x1) % DST_I_B1) * DST_I_SB1 + ((x1) / DST_I_B1) * DST_I_S1 \
            + ((x2) % DST_I_B2) * DST_I_SB2 + ((x2) / DST_I_B2) * DST_I_S2 \
            + ((x3) % DST_I_B3) * DST_I_SB3 + ((x3) / DST_I_B3) * DST_I_S3)
#define DST_I_C_OFF(x0, x1, x2, x3) \
    (((x0) % DST_I_C_B0) * DST_I_C_SB0 + ((x0) / DST_I_C_B0) * DST_I_C_S0 \
            + ((x1) % DST_I_C_B1) * DST_I_C_SB1 \
            + ((x1) / DST_I_C_B1) * DST_I_C_S1 \
            + ((x2) % DST_I_C_B2) * DST_I_C_SB2 \
            + ((x2) / DST_I_C_B2) * DST_I_C_S2 \
            + ((x3) % DST_I_C_B3) * DST_I_C_SB3 \
            + ((x3) / DST_I_C_B3) * DST_I_C_S3)
#define BIAS_OFF(x0, x1, x2, x3) \
    (((x0) % BIAS_B0) * BIAS_SB0 + ((x0) / BIAS_B0) * BIAS_S0 \
            + ((x1) % BIAS_B1) * BIAS_SB1 + ((x1) / BIAS_B1) * BIAS_S1 \
            + ((x2) % BIAS_B2) * BIAS_SB2 + ((x2) / BIAS_B2) * BIAS_S2 \
            + ((x3) % BIAS_B3) * BIAS_SB3 + ((x3) / BIAS_B3) * BIAS_S3)

#define DIFF_SRC_L_OFF(x0, x1, x2) \
    (((x0) % DIFF_SRC_L_B0) * DIFF_SRC_L_SB0 \
            + ((x0) / DIFF_SRC_L_B0) * DIFF_SRC_L_S0 \
            + ((x1) % DIFF_SRC_L_B1) * DIFF_SRC_L_SB1 \
            + ((x1) / DIFF_SRC_L_B1) * DIFF_SRC_L_S1 \
            + ((x2) % DIFF_SRC_L_B2) * DIFF_SRC_L_SB2 \
            + ((x2) / DIFF_SRC_L_B2) * DIFF_SRC_L_S2)
#define DIFF_DST_L_OFF(x0, x1, x2) \
    (((x0) % DIFF_DST_L_B0) * DIFF_DST_L_SB0 \
            + ((x0) / DIFF_DST_L_B0) * DIFF_DST_L_S0 \
            + ((x1) % DIFF_DST_L_B1) * DIFF_DST_L_SB1 \
            + ((x1) / DIFF_DST_L_B1) * DIFF_DST_L_S1 \
            + ((x2) % DIFF_DST_L_B2) * DIFF_DST_L_SB2 \
            + ((x2) / DIFF_DST_L_B2) * DIFF_DST_L_S2)
#define DIFF_SRC_I_OFF(x0, x1, x2, x3) \
    (((x0) % DIFF_SRC_I_B0) * DIFF_SRC_I_SB0 \
            + ((x0) / DIFF_SRC_I_B0) * DIFF_SRC_I_S0 \
            + ((x1) % DIFF_SRC_I_B1) * DIFF_SRC_I_SB1 \
            + ((x1) / DIFF_SRC_I_B1) * DIFF_SRC_I_S1 \
            + ((x2) % DIFF_SRC_I_B2) * DIFF_SRC_I_SB2 \
            + ((x2) / DIFF_SRC_I_B2) * DIFF_SRC_I_S2 \
            + ((x3) % DIFF_SRC_I_B3) * DIFF_SRC_I_SB3 \
            + ((x3) / DIFF_SRC_I_B3) * DIFF_SRC_I_S3)
#define DIFF_DST_I_OFF(x0, x1, x2, x3) \
    (((x0) % DIFF_DST_I_B0) * DIFF_DST_I_SB0 \
            + ((x0) / DIFF_DST_I_B0) * DIFF_DST_I_S0 \
            + ((x1) % DIFF_DST_I_B1) * DIFF_DST_I_SB1 \
            + ((x1) / DIFF_DST_I_B1) * DIFF_DST_I_S1 \
            + ((x2) % DIFF_DST_I_B2) * DIFF_DST_I_SB2 \
            + ((x2) / DIFF_DST_I_B2) * DIFF_DST_I_S2 \
            + ((x3) % DIFF_DST_I_B3) * DIFF_DST_I_SB3 \
            + ((x3) / DIFF_DST_I_B3) * DIFF_DST_I_S3)
#define DIFF_SRC_I_C_OFF(x0, x1, x2, x3) \
    (((x0) % DIFF_SRC_I_C_B0) * DIFF_SRC_I_C_SB0 \
            + ((x0) / DIFF_SRC_I_C_B0) * DIFF_SRC_I_C_S0 \
            + ((x1) % DIFF_SRC_I_C_B1) * DIFF_SRC_I_C_SB1 \
            + ((x1) / DIFF_SRC_I_C_B1) * DIFF_SRC_I_C_S1 \
            + ((x2) % DIFF_SRC_I_C_B2) * DIFF_SRC_I_C_SB2 \
            + ((x2) / DIFF_SRC_I_C_B2) * DIFF_SRC_I_C_S2 \
            + ((x3) % DIFF_SRC_I_C_B3) * DIFF_SRC_I_C_SB3 \
            + ((x3) / DIFF_SRC_I_C_B3) * DIFF_SRC_I_C_S3)
#define DIFF_DST_I_C_OFF(x0, x1, x2, x3) \
    (((x0) % DIFF_DST_I_C_B0) * DIFF_DST_I_C_SB0 \
            + ((x0) / DIFF_DST_I_C_B0) * DIFF_DST_I_C_S0 \
            + ((x1) % DIFF_DST_I_C_B1) * DIFF_DST_I_C_SB1 \
            + ((x1) / DIFF_DST_I_C_B1) * DIFF_DST_I_C_S1 \
            + ((x2) % DIFF_DST_I_C_B2) * DIFF_DST_I_C_SB2 \
            + ((x2) / DIFF_DST_I_C_B2) * DIFF_DST_I_C_S2 \
            + ((x3) % DIFF_DST_I_C_B3) * DIFF_DST_I_C_SB3 \
            + ((x3) / DIFF_DST_I_C_B3) * DIFF_DST_I_C_S3)
#define DIFF_BIAS_OFF(x0, x1, x2, x3) \
    (((x0) % DIFF_BIAS_B0) * DIFF_BIAS_SB0 \
            + ((x0) / DIFF_BIAS_B0) * DIFF_BIAS_S0 \
            + ((x1) % DIFF_BIAS_B1) * DIFF_BIAS_SB1 \
            + ((x1) / DIFF_BIAS_B1) * DIFF_BIAS_S1 \
            + ((x2) % DIFF_BIAS_B2) * DIFF_BIAS_SB2 \
            + ((x2) / DIFF_BIAS_B2) * DIFF_BIAS_S2 \
            + ((x3) % DIFF_BIAS_B3) * DIFF_BIAS_SB3 \
            + ((x3) / DIFF_BIAS_B3) * DIFF_BIAS_S3)

float one_m_square(float a) {
    return 1.0f - a * a;
}
float x_m_square(float a) {
    return (1.0f - a) * a;
}
float relu_fwd(float s, float alpha) {
    return s > 0 ? s : s * alpha;
}
float tanh_fwd(float s) {
    return tanh(s);
}
float logistic_fwd(float s) {
    return 1 / (1 + exp((float)-s));
}
float logistic_bwd(float s) {
    return x_m_square(s);
}
float relu_bwd(float s, float alpha) {
    return s > 0 ? 1.f : alpha;
}
float tanh_bwd(float s) {
    return (1 - s) * (1 + s);
}
float linear(float s, float alpha) {
    return alpha * s;
}

float relu_fwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return relu_fwd(s, alpha);
#else
    return linear(s, alpha);
#endif
}
float tanh_fwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return tanh(s);
#else
    return linear(s, alpha);
#endif
}
float logistic_fwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return logistic_fwd(s);
#else
    return linear(s, alpha);
#endif
}

float relu_bwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return relu_bwd(s, alpha);
#else
    return linear(s, alpha);
#endif
}
float tanh_bwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return tanh_bwd(s);
#else
    return linear(s, alpha);
#endif
}
float logistic_bwd_tm(float s, float alpha) {
#if !IS_TESTMODE
    return logistic_bwd(s);
#else
    return linear(s, alpha);
#endif
}

float activation_fwd(float s, float alpha, float cliping) {
#if CELL_KIND == VANILLA_LSTM
    // LSTM doesn't use activation function
    return 0.0f;
#else // VANILLA_RNN
#if ACTIVATION_KIND == ELTWISE_RELU
    return relu_fwd_tm(s, alpha);
#elif ACTIVATION_KIND == ELTWISE_TANH
    return tanh_fwd_tm(s, alpha);
#elif ACTIVATION_KIND == ELTWISE_LOGISTIC
    return logistic_fwd_tm(s, alpha);
#else
#error "Unsupported activation_kind"
#endif
#endif
}
float activation_bwd(float s, float alpha, float cliping) {
#if CELL_KIND == VANILLA_LSTM
    // LSTM doesn't use activation function
    return 0.0f;
#else // VANILLA_RNN
#if ACTIVATION_KIND == ELTWISE_RELU
    return relu_bwd_tm(s, alpha);
#elif ACTIVATION_KIND == ELTWISE_TANH
    return tanh_bwd_tm(s, alpha);
#elif ACTIVATION_KIND == ELTWISE_LOGISTIC
    return logistic_bwd_tm(s, alpha);
#else
#error "Unsupported activation_kind"
#endif
#endif
}

SRC_DATA_T maybe_q(INPUT_DATA_T f, float shift, float scale, int quantize) {
    if (quantize) {
        float qf = f * scale + shift;
        return TO_SRC(qf);
    } else
        return TO_SRC(f);
}

__kernel void ref_rnn_copy_init_layer(
        __global char *ws, __global char *src_base, int lr, int rl) {

#if IS_FWD

    const int it = get_global_id(2);
    const int b = get_global_id(1);
    const int c = get_global_id(0);
    __global SRC_DATA_T *dst;
    __global SRC_DATA_T *dst_base
            = (__global SRC_DATA_T *)(ws + WS_STATES_OFFSET);
    __global SRC_DATA_T *src = (__global SRC_DATA_T *)src_base
            + SRC_L_OFF(it, 0, 0) + b * SLC + c;

    if (lr) {
        dst = dst_base + OFF_WS_STATE(0, 0, it + 1, b, c);
        dst[0] = src[0];
    }
    if (rl) {
        dst = dst_base + OFF_WS_STATE(0, N_DIR - 1, N_ITER - it, b, c);
        dst[0] = src[0];
    }

#else // BWD

    const int it = get_global_id(1);
    const int b = get_global_id(0);

    __global PRECISE_DATA_T *dst
            = (__global PRECISE_DATA_T *)(ws + WS_DIFF_STATES_OFFSET);

#if DIRECTION_KIND == CONCAT
    __global SRC_DATA_T *src
            = (__global SRC_DATA_T *)src_base + DIFF_DST_L_OFF(it, b, 0);
    for (int s = 0; s < DIC; s++) {
        dst[OFF_WS_DIFF_STATES(N_LAYER, 0, N_STATES, it, b, s)] = src[s];
        dst[OFF_WS_DIFF_STATES(N_LAYER, 1, N_STATES, N_ITER - it - 1, b, s)]
                = src[DIC + s];
    }
#elif DIRECTION_KIND == SUM
    __global SRC_DATA_T *src
            = (__global SRC_DATA_T *)src_base + DIFF_DST_L_OFF(it, b, 0);
    for (int s = 0; s < DIC; s++) {
        dst[OFF_WS_DIFF_STATES(N_LAYER, 0, N_STATES, it, b, s)] = src[s];
        dst[OFF_WS_DIFF_STATES(N_LAYER, 1, N_STATES, N_ITER - it - 1, b, s)]
                = src[s];
    }
#elif DIRECTION_KIND == L2R
    __global SRC_DATA_T *src
            = (__global SRC_DATA_T *)src_base + DIFF_DST_L_OFF(it, b, 0);
    for (int s = 0; s < DIC; s++) {
        dst[OFF_WS_DIFF_STATES(N_LAYER, 0, N_STATES, it, b, s)] = src[s];
    }
#elif DIRECTION_KIND == R2L
    __global SRC_DATA_T *src = (__global SRC_DATA_T *)src_base
            + DIFF_DST_L_OFF(N_ITER - it - 1, b, 0);
    for (int s = 0; s < DIC; s++) {
        dst[OFF_WS_DIFF_STATES(N_LAYER, 0, N_STATES, it, b, s)] = src[s];
    }
#else
#error "Unsupported direction_kind"
#endif
#endif
}

__kernel void ref_rnn_copy_init_iter(
        __global char *ws, __global char *src_base, __global char *src_c_base
#if IS_FWD
        ,
        const float shift, const float scale, const int quantize
#endif
) {

    const int s = get_global_id(0);
    const int b = get_global_id(1);
    const int lay = get_global_id(2) / N_DIR;
    const int dir = get_global_id(2) % N_DIR;

#if IS_FWD
    __global INPUT_DATA_T *src = (__global INPUT_DATA_T *)(src_base);
    __global PRECISE_DATA_T *src_c = (__global PRECISE_DATA_T *)(src_c_base);

    __global SRC_DATA_T *dst = (__global SRC_DATA_T *)(ws + WS_STATES_OFFSET);
    if (s < SIC)
        dst[OFF_WS_STATE(lay + 1, dir, 0, b, s)] = src_base
                ? maybe_q(
                        src[SRC_I_OFF(lay, dir, b, s)], shift, scale, quantize)
                : TO_SRC(0.0f);
#if WITH_SRC_ITER_C
    __global PRECISE_DATA_T *dst_c
            = (__global PRECISE_DATA_T *)(ws + WS_C_STATE_OFFSET);
    if (s < DIC)
        dst_c[OFF_WS_STATE(lay + 1, dir, 0, b, s)] = src_c_base
                ? src_c[SRC_I_C_OFF(lay, dir, b, s)]
                : TO_SRC(0.0f);
#endif

#else // BWD

    __global INPUT_DATA_T *src = (__global INPUT_DATA_T *)(src_base);
    __global PRECISE_DATA_T *src_c = (__global PRECISE_DATA_T *)(src_c_base);

    __global PRECISE_DATA_T *dst
            = (__global PRECISE_DATA_T *)(ws + WS_DIFF_STATES_OFFSET);
    if (s < DIC)
        dst[OFF_WS_DIFF_STATES(lay, dir, 0, N_ITER, b, s)]
                = src_base ? src[DIFF_DST_I_OFF(lay, dir, b, s)] : 0.0f;
#if WITH_DST_ITER_C
    if (s < DIC)
        dst[OFF_WS_DIFF_STATES(lay, dir, 1, N_ITER, b, s)]
                = src_c_base ? src_c[DIFF_DST_I_C_OFF(lay, dir, b, s)] : 0.0f;
#endif
#endif
}

DST_DATA_T maybe_dq_l(SRC_DATA_T s, float shift, float scale, int dequantize) {
    if (dequantize) {
        return TO_DST(((float)s - shift) / scale);
    } else
        return TO_DST(s);
}

__kernel void ref_rnn_copy_res_layer(
        __global char *ws, __global char *dst_base, int lr, int rl
#if IS_FWD
        ,
        const float shift, const float scale, const int dequantize
#endif
) {

    const int it = get_global_id(2);
    const int b = get_global_id(1);
    const int s = get_global_id(0);

#if IS_FWD
    __global SRC_DATA_T *src = (__global SRC_DATA_T *)(ws + WS_STATES_OFFSET);
    __global DST_DATA_T *dst = (__global DST_DATA_T *)(dst_base);
    int dir = 0;
    if (lr) {
        dst[DST_L_OFF(it, b, dir * DIC + s)]
                = maybe_dq_l(src[OFF_WS_STATE(N_LAYER, dir, it + 1, b, s)],
                        shift, scale, dequantize);
        dir = 1;
    }
    if (rl) {
#if DIRECTION_KIND == SUM
        if (dequantize) {
            float val
                    = (float)src[OFF_WS_STATE(N_LAYER, dir, N_ITER - it, b, s)]
                    + dst[DST_L_OFF(it, b, s)];
            val = min(max(val, 0.f), 255.f);
            dst[DST_L_OFF(it, b, s)] = TO_DST((val - 2 * shift) / scale);

        } else {
#if defined(SRC_DT_U8) && defined(DST_DT_U8)
            dst[DST_L_OFF(it, b, s)] = convert_uchar_sat(
                    convert_short(
                            src[OFF_WS_STATE(N_LAYER, dir, N_ITER - it, b, s)])
                    + convert_short(dst[DST_L_OFF(it, b, s)]));
#else
            dst[DST_L_OFF(it, b, s)]
                    += src[OFF_WS_STATE(N_LAYER, dir, N_ITER - it, b, s)];
#endif
        }
#else
        dst[DST_L_OFF(it, b, dir * DIC + s)]
                = maybe_dq_l(src[OFF_WS_STATE(N_LAYER, dir, N_ITER - it, b, s)],
                        shift, scale, dequantize);
#endif
    }
#else // BWD

    __global PRECISE_DATA_T *src
            = (__global PRECISE_DATA_T *)(ws + WS_DIFF_STATES_OFFSET);
    __global OUTPUT_DATA_T *dst = (__global OUTPUT_DATA_T *)(dst_base);
    int dir = 0;

#if DIRECTION_KIND == R2L
    const int iter = N_ITER - 1 - it;
#else
    const int iter = it;
#endif
    PRECISE_DATA_T res = src[OFF_WS_DIFF_STATES(0, 0, N_STATES, it, b, s)];
#if N_DIR > 1
    res += src[OFF_WS_DIFF_STATES(0, 1, N_STATES, N_ITER - 1 - it, b, s)];
#endif
    dst[DIFF_SRC_L_OFF(iter, b, dir * SLC + s)] = res;
#endif
}

OUTPUT_DATA_T maybe_dq_i(
        SRC_DATA_T s, float shift, float scale, int dequantize) {
    if (dequantize) {
        return TO_OUTPUT(((float)s - shift) / scale);
    } else
        return TO_OUTPUT(s);
}

__kernel void ref_rnn_copy_res_iter(
        __global char *ws, __global char *dst_base, __global char *dst_c_base
#if IS_FWD
        ,
        const float shift, const float scale, const int dequantize
#endif
) {

    const int s = get_global_id(0);
    const int b = get_global_id(1);
    const int lay = get_global_id(2) / N_DIR;
    const int dir = get_global_id(2) % N_DIR;

#if IS_FWD
    __global SRC_DATA_T *src = (__global SRC_DATA_T *)(ws + WS_STATES_OFFSET);
    __global OUTPUT_DATA_T *dst = (__global OUTPUT_DATA_T *)(dst_base);

    if (dst_base && s < DIC) {
        dst[DST_I_OFF(lay, dir, b, s)]
                = maybe_dq_i(src[OFF_WS_STATE(lay + 1, dir, N_ITER, b, s)],
                        shift, scale, dequantize);
    }
#if WITH_DST_ITER_C
    __global PRECISE_DATA_T *src_c
            = (__global PRECISE_DATA_T *)(ws + WS_C_STATE_OFFSET);
    __global PRECISE_DATA_T *dst_c = (__global PRECISE_DATA_T *)(dst_c_base);
    if (dst_c_base && s < DIC) {
        dst_c[DST_I_C_OFF(lay, dir, b, s)]
                = src_c[OFF_WS_STATE(lay + 1, dir, N_ITER, b, s)];
    }
#endif

#else // BWD

    __global PRECISE_DATA_T *src
            = (__global PRECISE_DATA_T *)(ws + WS_DIFF_STATES_OFFSET);
    __global OUTPUT_DATA_T *dst = (__global OUTPUT_DATA_T *)(dst_base);
    __global PRECISE_DATA_T *dst_c = (__global PRECISE_DATA_T *)(dst_c_base);
    if (dst_base && s < SIC) {
        dst[DIFF_SRC_I_OFF(lay, dir, b, s)]
                = src[OFF_WS_DIFF_STATES(lay, dir, 0, 0, b, s)];
    }
#if WITH_SRC_ITER_C
    if (dst_base && s < DIC) {
        dst_c[DIFF_SRC_I_C_OFF(lay, dir, b, s)]
                = src[OFF_WS_DIFF_STATES(lay, dir, 1, 0, b, s)];
    }
#endif
#endif
}

__kernel void ref_rnn_ws_set(
        __global char *ws, OFFTYPE ws_offset, float val, int ws_part) {

    if (ws_part == WS_C_STATES || ws_part == WS_DIFF_STATES
            || ws_part == WS_BIAS) {
        __global PRECISE_DATA_T *dst
                = (__global PRECISE_DATA_T *)(ws + ws_offset);
        dst[get_global_id(0)] = CONVERT_DATA_T(val);
    } else if (ws_part == WS_GATES) {
        __global ACC_DATA_T *dst = (__global ACC_DATA_T *)(ws + ws_offset);
        dst[get_global_id(0)] = TO_ACC(val);
    } else { // ws_part == WS_STATES
        __global SRC_DATA_T *dst = (__global SRC_DATA_T *)(ws + ws_offset);
        dst[get_global_id(0)] = TO_SRC(val);
    }
}

// useful for debug
#if DEBUGPRINT
__kernel void ref_rnn_ws_print(const __global char *ws) {
    {
        __global ACC_DATA_T *wt = (__global ACC_DATA_T *)(ws + WS_GATES_OFFSET);
        printf("ws_gates: off %d\n", WS_GATES_OFFSET);
        printf("[lay,dir,iter,batch]\n");
        for_(int j = 0; j < N_LAYER; j++)
        for_(int dir = 0; dir < N_DIR; dir++)
        for_(int i = 0; i < N_ITER; i++)
        for (int b = 0; b < BATCH; b++) {
            printf("[%d,%d,%d,%d]: ", j, dir, i, b);
            for_(int g = 0; g < N_GATES; g++)
            for (int s = 0; s < DIC; s++) {
                printf(" %f",
                        TO_DATA_T(*(wt + OFF_WS_GATES(j, dir, i, b, g, s))));
            }
            printf("\n");
        }
    }
    {
        __global SRC_DATA_T *wt
                = (__global SRC_DATA_T *)(ws + WS_STATES_OFFSET);
        printf("ws_states (H): off %d\n", WS_STATES_OFFSET);
        printf("[lay,dir,iter]\n");
        for_(int j = 0; j < N_LAYER + 1; j++)
        for_(int dir = 0; dir < N_DIR; dir++)
        for (int i = 0; i < N_ITER + 1; i++) {
            printf("[%d,%d,%d] : ", j, dir, i);
            for_(int b = 0; b < BATCH; b++)
            for (int s = 0; s < WIC; s++) {
                printf(" %f", TO_DATA_T(*(wt + OFF_WS_STATE(j, dir, i, b, s))));
            }
            printf("\n");
        }
    }
#if IS_FWD
    {
        __global PRECISE_DATA_T *wt
                = (__global PRECISE_DATA_T *)(ws + WS_C_STATE_OFFSET);
        printf("ws_states (C): off %d\n", WS_C_STATE_OFFSET);
        printf("[lay,dir,iter]\n");
        for_(int j = 0; j < N_LAYER + 1; j++)
        for_(int dir = 0; dir < N_DIR; dir++)
        for (int i = 0; i < N_ITER + 1; i++) {
            printf("[%d,%d,%d] : ", j, dir, i);
            for_(int b = 0; b < BATCH; b++)
            for (int s = 0; s < WIC; s++) {
                printf(" %f", *(wt + OFF_WS_STATE(j, dir, i, b, s)));
            }
            printf("\n");
        }
    }
#endif
#if !IS_FWD
    {
        __global PRECISE_DATA_T *wt
                = (__global PRECISE_DATA_T *)(ws + WS_DIFF_STATES_OFFSET);
        printf("ws_diff_states: off %d\n", WS_DIFF_STATES_OFFSET);
        printf("[lay,dir,state,iter]\n");
        for_(int j = 0; j < N_LAYER + 1; j++)
        for_(int dir = 0; dir < N_DIR; dir++)
        for_(int st = 0; st < N_STATES + 1; st++)
        for (int i = 0; i < N_ITER + 1; i++) {
            printf("[%d,%d,%d,%d] : ", j, dir, st, i);
            for_(int b = 0; b < BATCH; b++)
            for (int s = 0; s < WIC; s++) {
                printf(" %f", *(wt + OFF_WS_DIFF_STATES(j, dir, st, i, b, s)));
            }
            printf("\n");
        }
    }
#endif
#if COPY_BIAS
    {
        __global PRECISE_DATA_T *wt
                = (__global PRECISE_DATA_T *)(ws + WS_BIAS_OFFSET);
        printf("ws_bias: off %d\n", WS_BIAS_OFFSET);
        printf("[lay,dir]\n");
        for_(int j = 0; j < N_LAYER; j++)
        for_(int dir = 0; dir < N_DIR; dir++)
        {
            printf("[%d,%d] : ", j, dir);
            for_(int nb = 0; nb < N_BIAS; nb++)
            for (int dic = 0; dic < DIC; dic++) {
                printf(" %f", *(wt + OFF_WS_BIAS(j, dir, nb, dic)));
            }
            printf("\n");
        }
    }
#endif
}
#endif

__kernel void ref_rnn_bias_prepare(__global char *ws, __global float *scales,
        __global char *wei_layer, __global char *wei_iter, __global float *bias,
        float data_shift, float data_scale) {
#if COPY_BIAS

    const int dic = get_global_id(0);
    const int nbias = get_global_id(1);
    const int layer = get_global_id(2) / N_DIR;
    const int dir = get_global_id(2) % N_DIR;

    __global float *ws_bias = (__global float *)(ws + WS_BIAS_OFFSET);

    const float wei_scale
#if WEI_QPARAM_MASK
            = scales[nbias * DIC + dic];
#else
            = scales[0];
#endif

#define COMP_OFF(i0, i1, i2, i3) \
    ((((i0) * (N_DIR) + (i1)) * (N_BIAS) + (i2)) * (DIC) + (i3))
#define COMP_WEI_LAYER_OFF (WEI_L_D0 * WEI_L_S0)
#define COMP_WEI_ITER_OFF (WEI_I_D0 * WEI_I_S0)

    __global char *temp = (__global char *)(wei_iter + COMP_WEI_ITER_OFF);
    __global float *wei_iter_comp
            = (__global float *)(((unsigned long)temp + (sizeof(float) - 1))
                    & -sizeof(float));
    temp = (__global char *)(wei_layer + COMP_WEI_LAYER_OFF);
    __global float *wei_layer_comp
            = (__global float *)(((unsigned long)temp + (sizeof(float) - 1))
                    & -sizeof(float));

    const int off = COMP_OFF(layer, dir, nbias, dic);
    const float comp = wei_layer_comp[off] + wei_iter_comp[off];
    ws_bias[OFF_WS_BIAS(layer, dir, nbias, dic)]
            = bias[BIAS_OFF(layer, dir, nbias, dic)]
            - comp * data_shift / (wei_scale * data_scale);

#endif
}

#if IS_INT8 && CELL_KIND == VANILLA_LSTM

SRC_DATA_T q_d(float f, float data_scale, float data_shift) {
    float qf = f * data_scale + data_shift;
    return TO_SRC(qf);
}

float deq_w(ACC_DATA_T s, int gate, int j, __global float *scales,
        float data_scale) {
#if WEI_QPARAM_MASK
    float wei_scale = scales[gate * DIC + j];
#else
    float wei_scale = scales[0];
#endif
    return (float)(s) / (wei_scale * data_scale);
}

// for int8 LSTM
__kernel void ref_rnn_elemwise_fwd(int dir, int lay, int iter,
        __global char *ws, __global float *scales, __global float *bias_base,
        float alpha, float data_shift, float data_scale,
        __global float *tm_scales, float tm_cscale) {

    const int i = get_global_id(1); // batch
    const int j = get_global_id(0); // dic

    const __global float *c_states_tm1_l
            = (__global float *)(ws + WS_C_STATE_OFFSET)
            + OFF_WS_STATE(lay + 1, dir, iter, 0, 0);
    __global float *ws_bias = (__global float *)(ws + WS_BIAS_OFFSET);

    __global ACC_DATA_T *ws_gates
            = (__global ACC_DATA_T *)(ws + WS_GATES_OFFSET)
            + OFF_WS_GATES(lay, dir, iter, 0, 0, 0);

    __global SRC_DATA_T *h_states_t_l
            = (__global SRC_DATA_T *)(ws + WS_STATES_OFFSET)
            + OFF_WS_STATE(lay + 1, dir, iter + 1, 0, 0);
    __global float *c_states_t_l = (__global float *)(ws + WS_C_STATE_OFFSET)
            + OFF_WS_STATE(lay + 1, dir, iter + 1, 0, 0);

    float G0 = logistic_fwd_tm(
            deq_w(ws_gates[CELL_WS_GATES(i, 0, j)], 0, j, scales, data_scale)
                    + ws_bias[OFF_WS_BIAS(lay, dir, 0, j)],
            tm_scales[0]);
    float G1 = logistic_fwd_tm(
            deq_w(ws_gates[CELL_WS_GATES(i, 1, j)], 1, j, scales, data_scale)
                    + ws_bias[OFF_WS_BIAS(lay, dir, 1, j)],
            tm_scales[1]);
    float G2 = tanh_fwd_tm(
            deq_w(ws_gates[CELL_WS_GATES(i, 2, j)], 2, j, scales, data_scale)
                    + ws_bias[OFF_WS_BIAS(lay, dir, 2, j)],
            tm_scales[2]);
    float G3 = logistic_fwd_tm(
            deq_w(ws_gates[CELL_WS_GATES(i, 3, j)], 3, j, scales, data_scale)
                    + ws_bias[OFF_WS_BIAS(lay, dir, 3, j)],
            tm_scales[3]);

    float tmp = G1 * c_states_tm1_l[CELL_WS_STATE(i, j)] + G0 * G2;

    h_states_t_l[CELL_WS_STATE(i, j)]
            = q_d(G3 * tanh_fwd_tm(tmp, tm_cscale), data_scale, data_shift);
    c_states_t_l[CELL_WS_STATE(i, j)] = tmp;
}

#else

__kernel void ref_rnn_elemwise_fwd(int dir, int lay, int iter,
        __global char *ws, __global PRECISE_DATA_T *bias_base, float alpha,
        __global float *tm_scales, float tm_cscale) {

    const int i = get_global_id(1); // batch
    const int j = get_global_id(0); // dic

    const __global PRECISE_DATA_T *c_states_tm1_l
            = (__global PRECISE_DATA_T *)(ws + WS_C_STATE_OFFSET)
            + OFF_WS_STATE(lay + 1, dir, iter, 0, 0);
    const __global PRECISE_DATA_T *bias = bias_base + BIAS_OFF(lay, dir, 0, 0);
    __global PRECISE_DATA_T *ws_gates
            = (__global PRECISE_DATA_T *)(ws + WS_GATES_OFFSET)
            + OFF_WS_GATES(lay, dir, iter, 0, 0, 0);

    __global SRC_DATA_T *h_states_t_l
            = (__global SRC_DATA_T *)(ws + WS_STATES_OFFSET)
            + OFF_WS_STATE(lay + 1, dir, iter + 1, 0, 0);
    __global PRECISE_DATA_T *c_states_t_l
            = (__global PRECISE_DATA_T *)(ws + WS_C_STATE_OFFSET)
            + OFF_WS_STATE(lay + 1, dir, iter + 1, 0, 0);

#if CELL_KIND == VANILLA_LSTM

    float g_i = logistic_fwd_tm(
            (float)ws_gates[CELL_WS_GATES(i, 0, j)] + bias[OFF_KER_BIAS(0, j)],
            tm_scales[0]);
    float g_f = logistic_fwd_tm(
            (float)ws_gates[CELL_WS_GATES(i, 1, j)] + bias[OFF_KER_BIAS(1, j)],
            tm_scales[1]);
    float g_z = tanh_fwd_tm(
            (float)ws_gates[CELL_WS_GATES(i, 2, j)] + bias[OFF_KER_BIAS(2, j)],
            tm_scales[2]);
    float g_o = logistic_fwd_tm(
            (float)ws_gates[CELL_WS_GATES(i, 3, j)] + bias[OFF_KER_BIAS(3, j)],
            tm_scales[3]);

    ws_gates[CELL_WS_GATES(i, 0, j)] = g_i;
    ws_gates[CELL_WS_GATES(i, 1, j)] = g_f;
    ws_gates[CELL_WS_GATES(i, 2, j)] = g_z;
    ws_gates[CELL_WS_GATES(i, 3, j)] = g_o;

    float Ct = g_f * c_states_tm1_l[CELL_WS_STATE(i, j)] + g_i * g_z;
    float Ht = g_o * tanh_fwd_tm(Ct, tm_cscale);

    h_states_t_l[CELL_WS_STATE(i, j)] = Ht;
    c_states_t_l[CELL_WS_STATE(i, j)] = Ct;

#elif CELL_KIND == VANILLA_RNN

    float g = activation_fwd(
            (float)ws_gates[CELL_WS_GATES(i, 0, j)] + bias[OFF_KER_BIAS(0, j)],
#if IS_TESTMODE
            tm_scales[0], 0);
#else
            alpha, 0);
#endif

    ws_gates[CELL_WS_GATES(i, 0, j)] = g;
    h_states_t_l[CELL_WS_STATE(i, j)] = g;

#else
#error "Wrong cell kind"
#endif
}
#endif

__kernel void ref_rnn_elemwise_bwd(int dir, int lay, int iter,
        __global char *ws, __global PRECISE_DATA_T *bias_base, float alpha,
        __global float *tm_scales, float tm_cscale) {

    const int i = get_global_id(1); // batch
    const int j = get_global_id(0); // dic

#if CELL_KIND == VANILLA_LSTM
    __global PRECISE_DATA_T *ws_gates
            = (__global PRECISE_DATA_T *)(ws + WS_GATES_OFFSET)
            + OFF_WS_GATES(lay, dir, iter, 0, 0, 0);
    __global PRECISE_DATA_T *c_states_t_l
            = (__global PRECISE_DATA_T *)(ws + WS_C_STATE_OFFSET)
            + OFF_WS_STATE(lay + 1, dir, iter + 1, 0, 0);
    __global PRECISE_DATA_T *c_states_tm1_l
            = (__global PRECISE_DATA_T *)(ws + WS_C_STATE_OFFSET)
            + OFF_WS_STATE(lay + 1, dir, iter, 0, 0);
    __global PRECISE_DATA_T *diff_states_t_l
            = (__global PRECISE_DATA_T *)(ws + WS_DIFF_STATES_OFFSET)
            + OFF_WS_DIFF_STATES(lay, dir, 0, iter, 0, 0);
    __global PRECISE_DATA_T *diff_states_tp1_l
            = (__global PRECISE_DATA_T *)(ws + WS_DIFF_STATES_OFFSET)
            + OFF_WS_DIFF_STATES(lay, dir, 0, iter + 1, 0, 0);
    __global PRECISE_DATA_T *diff_states_t_lp1
            = (__global PRECISE_DATA_T *)(ws + WS_DIFF_STATES_OFFSET)
            + OFF_WS_DIFF_STATES(lay + 1, dir, 0, iter, 0, 0);

    float Ct = c_states_t_l[CELL_WS_STATE(i, j)];
    /// @todo save it in the workspace in fwd pass or recompute it to
    /// save bw
    float tanhCt = tanh_fwd_tm(Ct, tm_cscale);
    // we have 2 incoming diffs on Ht
    float dHt = (float)diff_states_tp1_l[CELL_WS_DIFF_STATES(0, i, j)]
            + diff_states_t_lp1[CELL_WS_DIFF_STATES(N_STATES, i, j)];
    float dCt = (float)diff_states_tp1_l[CELL_WS_DIFF_STATES(1, i, j)]
            + one_m_square(tanhCt) * ws_gates[CELL_WS_GATES(i, 3, j)] * dHt;

    float dG1 = (float)c_states_tm1_l[CELL_WS_STATE(i, j)] * dCt
            * x_m_square((float)ws_gates[CELL_WS_GATES(i, 1, j)]);
    float dG0 = (float)ws_gates[CELL_WS_GATES(i, 2, j)] * dCt
            * x_m_square((float)ws_gates[CELL_WS_GATES(i, 0, j)]);
    float dG3 = tanhCt * dHt
            * x_m_square((float)ws_gates[CELL_WS_GATES(i, 3, j)]);
    float dG2 = ws_gates[CELL_WS_GATES(i, 0, j)] * dCt
            * one_m_square((float)ws_gates[CELL_WS_GATES(i, 2, j)]);

    diff_states_t_l[CELL_WS_DIFF_STATES(1, i, j)]
            = dCt * ws_gates[CELL_WS_GATES(i, 1, j)];

    ws_gates[CELL_WS_GATES(i, 0, j)] = dG0;
    ws_gates[CELL_WS_GATES(i, 1, j)] = dG1;
    ws_gates[CELL_WS_GATES(i, 2, j)] = dG2;
    ws_gates[CELL_WS_GATES(i, 3, j)] = dG3;

#elif CELL_KIND == VANILLA_RNN
    __global PRECISE_DATA_T *ws_gates
            = (__global PRECISE_DATA_T *)(ws + WS_GATES_OFFSET)
            + OFF_WS_GATES(lay, dir, iter, i, 0, j);
    __global PRECISE_DATA_T *ws_diff_states
            = (__global PRECISE_DATA_T *)(ws + WS_DIFF_STATES_OFFSET);
    __global PRECISE_DATA_T *diff_states_t_lp1 = ws_diff_states
            + OFF_WS_DIFF_STATES(lay + 1, dir, N_STATES, iter, i, j);
    __global PRECISE_DATA_T *diff_states_tp1_l
            = ws_diff_states + OFF_WS_DIFF_STATES(lay, dir, 0, iter + 1, i, j);

    const float dH = (float)diff_states_t_lp1[0] + diff_states_tp1_l[0];

    float g = ws_gates[0];
#if IS_TESTMODE
    ws_gates[0] = dH * activation_bwd(g, tm_scales[0], 0.);
#else
    ws_gates[0] = dH * activation_bwd(g, alpha, 0.);
#endif

#else
#error "Wrong cell kind"
#endif
}

__kernel void ref_rnn_gates_reduction(int dir, int lay, int iter,
        __global PRECISE_DATA_T *diff_bias_base, __global char *ws) {
#if !IS_FWD
    const int i = get_global_id(0); // n_gates
    const int k = get_global_id(1); // dic

    __global PRECISE_DATA_T *diff_bias
            = diff_bias_base + DIFF_BIAS_OFF(lay, dir, 0, 0);
    __global PRECISE_DATA_T *ws_gates
            = (__global PRECISE_DATA_T *)(ws + WS_GATES_OFFSET)
            + OFF_WS_GATES(lay, dir, iter, 0, 0, 0);

    for (int j = 0; j < BATCH; j++) {
        diff_bias[i * DIC + k] += ws_gates[j * GATES_WS_LD + i * DIC + k];
    }
#endif
}
