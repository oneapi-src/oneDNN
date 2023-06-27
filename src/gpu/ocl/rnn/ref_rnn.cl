/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/rnn/rnn_types.h"

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
float activation_bwd(float s, float alpha, float cliping) {
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

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
ref_rnn_copy_init_layer(__global WS_STATE_DATA_T *dst_base,
        __global char *src_base, __global DIFF_DATA_T *scratch_diff_states,
        int lr, int rl, int batch, int dhc, int slc, int n_iter, int n_layer,
        int n_dir, int n_states, int states_ws_ld, int scratch_diff_states_ld,
#if IS_FWD
        int it_stride
#else
        int it_stride, int b_stride
#endif
) {

#if IS_FWD

    const int it = get_global_id(2);
    const int b = get_global_id(1);
    const int c = get_global_id(0);
    if (c >= slc || b >= batch || it >= n_iter) return;

    __global WS_STATE_DATA_T *dst;
    __global WS_STATE_DATA_T *src = (__global WS_STATE_DATA_T *)src_base
            + src_l_off(it_stride, it) + b * slc + c;

    if (lr) {
        dst = dst_base
                + off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld, 0,
                        0, it + 1, b, c);
        dst[0] = src[0];
    }
    if (rl) {
        dst = dst_base
                + off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld, 0,
                        n_dir - 1, n_iter - it, b, c);
        dst[0] = src[0];
    }

#else // BWD

    const int it = get_global_id(2);
    const int b = get_global_id(1);
    const int s = get_global_id(0);
    if (s >= dhc || b >= batch || it >= n_iter) return;

    __global DIFF_DATA_T *dst = scratch_diff_states;
    const param4 strides = {it_stride, b_stride, 0, 0};

#if DIRECTION_KIND == CONCAT
    __global DIFF_DATA_T *src
            = (__global DIFF_DATA_T *)src_base + diff_dst_l_off(strides, it, b);
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 0, n_states, it, b, s)]
            = src[s];
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 1, n_states, n_iter - it - 1, b,
            s)]
            = src[dhc + s];
#elif DIRECTION_KIND == SUM
    __global DIFF_DATA_T *src
            = (__global DIFF_DATA_T *)src_base + diff_dst_l_off(strides, it, b);
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 0, n_states, it, b, s)]
            = src[s];
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 1, n_states, n_iter - it - 1, b,
            s)]
            = src[s];
#elif DIRECTION_KIND == L2R
    __global DIFF_DATA_T *src
            = (__global DIFF_DATA_T *)src_base + diff_dst_l_off(strides, it, b);
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 0, n_states, it, b, s)]
            = src[s];
#elif DIRECTION_KIND == R2L
    __global DIFF_DATA_T *src = (__global DIFF_DATA_T *)src_base
            + diff_dst_l_off(strides, n_iter - it - 1, b);
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 0, n_states, it, b, s)]
            = src[s];
#else
#error "Unsupported direction_kind"
#endif
#endif
}

__kernel void ref_rnn_copy_init_iter(__global WS_STATE_DATA_T *dst_base,
        __global AUX_DATA_T *dst_c_base, __global char *src_base,
        __global char *src_c_base, __global char *scratch_diff_states,
        int batch, int dhc, int sic, int n_iter, int n_layer, int n_dir,
        int n_states, int states_ws_ld, int scratch_diff_states_ld,
#if IS_FWD
        int lay_stride, int dir_stride, int b_stride, int s_stride
#if WITH_SRC_ITER_C
        ,
        int lay_c_stride, int dir_c_stride, int b_c_stride, int s_c_stride
#endif // WITH_SRC_ITER_C
        ,
        const float shift, const float scale, const int quantize
#else // BWD
        int lay_stride, int dir_stride, int b_stride, int s_stride
#if WITH_DST_ITER_C
        ,
        int lay_c_stride, int dir_c_stride, int b_c_stride, int s_c_stride
#endif // WITH_DST_ITER_C
#endif // IS_FWD
) {

    const int s = get_global_id(0);
    const int b = get_global_id(1);
    const int lay = get_global_id(2) / n_dir;
    const int dir = get_global_id(2) % n_dir;
    const param4 strides = {lay_stride, dir_stride, b_stride, s_stride};

#if IS_FWD
    __global INPUT_DATA_T *src = (__global INPUT_DATA_T *)(src_base);
    __global WS_STATE_DATA_T *dst = dst_base;
    int ws_state_offset = off_ws_state(
            n_layer, n_dir, n_iter, batch, states_ws_ld, lay + 1, dir, 0, b, s);
    if (s < sic) {
        int src_i_offset = src_i_off(strides, lay, dir, b, s);
        dst[ws_state_offset] = src_base
                ? (quantize ? TO_WS_STATE(src[src_i_offset] * scale + shift)
                            : src[src_i_offset])
                : TO_WS_STATE(0.0f);
    }
#if WITH_SRC_ITER_C
    __global AUX_DATA_T *src_c = (__global AUX_DATA_T *)(src_c_base);
    __global AUX_DATA_T *dst_c = dst_c_base;
    if (s < dhc) {
        const param4 c_strides
                = {lay_c_stride, dir_c_stride, b_c_stride, s_c_stride};
        dst_c[ws_state_offset] = src_c_base
                ? src_c[src_i_c_off(c_strides, lay, dir, b, s)]
                : TO_WS_STATE(0.0f);
    }
#endif

#else // BWD

    __global DIFF_DATA_T *src = (__global DIFF_DATA_T *)(src_base);
    __global DIFF_DATA_T *dst = (__global DIFF_DATA_T *)scratch_diff_states;

    if (s < dhc)
        dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
                scratch_diff_states_ld, lay, dir, 0, n_iter, b, s)]
                = src_base ? src[diff_dst_i_off(strides, lay, dir, b, s)]
                           : 0.0f;
#if WITH_DST_ITER_C
    __global DIFF_DATA_T *src_c = (__global DIFF_DATA_T *)(src_c_base);
    if (s < dhc) {
        const param4 c_strides
                = {lay_c_stride, dir_c_stride, b_c_stride, s_c_stride};
        dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
                scratch_diff_states_ld, lay, dir, 1, n_iter, b, s)]
                = src_c_base
                ? src_c[diff_dst_i_c_off(c_strides, lay, dir, b, s)]
                : 0.0f;
    }
#endif
#endif
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
ref_rnn_copy_res_layer(__global WS_STATE_DATA_T *src_base,
        __global char *dst_base, __global char *scratch_diff_states, int lr,
        int rl, int batch, int dhc, int slc, int n_iter, int n_layer, int n_dir,
        int n_states, int states_ws_ld, int scratch_diff_states_ld,
#if IS_FWD
        int it_stride, int b_stride, int s_stride, const float shift,
        const float scale, const int dequantize
#else
        int it_stride, int b_stride, int s_stride
#endif
) {

    const int it = get_global_id(2);
    const int b = get_global_id(1);
    const int s = get_global_id(0);
    const param4 strides = {it_stride, b_stride, s_stride, 0};

#if IS_FWD
    if (s >= dhc || b >= batch || it >= n_iter) return;
    __global WS_STATE_DATA_T *src = src_base;
    __global DST_DATA_T *dst = (__global DST_DATA_T *)(dst_base);
    int dir = 0;
    if (lr) {
        bool dequantize_at_copy = dequantize && DIRECTION_KIND != SUM;
        dst[dst_l_off(strides, it, b, dir * dhc + s)] = dequantize_at_copy
                ? TO_DST(((float)src[off_ws_state(n_layer, n_dir, n_iter, batch,
                                  states_ws_ld, n_layer, dir, it + 1, b, s)]
                                 - shift)
                        / scale)
                : src[off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld,
                        n_layer, dir, it + 1, b, s)];
        dir = 1;
    }
    if (rl) {
#if DIRECTION_KIND == SUM
        if (dequantize) {
            float val = (float)src[off_ws_state(n_layer, n_dir, n_iter, batch,
                                states_ws_ld, n_layer, dir, n_iter - it, b, s)]
                    + dst[dst_l_off(strides, it, b, s)];
            val = min(max(val, 0.f), 255.f);
            dst[dst_l_off(strides, it, b, s)]
                    = TO_DST((val - 2 * shift) / scale);
        } else {
#if defined(SRC_DT_U8) && defined(DST_DT_U8)
            dst[dst_l_off(strides, it, b, s)] = convert_uchar_sat(
                    convert_short(src[off_ws_state(n_layer, n_dir, n_iter,
                            batch, states_ws_ld, n_layer, dir, n_iter - it, b,
                            s)])
                    + convert_short(dst[dst_l_off(strides, it, b, s)]));
#else
            ACC_DATA_T temp_src = DST_TO_REF(dst[dst_l_off(strides, it, b, s)]);
            temp_src += DST_TO_REF(src[off_ws_state(n_layer, n_dir, n_iter,
                    batch, states_ws_ld, n_layer, dir, n_iter - it, b, s)]);
            dst[dst_l_off(strides, it, b, s)] = REF_TO_DST(temp_src);
#endif
        }
#else
        dst[dst_l_off(strides, it, b, dir * dhc + s)] = dequantize
                ? TO_DST(
                        ((float)src[off_ws_state(n_layer, n_dir, n_iter, batch,
                                 states_ws_ld, n_layer, dir, n_iter - it, b, s)]
                                - shift)
                        / scale)
                : src[off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld,
                        n_layer, dir, n_iter - it, b, s)];
#endif
    }
#else // BWD

    if (s >= slc || b >= batch || it >= n_iter) return;
    __global DIFF_DATA_T *src = (__global DIFF_DATA_T *)(scratch_diff_states);
    __global DIFF_DATA_T *dst = (__global DIFF_DATA_T *)(dst_base);
    int dir = 0;

#if DIRECTION_KIND == R2L
    const int iter = n_iter - 1 - it;
#else
    const int iter = it;
#endif
    DIFF_DATA_T res = src[off_scratch_diff_states(n_layer, n_dir, n_states,
            n_iter, batch, scratch_diff_states_ld, 0, 0, n_states, it, b, s)];
    if (n_dir > 1) {
        res += src[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                batch, scratch_diff_states_ld, 0, 1, n_states, n_iter - 1 - it,
                b, s)];
    }
    dst[diff_src_l_off(strides, iter, b, dir * slc + s)] = res;
#endif
}

__kernel void ref_rnn_copy_res_iter(__global WS_STATE_DATA_T *src_base,
        __global AUX_DATA_T *src_c_base, __global char *dst_base,
        __global char *dst_c_base, __global char *scratch_diff_states,
        int batch, int dhc, int sic, int n_iter, int n_layer, int n_dir,
        int n_states, int states_ws_ld, int scratch_diff_states_ld,
#if IS_FWD
        int lay_stride, int dir_stride, int b_stride, int s_stride
#if WITH_DST_ITER_C
        ,
        int lay_c_stride, int dir_c_stride, int b_c_stride, int s_c_stride
#endif // WITH_DST_ITER_C
        ,
        const float shift, const float scale, const int dequantize
#else // BWD
        int lay_stride, int dir_stride, int b_stride, int s_stride
#if WITH_SRC_ITER_C
        ,
        int lay_c_stride, int dir_c_stride, int b_c_stride, int s_c_stride
#endif // WITH_SRC_ITER_C
#endif
) {

    const int s = get_global_id(0);
    const int b = get_global_id(1);
    const int lay = get_global_id(2) / n_dir;
    const int dir = get_global_id(2) % n_dir;
    const param4 strides = {lay_stride, dir_stride, b_stride, s_stride};

#if IS_FWD
    __global WS_STATE_DATA_T *src = src_base;
    __global OUTPUT_DATA_T *dst = (__global OUTPUT_DATA_T *)(dst_base);

    if (dst_base && s < dhc) {
        dst[dst_i_off(strides, lay, dir, b, s)] = dequantize
                ? TO_OUTPUT(
                        ((float)src[off_ws_state(n_layer, n_dir, n_iter, batch,
                                 states_ws_ld, lay + 1, dir, n_iter, b, s)]
                                - shift)
                        / scale)
                : TO_OUTPUT(src[off_ws_state(n_layer, n_dir, n_iter, batch,
                        states_ws_ld, lay + 1, dir, n_iter, b, s)]);
    }
#if WITH_DST_ITER_C
    __global AUX_DATA_T *src_c = src_c_base;
    __global AUX_DATA_T *dst_c = (__global AUX_DATA_T *)(dst_c_base);
    const param4 c_strides
            = {lay_c_stride, dir_c_stride, b_c_stride, s_c_stride};
    if (dst_c_base && s < dhc) {
        dst_c[dst_i_c_off(c_strides, lay, dir, b, s)]
                = src_c[off_ws_state(n_layer, n_dir, n_iter, batch,
                        states_ws_ld, lay + 1, dir, n_iter, b, s)];
    }
#endif

#else // BWD

    __global DIFF_DATA_T *src = (__global DIFF_DATA_T *)(scratch_diff_states);
    __global DIFF_DATA_T *dst = (__global DIFF_DATA_T *)(dst_base);
    __global DIFF_DATA_T *dst_c = (__global DIFF_DATA_T *)(dst_c_base);
    if (dst_base && s < sic) {
        dst[diff_src_i_off(strides, lay, dir, b, s)]
                = src[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 0, 0, b, s)];
    }
#if WITH_SRC_ITER_C
    if (dst_base && s < dhc) {
        const param4 c_strides
                = {lay_c_stride, dir_c_stride, b_c_stride, s_c_stride};
        dst_c[diff_src_i_c_off(c_strides, lay, dir, b, s)]
                = src[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 1, 0, b, s)];
    }
#endif
#endif
}

__kernel void ref_rnn_ws_set(
        __global char *ws, OFFTYPE ws_offset, float val, int ws_part) {

    if (ws_part == WS_C_STATES || ws_part == WS_BIAS) {
        __global DIFF_DATA_T *dst = (__global DIFF_DATA_T *)(ws + ws_offset);
        dst[get_global_id(0)] = CONVERT_DATA_T(val);
    } else if (ws_part == WS_GATES) {
        __global ACC_DATA_T *dst = (__global ACC_DATA_T *)(ws + ws_offset);
        dst[get_global_id(0)] = TO_ACC(val);
    } else { // ws_part == WS_STATES
        __global WS_STATE_DATA_T *dst
                = (__global WS_STATE_DATA_T *)(ws + ws_offset);
        dst[get_global_id(0)] = TO_WS_STATE(val);
    }
}

// useful for debug
#if DEBUGPRINT
__kernel void ref_rnn_ws_print(__global ACC_DATA_T *gates_base,
        __global WS_STATE_DATA_T *states_base,
        __global AUX_DATA_T *c_states_base, __global AUX_DATA_T *bias_base,
        __global ACC_DATA_T *grid_comp_base, int batch, int n_layer, int n_dir,
        int n_iter, int n_bias, int dhc, int n_gates, int states_ws_ld,
        int gates_ws_ld, int wic) {
    {
        __global ACC_DATA_T *wt = gates_base;
        printf("[lay,dir,iter,batch]\n");
        for_(int j = 0; j < n_layer; j++)
        for_(int dir = 0; dir < n_dir; dir++)
        for_(int i = 0; i < n_iter; i++)
        for (int b = 0; b < batch; b++) {
            printf("[%d,%d,%d,%d]: ", j, dir, i, b);
            for_(int g = 0; g < n_gates; g++)
            for (int s = 0; s < dhc; s++) {
                printf(" %f",
                        SRC_TO_REF(*(wt
                                + off_ws_gates(n_dir, n_iter, batch,
                                        gates_ws_ld, dhc, j, dir, i, b, g,
                                        s))));
            }
            printf("\n");
        }
    }
    {
        __global WS_STATE_DATA_T *wt = states_base;
        printf("[lay,dir,iter]\n");
        for_(int j = 0; j < n_layer + 1; j++)
        for_(int dir = 0; dir < n_dir; dir++)
        for (int i = 1; i < n_iter + 1; i++) {
            printf("[%d,%d,%d] : ", j, dir, i);
            for_(int b = 0; b < batch; b++)
            for (int s = 0; s < wic; s++) {
                printf(" %f",
                        SRC_TO_REF(*(wt
                                + off_ws_state(n_layer, n_dir, n_iter, batch,
                                        states_ws_ld, j, dir, i, b, s))));
            }
            printf("\n");
        }
    }
#if IS_TRAINING && CELL_KIND == LBR_GRU
    {
        __global ACC_DATA_T *wt = grid_comp_base;
        printf("[lay,dir,iter,batch]\n");
        for_(int j = 0; j < n_layer; j++)
        for_(int dir = 0; dir < n_dir; dir++)
        for_(int i = 0; i < n_iter; i++)
        for (int b = 0; b < batch; b++) {
            printf("[%d,%d,%d,%d]: ", j, dir, i, b);
            for (int s = 0; s < dhc; s++) {
                printf(" %f",
                        *(wt
                                + off_ws_grid_offset(n_layer, n_dir, n_iter,
                                        batch, dhc, j, dir, i, b, s)));
            }
            printf("\n");
        }
    }
#endif
#if IS_FWD && CELL_KIND == VANILLA_LSTM
    {
        __global AUX_DATA_T *wt = c_states_base;
        printf("[lay,dir,iter]\n");
        for_(int j = 0; j < n_layer; j++)
        for_(int dir = 0; dir < n_dir; dir++)
        for (int i = 0; i < n_iter + 1; i++) {
            printf("[%d,%d,%d] : ", j, dir, i);
            for_(int b = 0; b < batch; b++)
            for (int s = 0; s < wic; s++) {
                printf(" %f",
                        *(wt
                                + off_ws_state(n_layer, n_dir, n_iter, batch,
                                        states_ws_ld, j, dir, i, b, s)));
            }
            printf("\n");
        }
    }
#endif
#if COPY_BIAS
    {
        __global AUX_DATA_T *wt = bias_base;
        printf("[lay,dir]\n");
        for_(int j = 0; j < n_layer; j++)
        for_(int dir = 0; dir < n_dir; dir++)
        {
            printf("[%d,%d] : ", j, dir);
            for_(int nb = 0; nb < n_bias; nb++)
            for (int dhc_ = 0; dhc_ < dhc; dhc_++) {
                printf(" %f",
                        *(wt
                                + off_ws_bias(n_layer, n_dir, n_bias, dhc, j,
                                        dir, nb, dhc_)));
            }
            printf("\n");
        }
    }
#endif
}
#endif

__kernel void ref_rnn_bias_prepare(__global float *ws_bias,
        __global float *scales, __global char *wei_layer,
        __global char *wei_iter, __global float *bias, int dhc, int n_layer,
        int n_dir, int n_bias, float data_shift, float data_scale,
        int wei_l_comp_off, int wei_i_comp_off, int lay_stride, int dir_stride,
        int nbias_stride, int dhc_stride) {
#if COPY_BIAS

    const int dhc_ = get_global_id(0);
    const int nbias_ = get_global_id(1);
    const int layer_ = get_global_id(2) / n_dir;
    const int dir_ = get_global_id(2) % n_dir;
    const param4 strides = {lay_stride, dir_stride, nbias_stride, dhc_stride};

    const float wei_scale
#if WEI_QPARAM_MASK
            = scales[nbias_ * dhc + dhc_];
#else
            = scales[0];
#endif
    __global char *temp = (__global char *)(wei_iter + wei_i_comp_off);
    __global float *wei_iter_comp
            = (__global float *)(((unsigned long)temp + (sizeof(float) - 1))
                    & -sizeof(float));
    temp = (__global char *)(wei_layer + wei_l_comp_off);
    __global float *wei_layer_comp
            = (__global float *)(((unsigned long)temp + (sizeof(float) - 1))
                    & -sizeof(float));
    const int off = comp_off(n_dir, n_bias, dhc, layer_, dir_, nbias_, dhc_);
    const float comp = wei_layer_comp[off] + wei_iter_comp[off];
    ws_bias[off_ws_bias(
            n_layer, n_dir, n_bias, dhc, layer_, dir_, nbias_, dhc_)]
            = bias[bias_off(strides, layer_, dir_, nbias_, dhc_)]
            - comp * data_shift / (wei_scale * data_scale);

#endif
}

#if IS_INT8 && CELL_KIND == VANILLA_LSTM

WS_STATE_DATA_T q_d(float f, float data_scale, float data_shift) {
    float qf = f * data_scale + data_shift;
    return TO_WS_STATE(qf);
}

float deq_w(ACC_DATA_T s, int gate, int j, __global float *scales,
        float data_scale, int dhc) {
#if WEI_QPARAM_MASK
    float wei_scale = scales[gate * dhc + j];
#else
    float wei_scale = scales[0];
#endif
    return (float)(s) / (wei_scale * data_scale);
}

// for int8 LSTM
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
ref_rnn_elemwise_fwd(int dir, int lay, int iter, __global char *scr_gates,
        __global float *scales, __global float *bias_base, float alpha,
        float data_shift, float data_scale, __global float *tm_scales,
        __global WS_STATE_DATA_T *h_states_t_l, __global float *c_states_t_l,
        __global float *c_states_tm1_l, __global AUX_DATA_T *ws_gates,
        __global float *ws_bias, int states_ws_ld, int scratch_gates_ld,
        int batch, int dhc, int n_layer, int n_dir, int n_bias,
        int n_iter_scratch_gates, float tm_cscale) {
    const int i = get_global_id(1); // batch
    const int j = get_global_id(0); // dhc

    if (j >= dhc || i >= batch) return;

    __global ACC_DATA_T *scratch_gates = (__global ACC_DATA_T *)(scr_gates)
            + off_scratch_mem_iter(
                    n_iter_scratch_gates, batch, scratch_gates_ld, dhc, iter);

    float G0 = logistic_fwd_tm(deq_w(scratch_gates[cell_scratch_mem(
                                             scratch_gates_ld, dhc, i, 0, j)],
                                       0, j, scales, data_scale, dhc)
                    + ws_bias[off_ws_bias(
                            n_layer, n_dir, n_bias, dhc, lay, dir, 0, j)],
            tm_scales[0]);
    float G1 = logistic_fwd_tm(deq_w(scratch_gates[cell_scratch_mem(
                                             scratch_gates_ld, dhc, i, 1, j)],
                                       1, j, scales, data_scale, dhc)
                    + ws_bias[off_ws_bias(
                            n_layer, n_dir, n_bias, dhc, lay, dir, 1, j)],
            tm_scales[1]);
    float G2 = tanh_fwd_tm(deq_w(scratch_gates[cell_scratch_mem(
                                         scratch_gates_ld, dhc, i, 2, j)],
                                   2, j, scales, data_scale, dhc)
                    + ws_bias[off_ws_bias(
                            n_layer, n_dir, n_bias, dhc, lay, dir, 2, j)],
            tm_scales[2]);
    float G3 = logistic_fwd_tm(deq_w(scratch_gates[cell_scratch_mem(
                                             scratch_gates_ld, dhc, i, 3, j)],
                                       3, j, scales, data_scale, dhc)
                    + ws_bias[off_ws_bias(
                            n_layer, n_dir, n_bias, dhc, lay, dir, 3, j)],
            tm_scales[3]);

    float tmp
            = G1 * c_states_tm1_l[cell_ws_state(states_ws_ld, i, j)] + G0 * G2;

    h_states_t_l[cell_ws_state(states_ws_ld, i, j)]
            = q_d(G3 * tanh_fwd_tm(tmp, tm_cscale), data_scale, data_shift);
    c_states_t_l[cell_ws_state(states_ws_ld, i, j)] = tmp;
}

#else

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
ref_rnn_elemwise_fwd(int dir, int lay, int iter, __global char *scr_gates,
        __global AUX_DATA_T *bias_base, float alpha, __global float *tm_scales,
        __global WS_STATE_DATA_T *h_states_t_l,
        __global AUX_DATA_T *c_states_t_l, __global AUX_DATA_T *c_states_tm1_l,
        __global AUX_DATA_T *ws_gates, __global AUX_DATA_T *ws_grid,
        int states_ws_ld, int gates_ws_ld, int scratch_gates_ld, int batch,
        int dhc, int n_layer, int n_gates, int n_iter_scratch_gates,
        int bias_lay_stride, int bias_dir_stride, int bias_s2, int bias_s3,
#if CELL_KIND == VANILLA_LSTM || CELL_KIND == VANILLA_RNN
        float tm_cscale
#elif CELL_KIND == LBR_GRU
        __global WS_STATE_DATA_T *h_states_tm_l, __global char *scr_cell
#elif CELL_KIND == VANILLA_GRU
        __global WS_STATE_DATA_T *h_states_tm_l, int n_part
#endif
) {
    const int i = get_global_id(1); // batch
    const int j = get_global_id(0); // dhc

    if (j >= dhc || i >= batch) return;
    const param4 strides = {bias_lay_stride, bias_dir_stride, bias_s2, bias_s3};

    const __global AUX_DATA_T *bias
            = bias_base + bias_off(strides, lay, dir, 0, 0);
    __global AUX_DATA_T *scratch_gates = (__global AUX_DATA_T *)(scr_gates)
            + off_scratch_mem_iter(
                    n_iter_scratch_gates, batch, scratch_gates_ld, dhc, iter);

#if CELL_KIND == VANILLA_LSTM
    float g_i = logistic_fwd_tm((float)scratch_gates[cell_scratch_mem(
                                        scratch_gates_ld, dhc, i, 0, j)]
                    + bias[off_ker_bias(n_gates, dhc, 0, j)],
            tm_scales[0]);
    float g_f = logistic_fwd_tm((float)scratch_gates[cell_scratch_mem(
                                        scratch_gates_ld, dhc, i, 1, j)]
                    + bias[off_ker_bias(n_gates, dhc, 1, j)],
            tm_scales[1]);
    float g_z = tanh_fwd_tm((float)scratch_gates[cell_scratch_mem(
                                    scratch_gates_ld, dhc, i, 2, j)]
                    + bias[off_ker_bias(n_gates, dhc, 2, j)],
            tm_scales[2]);
    float g_o = logistic_fwd_tm((float)scratch_gates[cell_scratch_mem(
                                        scratch_gates_ld, dhc, i, 3, j)]
                    + bias[off_ker_bias(n_gates, dhc, 3, j)],
            tm_scales[3]);

#if IS_TRAINING
    ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)] = g_i;
    ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)] = g_f;
    ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)] = g_z;
    ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 3, j)] = g_o;
#endif

    float Ct = g_f * c_states_tm1_l[cell_ws_state(states_ws_ld, i, j)]
            + g_i * g_z;
    float Ht = g_o * tanh_fwd_tm(Ct, tm_cscale);

    h_states_t_l[cell_ws_state(states_ws_ld, i, j)] = TO_INPUT(Ht);
    c_states_t_l[cell_ws_state(states_ws_ld, i, j)] = Ct;

#elif CELL_KIND == VANILLA_RNN

    float g = activation_fwd((float)scratch_gates[cell_scratch_mem(
                                     scratch_gates_ld, dhc, i, 0, j)]
                    + bias[off_ker_bias(n_gates, dhc, 0, j)],
#if IS_TESTMODE
            tm_scales[0], 0);
#else
            alpha, 0);
#endif

#if IS_TRAINING
    ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)] = g;
#endif
    h_states_t_l[cell_ws_state(states_ws_ld, i, j)] = TO_INPUT(g);

#elif CELL_KIND == LBR_GRU
    // AUX and SCRATCH data type is same for fwd prop
    __global AUX_DATA_T *scratch_cell = (__global AUX_DATA_T *)(scr_cell);
    __global WS_STATE_DATA_T *src_iter = h_states_tm_l;

    float Wh_b = (float)scratch_cell[cell_scratch_mem(
                         scratch_gates_ld, dhc, i, 2, j)]
            + bias[off_ker_bias(n_gates, dhc, 3, j)];
    float G0 = logistic_fwd_tm((float)scratch_gates[cell_scratch_mem(
                                       scratch_gates_ld, dhc, i, 0, j)]
                    + (float)scratch_cell[cell_scratch_mem(
                            scratch_gates_ld, dhc, i, 0, j)]
                    + bias[off_ker_bias(n_gates, dhc, 0, j)],
            tm_scales[0]);
    float G1 = logistic_fwd_tm((float)scratch_gates[cell_scratch_mem(
                                       scratch_gates_ld, dhc, i, 1, j)]
                    + (float)scratch_cell[cell_scratch_mem(
                            scratch_gates_ld, dhc, i, 1, j)]
                    + bias[off_ker_bias(n_gates, dhc, 1, j)],
            tm_scales[1]);

    float G2 = tanh_fwd_tm((float)scratch_gates[cell_scratch_mem(
                                   scratch_gates_ld, dhc, i, 2, j)]
                    + G1 * Wh_b + bias[off_ker_bias(n_gates, dhc, 2, j)],
            tm_scales[2]);
    float Ht = G0 * TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)])
            + (1 - G0) * G2;

    h_states_t_l[cell_ws_state(states_ws_ld, i, j)] = TO_INPUT(Ht);

#if IS_TRAINING
    ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)] = G0;
    ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)] = G1;
    ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)] = G2;
    ws_grid[cell_ws_grid_comp(dhc, i, j)] = Wh_b;
#endif
#elif CELL_KIND == VANILLA_GRU
    __global WS_STATE_DATA_T *src_iter = h_states_tm_l;

    if (n_part == 1) {
        float G0 = logistic_fwd_tm(
                scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 0, j)]
                        + bias[off_ker_bias(n_gates, dhc, 0, j)],
                tm_scales[0]);
        float G1 = logistic_fwd_tm(
                scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 1, j)]
                        + bias[off_ker_bias(n_gates, dhc, 1, j)],
                tm_scales[1]);

        /* TODO from CPU: Can be optimized for fwd_training by using
        ws_gates instead of scratch_gates in p2 */
        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 0, j)]
                = TO_INPUT(G0);
        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 1, j)]
                = TO_INPUT(G1);
        float tmp = TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)]);
        h_states_t_l[cell_ws_state(states_ws_ld, i, j)] = TO_INPUT(tmp * G1);
#if IS_TRAINING
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)] = G0;
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)] = G1;
#endif
    } else if (n_part == 2) {
        float G0 = TO_REF(scratch_gates[cell_scratch_mem(
                scratch_gates_ld, dhc, i, 0, j)]);
        float G2 = tanh_fwd_tm(
                scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 2, j)]
                        + bias[off_ker_bias(n_gates, dhc, 2, j)],
                tm_scales[2]);
        float tmp = TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)]);
        h_states_t_l[cell_ws_state(states_ws_ld, i, j)]
                = TO_INPUT(tmp * G0 + (1.0f - G0) * G2);
#if IS_TRAINING
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)] = G2;
#endif
    }
#else
#error "Wrong Cell Kind"
#endif
}
#endif

#if NEED_BIAS_ATOMIC_REDUCE
#define MAYBE_ATOMIC volatile __global
#define DIFF_BIAS_DATA_T CONCAT2(atomic_, DIFF_DATA_T)
#else
#define MAYBE_ATOMIC __global
#define DIFF_BIAS_DATA_T DIFF_DATA_T
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
ref_rnn_elemwise_bwd(int dir, int lay, int iter, __global char *scr_gates,
        __global AUX_DATA_T *bias_base, float alpha, __global float *tm_scales,
        __global WS_STATE_DATA_T *states_tm1_l,
        __global AUX_DATA_T *c_states_t_l, __global AUX_DATA_T *c_states_tm1_l,
        __global AUX_DATA_T *ws_gates, __global AUX_DATA_T *ws_grid,
        int states_ws_ld, int gates_ws_ld, int scratch_gates_ld, int batch,
        int dhc, int n_layer, int n_gates, int n_iter_scratch_gates, int n_dir,
        int n_states, int n_iter, int scratch_diff_states_ld,
#if CELL_KIND == VANILLA_LSTM || CELL_KIND == VANILLA_RNN
        float tm_cscale,
#elif CELL_KIND == LBR_GRU
        __global char *scr_gate_r,
#elif CELL_KIND == VANILLA_GRU
        int n_part, __global char *scr_cell, __global char *scratch_dhG1,
#endif
        __global DIFF_DATA_T *diff_states,
        MAYBE_ATOMIC DIFF_BIAS_DATA_T *diff_bias_base, int lay_stride,
        int dir_stride, int diff_bias_s2, int diff_bias_s3) {
#if !IS_FWD
    const int i_ = get_global_id(1) * ELEMWISE_BWD_BATCH_BLOCK; // batch
    const int j = get_global_id(0); // dhc

    const param4 strides = {lay_stride, dir_stride, diff_bias_s2, diff_bias_s3};

    MAYBE_ATOMIC DIFF_BIAS_DATA_T *diff_bias
            = diff_bias_base + diff_bias_off(strides, lay, dir, 0, 0);

    if (j >= dhc) return;

    DIFF_DATA_T diff_bias_acc[N_BIAS] = {0};
    for (int batch_id = 0; batch_id < ELEMWISE_BWD_BATCH_BLOCK; batch_id++) {
        int i = i_ + batch_id;
        if (i >= batch) break;

#if CELL_KIND == VANILLA_LSTM

        __global SRC_DATA_T *scratch_gates = (__global SRC_DATA_T *)(scr_gates)
                + off_scratch_mem_iter(n_iter_scratch_gates, batch,
                        scratch_gates_ld, dhc, iter);

        __global DIFF_DATA_T *diff_states_t_l = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 0, iter, 0, 0);
        __global DIFF_DATA_T *diff_states_tp1_l = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 0, iter + 1, 0,
                        0);
        __global DIFF_DATA_T *diff_states_t_lp1 = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay + 1, dir, 0, iter, 0,
                        0);

        float Ct = c_states_t_l[cell_ws_state(states_ws_ld, i, j)];
        /// @todo save it in the workspace in fwd pass or recompute it to
        /// save bw
        float tanhCt = tanh_fwd_tm(Ct, tm_cscale);
        // we have 2 incoming diffs on Ht
        float dHt = (float)diff_states_tp1_l[cell_scratch_diff_states(
                            n_iter, batch, scratch_diff_states_ld, 0, i, j)]
                + diff_states_t_lp1[cell_scratch_diff_states(
                        n_iter, batch, scratch_diff_states_ld, n_states, i, j)];

        float dCt = (float)diff_states_tp1_l[cell_scratch_diff_states(
                            n_iter, batch, scratch_diff_states_ld, 1, i, j)]
                + one_m_square(tanhCt)
                        * ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 3, j)]
                        * dHt;

        float dG1 = (float)c_states_tm1_l[cell_ws_state(states_ws_ld, i, j)]
                * dCt
                * x_m_square(
                        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)]);
        float dG0 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)] * dCt
                * x_m_square(
                        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)]);
        float dG3 = tanhCt * dHt
                * x_m_square(
                        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 3, j)]);
        float dG2 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)] * dCt
                * one_m_square(
                        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)]);

        diff_states_t_l[cell_scratch_diff_states(
                n_iter, batch, scratch_diff_states_ld, 1, i, j)]
                = dCt * ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)];

        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 0, j)]
                = TO_INPUT(dG0);
        diff_bias_acc[0] += dG0;
        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 1, j)]
                = TO_INPUT(dG1);
        diff_bias_acc[1] += dG1;
        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 2, j)]
                = TO_INPUT(dG2);
        diff_bias_acc[2] += dG2;
        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 3, j)]
                = TO_INPUT(dG3);
        diff_bias_acc[3] += dG3;

#if N_BIAS != 4
#error "Unexpected N_BIAS for VANILLA_LSTM"
#endif

#elif CELL_KIND == LBR_GRU
        __global SRC_DATA_T *scratch_gates = (__global SRC_DATA_T *)(scr_gates)
                + off_scratch_mem_iter(n_iter_scratch_gates, batch,
                        scratch_gates_ld, dhc, iter);
        __global SRC_DATA_T *scratch_gate_r
                = (__global SRC_DATA_T *)(scr_gate_r);

        __global DIFF_DATA_T *diff_src_iter = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 0, iter, 0, 0);
        __global DIFF_DATA_T *diff_dst_iter = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 0, iter + 1, 0,
                        0);
        __global DIFF_DATA_T *diff_dst_layer = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay + 1, dir, 0, iter, 0,
                        0);
        __global WS_STATE_DATA_T *src_iter //h_states_tm1_l
                = states_tm1_l;

        float h = TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)]);
        float Wh_b = ws_grid[cell_ws_grid_comp(dhc, i, j)];
        float dHt = diff_dst_iter[cell_scratch_diff_states(
                            n_iter, batch, scratch_diff_states_ld, 0, i, j)]
                + diff_dst_layer[cell_scratch_diff_states(
                        n_iter, batch, scratch_diff_states_ld, n_states, i, j)];

        float dG0 = (h - ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)])
                * dHt
                * x_m_square(
                        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)]);
        float dG2 = (1.0f - ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)])
                * one_m_square(
                        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)])
                * dHt;
        float dG1 = Wh_b * dG2
                * x_m_square(
                        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)]);

        diff_src_iter[cell_scratch_diff_states(
                n_iter, batch, scratch_diff_states_ld, 0, i, j)]
                = dHt * ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)];

        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 0, j)]
                = TO_INPUT(dG0);
        diff_bias_acc[0] += dG0;
        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 1, j)]
                = TO_INPUT(dG1);
        diff_bias_acc[1] += dG1;
        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 2, j)]
                = TO_INPUT(dG2);
        diff_bias_acc[2] += dG2;

        scratch_gate_r[cell_scratch_mem(scratch_gates_ld, dhc, i, 0, j)]
                = TO_INPUT(dG0);
        scratch_gate_r[cell_scratch_mem(scratch_gates_ld, dhc, i, 1, j)]
                = TO_INPUT(dG1);
        float tmp = dG2 * ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)];
        scratch_gate_r[cell_scratch_mem(scratch_gates_ld, dhc, i, 2, j)]
                = TO_INPUT(tmp);
        diff_bias_acc[3] += tmp;
#if N_BIAS != 4
#error "Unexpected N_BIAS for LBR_GRU"
#endif

#elif CELL_KIND == VANILLA_RNN

        __global AUX_DATA_T *ws_gates_run
                = ws_gates + cell_ws_gates(gates_ws_ld, dhc, i, 0, j);
        __global SRC_DATA_T *scratch_gates = (__global SRC_DATA_T *)(scr_gates)
                + off_scratch_mem(n_iter_scratch_gates, batch, scratch_gates_ld,
                        dhc, iter, i, 0, j);

        __global DIFF_DATA_T *diff_states_t_lp1 = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay + 1, dir, n_states,
                        iter, i, j);
        __global DIFF_DATA_T *diff_states_tp1_l = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 0, iter + 1, i,
                        j);
        const float dH = (float)diff_states_t_lp1[0] + diff_states_tp1_l[0];

        float g = ws_gates_run[0];
#if IS_TESTMODE
        float tmp = = dH * activation_bwd(g, tm_scales[0], 0.);
        scratch_gates[0] = TO_INPUT(tmp);
        diff_bias_acc[0] += tmp;
#else
        float tmp = dH * activation_bwd(g, alpha, 0.);
        scratch_gates[0] = TO_INPUT(tmp);
        diff_bias_acc[0] += tmp;
#endif
#if N_BIAS != 1
#error "Unexpected N_BIAS for VANILLA_RNN"
#endif
#elif CELL_KIND == VANILLA_GRU
        __global SRC_DATA_T *scratch_gates = (__global SRC_DATA_T *)(scr_gates)
                + off_scratch_mem_iter(n_iter_scratch_gates, batch,
                        scratch_gates_ld, dhc, iter);
        __global DIFF_DATA_T *diff_src_iter = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 0, iter, 0, 0);
        __global DIFF_DATA_T *diff_dst_iter = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 0, iter + 1, 0,
                        0);
        __global DIFF_DATA_T *diff_dst_layer = diff_states
                + off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay + 1, dir, 0, iter, 0,
                        0);
        __global WS_STATE_DATA_T *src_iter //h_states_tm1_l
                = states_tm1_l;

        float h = TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)]);
        if (n_part == 1) {
            float dHt = diff_dst_iter[cell_scratch_diff_states(
                                n_iter, batch, scratch_diff_states_ld, 0, i, j)]
                    + diff_dst_layer[cell_scratch_diff_states(n_iter, batch,
                            scratch_diff_states_ld, n_states, i, j)];
            float dG2 = (1.0f
                                - ws_gates[cell_ws_gates(
                                        gates_ws_ld, dhc, i, 0, j)])
                    * dHt
                    * one_m_square(
                            ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)]);
            float dG0 = (h - ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)])
                    * dHt
                    * x_m_square(
                            ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)]);
            diff_src_iter[cell_scratch_diff_states(
                    n_iter, batch, scratch_diff_states_ld, 0, i, j)]
                    = dHt * ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)];

            scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 0, j)]
                    = TO_INPUT(dG0);
            diff_bias_acc[0] += dG0;
            scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 2, j)]
                    = TO_INPUT(dG2);
            diff_bias_acc[2] += dG2;
        } else if (n_part == 2) {
            __global SRC_DATA_T *scratch_cell
                    = (__global SRC_DATA_T *)(scr_cell);
            __global DIFF_DATA_T *dhG1 = (__global DIFF_DATA_T *)scratch_dhG1;

            float dG1 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)];
            diff_src_iter[cell_scratch_diff_states(
                    n_iter, batch, scratch_diff_states_ld, 0, i, j)]
                    += dhG1[off_scratch_dhg1(
                               batch, scratch_diff_states_ld, i, j)]
                    * dG1;

            float tmp = dhG1[off_scratch_dhg1(
                                batch, scratch_diff_states_ld, i, j)]
                    * h * x_m_square(dG1);
            scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 1, j)]
                    = TO_INPUT(tmp);
            diff_bias_acc[1] += tmp;
            scratch_cell[off_scratch_cell(batch, states_ws_ld, i, j)]
                    = TO_INPUT(dG1 * h);
        }
#if N_BIAS != 3
#error "Unexpected N_BIAS for VANILLA_GRU"
#endif

#else
#error "Wrong Cell Kind"
#endif
    }
    unroll_for(int k = 0; k < N_BIAS; k++) {
#if NEED_BIAS_ATOMIC_REDUCE
        atomic_add_global(&diff_bias[k * dhc + j], diff_bias_acc[k]);
#else
        diff_bias[k * dhc + j] += diff_bias_acc[k];
#endif
    }
#endif // !IS_FWD
}
