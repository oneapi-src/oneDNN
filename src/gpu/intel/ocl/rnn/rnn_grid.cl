/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/intel/ocl/rnn/cell_compute.h"
#include "gpu/intel/ocl/rnn/cell_kind_utility.h"
#include "gpu/intel/ocl/rnn/rnn_common.h"

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
simple_rnn_copy_init_layer(__global WS_STATE_DATA_T *dst_base,
        __global char *src_base, __global AUX_DATA_T *scratch_diff_states,
        int lr, int rl, int batch, int dhc, int slc, int n_iter, int n_layer,
        int n_dir, int n_states, int states_ws_ld, int scratch_diff_states_ld,
        int64x3_t strides) {

#if IS_FWD

    const int it = get_global_id(2);
    const int b = get_global_id(1);
    const int c = get_global_id(0);
    if (c >= slc || b >= batch || it >= n_iter) return;

    __global WS_STATE_DATA_T *dst;
    __global WS_STATE_DATA_T *src = (__global WS_STATE_DATA_T *)src_base
            + src_l_off(strides, it, b, c);

    if (lr) {
        dst = dst_base
                + off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld, -1,
                        0, it, b, c);
        dst[0] = src[0];
    }
    if (rl) {
        dst = dst_base
                + off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld, -1,
                        n_dir - 1, n_iter - it - 1, b, c);
        dst[0] = src[0];
    }

#else // BWD

    const int it = get_global_id(2);
    const int b = get_global_id(1);
    const int s = get_global_id(0);
    if (s >= dhc || b >= batch || it >= n_iter) return;

    __global AUX_DATA_T *dst = scratch_diff_states;

#if DIRECTION_KIND == CONCAT
    __global DIFF_DATA_T *src = (__global DIFF_DATA_T *)src_base
            + diff_dst_l_off(strides, it, b, s);
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 0, n_states, it, b, s)]
            = src[0];
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 1, n_states, n_iter - it - 1, b,
            s)]
            = src[dhc];
#elif DIRECTION_KIND == SUM
    __global DIFF_DATA_T *src = (__global DIFF_DATA_T *)src_base
            + diff_dst_l_off(strides, it, b, s);
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 0, n_states, it, b, s)]
            = src[0];
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 1, n_states, n_iter - it - 1, b,
            s)]
            = src[0];
#elif DIRECTION_KIND == L2R
    __global DIFF_DATA_T *src = (__global DIFF_DATA_T *)src_base
            + diff_dst_l_off(strides, it, b, s);
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 0, n_states, it, b, s)]
            = src[0];
#elif DIRECTION_KIND == R2L
    __global DIFF_DATA_T *src = (__global DIFF_DATA_T *)src_base
            + diff_dst_l_off(strides, n_iter - it - 1, b, s);
    dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
            scratch_diff_states_ld, n_layer, 0, n_states, it, b, s)]
            = src[0];
#else
#error "Unsupported direction_kind"
#endif
#endif
}

__kernel void simple_rnn_copy_init_iter(__global WS_STATE_DATA_T *dst_base,
        __global AUX_DATA_T *dst_c_base, __global char *src_base,
        __global char *src_c_base, __global AUX_DATA_T *scratch_diff_states,
        int batch, int dhc, int sic, int n_iter, int n_layer, int n_dir,
        int n_states, int states_ws_ld,
#if IS_FWD
        int64x4_t src_iter_strides,
#if WITH_SRC_ITER_C
        int64x4_t src_iter_c_strides,
#endif // WITH_SRC_ITER_C
        const float shift, const float scale, const int quantize,
#else // BWD
        int64x4_t diff_dst_iter_strides,
#if WITH_DST_ITER_C
        int64x4_t diff_dst_iter_c_strides,
#endif // WITH_DST_ITER_C

#endif // IS_FWD
        int scratch_diff_states_ld) {

    const int s = get_global_id(0);
    const int b = get_global_id(1);
    const int lay = get_global_id(2) / n_dir;
    const int dir = get_global_id(2) % n_dir;

#if IS_FWD
    __global INPUT_DATA_T *src = (__global INPUT_DATA_T *)(src_base);
    __global WS_STATE_DATA_T *dst = dst_base;
    int ws_state_offset = off_ws_state(
            n_layer, n_dir, n_iter, batch, states_ws_ld, lay, dir, -1, b, s);
    if (s < sic) {
        int src_i_offset = src_i_off(src_iter_strides, lay, dir, b, s);
        dst[ws_state_offset] = src_base
                ? (quantize ? TO_WS_STATE(src[src_i_offset] * scale + shift)
                            : src[src_i_offset])
                : TO_WS_STATE(0.0f);
    }
#if WITH_SRC_ITER_C
    __global SRC_C_DATA_T *src_c = (__global SRC_C_DATA_T *)(src_c_base);
    __global AUX_DATA_T *dst_c = dst_c_base;
    if (s < dhc) {
        int ws_c_state_offset = off_ws_c_state(n_layer, n_dir, n_iter, batch,
                states_ws_ld, lay, dir, -1, b, s);
        dst_c[ws_c_state_offset] = src_c_base
                ? TO_AUX(src_c[src_i_c_off(src_iter_c_strides, lay, dir, b, s)])
                : TO_AUX(0.0f);
    }
#endif

#else // BWD

    __global DIFF_DATA_T *src = (__global DIFF_DATA_T *)(src_base);
    __global AUX_DATA_T *dst = scratch_diff_states;

    if (s < dhc)
        dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
                scratch_diff_states_ld, lay, dir, 0, n_iter, b, s)]
                = src_base
                ? src[diff_dst_i_off(diff_dst_iter_strides, lay, dir, b, s)]
                : 0.0f;
#if WITH_DST_ITER_C
    __global DIFF_DATA_T *src_c = (__global DIFF_DATA_T *)(src_c_base);
    if (s < dhc) {
        dst[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter, batch,
                scratch_diff_states_ld, lay, dir, 1, n_iter, b, s)]
                = src_c_base ? src_c[diff_dst_i_c_off(
                          diff_dst_iter_c_strides, lay, dir, b, s)]
                             : 0.0f;
    }
#endif
#endif
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
simple_rnn_copy_res_layer(
        __global WS_STATE_DATA_T *src_base, __global char *dst_base,
        __global AUX_DATA_T *scratch_diff_states, int lr, int rl, int batch,
        int dhc, int slc, int n_iter, int n_layer, int n_dir, int n_states,
        int states_ws_ld, int scratch_diff_states_ld, int64x3_t strides
#if IS_FWD
        ,
        const float shift, const float scale, const int dequantize
#endif
) {

    const int it = get_global_id(2);
    const int b = get_global_id(1);
    const int s = get_global_id(0);

#if IS_FWD
    if (s >= dhc || b >= batch || it >= n_iter) return;
    __global WS_STATE_DATA_T *src = src_base;
    __global DST_DATA_T *dst = (__global DST_DATA_T *)(dst_base);
    int dir = 0;
    if (lr) {
        bool dequantize_at_copy = dequantize && DIRECTION_KIND != SUM;
        dst[dst_l_off(strides, it, b, dir * dhc + s)] = dequantize_at_copy
                ? TO_DST(((float)src[off_ws_state(n_layer, n_dir, n_iter, batch,
                                  states_ws_ld, n_layer - 1, dir, it, b, s)]
                                 - shift)
                        / scale)
                : src[off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld,
                        n_layer - 1, dir, it, b, s)];
        dir = 1;
    }
    if (rl) {
#if DIRECTION_KIND == SUM
        if (dequantize) {
            float val = (float)src[off_ws_state(n_layer, n_dir, n_iter, batch,
                                states_ws_ld, n_layer - 1, dir, n_iter - it - 1,
                                b, s)]
                    + dst[dst_l_off(strides, it, b, s)];
            val = min(max(val, 0.f), 255.f);
            dst[dst_l_off(strides, it, b, s)]
                    = TO_DST((val - 2 * shift) / scale);
        } else {
#if defined(SRC_DT_U8) && defined(DST_DT_U8)
            dst[dst_l_off(strides, it, b, s)] = convert_uchar_sat(
                    convert_short(src[off_ws_state(n_layer, n_dir, n_iter,
                            batch, states_ws_ld, n_layer - 1, dir,
                            n_iter - it - 1, b, s)])
                    + convert_short(dst[dst_l_off(strides, it, b, s)]));
#else
            ACC_DATA_T temp_src = DST_TO_REF(dst[dst_l_off(strides, it, b, s)]);
            temp_src += DST_TO_REF(src[off_ws_state(n_layer, n_dir, n_iter,
                    batch, states_ws_ld, n_layer - 1, dir, n_iter - it - 1, b,
                    s)]);
            dst[dst_l_off(strides, it, b, s)] = REF_TO_DST(temp_src);
#endif
        }
#else
        dst[dst_l_off(strides, it, b, dir * dhc + s)] = dequantize
                ? TO_DST(((float)src[off_ws_state(n_layer, n_dir, n_iter, batch,
                                  states_ws_ld, n_layer - 1, dir,
                                  n_iter - it - 1, b, s)]
                                 - shift)
                        / scale)
                : src[off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld,
                        n_layer - 1, dir, n_iter - it - 1, b, s)];
#endif
    }
#else // BWD

    if (s >= slc || b >= batch || it >= n_iter) return;
    __global AUX_DATA_T *src = scratch_diff_states;
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

__kernel void simple_rnn_copy_res_iter(
        __global WS_STATE_DATA_T *src_base, __global AUX_DATA_T *src_c_base,
        __global char *dst_base, __global char *dst_c_base,
        __global AUX_DATA_T *scratch_diff_states, int batch, int dhc, int sic,
        int n_iter, int n_layer, int n_dir, int n_states, int states_ws_ld,
        int scratch_diff_states_ld, int64x4_t strides
#if (IS_FWD && WITH_DST_ITER_C) || (!IS_FWD && WITH_SRC_ITER_C)
        ,
        int64x4_t c_strides
#endif // WITH_DST_ITER_C
#if IS_FWD
        ,
        const float shift, const float scale, const int dequantize
#endif
) {

    const int s = get_global_id(0);
    const int b = get_global_id(1);
    const int lay = get_global_id(2) / n_dir;
    const int dir = get_global_id(2) % n_dir;

#if IS_FWD
    __global WS_STATE_DATA_T *src = src_base;
    __global OUTPUT_DATA_T *dst = (__global OUTPUT_DATA_T *)(dst_base);

    if (dst_base && s < dhc) {
        dst[dst_i_off(strides, lay, dir, b, s)] = dequantize
                ? TO_OUTPUT(
                        ((float)src[off_ws_state(n_layer, n_dir, n_iter, batch,
                                 states_ws_ld, lay, dir, n_iter - 1, b, s)]
                                - shift)
                        / scale)
                : TO_OUTPUT(src[off_ws_state(n_layer, n_dir, n_iter, batch,
                        states_ws_ld, lay, dir, n_iter - 1, b, s)]);
    }
#if WITH_DST_ITER_C
    __global AUX_DATA_T *src_c = src_c_base;
    __global DST_C_DATA_T *dst_c = (__global DST_C_DATA_T *)(dst_c_base);
    if (dst_c_base && s < dhc) {
        dst_c[dst_i_c_off(c_strides, lay, dir, b, s)]
                = src_c[off_ws_c_state(n_layer, n_dir, n_iter, batch,
                        states_ws_ld, lay, dir, n_iter - 1, b, s)];
    }
#endif

#else // BWD

    __global AUX_DATA_T *src = scratch_diff_states;
    __global DIFF_DATA_T *dst = (__global DIFF_DATA_T *)(dst_base);
    __global DIFF_DATA_T *dst_c = (__global DIFF_DATA_T *)(dst_c_base);
    if (dst_base && s < sic) {
        dst[diff_src_i_off(strides, lay, dir, b, s)]
                = src[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 0, 0, b, s)];
    }
#if WITH_SRC_ITER_C
    if (dst_base && s < dhc) {
        dst_c[diff_src_i_c_off(c_strides, lay, dir, b, s)]
                = src[off_scratch_diff_states(n_layer, n_dir, n_states, n_iter,
                        batch, scratch_diff_states_ld, lay, dir, 1, 0, b, s)];
    }
#endif
#endif
}

__kernel void rnn_bias_prepare(__global float *ws_bias, __global float *scales,
        __global char *wei_layer, __global char *wei_iter,
        __global BIAS_DATA_T *bias, int dhc, int n_layer, int n_dir,
        float data_shift, float data_scale, int wei_l_comp_off,
        int wei_i_comp_off, int64x4_t bias_strides) {
#if COPY_BIAS

    const int dhc_ = get_global_id(0);
    const int nbias_ = get_global_id(1);
    const int layer_ = get_global_id(2) / n_dir;
    const int dir_ = get_global_id(2) % n_dir;

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
    const int off = comp_off(n_dir, dhc, layer_, dir_, nbias_, dhc_);
    const float comp = wei_layer_comp[off] + wei_iter_comp[off];
    ws_bias[off_ws_bias(n_layer, n_dir, dhc, layer_, dir_, nbias_, dhc_)]
            = bias[bias_off(bias_strides, layer_, dir_, nbias_, dhc_)]
            - comp * data_shift / (wei_scale * data_scale);

#endif
}

// for int8 LSTM
#if IS_INT8 && CELL_KIND == VANILLA_LSTM
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
simple_rnn_elemwise_fwd(int dir, int lay, int iter,
        __global ACC_DATA_T *scratch_gates_, dim_t scratch_gates_off,
        __global float *scales, float alpha, float data_shift, float data_scale,
        __global float *tm_scales, __global WS_STATE_DATA_T *h_states_t_l_,
        dim_t h_states_t_l_off, __global float *c_states_t_l_,
        dim_t c_states_t_l_off, __global float *c_states_tm1_l_,
        dim_t c_states_tm1_l_off, __global AUX_DATA_T *ws_gates_,
        dim_t ws_gates_off, __global float *ws_bias, int states_ws_ld,
        int scratch_gates_ld, int batch, int dhc, int n_layer, int n_dir,
        float tm_cscale) {
    const int i = get_global_id(1); // batch
    const int j = get_global_id(0); // dhc

    if (j >= dhc || i >= batch) return;

    __global ACC_DATA_T *scratch_gates = scratch_gates_ + scratch_gates_off;
    __global WS_STATE_DATA_T *h_states_t_l = h_states_t_l_ + h_states_t_l_off;
    __global float *c_states_t_l = c_states_t_l_ + c_states_t_l_off;
    __global float *c_states_tm1_l = c_states_tm1_l_ + c_states_tm1_l_off;
    __global AUX_DATA_T *ws_gates = ws_gates_ + ws_gates_off;

    float G0 = logistic_fwd_tm(deq_w(scratch_gates[cell_scratch_mem(
                                             scratch_gates_ld, dhc, i, 0, j)],
                                       0, j, scales, data_scale, dhc)
                    + ws_bias[off_ws_bias(n_layer, n_dir, dhc, lay, dir, 0, j)],
            tm_scales[0]);
    float G1 = logistic_fwd_tm(deq_w(scratch_gates[cell_scratch_mem(
                                             scratch_gates_ld, dhc, i, 1, j)],
                                       1, j, scales, data_scale, dhc)
                    + ws_bias[off_ws_bias(n_layer, n_dir, dhc, lay, dir, 1, j)],
            tm_scales[1]);
    float G2 = tanh_fwd_tm(deq_w(scratch_gates[cell_scratch_mem(
                                         scratch_gates_ld, dhc, i, 2, j)],
                                   2, j, scales, data_scale, dhc)
                    + ws_bias[off_ws_bias(n_layer, n_dir, dhc, lay, dir, 2, j)],
            tm_scales[2]);
    float G3 = logistic_fwd_tm(deq_w(scratch_gates[cell_scratch_mem(
                                             scratch_gates_ld, dhc, i, 3, j)],
                                       3, j, scales, data_scale, dhc)
                    + ws_bias[off_ws_bias(n_layer, n_dir, dhc, lay, dir, 3, j)],
            tm_scales[3]);

    float tmp
            = G1 * c_states_tm1_l[cell_ws_state(states_ws_ld, i, j)] + G0 * G2;

    h_states_t_l[cell_ws_state(states_ws_ld, i, j)]
            = q_d(G3 * tanh_fwd_tm(tmp, tm_cscale), data_scale, data_shift);
    c_states_t_l[cell_ws_state(states_ws_ld, i, j)] = tmp;
}

#else

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
simple_rnn_elemwise_fwd(__global ACC_DATA_T *scratch_gates_,
        dim_t scratch_gates_off, __global BIAS_DATA_T *bias_, dim_t bias_off,
        float alpha, __global float *tm_scales,
        __global WS_STATE_DATA_T *h_states_t_l_, dim_t h_states_t_l_off,
        __global AUX_DATA_T *c_states_t_l_, dim_t c_states_t_l_off,
        __global AUX_DATA_T *c_states_tm1_l_, dim_t c_states_tm1_l_off,
        __global AUX_DATA_T *ws_gates_, dim_t ws_gates_off,
        __global AUX_DATA_T *ws_grid_, dim_t ws_grid_off, int states_ws_ld,
        int gates_ws_ld, int scratch_gates_ld, int batch, int dhc,
#if CELL_KIND == VANILLA_LSTM || CELL_KIND == VANILLA_RNN
        float tm_cscale
#elif CELL_KIND == LBR_GRU
        __global WS_STATE_DATA_T *h_states_tm_l_, dim_t h_states_tm_l_off,
        __global char *scr_cell
#elif CELL_KIND == VANILLA_GRU
        __global WS_STATE_DATA_T *h_states_tm_l_, dim_t h_states_tm_l_off,
        int n_part
#endif
) {
    const int i = get_global_id(1); // batch
    const int j = get_global_id(0); // dhc

    if (j >= dhc || i >= batch) return;

    __global ACC_DATA_T *scratch_gates = scratch_gates_ + scratch_gates_off;
    __global BIAS_DATA_T *bias = bias_ + bias_off;
    __global WS_STATE_DATA_T *h_states_t_l = h_states_t_l_ + h_states_t_l_off;
    __global AUX_DATA_T *c_states_t_l = c_states_t_l_ + c_states_t_l_off;
    __global AUX_DATA_T *c_states_tm1_l = c_states_tm1_l_ + c_states_tm1_l_off;
    __global AUX_DATA_T *ws_gates = ws_gates_ + ws_gates_off;
    __global AUX_DATA_T *ws_grid = ws_grid_ + ws_grid_off;
#if CELL_KIND == LBR_GRU || CELL_KIND == VANILLA_GRU
    __global WS_STATE_DATA_T *h_states_tm_l
            = h_states_tm_l_ + h_states_tm_l_off;
#endif

#if CELL_KIND == VANILLA_LSTM
    float G[vanilla_lstm_n_gates];
    float B[vanilla_lstm_n_bias];
    for (int gate_idx = 0; gate_idx < vanilla_lstm_n_gates; gate_idx++) {
        G[gate_idx] = convert_float(scratch_gates[cell_scratch_mem(
                scratch_gates_ld, dhc, i, gate_idx, j)]);
        B[gate_idx] = convert_float(bias[off_ker_bias(dhc, gate_idx, j)]);
    }
    vanilla_lstm_gates_t gates = vanilla_lstm_compute_gates(G, B, tm_scales);
    vanilla_lstm_store(ws_gates, gates_ws_ld, h_states_t_l, c_states_t_l,
            c_states_tm1_l, states_ws_ld, dhc, i, j, tm_cscale, gates);

#elif CELL_KIND == VANILLA_RNN
    float g = vanilla_rnn_compute_gates(
            convert_float(scratch_gates[cell_scratch_mem(
                    scratch_gates_ld, dhc, i, 0, j)]),
            convert_float(bias[off_ker_bias(dhc, 0, j)]), alpha, tm_scales);
    store_vanilla_rnn(
            ws_gates, gates_ws_ld, h_states_t_l, states_ws_ld, dhc, i, j, g);

#elif CELL_KIND == LBR_GRU
    // AUX and SCRATCH data type is same for fwd prop
    __global AUX_DATA_T *scratch_cell = (__global AUX_DATA_T *)(scr_cell);
    __global WS_STATE_DATA_T *src_iter = h_states_tm_l;

    lbr_gru_gates_t gates = compute_gates_lbr_gru(scratch_gates, scratch_cell,
            bias, tm_scales, scratch_gates_ld, dhc, i, j);
    float Wh_b = gates.Wh_b;
    float G0 = gates.G[0];
    float G1 = gates.G[1];
    float G2 = gates.G[2];

    float Ht = G0 * TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)])
            + (1 - G0) * G2;

    h_states_t_l[cell_ws_state(states_ws_ld, i, j)] = TO_WS_STATE(Ht);

    if (!RECOMPUTE_GATES && IS_TRAINING) {
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)] = G0;
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)] = G1;
        ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)] = G2;
        ws_grid[cell_ws_grid_comp(dhc, i, j)] = Wh_b;
    }
#elif CELL_KIND == VANILLA_GRU
    __global WS_STATE_DATA_T *src_iter = h_states_tm_l;

    if (n_part == 1) {
        float G0 = logistic_fwd_tm(
                scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 0, j)]
                        + bias[off_ker_bias(dhc, 0, j)],
                tm_scales[0]);
        float G1 = logistic_fwd_tm(
                scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 1, j)]
                        + bias[off_ker_bias(dhc, 1, j)],
                tm_scales[1]);

        /* TODO from CPU: Can be optimized for fwd_training by using
        ws_gates instead of scratch_gates in p2 */
        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 0, j)]
                = TO_ACC(G0);
        scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 1, j)]
                = TO_ACC(G1);
        float tmp = TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)]);
        h_states_t_l[cell_ws_state(states_ws_ld, i, j)] = TO_WS_STATE(tmp * G1);
        if (!RECOMPUTE_GATES && IS_TRAINING) {
            ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)] = G0;
            ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)] = G1;
        }
    } else if (n_part == 2) {
        float G0 = convert_float(scratch_gates[cell_scratch_mem(
                scratch_gates_ld, dhc, i, 0, j)]);
        float G2 = tanh_fwd_tm(
                scratch_gates[cell_scratch_mem(scratch_gates_ld, dhc, i, 2, j)]
                        + bias[off_ker_bias(dhc, 2, j)],
                tm_scales[2]);
        float tmp = TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)]);
        h_states_t_l[cell_ws_state(states_ws_ld, i, j)]
                = TO_WS_STATE(tmp * G0 + (1.0f - G0) * G2);
        if (!RECOMPUTE_GATES && IS_TRAINING) {
            ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)] = G2;
        }
    }
#else
#error "Wrong Cell Kind"
#endif
}
#endif

#if !IS_FWD
// The scratch_diff_gates and scratch_gates buffers may refer to the
// same memory when sizeof(SRC_DATA_T) == sizeof(AUX_DATA_T) or when
// scratch_gates is unused in order to reduce memory usage
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
simple_rnn_elemwise_bwd(int dir, int lay, int iter,
        __global SRC_DATA_T *scratch_diff_gates_, dim_t scratch_diff_gates_off,
        __global AUX_DATA_T *scratch_gates_, dim_t scratch_gates_off,
        __global BIAS_DATA_T *bias_, dim_t bias_off, float alpha,
        __global float *tm_scales, __global WS_STATE_DATA_T *states_tm1_l_,
        dim_t states_tm1_l_off, __global AUX_DATA_T *c_states_t_l_,
        dim_t c_states_t_l_off, __global AUX_DATA_T *c_states_tm1_l_,
        dim_t c_states_tm1_l_off, __global AUX_DATA_T *ws_gates_,
        dim_t ws_gates_off, __global AUX_DATA_T *ws_grid_, dim_t ws_grid_off,
        int states_ws_ld, int gates_ws_ld, int scratch_diff_gates_ld,
        int scratch_gates_ld, int batch, int dhc, int scratch_diff_states_ld,
        int diff_states_layer_ld,
#if CELL_KIND == VANILLA_LSTM || CELL_KIND == VANILLA_RNN
        float tm_cscale,
#elif CELL_KIND == LBR_GRU
        __global char *scr_gate_r,
#elif CELL_KIND == VANILLA_GRU
        int n_part, __global char *scr_cell, __global DIFF_DATA_T *dhG1_,
        dim_t dhG1_off,
#endif
        __global DIFF_DATA_T *diff_states_t_l_, dim_t diff_states_t_l_off,
        __global DIFF_DATA_T *diff_states_tp1_l_, dim_t diff_states_tp1_l_off,
        __global DIFF_DATA_T *diff_states_t_lp1_, dim_t diff_states_t_lp1_off,
#if CELL_KIND == VANILLA_LSTM
        __global DIFF_DATA_T *diff_states_t_l_s1_, dim_t diff_states_t_l_s1_off,
        __global DIFF_DATA_T *diff_states_tp1_l_s1_,
        dim_t diff_states_tp1_l_s1_off,
#endif
        MAYBE_ATOMIC DIFF_BIAS_DATA_T *diff_bias_base,
        int64x4_t diff_bias_strides) {
    const int i_ = get_global_id(1) * ELEMWISE_BWD_BATCH_BLOCK; // batch
    const int j = get_global_id(0); // dhc

    MAYBE_ATOMIC DIFF_BIAS_DATA_T *diff_bias
            = diff_bias_base + diff_bias_off(diff_bias_strides, lay, dir, 0, 0);

    if (j >= dhc) return;

    __global SRC_DATA_T *scratch_diff_gates
            = scratch_diff_gates_ + scratch_diff_gates_off;
    __global AUX_DATA_T *scratch_gates = scratch_gates_ + scratch_gates_off;
    __global BIAS_DATA_T *bias = bias_ + bias_off;
    __global WS_STATE_DATA_T *states_tm1_l = states_tm1_l_ + states_tm1_l_off;
    __global AUX_DATA_T *c_states_t_l = c_states_t_l_ + c_states_t_l_off;
    __global AUX_DATA_T *c_states_tm1_l = c_states_tm1_l_ + c_states_tm1_l_off;
    __global AUX_DATA_T *ws_gates = ws_gates_ + ws_gates_off;
    __global AUX_DATA_T *ws_grid = ws_grid_ + ws_grid_off;
    __global DIFF_DATA_T *diff_states_t_l
            = diff_states_t_l_ + diff_states_t_l_off;
    __global DIFF_DATA_T *diff_states_tp1_l
            = diff_states_tp1_l_ + diff_states_tp1_l_off;
    __global DIFF_DATA_T *diff_states_t_lp1
            = diff_states_t_lp1_ + diff_states_t_lp1_off;
#if CELL_KIND == VANILLA_LSTM
    __global DIFF_DATA_T *diff_states_t_l_s1
            = diff_states_t_l_s1_ + diff_states_t_l_s1_off;
    __global DIFF_DATA_T *diff_states_tp1_l_s1
            = diff_states_tp1_l_s1_ + diff_states_tp1_l_s1_off;
#endif

    DIFF_DATA_T diff_bias_acc[n_bias] = {0};

    for (int batch_id = 0; batch_id < ELEMWISE_BWD_BATCH_BLOCK; batch_id++) {
        int i = i_ + batch_id;
        if (i >= batch) break;

#if CELL_KIND == VANILLA_LSTM

#if !RECOMPUTE_GATES
        float G0 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)];
        float G1 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)];
        float G2 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)];
        float G3 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 3, j)];
#else
        float G[vanilla_lstm_n_gates];
        float B[vanilla_lstm_n_bias];
        for (int gate_idx = 0; gate_idx < vanilla_lstm_n_gates; gate_idx++) {
            G[gate_idx] = convert_float(scratch_gates[cell_scratch_mem(
                    scratch_gates_ld, dhc, i, gate_idx, j)]);
            B[gate_idx] = convert_float(bias[off_ker_bias(dhc, gate_idx, j)]);
        }
        vanilla_lstm_gates_t gates
                = vanilla_lstm_compute_gates(G, B, tm_scales);
        float G0 = gates.G[0];
        float G1 = gates.G[1];
        float G2 = gates.G[2];
        float G3 = gates.G[3];
#endif

        float Ct = c_states_t_l[cell_ws_state(states_ws_ld, i, j)];
        /// @todo save it in the workspace in fwd pass or recompute it to
        /// save bw
        float tanhCt = tanh_fwd_tm(Ct, tm_cscale);
        // we have 2 incoming diffs on Ht
        float dHt = (float)diff_states_tp1_l[cell_scratch_diff_states(
                            batch, scratch_diff_states_ld, i, j)]
                + diff_states_t_lp1[cell_scratch_diff_states(
                        batch, diff_states_layer_ld, i, j)];

        float dCt = (float)diff_states_tp1_l_s1[cell_scratch_diff_states(
                            batch, scratch_diff_states_ld, i, j)]
                + one_m_square(tanhCt) * G3 * dHt;

        float dG1 = (float)c_states_tm1_l[cell_ws_state(states_ws_ld, i, j)]
                * dCt * x_m_square(G1);
        float dG0 = G2 * dCt * x_m_square(G0);
        float dG3 = tanhCt * dHt * x_m_square(G3);
        float dG2 = G0 * dCt * one_m_square(G2);

        diff_states_t_l_s1[cell_scratch_diff_states(
                batch, scratch_diff_states_ld, i, j)]
                = dCt * G1;

        scratch_diff_gates[cell_scratch_mem(
                scratch_diff_gates_ld, dhc, i, 0, j)]
                = TO_SRC(dG0);
        diff_bias_acc[0] += dG0;
        scratch_diff_gates[cell_scratch_mem(
                scratch_diff_gates_ld, dhc, i, 1, j)]
                = TO_SRC(dG1);
        diff_bias_acc[1] += dG1;
        scratch_diff_gates[cell_scratch_mem(
                scratch_diff_gates_ld, dhc, i, 2, j)]
                = TO_SRC(dG2);
        diff_bias_acc[2] += dG2;
        scratch_diff_gates[cell_scratch_mem(
                scratch_diff_gates_ld, dhc, i, 3, j)]
                = TO_SRC(dG3);
        diff_bias_acc[3] += dG3;

#elif CELL_KIND == LBR_GRU
        __global SRC_DATA_T *scratch_gate_r
                = (__global SRC_DATA_T *)(scr_gate_r);
        __global WS_STATE_DATA_T *src_iter = states_tm1_l; //h_states_tm1_l

        float h = TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)]);
#if !RECOMPUTE_GATES
        float Wh_b = ws_grid[cell_ws_grid_comp(dhc, i, j)];
        float G0 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)];
        float G1 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)];
        float G2 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 2, j)];
#else
        lbr_gru_gates_t gates = compute_gates_lbr_gru(scratch_gates,
                scratch_cell, bias, tm_scales, scratch_gates_ld, dhc, i, j);
        float Wh_b = gates.Wh_b;
        float G0 = gates.G[0];
        float G1 = gates.G[1];
        float G2 = gates.G[2];
#endif

        float dHt = diff_states_tp1_l[cell_scratch_diff_states(
                            batch, scratch_diff_states_ld, i, j)]
                + diff_states_t_lp1[cell_scratch_diff_states(
                        batch, diff_states_layer_ld, i, j)];

        float dG0 = (h - G2) * dHt * x_m_square(G0);
        float dG2 = (1.0f - G0) * one_m_square(G2) * dHt;
        float dG1 = Wh_b * dG2 * x_m_square(G1);

        diff_states_t_l[cell_scratch_diff_states(
                batch, scratch_diff_states_ld, i, j)]
                = dHt * G0;

        scratch_diff_gates[cell_scratch_mem(
                scratch_diff_gates_ld, dhc, i, 0, j)]
                = TO_SRC(dG0);
        diff_bias_acc[0] += dG0;
        scratch_diff_gates[cell_scratch_mem(
                scratch_diff_gates_ld, dhc, i, 1, j)]
                = TO_SRC(dG1);
        diff_bias_acc[1] += dG1;
        scratch_diff_gates[cell_scratch_mem(
                scratch_diff_gates_ld, dhc, i, 2, j)]
                = TO_SRC(dG2);
        diff_bias_acc[2] += dG2;

        scratch_gate_r[cell_scratch_mem(scratch_diff_gates_ld, dhc, i, 0, j)]
                = TO_SRC(dG0);
        scratch_gate_r[cell_scratch_mem(scratch_diff_gates_ld, dhc, i, 1, j)]
                = TO_SRC(dG1);
        float tmp = dG2 * ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)];
        scratch_gate_r[cell_scratch_mem(scratch_diff_gates_ld, dhc, i, 2, j)]
                = TO_SRC(tmp);
        diff_bias_acc[3] += tmp;

#elif CELL_KIND == VANILLA_RNN
        float dH = diff_states_tp1_l[cell_scratch_diff_states(
                           batch, scratch_diff_states_ld, i, j)]
                + diff_states_t_lp1[cell_scratch_diff_states(
                        batch, diff_states_layer_ld, i, j)];

#if !RECOMPUTE_GATES
        float g = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)];
#else
        float g = vanilla_rnn_compute_gates(
                convert_float(scratch_gates[cell_scratch_mem(
                        scratch_gates_ld, dhc, i, 0, j)]),
                convert_float(bias[off_ker_bias(dhc, 0, j)]), alpha, tm_scales);
#endif
#if IS_TESTMODE
        float tmp = = dH * activation_bwd(g, tm_scales[0], 0.);
        scratch_diff_gates[cell_scratch_mem(
                scratch_diff_gates_ld, dhc, i, 0, j)]
                = TO_SRC(tmp);
        diff_bias_acc[0] += tmp;
#else
        float tmp = dH * activation_bwd(g, alpha, 0.);
        scratch_diff_gates[cell_scratch_mem(
                scratch_diff_gates_ld, dhc, i, 0, j)]
                = TO_SRC(tmp);
        diff_bias_acc[0] += tmp;
#endif
#elif CELL_KIND == VANILLA_GRU
        __global WS_STATE_DATA_T *src_iter = states_tm1_l; // h_states_tm1_l

        float h = TO_REF(src_iter[cell_ws_state(states_ws_ld, i, j)]);
        if (n_part == 1) {
            float dHt = diff_states_tp1_l[cell_scratch_diff_states(
                                batch, scratch_diff_states_ld, i, j)]
                    + diff_states_t_lp1[cell_scratch_diff_states(
                            batch, diff_states_layer_ld, i, j)];
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
            diff_states_t_l[cell_scratch_diff_states(
                    batch, scratch_diff_states_ld, i, j)]
                    = dHt * ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 0, j)];

            scratch_diff_gates[cell_scratch_mem(
                    scratch_diff_gates_ld, dhc, i, 0, j)]
                    = TO_SRC(dG0);
            diff_bias_acc[0] += dG0;
            scratch_diff_gates[cell_scratch_mem(
                    scratch_diff_gates_ld, dhc, i, 2, j)]
                    = TO_SRC(dG2);
            diff_bias_acc[2] += dG2;
        } else if (n_part == 2) {
            __global SRC_DATA_T *scratch_cell
                    = (__global SRC_DATA_T *)(scr_cell);
            __global DIFF_DATA_T *dhG1 = dhG1_ + dhG1_off;

            float dG1 = ws_gates[cell_ws_gates(gates_ws_ld, dhc, i, 1, j)];
            diff_states_t_l[cell_scratch_diff_states(
                    batch, scratch_diff_states_ld, i, j)]
                    += dhG1[off_scratch_dhg1(
                               batch, scratch_diff_states_ld, i, j)]
                    * dG1;

            float tmp = dhG1[off_scratch_dhg1(
                                batch, scratch_diff_states_ld, i, j)]
                    * h * x_m_square(dG1);
            scratch_diff_gates[cell_scratch_mem(
                    scratch_diff_gates_ld, dhc, i, 1, j)]
                    = TO_SRC(tmp);
            diff_bias_acc[1] += tmp;
            scratch_cell[off_scratch_cell(batch, states_ws_ld, i, j)]
                    = TO_SRC(dG1 * h);
        }
#else
#error "Wrong Cell Kind"
#endif
    }
    unroll_for(int k = 0; k < n_bias; k++) {
#if NEED_BIAS_ATOMIC_REDUCE
        atomic_add_global(&diff_bias[k * dhc + j], diff_bias_acc[k]);
#else
        diff_bias[k * dhc + j] += diff_bias_acc[k];
#endif
    }
}
#else
__kernel void simple_rnn_elemwise_bwd() {}
#endif // !IS_FWD

#if CELL_COMP_ENABLED

void gemm_sum_inner(float(C)[M_THR_BLOCK][N_THR_BLOCK],
        const __global WS_STATE_DATA_T *restrict A, const int a_stride,
        const __global WEI_LAYER_DATA_T *restrict B, const int b_stride,
        const int m_thr_stride, const int n_thr_stride, int m_l_end,
        int k_l_end, int n_l_end, bool mn_valid) {

    // Load A - Invariant across the subgroup, can do cooperative load
    float A_l[M_THR_BLOCK] = {};
    unroll_for(int m_l = 0; m_l < M_THR_BLOCK; m_l++) {
        load(A_l + m_l, A + m_l * m_thr_stride * a_stride,
                (mn_valid || (m_l * m_thr_stride) < m_l_end)
                        && (int)get_sub_group_local_id() < k_l_end);
    }

    // Load B
    float B_l[gemm_k_block][N_THR_BLOCK] = {};
    unroll_for(int n_l = 0; n_l < N_THR_BLOCK; n_l++) {
        unroll_for(int k_l = 0; k_l < gemm_k_block; k_l++) {
            load(&B_l[k_l][n_l], &B[k_l * b_stride + n_l * n_thr_stride],
                    k_l < k_l_end
                            && (mn_valid
                                    || (int)get_sub_group_local_id()
                                            < (n_l_end - n_l * n_thr_stride)));
        }
    }

    // Compute
    unroll_for(int m_l = 0; m_l < M_THR_BLOCK; m_l++) {
        unroll_for(int k_l = 0; k_l < gemm_k_block; k_l++) {
            unroll_for(int n_l = 0; n_l < N_THR_BLOCK; n_l++) {
                C[m_l][n_l] += B_l[k_l][n_l] * sg_get(A_l[m_l], k_l);
            }
        }
    }
}

// Perform C += A * B where all matrices are in row major layout
void gemm_sum(float(C)[N_OUTER_BLOCK][M_THR_BLOCK][N_THR_BLOCK],
        const __global WS_STATE_DATA_T *restrict A, const int a_stride,
        const __global WEI_LAYER_DATA_T *restrict B, const int b_stride,
        gemm_dims_t size, int m_sg, int m_thr_stride, int n_sg,
        int n_thr_stride, bool enable_m_tail, bool enable_k_tail,
        bool enable_n_tail) {

    // Optimization opportunity: Loads across the m and n dimension can overflow
    // so long as they do not cross the end of the buffer.
    bool valid_mn
            = (!enable_m_tail || m_sg + m_thr_stride * M_THR_BLOCK <= size.m)
            && (!enable_n_tail
                    || n_sg + n_thr_stride * N_THR_BLOCK <= size.n_inner);
    if (valid_mn) {
        int k_outer = 0;
        while (valid_mn && k_outer < size.k - gemm_k_block + 1) {
            int k_l_end = gemm_k_block;
            const int m_l_end = m_thr_stride * M_THR_BLOCK;
            const int n_l_end = n_thr_stride * N_THR_BLOCK;

            const int a_off_base = m_sg * a_stride + k_outer;
            for (int n_outer = 0; n_outer < N_OUTER_BLOCK; n_outer++) {
                const int b_off_base
                        = n_outer * size.n_inner + k_outer * b_stride + n_sg;
                gemm_sum_inner(C[n_outer], A + a_off_base, a_stride,
                        B + b_off_base, b_stride, m_thr_stride, n_thr_stride,
                        m_l_end, k_l_end, n_l_end, true);
            }
            k_outer += gemm_k_block;
        }
        while (enable_k_tail && k_outer < size.k) {
            int k_l_end = size.k - k_outer;
            const int m_l_end = size.m - m_sg;
            const int n_l_end = size.n_inner - n_sg;

            const int a_off_base = m_sg * a_stride + k_outer;
            for (int n_outer = 0; n_outer < N_OUTER_BLOCK; n_outer++) {
                const int b_off_base
                        = n_outer * size.n_inner + k_outer * b_stride + n_sg;
                gemm_sum_inner(C[n_outer], A + a_off_base, a_stride,
                        B + b_off_base, b_stride, m_thr_stride, n_thr_stride,
                        m_l_end, k_l_end, n_l_end, true);
            }
            k_outer += gemm_k_block;
        }
    } else {
        int k_outer = 0;
        while (k_outer < size.k) {
            int k_l_end = enable_k_tail ? size.k - k_outer : gemm_k_block;
            const int m_l_end = size.m - m_sg;
            const int n_l_end = size.n_inner - n_sg;

            const int a_off_base = m_sg * a_stride + k_outer;
            for (int n_outer = 0; n_outer < N_OUTER_BLOCK; n_outer++) {
                const int b_off_base
                        = n_outer * size.n_inner + k_outer * b_stride + n_sg;
                gemm_sum_inner(C[n_outer], A + a_off_base, a_stride,
                        B + b_off_base, b_stride, m_thr_stride, n_thr_stride,
                        m_l_end, k_l_end, n_l_end, false);
            }
            k_outer += gemm_k_block;
        }
    }
}

void cell_common_inner(const_wei_layer_cell_t wei_layer,
        const_wei_iter_cell_t wei_iter, const_ws_state_cell_t cell_layer,
        const_ws_state_cell_t cell_iter, aux_cell_t gates,
        ws_state_cell_t states, const_aux_cell_t scratch_gates, cell_ctx_t ctx,
        cell_dims_t outer, cell_dims_t dims) {
    // Extract local id from the subgroup id rather than `get_local_id` as the
    // mapping from subgroups to the local work group is not well defined.
    const int local_sgid0
            = get_sub_group_id() * SUBGROUP_SIZE % get_local_size(0);
    const int local_sgid1
            = get_sub_group_id() * SUBGROUP_SIZE / get_local_size(0);

    const int c_tg = outer.dhc + get_group_id(0) * DHC_LOCAL;
    const int c_sg = c_tg + local_sgid0 * CELL_DHC_THR;
    const int c_thr = c_sg + get_sub_group_local_id();
    const int c_thr_stride = SUBGROUP_SIZE;

    const int n_tg = outer.mb + get_group_id(1) * BATCH_LOCAL;
    const int n_sg = n_tg + local_sgid1 * CELL_BATCH_THR;
    const int n_thr = n_sg;
    const int n_thr_stride = 1;

    float C[n_gates][CELL_BATCH_THR][CELL_DHC_THR] = {};

    if (NEED_SCRATCH_GATES) {
        // GEMM operations may be calculated in an separate external kernel and are
        // passed in via `scratch_gates`
        for_(int gate_idx = 0; gate_idx < n_gates; gate_idx++)
        for (int n_l = 0; n_l < CELL_BATCH_THR; n_l++) {
            int n = n_thr + n_l * n_thr_stride;
            if (CELL_MB_TAIL && n >= dims.mb) break;
            for (int c_l = 0; c_l < CELL_DHC_THR; c_l++) {
                int c = c_thr + c_l * c_thr_stride;
                if (CELL_DHC_TAIL && c >= dims.dhc) break;
                C[gate_idx][n_l]
                 [c_l] = convert_float(scratch_gates.ptr[cell_scratch_mem(
                         scratch_gates.strides.mb, dims.dhc, n, gate_idx, c)]);
            }
        }
    }

    if (CELL_COMPUTE_GEMM_LAYER) {
        // cell_states = batch x slc
        // wei_layer = slc x (gates x dhc)
        // C = batch x (gates x dhc)
        gemm_dims_t size = {.m = dims.mb,
                .k = dims.slc,
                .n_inner = dims.dhc,
                .n_outer = n_gates};
        gemm_sum(C, cell_layer.ptr, cell_layer.strides.mb, wei_layer.ptr,
                wei_layer.strides.slc, size, n_sg, n_thr_stride, c_sg,
                c_thr_stride, CELL_DHC_TAIL, CELL_GEMM_LAYER_K_TAIL,
                CELL_MB_TAIL);
    }

    if (CELL_COMPUTE_GEMM_ITER) {
        // cell_states = batch x sic
        // wei_iter  = sic x (gates x dhc)
        // C = batch x (gates x dhc)
        gemm_dims_t size = {.m = dims.mb,
                .k = dims.sic,
                .n_inner = dims.dhc,
                .n_outer = n_gates};
        gemm_sum(C, cell_iter.ptr, cell_iter.strides.mb, wei_iter.ptr,
                wei_iter.strides.sic, size, n_sg, n_thr_stride, c_sg,
                c_thr_stride, CELL_DHC_TAIL, CELL_GEMM_ITER_K_TAIL,
                CELL_MB_TAIL);
    }

    for (int n_l = 0; n_l < CELL_BATCH_THR; n_l++) {
        int n = n_thr + n_l * n_thr_stride;
        if (CELL_MB_TAIL && n >= dims.mb) break;
        for (int c_l = 0; c_l < CELL_DHC_THR; c_l++) {
            int c = c_thr + c_l * c_thr_stride;
            if (CELL_DHC_TAIL && c >= dims.dhc) break;
            if (CELL_KIND == VANILLA_LSTM) {
                float G[vanilla_lstm_n_gates];
                float B[vanilla_lstm_n_bias];
                for (int gate_idx = 0; gate_idx < vanilla_lstm_n_gates;
                        gate_idx++) {
                    G[gate_idx] = C[gate_idx][n_l][c_l];
                    B[gate_idx] = convert_float(
                            ctx.lstm.bias[off_ker_bias(dims.dhc, gate_idx, c)]);
                }
                vanilla_lstm_gates_t g
                        = vanilla_lstm_compute_gates(G, B, ctx.lstm.tm_scales);
                vanilla_lstm_store(gates.ptr, gates.strides.mb, states.ptr,
                        ctx.lstm.c_states, ctx.lstm.c_states_iter,
                        states.strides.mb, dims.dhc, n, c, ctx.lstm.tm_cscale,
                        g);
            } else if (CELL_KIND == VANILLA_RNN) {
                float g = vanilla_rnn_compute_gates(C[0][n_l][c_l],
                        ctx.rnn.bias[off_ker_bias(dims.dhc, 0, c)],
                        ctx.rnn.alpha, ctx.rnn.tm_scales);
                store_vanilla_rnn(gates.ptr, gates.strides.mb, states.ptr,
                        states.strides.mb, dims.dhc, n, c, g);
            }
        }
    }
}

void cell_common(const_wei_layer_cell_t wei_layer,
        const_wei_iter_cell_t wei_iter, const_ws_state_cell_t cell_layer,
        const_ws_state_cell_t cell_iter, aux_cell_t gates,
        ws_state_cell_t states, const_aux_cell_t scratch_gates, cell_ctx_t ctx,
        cell_dims_t dims, cell_loops_t loops) {

    for_(cell_dim_t mb_outer = 0; mb_outer < loops.mb; mb_outer += BATCH_LOCAL)
    for (cell_dim_t dhc_outer = 0; dhc_outer < loops.dhc;
            dhc_outer += DHC_LOCAL) {
        cell_dims_t outer = {.mb = mb_outer, .dhc = dhc_outer};
        cell_common_inner(wei_layer, wei_iter, cell_layer, cell_iter, gates,
                states, scratch_gates, ctx, outer, dims);
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
simple_rnn_cell_fwd(__global const WEI_LAYER_DATA_T *wei_layer_,
        dim_t wei_layer_off, int64x5_t wei_layer_strides_,
        __global const WEI_ITER_DATA_T *wei_iter_, dim_t wei_iter_off,
        int64x5_t wei_iter_strides_,
        __global const WS_STATE_DATA_T *cell_layer_, dim_t cell_layer_off,
        int64x2_t cell_layer_strides_,
        __global const WS_STATE_DATA_T *cell_iter_, dim_t cell_iter_off,
        int64x2_t cell_iter_strides_, __global AUX_DATA_T *gates_,
        dim_t gates_off, int64x2_t gates_strides_,
        __global WS_STATE_DATA_T *states_, dim_t states_off,
        int64x2_t states_strides_,
#if CELL_KIND == VANILLA_LSTM
        __global AUX_DATA_T *c_states_, dim_t c_states_off,
        __global const AUX_DATA_T *c_states_iter_, dim_t c_states_iter_off,
        float tm_cscale,
#endif
#if NEED_SCRATCH_GATES
        __global AUX_DATA_T *scratch_gates_, dim_t scratch_gates_off,
        int64x2_t scratch_gates_strides_,
#endif
#if CELL_ENABLE_ITER_BLOCK
        dim_t iter_loop,
#endif
        __global BIAS_DATA_T *bias_, dim_t bias_off, float alpha,
        __global float *tm_scales, dim_t mb, dim_t dhc, dim_t slc, dim_t sic,
        dim_t dhc_loop) {

#if !NEED_SCRATCH_GATES
    __global AUX_DATA_T *scratch_gates_ = NULL;
    dim_t scratch_gates_off = 0;
    int64x2_t scratch_gates_strides_ = {};
#endif
#if !CELL_ENABLE_ITER_BLOCK
    const dim_t iter_loop = 1;
#endif

    grid_strides_t wei_layer_strides
            = {.iter = 0, .cell = {.slc = wei_layer_strides_.array[2]}};
    grid_strides_t wei_iter_strides
            = {.iter = 0, .cell = {.sic = wei_iter_strides_.array[2]}};
    grid_strides_t cell_layer_strides = {.iter = cell_layer_strides_.array[0],
            .cell = {.mb = cell_layer_strides_.array[1]}};
    grid_strides_t cell_iter_strides = {.iter = cell_iter_strides_.array[0],
            .cell = {.mb = cell_iter_strides_.array[1]}};
    grid_strides_t gates_strides = {.iter = gates_strides_.array[0],
            .cell = {.mb = gates_strides_.array[1]}};
    grid_strides_t states_strides = {.iter = states_strides_.array[0],
            .cell = {.mb = states_strides_.array[1]}};
    grid_strides_t scratch_gates_strides
            = {.iter = scratch_gates_strides_.array[0],
                    .cell = {.mb = scratch_gates_strides_.array[1]}};

    cell_dims_t dims = {.mb = mb, .dhc = dhc, .slc = slc, .sic = sic};
    cell_loops_t cell_loops = {.mb = BATCH_LOCAL, .dhc = dhc_loop};

    // Optimization Opportunity: The weights buffers are reused across
    // iterations. Because of this, we can load and reorder the weights buffers
    // into SLM so that we can get optimal load patterns in the GEMM operations
    // below.
    const_wei_layer_cell_t wei_layer = {.ptr = wei_layer_ + wei_layer_off,
            .strides = wei_layer_strides.cell};
    const_wei_iter_cell_t wei_iter = {
            .ptr = wei_iter_ + wei_iter_off, .strides = wei_iter_strides.cell};

    // Optimization Opportunity: bias can be preloaded to a register if n_gates*dhc
    // is small enough.
    __global BIAS_DATA_T *bias = bias_ + bias_off;

    for (dim_t iter = 0; iter < iter_loop; iter++) {
        const_ws_state_cell_t cell_layer = {.ptr
                = cell_layer_ + cell_layer_off + cell_layer_strides.iter * iter,
                .strides = cell_layer_strides.cell};
        const_ws_state_cell_t cell_iter = {.ptr
                = cell_iter_ + cell_iter_off + cell_iter_strides.iter * iter,
                .strides = cell_iter_strides.cell};
        aux_cell_t gates
                = {.ptr = gates_ + gates_off + gates_strides.iter * iter,
                        .strides = gates_strides.cell};
        ws_state_cell_t states
                = {.ptr = states_ + states_off + states_strides.iter * iter,
                        .strides = states_strides.cell};
        const_aux_cell_t scratch_gates = {.ptr = scratch_gates_
                        + scratch_gates_off + scratch_gates_strides.iter * iter,
                .strides = scratch_gates_strides.cell};

#if CELL_KIND == VANILLA_RNN
        cell_ctx_t cell_ctx = {
                .rnn = {.alpha = alpha, .bias = bias, .tm_scales = tm_scales}};
        __global AUX_DATA_T *c_states_ = NULL;
#elif CELL_KIND == VANILLA_LSTM
        cell_ctx_t cell_ctx
                = {.lstm = {.c_states = c_states_ + c_states_off
                                   + states_strides.iter * iter,
                           .c_states_iter = c_states_iter_ + c_states_iter_off
                                   + states_strides.iter * iter,
                           .bias = bias,
                           .tm_scales = tm_scales,
                           .tm_cscale = tm_cscale}};

#endif

        cell_common(wei_layer, wei_iter, cell_layer, cell_iter, gates, states,
                scratch_gates, cell_ctx, dims, cell_loops);

        if (iter < iter_loop - 1) barrier(CLK_GLOBAL_MEM_FENCE);
    }

    return;
}

#else
__kernel void simple_rnn_cell_fwd() {}
#endif
