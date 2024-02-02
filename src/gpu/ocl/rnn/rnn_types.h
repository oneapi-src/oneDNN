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

#ifndef GPU_OCL_RNN_RNN_TYPES_H
#define GPU_OCL_RNN_RNN_TYPES_H

#include "gpu/ocl/ocl_types.h"

#if OUTPUT_DT_U8
#define TO_OUTPUT(x) convert_uchar_sat_rte(x)
#elif OUTPUT_DT_S8
#define TO_OUTPUT(x) convert_char_sat_rte(x)
#elif OUTPUT_DT_S32
#define TO_OUTPUT(x) convert_int_sat_rte(x)
#else
#define TO_OUTPUT(x) (x)
#endif

#if INPUT_DT_BF16
#define TO_INPUT(x) cvt_f32_to_bf16(x)
#define TO_REF(x) cvt_bf16_to_f32(x)
#else
#define TO_INPUT(x) (x)
#define TO_REF(x) (float)(x)
#endif

#if DT_F16 && !IS_FWD
#error "FP16 is not supported for BWD"
#endif

#define OFFTYPE ulong
#define TO_WS_STATE(x) TO_SRC(x)

typedef struct param4 {
    int s0;
    int s1;
    int s2;
    int s3;
} param4;

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

#if CELL_KIND == VANILLA_RNN
const int n_gates = 1;
const int n_bias = 1;
#elif CELL_KIND == VANILLA_LSTM
const int n_gates = 4;
const int n_bias = 4;
#elif CELL_KIND == LBR_GRU
const int n_gates = 3;
const int n_bias = 4;
#elif CELL_KIND == VANILLA_GRU
const int n_gates = 3;
const int n_bias = 3;
#else
#error "Unimplemented cell kind"
#endif

int comp_off(int n_dir, int dhc, int i0, int i1, int i2, int i3) {
    return (((i0 * n_dir + i1) * n_bias + i2) * dhc + i3);
}

int off_ws_state(int n_layer, int n_dir, int n_iter, int batch,
        int states_ws_ld, int i0_, int i1, int i2_, int i3, int i4) {
    int i0 = COPY_SRC_LAYER ? i0_ + 1 : i0_;
    int i0_size = COPY_SRC_LAYER ? n_layer + 1 : n_layer;
    int i2 = i2_ + 1;
    return OFF5(i0, i0_size, i1, n_dir, i2, n_iter + 1, i3, batch, i4,
            states_ws_ld);
}
int off_ws_c_state(int n_layer, int n_dir, int n_iter, int batch,
        int states_ws_ld, int i0_, int i1, int i2_, int i3, int i4) {
    int i0 = i0_;
    int i0_size = n_layer;
    int i2 = i2_ + 1;
    return OFF5(i0, i0_size, i1, n_dir, i2, n_iter + 1, i3, batch, i4,
            states_ws_ld);
}
// cannot be presented by OFF6 due to leading dimension across two dims
int off_ws_gates(int n_dir, int n_iter, int batch, int gates_ws_ld, int dhc,
        int i0, int i1, int i2, int i3, int i4, int i5) {
    return i0 * n_dir * n_iter * batch * gates_ws_ld
            + i1 * n_iter * batch * gates_ws_ld + i2 * batch * gates_ws_ld
            + i3 * gates_ws_ld + i4 * dhc + i5;
}
int off_ws_bias(
        int n_layer, int n_dir, int dhc, int i0, int i1, int i2, int i3) {
    return OFF4(i0, n_layer, i1, n_dir, i2, n_bias, i3, dhc);
}
// grid offset for lbr GRU, LD = DHC
int off_ws_grid_offset(int n_layer, int n_dir, int n_iter, int batch, int dhc,
        int i0, int i1, int i2, int i3, int i4) {
    return OFF5(i0, n_layer, i1, n_dir, i2, n_iter, i3, batch, i4, dhc);
}

int off_ker_bias(int dhc, int i0, int i1) {
    return OFF2(i0, n_gates, i1, dhc);
}
int off_scratch_diff_states(int n_layer, int n_dir, int n_states, int n_iter,
        int batch, int scratch_diff_states_ld, int i0, int i1, int i2, int i3,
        int i4, int i5) {
    bool have_result_layer = COPY_DIFF_SRC_LAYER || n_layer > 1;
    int i0_size = n_layer - 1 + COPY_DIFF_DST_LAYER + have_result_layer;
    if (!have_result_layer) { i0--; }
    int i2_size = COPY_DIFF_DST_LAYER || COPY_DIFF_SRC_LAYER || n_layer != 1
            ? n_states + 1
            : n_states;

    int i3_size = n_iter + 1;
    if (i0_size == 0) {
        i3_size = 2;
        i3 %= i3_size;
        return OFF5(i1, conf_.n_dir, i2, i2_size, i3, i3_size, i4, batch, i5,
                scratch_diff_states_ld);
    }

    return OFF6(i0, i0_size, i1, n_dir, i2, i2_size, i3, i3_size, i4, batch, i5,
            scratch_diff_states_ld);
}
int off_scratch_dhg1(int batch, int scratch_diff_states_ld, int i0, int i1) {
    return OFF2(i0, batch, i1, scratch_diff_states_ld);
}
int off_scratch_cell(int batch, int states_ws_ld, int i0, int i1) {
    return OFF2(i0, batch, i1, states_ws_ld);
}

// for cell - shorter forms

int cell_ws_state(int states_ws_ld, int i, int j) {
    // OFF_WS_STATE(0, 0, 0, i4, i5)
    return i * states_ws_ld + j;
}
int cell_ws_gates(int gates_ws_ld, int dhc, int i, int n, int j) {
    // OFF_WS_GATES(0, 0, 0, i3, i4, i5)
    return i * gates_ws_ld + n * dhc + j;
}
int cell_ws_grid_comp(int dhc, int i3, int i4) {
    // OFF_WS_GRID_OFFSET(0, 0, 0, i3, i4)
    return i3 * dhc + i4;
}
int cell_scratch_mem(int scratch_gates_ld, int dhc, int i, int n, int j) {
    // OFF_SCRATCH_MEM(0, i1, i2, i3)
    return i * scratch_gates_ld + n * dhc + j;
}
int cell_scratch_diff_states(
        int batch, int scratch_diff_states_ld, int i4, int i5) {
    // OFF_SCRATCH_DIFF_STATES(0, 0, i2, 0, i4, i5)
    return (i4 * scratch_diff_states_ld + i5);
}

int bias_off(param4 bias, int x0, int x1, int x2, int x3) {
    return ((x0 % BIAS_B0) * BIAS_SB0 + (x0 / BIAS_B0) * bias.s0
            + (x1 % BIAS_B1) * BIAS_SB1 + (x1 / BIAS_B1) * bias.s1
            + (x2 % BIAS_B2) * BIAS_SB2 + (x2 / BIAS_B2) * bias.s2
            + (x3 % BIAS_B3) * BIAS_SB3 + (x3 / BIAS_B3) * bias.s3);
}
int src_l_off(int src_l_s0, int iter) {
    return ((iter % SRC_L_B0) * SRC_L_SB0 + (iter / SRC_L_B0) * src_l_s0);
}
int src_i_off(param4 src_i, int x0, int x1, int x2, int x3) {
    return ((x0 % SRC_I_B0) * SRC_I_SB0 + (x0 / SRC_I_B0) * src_i.s0
            + (x1 % SRC_I_B1) * SRC_I_SB1 + (x1 / SRC_I_B1) * src_i.s1
            + (x2 % SRC_I_B2) * SRC_I_SB2 + (x2 / SRC_I_B2) * src_i.s2
            + (x3 % SRC_I_B3) * SRC_I_SB3 + (x3 / SRC_I_B3) * src_i.s3);
}
#if WITH_SRC_ITER_C
int src_i_c_off(param4 src_i_c, int x0, int x1, int x2, int x3) {
    return ((x0 % SRC_I_C_B0) * SRC_I_C_SB0 + (x0 / SRC_I_C_B0) * src_i_c.s0
            + (x1 % SRC_I_C_B1) * SRC_I_C_SB1 + (x1 / SRC_I_C_B1) * src_i_c.s1
            + (x2 % SRC_I_C_B2) * SRC_I_C_SB2 + (x2 / SRC_I_C_B2) * src_i_c.s2
            + (x3 % SRC_I_C_B3) * SRC_I_C_SB3 + (x3 / SRC_I_C_B3) * src_i_c.s3);
}
#endif
int dst_l_off(param4 dst_l, int x0, int x1, int x2) {
    return ((x0 % DST_L_B0) * DST_L_SB0 + (x0 / DST_L_B0) * dst_l.s0
            + (x1 % DST_L_B1) * DST_L_SB1 + (x1 / DST_L_B1) * dst_l.s1
            + (x2 % DST_L_B2) * DST_L_SB2 + (x2 / DST_L_B2) * dst_l.s2);
}
int dst_i_off(param4 dst_i, int x0, int x1, int x2, int x3) {
    return ((x0 % DST_I_B0) * DST_I_SB0 + (x0 / DST_I_B0) * dst_i.s0
            + (x1 % DST_I_B1) * DST_I_SB1 + (x1 / DST_I_B1) * dst_i.s1
            + (x2 % DST_I_B2) * DST_I_SB2 + (x2 / DST_I_B2) * dst_i.s2
            + (x3 % DST_I_B3) * DST_I_SB3 + (x3 / DST_I_B3) * dst_i.s3);
}
#if WITH_DST_ITER_C
int dst_i_c_off(param4 dst_i_c, int x0, int x1, int x2, int x3) {
    return ((x0 % DST_I_C_B0) * DST_I_C_SB0 + (x0 / DST_I_C_B0) * dst_i_c.s0
            + (x1 % DST_I_C_B1) * DST_I_C_SB1 + (x1 / DST_I_C_B1) * dst_i_c.s1
            + (x2 % DST_I_C_B2) * DST_I_C_SB2 + (x2 / DST_I_C_B2) * dst_i_c.s2
            + (x3 % DST_I_C_B3) * DST_I_C_SB3 + (x3 / DST_I_C_B3) * dst_i_c.s3);
}
#endif
#if !IS_FWD
int diff_src_l_off(param4 diff_src_l, int x0, int x1, int x2) {
    return ((x0 % DIFF_SRC_L_B0) * DIFF_SRC_L_SB0
            + (x0 / DIFF_SRC_L_B0) * diff_src_l.s0
            + (x1 % DIFF_SRC_L_B1) * DIFF_SRC_L_SB1
            + (x1 / DIFF_SRC_L_B1) * diff_src_l.s1
            + (x2 % DIFF_SRC_L_B2) * DIFF_SRC_L_SB2
            + (x2 / DIFF_SRC_L_B2) * diff_src_l.s2);
}
int diff_dst_l_off(param4 diff_dst_l, int iter, int batch) {
    return ((iter % DIFF_DST_L_B0) * DIFF_DST_L_SB0
            + (iter / DIFF_DST_L_B0) * diff_dst_l.s0
            + (batch % DIFF_DST_L_B1) * DIFF_DST_L_SB1
            + (batch / DIFF_DST_L_B1) * diff_dst_l.s1);
}
int diff_src_i_off(param4 diff_src_i, int x0, int x1, int x2, int x3) {
    return ((x0 % DIFF_SRC_I_B0) * DIFF_SRC_I_SB0
            + (x0 / DIFF_SRC_I_B0) * diff_src_i.s0
            + (x1 % DIFF_SRC_I_B1) * DIFF_SRC_I_SB1
            + (x1 / DIFF_SRC_I_B1) * diff_src_i.s1
            + (x2 % DIFF_SRC_I_B2) * DIFF_SRC_I_SB2
            + (x2 / DIFF_SRC_I_B2) * diff_src_i.s2
            + (x3 % DIFF_SRC_I_B3) * DIFF_SRC_I_SB3
            + (x3 / DIFF_SRC_I_B3) * diff_src_i.s3);
}
int diff_dst_i_off(param4 diff_dst_i, int x0, int x1, int x2, int x3) {
    return ((x0 % DIFF_DST_I_B0) * DIFF_DST_I_SB0
            + (x0 / DIFF_DST_I_B0) * diff_dst_i.s0
            + (x1 % DIFF_DST_I_B1) * DIFF_DST_I_SB1
            + (x1 / DIFF_DST_I_B1) * diff_dst_i.s1
            + (x2 % DIFF_DST_I_B2) * DIFF_DST_I_SB2
            + (x2 / DIFF_DST_I_B2) * diff_dst_i.s2
            + (x3 % DIFF_DST_I_B3) * DIFF_DST_I_SB3
            + (x3 / DIFF_DST_I_B3) * diff_dst_i.s3);
}
#if WITH_SRC_ITER_C
int diff_src_i_c_off(param4 diff_src_i_c, int x0, int x1, int x2, int x3) {
    return ((x0 % DIFF_SRC_I_C_B0) * DIFF_SRC_I_C_SB0
            + (x0 / DIFF_SRC_I_C_B0) * diff_src_i_c.s0
            + (x1 % DIFF_SRC_I_C_B1) * DIFF_SRC_I_C_SB1
            + (x1 / DIFF_SRC_I_C_B1) * diff_src_i_c.s1
            + (x2 % DIFF_SRC_I_C_B2) * DIFF_SRC_I_C_SB2
            + (x2 / DIFF_SRC_I_C_B2) * diff_src_i_c.s2
            + (x3 % DIFF_SRC_I_C_B3) * DIFF_SRC_I_C_SB3
            + (x3 / DIFF_SRC_I_C_B3) * diff_src_i_c.s3);
}
#endif
#if WITH_DST_ITER_C
int diff_dst_i_c_off(param4 diff_dst_i_c, int x0, int x1, int x2, int x3) {
    return ((x0 % DIFF_DST_I_C_B0) * DIFF_DST_I_C_SB0
            + (x0 / DIFF_DST_I_C_B0) * diff_dst_i_c.s0
            + (x1 % DIFF_DST_I_C_B1) * DIFF_DST_I_C_SB1
            + (x1 / DIFF_DST_I_C_B1) * diff_dst_i_c.s1
            + (x2 % DIFF_DST_I_C_B2) * DIFF_DST_I_C_SB2
            + (x2 / DIFF_DST_I_C_B2) * diff_dst_i_c.s2
            + (x3 % DIFF_DST_I_C_B3) * DIFF_DST_I_C_SB3
            + (x3 / DIFF_DST_I_C_B3) * diff_dst_i_c.s3);
}
#endif
int diff_bias_off(param4 diff_bias, int x0, int x1, int x2, int x3) {
    return ((x0 % DIFF_BIAS_B0) * DIFF_BIAS_SB0
            + (x0 / DIFF_BIAS_B0) * diff_bias.s0
            + (x1 % DIFF_BIAS_B1) * DIFF_BIAS_SB1
            + (x1 / DIFF_BIAS_B1) * diff_bias.s1
            + (x2 % DIFF_BIAS_B2) * DIFF_BIAS_SB2
            + (x2 / DIFF_BIAS_B2) * diff_bias.s2
            + (x3 % DIFF_BIAS_B3) * DIFF_BIAS_SB3
            + (x3 / DIFF_BIAS_B3) * diff_bias.s3);
}
#endif // !IS_FWD
#endif
