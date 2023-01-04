/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#include "gpu/ocl/gen9_bnorm.h"

// Two sets of kernels are implemented for BWD bnorm
// 1) The kernels that support both blocked and NHWC layouts (USE_NHWC).
//    These kernels perform IC tail processing for NHWC and for ic % 8 == 0
//    cases only.
// 2) Specially optimized for NHWC kernels (under NHWC_OPTIMIZED).
//    Not supported IC tail processing.
//
// For both algorithms, two types of reduction are implemented:
// 1) Reduction over scratchpad (reduce_temp) with SLM use, implemented by
//    gen9_reduce_xxxxx kernels.
// 2) Atomics-based reduction with SLM use, implemented as part of calc kernels
//    (gem9_calc_xxxxx_fused_reduction() functions). It also requires
//    zeroing and finalization steps, see gen9_fused_reduce_xxxxx kernels.
//    Under FUSED_ATOMICS_REDUCTION definition.

#define LOAD_DATA_Nx16_USING_LOOP(n, dest, src) \
    { \
        for (int k = 0; k < n; ++k) { \
            dest[k] = LOAD_DATA_1x16(&src[k * IC_BLOCK_STRIDE]); \
        } \
    }

#define LOAD_UINT_Nx16_USING_LOOP(n, dest, src) \
    { \
        for (int k = 0; k < n; ++k) { \
            dest[k] = LOAD_UINT_1x16(&src[k * IC_BLOCK_STRIDE]); \
        } \
    }

#define LOAD_CHAR_Nx16_USING_LOOP(n, dest, src) \
    { \
        for (int k = 0; k < n; ++k) { \
            dest[k] = LOAD_CHAR_1x16(&src[k * IC_BLOCK_STRIDE]); \
        } \
    }

#define LOAD_DATA_8x16_USING_LAYOUT(dest, src) \
    { \
        if (USE_NHWC) { \
            LOAD_DATA_Nx16_USING_LOOP(8, dest, src); \
        } else { \
            dest = LOAD_DATA_8x16(src); \
        } \
    }

#define LOAD_UINT_8x16_USING_LAYOUT(dest, src) \
    { \
        if (USE_NHWC) { \
            LOAD_UINT_Nx16_USING_LOOP(8, dest, src); \
        } else { \
            dest = LOAD_UINT_8x16(src); \
        } \
    }

#define LOAD_CHAR_8x16_USING_LAYOUT(dest, src) \
    { \
        if (USE_NHWC) { \
            LOAD_CHAR_Nx16_USING_LOOP(8, dest, src); \
        } else { \
            dest = LOAD_CHAR_8x16(src); \
        } \
    }

#define LOAD_DATA_Nx16_USING_LOOP_HALF(n, dest, src) \
    { \
        for (int k = 0; k < n; k += 2) { \
            dest[k] = LOAD_DATA_1x16(&src[k * IC_BLOCK_STRIDE]); \
        } \
    }

#if NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calculate_stats(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global char *ws,
        __global float *temp_reduce, volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int ic_block_offset = (c / 16) * IC_BLOCK;
    const int offset = ic_block_offset + sp_block_idx * STAT_SP_BLOCK * IC;

    mean += ic_block_offset;
    src += offset;
    diff_dst += offset;
    ws += offset;

    float v_mean[IC_BLOCK_SGROUPS];
    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        v_mean[sg] = as_float(intel_sub_group_block_read(
                (const __global uint *)(&mean[(sg * 16)])));
    }

    VECT_FLOAT_T diff_gamma[IC_BLOCK_SGROUPS / VECT_SIZE] = {0.0f};
    VECT_FLOAT_T diff_beta[IC_BLOCK_SGROUPS / VECT_SIZE] = {0.0f};
#if HAS_IC_VECT_TAIL
    float diff_gamma_tail[IC_TAIL_SGROUPS] = {0.0f};
    float diff_beta_tail[IC_TAIL_SGROUPS] = {0.0f};
#else
    float *diff_gamma_tail = NULL;
    float *diff_beta_tail = NULL;
#endif

#if HAS_STAT_SP_BLOCK_TAIL
    for (int sp = 0; sp < min(STAT_SP_BLOCK, SP - sp_block_idx * STAT_SP_BLOCK);
            ++sp) {
#else
    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
#endif
        for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
            const int sg_idx = sg * 16 * VECT_SIZE;
#if FUSE_BN_RELU
            VECT_CHAR_T ws_vect = LOAD_VECT_CHAR(&ws[sg_idx]);
#endif
            VECT_FLOAT_T src_vect = LOAD_VECT_DATA(&src[sg_idx]);
            VECT_FLOAT_T dd_vect = LOAD_VECT_DATA(&diff_dst[sg_idx]);
            VECT_FLOAT_T v0;
            for (int vect = 0; vect < VECT_SIZE; ++vect) {
                int sg_idx = sg * VECT_SIZE + vect;
#if VECT_SIZE > 1
                v0[vect] = src_vect[vect] - v_mean[sg_idx];
#else
                v0 = src_vect - v_mean[sg_idx];
#endif
            }

#if FUSE_BN_RELU
            dd_vect = select(
                    (VECT_FLOAT_T)0.0f, dd_vect, CONVERT_VECT_INT_T(ws_vect));
#endif

            diff_gamma[sg] = fma(v0, dd_vect, diff_gamma[sg]);
            diff_beta[sg] += dd_vect;
        }
#if HAS_IC_VECT_TAIL
        for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
            const int sg_idx = IC_VECT_SGROUPS + sg;
#if FUSE_BN_RELU
            char ws_tail = LOAD_CHAR_1x16(&ws[sg_idx * 16]);
#endif
            float src_tail = LOAD_DATA_1x16(&src[sg_idx * 16]);
            float dd_tail = LOAD_DATA_1x16(&diff_dst[sg_idx * 16]);
            float v0 = src_tail - v_mean[sg_idx];
#if FUSE_BN_RELU
            dd_tail = select(0.0f, dd_tail, convert_int(ws_tail));
#endif
            diff_gamma_tail[sg] = fma(v0, dd_tail, diff_gamma_tail[sg]);
            diff_beta_tail[sg] += dd_tail;
        }
#endif
        src += IC;
        diff_dst += IC;
#if FUSE_BN_RELU
        ws += IC;
#endif
    }

#if FUSED_ATOMICS_REDUCTION
    __local float local_gamma[2 * CALC_SLM_SIZE];
    __local float *local_beta = local_gamma + CALC_SLM_SIZE;
    gen9_calc_fused_reduction(diff_scale, diff_shift, ic_block_offset,
            diff_gamma, diff_beta, diff_gamma_tail, diff_beta_tail, local_gamma,
            local_beta);
#else
    // scratchpad layout:
    // IC16 - diff_gamma reduction, wrote by gen9_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * IC16 - diff_gamma stats calculated by this kernel
    // IC16 - diff_beta reduction, wrote by gen9_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * IC16 - diff_beta stats calculated by this kernel

    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        const int reduce_off = sp_block_idx * 16
                + REDUCE_STAT_NBLOCKS * 16
                        * (sg + (int)(c / 16) * (IC_BLOCK / 16));

        const int diff_gamma_offset = IC16 + reduce_off;
        const int diff_beta_offset
                = 2 * IC16 + REDUCE_STAT_NBLOCKS * IC16 + reduce_off;

#if HAS_IC_VECT_TAIL
        if (sg >= IC_VECT_SGROUPS) {
            STORE_FLOAT_1x16(&temp_reduce[diff_gamma_offset],
                    diff_gamma_tail[sg - IC_VECT_SGROUPS]);
            STORE_FLOAT_1x16(&temp_reduce[diff_beta_offset],
                    diff_beta_tail[sg - IC_VECT_SGROUPS]);
        } else
#endif
        {
#if VECT_SIZE > 1
            STORE_FLOAT_1x16(&temp_reduce[diff_gamma_offset],
                    diff_gamma[sg / VECT_SIZE][sg % VECT_SIZE]);
            STORE_FLOAT_1x16(&temp_reduce[diff_beta_offset],
                    diff_beta[sg / VECT_SIZE][sg % VECT_SIZE]);
#else
            STORE_FLOAT_1x16(&temp_reduce[diff_gamma_offset], diff_gamma[sg]);
            STORE_FLOAT_1x16(&temp_reduce[diff_beta_offset], diff_beta[sg]);
#endif
        }
    }
#endif // FUSED_ATOMICS_REDUCTION
}

#else // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calculate_stats(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global char *ws,
        __global float *temp_reduce, volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int mb_sp_idx = mb * STAT_SP_NBLOCKS + sp_block_idx;
    const int group_c_offset = REDUCE_STAT_NBLOCKS * 16 * (int)(c / 16);
    const int simd_id = get_sub_group_local_id();
#if HAS_IC_TAIL
    const bool is_last_ic_block = c + 16 > IC;
    const bool is_last_sp_block = (sp_block_idx == STAT_SP_NBLOCKS - 1);
#endif

    temp_reduce += group_c_offset;

#if USE_NHWC
    const int offset = c + sp_block_idx * STAT_SP_BLOCK * IC;
#else
    const int offset = (c & 15) + sp_block_idx * STAT_SP_BLOCK * 16
            + (c & ~15) * SP + mb * SP * IC;
#endif
    src += offset;
    diff_dst += offset;
    ws += offset;

    float v_mean = MAYBE_LAST_IC_LOAD_FLOAT_1x16(mean, c);

    float8 diff_gamma = 0.0f;
    float8 diff_beta = 0.0f;

#if HAS_STAT_SP_TAIL
    int sp;
    if (sp_block_idx == STAT_SP_TAIL) {
        sp = SP - STAT_SP_TAIL * STAT_SP_BLOCK;
    } else {
        sp = STAT_SP_BLOCK;
    }
#else
    int sp = STAT_SP_BLOCK;
#endif

    const int C_PARALLEL_FACTOR = 8;

    for (; sp > C_PARALLEL_FACTOR - 1; sp -= C_PARALLEL_FACTOR) {
        float8 src_data;
        float8 dd_data;

#if FUSE_BN_RELU == 1
        char8 ws_data;
        LOAD_CHAR_8x16_USING_LAYOUT(ws_data, ws);
#endif // #if FUSE_BN_RELU == 1

#if IS_IC_EQ_8
        LOAD_DATA_Nx16_USING_LOOP_HALF(8, src_data, src);
        LOAD_DATA_Nx16_USING_LOOP_HALF(8, dd_data, diff_dst);
        float8 t_src = intel_sub_group_shuffle_down(src_data, src_data, 8);
        float8 t_dd = intel_sub_group_shuffle_down(dd_data, dd_data, 8);
        for (int k = 0; k < 7; k += 2) {
            dd_data[k + 1] = t_dd[k];
            src_data[k + 1] = t_src[k];
        }
#elif HAS_IC_TAIL
        const bool is_last_sp = sp - C_PARALLEL_FACTOR <= C_PARALLEL_FACTOR - 1;
        if (is_last_sp && is_last_ic_block && is_last_sp_block) {
            LOAD_DATA_Nx16_USING_LOOP(7, src_data, src);
            LOAD_DATA_Nx16_USING_LOOP(7, dd_data, diff_dst);
            dd_data[7] = simd_id < 8
                    ? CONVERT_FLOAT_T(diff_dst[7 * IC_BLOCK_STRIDE + simd_id])
                    : 0.0f;
            src_data[7] = simd_id < 8
                    ? CONVERT_FLOAT_T(src[7 * IC_BLOCK_STRIDE + simd_id])
                    : 0.0f;
        } else {
            LOAD_DATA_Nx16_USING_LOOP(8, src_data, src);
            LOAD_DATA_Nx16_USING_LOOP(8, dd_data, diff_dst);
        }
#else
        LOAD_DATA_8x16_USING_LAYOUT(src_data, src);
        LOAD_DATA_8x16_USING_LAYOUT(dd_data, diff_dst);
#endif

        src += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
        diff_dst += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#if FUSE_BN_RELU == 1
        ws += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#endif // #if FUSE_BN_RELU == 1

#if FUSE_BN_RELU == 1
        const float8 C_ZERO = 0.0;
        dd_data = select(C_ZERO, dd_data, convert_int8(ws_data));
#endif // #if FUSE_BN_RELU == 1

        const float8 v0 = src_data - v_mean;
        diff_gamma = fma(v0, dd_data, diff_gamma);
        diff_beta += dd_data;
    }

#if HAS_STAT_SP_TAIL
    if (sp_block_idx == STAT_SP_TAIL) {
        sp = (SP - STAT_SP_TAIL * STAT_SP_BLOCK) % C_PARALLEL_FACTOR;
        while (sp-- >= 1) {
#if FUSE_BN_RELU == 1
            const char ws_data = LOAD_CHAR_1x16(&ws[0]);
#else
            const char ws_data = 1;
#endif // #if FUSE_BN_RELU == 1

#if HAS_IC_TAIL
            float src_data, dd_data;
            if (sp == 0 && is_last_ic_block) {
                src_data = simd_id < 8 ? CONVERT_FLOAT_T(src[simd_id]) : 0.0f;
                dd_data = simd_id < 8 ? CONVERT_FLOAT_T(diff_dst[simd_id])
                                      : 0.0f;
            } else {
                src_data = LOAD_DATA_1x16(&src[0]);
                dd_data = LOAD_DATA_1x16(&diff_dst[0]);
            }
#else
            const float src_data = LOAD_DATA_1x16(&src[0]);
            const float dd_data = LOAD_DATA_1x16(&diff_dst[0]);
#endif

            src += IC_BLOCK_STRIDE;
            diff_dst += IC_BLOCK_STRIDE;
#if FUSE_BN_RELU == 1
            ws += IC_BLOCK_STRIDE;
#endif // #if FUSE_BN_RELU == 1

            if (ws_data != 0) {
                const float v0 = src_data - v_mean;
                const float diff_gamma_tmp = fma(v0, dd_data, diff_gamma[0]);
                diff_gamma[0] = diff_gamma_tmp;
                diff_beta[0] += dd_data;
            }
        }
    }
#endif // #if HAS_STAT_SP_TAIL

    for (int i = 1; i < 8; i++) {
        diff_gamma[0] += diff_gamma[i];
        diff_beta[0] += diff_beta[i];
    }

#if FUSED_ATOMICS_REDUCTION
    __local float local_gamma[2 * CALC_SLM_SIZE];
    __local float *local_beta = local_gamma + CALC_SLM_SIZE;
    gen9_calc_fused_reduction(diff_scale, diff_shift, c, &diff_gamma,
            &diff_beta, NULL, NULL, local_gamma, local_beta);
#else
    // scratchpad layout:
    // IC16 - diff_gamma reduction, wrote by gen9_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * IC16 - diff_gamma stats calculated by this kernel
    // IC16 - diff_beta reduction, wrote by gen9_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * IC16 - diff_beta stats calculated by this kernel
    STORE_FLOAT_1x16(&temp_reduce[IC16 + mb_sp_idx * 16], diff_gamma[0]);
    STORE_FLOAT_1x16(&temp_reduce[2 * IC16 + REDUCE_STAT_NBLOCKS * IC16
                             + mb_sp_idx * 16],
            diff_beta[0]);
#endif // FUSED_ATOMICS_REDUCTION
}

#endif // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_stats(__global float *temp_reduce,
        __global float *diff_scale, __global float *diff_shift,
        __global float *variance, float eps) {

    __local float local_gamma[16 * REDUCE_IC_SUB_GROUPS];
    __local float local_beta[16 * REDUCE_IC_SUB_GROUPS];
    const int ic_sub_group = get_global_id(0) / 16;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * 16 + simd_id;
    float diff_gamma = 0.0f;
    float diff_beta = 0.0f;

    temp_reduce += IC16 + REDUCE_STAT_NBLOCKS * 16 * group_c
            + REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS * 16 * ic_sub_group
            + simd_id;
    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        diff_gamma += temp_reduce[i * 16];
    }
    temp_reduce += IC16 + IC16 * REDUCE_STAT_NBLOCKS;
    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        diff_beta += temp_reduce[i * 16];
    }

    if (ic_sub_group > 0) {
        local_gamma[ic_sub_group * 16 + simd_id] = diff_gamma;
        local_beta[ic_sub_group * 16 + simd_id] = diff_beta;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ic_sub_group == 0) {
        for (int i = 1; i < REDUCE_IC_SUB_GROUPS; i++) {
            diff_gamma += local_gamma[i * 16 + simd_id];
            diff_beta += local_beta[i * 16 + simd_id];
        }

        float sqrt_variance = 1.0f / sqrt(variance[c] + eps);

#if HAS_IC_TAIL
        const bool is_last_ic_block = group_c * 16 + 16 > IC;
        if (!is_last_ic_block || (is_last_ic_block && simd_id < 8))
#endif
        {
            diff_scale[c] = diff_gamma * sqrt_variance;
#if DIFF_SHIFT == 1
            diff_shift[c] = diff_beta;
#else
            diff_shift[IC + IC * REDUCE_STAT_NBLOCKS + c] = diff_beta;
#endif // #if DIFF_SHIFT == 1
        }
    }
}

#if NHWC_OPTIMIZED

KERNEL_ATTR
__kernel void gen9_bnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scaleshift, __global char *ws,
        __global DATA_T *diff_src, __global float *diff_scale,
        __global float *diff_shift, float eps, __global DATA_T *diff_src_add) {

    const int c = GWS_GET_IC();
    const int ic_block_offset = (c / 16) * IC_BLOCK;

    variance += ic_block_offset;
    mean += ic_block_offset;
    diff_scale += ic_block_offset;
    diff_shift += ic_block_offset;
    scaleshift += ic_block_offset;

    VECT_FLOAT_T v_variance[IC_BLOCK_SGROUPS / VECT_SIZE],
            v_mean[IC_BLOCK_SGROUPS / VECT_SIZE],
            diff_gamma[IC_BLOCK_SGROUPS / VECT_SIZE],
            diff_beta[IC_BLOCK_SGROUPS / VECT_SIZE],
            sqrt_variance[IC_BLOCK_SGROUPS / VECT_SIZE],
            gamma[IC_BLOCK_SGROUPS / VECT_SIZE];
    for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
        const int sg_idx = sg * 16 * VECT_SIZE;
        v_variance[sg] = LOAD_VECT_FLOAT(&variance[sg_idx]);
#if CALCULATE_DIFF_STATS == 1
        v_mean[sg] = LOAD_VECT_FLOAT(&mean[sg_idx]);
        diff_gamma[sg] = LOAD_VECT_FLOAT(&diff_scale[sg_idx]);
#if DIFF_SHIFT == 1
        diff_beta[sg] = LOAD_VECT_FLOAT(&diff_shift[sg_idx]);
#else
        diff_beta[sg] = LOAD_VECT_FLOAT(
                &diff_shift[IC + REDUCE_STAT_NBLOCKS * IC + sg_idx]);
#endif // #if DIFF_SHIFT == 1
#endif // #if CALCULATE_DIFF_STATS == 1
#if USE_SCALE == 1
        gamma[sg] = LOAD_VECT_FLOAT(&scaleshift[sg_idx]);
#else
        gamma[sg] = (VECT_FLOAT_T)1.0f;
#endif
        sqrt_variance[sg]
                = (VECT_FLOAT_T)1.0f / sqrt(v_variance[sg] + (VECT_FLOAT_T)eps);
    }
#if HAS_IC_VECT_TAIL
    float v_variance_tail[IC_TAIL_SGROUPS], v_mean_tail[IC_TAIL_SGROUPS],
            diff_gamma_tail[IC_TAIL_SGROUPS], diff_beta_tail[IC_TAIL_SGROUPS],
            sqrt_variance_tail[IC_TAIL_SGROUPS], gamma_tail[IC_TAIL_SGROUPS];
    for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
        const int sg_idx = (IC_VECT_SGROUPS + sg) * 16;
        v_variance_tail[sg] = LOAD_FLOAT_1x16(&variance[sg_idx]);

#if CALCULATE_DIFF_STATS == 1
        v_mean_tail[sg] = LOAD_FLOAT_1x16(&mean[sg_idx]);
        diff_gamma_tail[sg] = LOAD_FLOAT_1x16(&diff_scale[sg_idx]);
#if DIFF_SHIFT == 1
        diff_beta_tail[sg] = LOAD_FLOAT_1x16(&diff_shift[sg_idx]);
#else
        diff_beta_tail[sg] = LOAD_FLOAT_1x16(
                &diff_shift[IC + REDUCE_STAT_NBLOCKS * IC + sg_idx]);
#endif // #if DIFF_SHIFT == 1
#endif // #if CALCULATE_DIFF_STATS == 1
#if USE_SCALE == 1
        gamma_tail[sg] = LOAD_FLOAT_1x16(&scaleshift[sg_idx]);
#else
        gamma_tail[sg] = 1.0f;
#endif
        sqrt_variance_tail[sg] = 1.0f / sqrt(v_variance_tail[sg] + eps);
    }
#endif

    const int sp_block_idx = GWS_GET_SP();
    const int offset = ic_block_offset + sp_block_idx * STAT_SP_BLOCK * IC;

    src += offset;
    diff_dst += offset;
    ws += offset;
    diff_src += offset;
#if FUSE_BN_ADD_RELU
    diff_src_add += offset;
#endif

#if HAS_STAT_SP_BLOCK_TAIL
    for (int sp = 0; sp < min(STAT_SP_BLOCK, SP - sp_block_idx * STAT_SP_BLOCK);
            ++sp) {
#else
    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
#endif
        for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
            const int sg_idx = sg * 16 * VECT_SIZE;
            VECT_FLOAT_T src_vect = LOAD_VECT_DATA(&src[sg_idx]);
            VECT_FLOAT_T dd_vect = LOAD_VECT_DATA(&diff_dst[sg_idx]);

#if FUSE_BN_RELU
            VECT_CHAR_T ws_vect = LOAD_VECT_CHAR(&ws[sg_idx]);
            dd_vect = select(
                    (VECT_FLOAT_T)0.0f, dd_vect, CONVERT_VECT_INT_T(ws_vect));
#if FUSE_BN_ADD_RELU
            STORE_VECT_DATA(&diff_src_add[sg_idx], dd_vect);
#endif
#endif

#if CALCULATE_DIFF_STATS == 1
            dd_vect -= (diff_beta[sg]
                               + (src_vect - v_mean[sg]) * diff_gamma[sg]
                                       * sqrt_variance[sg])
                    / (MB * ID * IH * IW);
#endif // #if CALCULATE_DIFF_STATS == 1
            dd_vect *= gamma[sg] * sqrt_variance[sg];
            STORE_VECT_DATA(&diff_src[sg_idx], dd_vect);
        }
#if HAS_IC_VECT_TAIL
        for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
            const int sg_idx = (IC_VECT_SGROUPS + sg) * 16;
            float src_tail = LOAD_DATA_1x16(&src[sg_idx]);
            float dd_tail = LOAD_DATA_1x16(&diff_dst[sg_idx]);
#if FUSE_BN_RELU
            char ws_tail = LOAD_CHAR_1x16(&ws[sg_idx]);
            dd_tail = select(0.0f, dd_tail, convert_int(ws_tail));
#if FUSE_BN_ADD_RELU
            STORE_DATA_1x16(&diff_src_add[sg_idx], dd_tail);
#endif
#endif
#if CALCULATE_DIFF_STATS == 1
            dd_tail -= (diff_beta_tail[sg]
                               + (src_tail - v_mean_tail[sg])
                                       * diff_gamma_tail[sg]
                                       * sqrt_variance_tail[sg])
                    / (MB * ID * IH * IW);
#endif // #if CALCULATE_DIFF_STATS == 1
            dd_tail *= gamma_tail[sg] * sqrt_variance_tail[sg];
            STORE_DATA_1x16(&diff_src[sg_idx], dd_tail);
        }
#endif
        src += IC;
        diff_dst += IC;
        diff_src += IC;
#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
        diff_src_add += IC;
#endif
        ws += IC;
#endif
    }
}

#else // NHWC_OPTIMIZED

inline void write_8x16_block(__global DATA_T *ptr, int c, float8 val) {
#if USE_NHWC
#if HAS_IC_TAIL
    const int simd_id = get_sub_group_local_id();
    const bool is_last_ic_block = c + 16 > IC;
    if (is_last_ic_block) {
        if (simd_id < 8) {
            for (int k = 0; k < 8; ++k)
                ptr[k * IC_BLOCK_STRIDE + simd_id] = CONVERT_DATA_T(val[k]);
        }
    } else
#endif // HAS_IC_TAIL
        for (int k = 0; k < 8; ++k)
            STORE_DATA_1x16(&ptr[k * IC_BLOCK_STRIDE], val[k]);
#else
    STORE_DATA_8x16(&ptr[0], val);
#endif // #if USE_NHWC
}
inline void write_1x16_block(__global DATA_T *ptr, int c, float val) {
#if HAS_IC_TAIL
    const int simd_id = get_sub_group_local_id();
    const bool is_last_ic_block = c + 16 > IC;
    if (!is_last_ic_block) {
        STORE_DATA_1x16(&ptr[0], val);
    } else {
        if (simd_id < 8) { ptr[simd_id] = CONVERT_DATA_T(val); }
    }
#else
    STORE_DATA_1x16(&ptr[0], val);
#endif
}

KERNEL_ATTR
__kernel void gen9_bnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scaleshift, __global char *ws,
        __global DATA_T *diff_src, __global float *diff_scale,
        __global float *diff_shift, float eps, __global DATA_T *diff_src_add) {

    const int c = GWS_GET_IC();
    const int simd_id = get_sub_group_local_id();
#if HAS_IC_TAIL
    const bool is_last_ic_block = c + 16 > IC;
#endif

    const float v_variance = MAYBE_LAST_IC_LOAD_FLOAT_1x16(variance, c);
#if CALCULATE_DIFF_STATS == 1
    const float v_mean = MAYBE_LAST_IC_LOAD_FLOAT_1x16(mean, c);
    const float diff_gamma = MAYBE_LAST_IC_LOAD_FLOAT_1x16(diff_scale, c);
#if DIFF_SHIFT == 1
    const float diff_beta = MAYBE_LAST_IC_LOAD_FLOAT_1x16(diff_shift, c);
#else
    const float diff_beta = MAYBE_LAST_IC_LOAD_FLOAT_1x16(
            diff_shift, IC + REDUCE_STAT_NBLOCKS * IC + c);
#endif // #if DIFF_SHIFT == 1
#endif // #if CALCULATE_DIFF_STATS == 1

#if USE_SCALE == 1
    const float gamma = MAYBE_LAST_IC_LOAD_FLOAT_1x16(scaleshift, c);
#else
    const float gamma = 1;
#endif // #if USE_SCALE == 1

    const int sp_block_idx = GWS_GET_SP();
#if USE_NHWC
    const int offset = c + sp_block_idx * VECT_SIZE * IC;
#else
    const int mb = GWS_GET_MB();
    const int offset = (c & 15) + sp_block_idx * VECT_SIZE * 16 + (c & ~15) * SP
            + mb * SP * IC;
#endif

#if HAS_IC_TAIL
    const bool is_last_sp_block = sp_block_idx == SP / VECT_SIZE - 1;
#endif

    src += offset;
    diff_dst += offset;
    ws += offset;
    diff_src += offset;
#if FUSE_BN_ADD_RELU
    diff_src_add += offset;
#endif

#if HAS_SP_TAIL
    int sp;
    if (sp_block_idx == SP_TAIL / VECT_SIZE) {
        sp = SP - SP_TAIL;
    } else {
        sp = VECT_SIZE;
    }
#else
    int sp = VECT_SIZE;
#endif

    const float sqrt_variance = 1.0f / sqrt(v_variance + eps);

    const int C_PARALLEL_FACTOR = 8;
    for (; sp > C_PARALLEL_FACTOR - 1; sp -= C_PARALLEL_FACTOR) {
        float8 src_data;
        float8 dd_data;

#if FUSE_BN_RELU == 1
        char8 ws_data;
        LOAD_CHAR_8x16_USING_LAYOUT(ws_data, ws);
#endif // #if FUSE_BN_RELU == 1

#if IS_IC_EQ_8
        LOAD_DATA_Nx16_USING_LOOP_HALF(8, src_data, src);
        LOAD_DATA_Nx16_USING_LOOP_HALF(8, dd_data, diff_dst);
        float8 t_dd = intel_sub_group_shuffle_down(dd_data, dd_data, 8);
        float8 t_src = intel_sub_group_shuffle_down(src_data, src_data, 8);
        for (int k = 0; k < 7; k += 2) {
            dd_data[k + 1] = t_dd[k];
            src_data[k + 1] = t_src[k];
        }
#elif HAS_IC_TAIL && !HAS_SP_TAIL
        const bool is_last_sp = sp - C_PARALLEL_FACTOR <= C_PARALLEL_FACTOR - 1;
        if (is_last_sp && is_last_ic_block && is_last_sp_block) {
            LOAD_DATA_Nx16_USING_LOOP(7, src_data, src);
            LOAD_DATA_Nx16_USING_LOOP(7, dd_data, diff_dst);
            dd_data[7] = simd_id < 8
                    ? CONVERT_FLOAT_T(diff_dst[7 * IC_BLOCK_STRIDE + simd_id])
                    : 0.0f;
            src_data[7] = simd_id < 8
                    ? CONVERT_FLOAT_T(src[7 * IC_BLOCK_STRIDE + simd_id])
                    : 0.0f;
        } else {
            LOAD_DATA_Nx16_USING_LOOP(8, src_data, src);
            LOAD_DATA_Nx16_USING_LOOP(8, dd_data, diff_dst);
        }
#else
        LOAD_DATA_8x16_USING_LAYOUT(dd_data, diff_dst);
        LOAD_DATA_8x16_USING_LAYOUT(src_data, src);
#endif // IS_IC_EQ_8

        src += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
        diff_dst += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#if FUSE_BN_RELU == 1
        ws += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#endif // #if FUSE_BN_RELU == 1

#if FUSE_BN_RELU == 1
        const float8 C_ZERO = 0.0;
        dd_data = select(C_ZERO, dd_data, convert_int8(ws_data));
#if FUSE_BN_ADD_RELU
        write_8x16_block(diff_src_add, c, dd_data);
#endif
#endif // #if FUSE_BN_RELU == 1

#if CALCULATE_DIFF_STATS == 1
        dd_data -= (diff_beta
                           + (src_data - v_mean) * diff_gamma * sqrt_variance)
                / (MB * ID * IH * IW);
#endif // #if CALCULATE_DIFF_STATS == 1

        dd_data *= gamma * sqrt_variance;
        write_8x16_block(diff_src, c, dd_data);

        diff_src += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#if FUSE_BN_ADD_RELU
        diff_src_add += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#endif
    }

#if HAS_SP_TAIL
    if (sp_block_idx == SP_TAIL / VECT_SIZE) {
        sp = (SP - SP_TAIL) % C_PARALLEL_FACTOR;
        while (sp-- >= 1) {
#if FUSE_BN_RELU == 1
            const char ws_data = LOAD_CHAR_1x16(&ws[0]);
#endif // #if FUSE_BN_RELU == 1

#if HAS_IC_TAIL
            float dd_data;
            if (sp == 0 && is_last_ic_block)
                dd_data = simd_id < 8 ? CONVERT_FLOAT_T(diff_dst[simd_id])
                                      : 0.0f;
            else
                dd_data = LOAD_DATA_1x16(&diff_dst[0]);
#if CALCULATE_DIFF_STATS == 1
            float src_data;
            if (sp == 0 && is_last_ic_block)
                src_data = simd_id < 8 ? CONVERT_FLOAT_T(src[simd_id]) : 0.0f;
            else
                src_data = LOAD_DATA_1x16(&src[0]);
#endif // #if CALCULATE_DIFF_STATS == 1
#else
            float dd_data = LOAD_DATA_1x16(&diff_dst[0]);
#if CALCULATE_DIFF_STATS == 1
            const float src_data = LOAD_DATA_1x16(&src[0]);
#endif // #if CALCULATE_DIFF_STATS == 1
#endif // HAS_IC_TAIL

            src += IC_BLOCK_STRIDE;
            diff_dst += IC_BLOCK_STRIDE;
#if FUSE_BN_RELU == 1
            ws += IC_BLOCK_STRIDE;
#endif // #if FUSE_BN_RELU == 1

#if FUSE_BN_RELU == 1
            if (ws_data == 0) dd_data = 0;
#if FUSE_BN_ADD_RELU
            write_1x16_block(diff_src_add, c, dd_data);
#endif
#endif // #if FUSE_BN_RELU == 1

#if CALCULATE_DIFF_STATS == 1
            dd_data -= (diff_beta
                               + (src_data - v_mean) * diff_gamma
                                       * sqrt_variance)
                    / (MB * ID * IH * IW);
#endif // #if CALCULATE_DIFF_STATS == 1

            dd_data *= gamma * sqrt_variance;
            write_1x16_block(diff_src, c, dd_data);

            diff_src += IC_BLOCK_STRIDE;
#if FUSE_BN_ADD_RELU
            diff_src_add += IC_BLOCK_STRIDE;
#endif
        }
    }
#endif // #if HAS_SP_TAIL
}
#endif // NHWC_OPTIMIZED
