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
#include "gpu/ocl/bnorm/gen9_bnorm.h"

// FWD kernels for regular and 1-pass (under USE_STATS_ONE_PASS) bnorm
// algorithms that support both blocked and NHWC layouts (USE_NHWC definition).
// These kernels perform IC tail processing for NHWC and for ic % 8 == 0
// cases only.
// For both algorithms, two types of reduction are implemented:
// 1) Reduction over scratchpad (reduce_temp) with SLM use, implemented by
//    gen9_reduce_* kernels.
// 2) Atomics-based reduction with SLM use (FUSED_ATOMICS_REDUCTION definition),
//    implemented as part of calc kernels, see gen9_*_fused_reduction()
//    functions in gen9_bnorm.h. This reduction implementation requires
//    zeroing and finalization steps, see gen9_fused_reduce_* kernels
//    in gen9_bnorm_reduce.cl

#define LOAD_DATA_Nx16_USING_LOOP_IDX(n, dest, src, idx) \
    { \
        for (int k = 0; k < n; ++k) { \
            dest[k] = LOAD_DATA_1x16(&src[(k + idx) * IC]); \
        } \
    }
#define LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(n, dest, src, idx) \
    { \
        for (int k = 0; k < n; k += 2) { \
            dest[k] = LOAD_DATA_1x16(&src[(k + idx) * IC]); \
        } \
    }

#if USE_STATS_ONE_PASS

// IC tail processing is not implemented for one pass algorithm
// Calculates partial sums of values and squares-of-values per channel

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean_var(__global DATA_T *src,
        __global ACCUM_DATA_T *reduce_temp,
        volatile __global atomic_float *mean,
        volatile __global atomic_float *variance) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int mb_sp_idx = mb * STAT_SP_NBLOCKS + sp_block_idx;
    const int group_c_offset = REDUCE_STAT_NBLOCKS * 16 * (int)(c / 16);
    const int simd_id = get_sub_group_local_id();
    const int ver_offs = REDUCE_STAT_NBLOCKS * IC;

#if USE_NHWC
    src += c + sp_block_idx * STAT_SP_BLOCK * IC;
#else
    src += (c & 15) + sp_block_idx * STAT_SP_BLOCK * 16 + (c & ~15) * SP
            + mb * SP * IC;
#endif

    SUM_DATA_T sum;
    SUM_DATA_T sum_sq;
    sum.s0 = 0;
    sum.s1 = 0;
    sum_sq.s0 = 0;
    sum_sq.s1 = 0;

#if HAS_STAT_SP_TAIL
    if (sp_block_idx == STAT_SP_TAIL) {
        int sp = SP - STAT_SP_TAIL * STAT_SP_BLOCK;
        while (sp >= 16) {
#if USE_NHWC
            float8 s0, s1;
            for (int k = 0; k < 8; ++k)
                s0[k] = LOAD_DATA_1x16(&src[k * IC]);
            for (int k = 0; k < 8; ++k)
                s1[k] = LOAD_DATA_1x16(&src[(k + 8) * IC]);
#else
            float8 s0 = LOAD_DATA_8x16(&src[0]);
            float8 s1 = LOAD_DATA_8x16(&src[8 * 16]);
#endif
            for (int i = 0; i < 8; i++) {
                sum = summation(s0[i], sum);
                sum = summation(s1[i], sum);
                sum_sq = summation(s0[i] * s0[i], sum_sq);
                sum_sq = summation(s1[i] * s1[i], sum_sq);
            }

            src += 16 * IC_BLOCK_STRIDE;
            sp -= 16;
        }
        while (sp >= 1) {
            float s0 = LOAD_DATA_1x16(&src[0]);
            sum = summation(s0, sum);
            sum_sq = summation(s0 * s0, sum_sq);
            src += IC_BLOCK_STRIDE;
            --sp;
        }
    } else
#endif
    {
        for (int sp = 0; sp < STAT_SP_BLOCK / 16; ++sp) {
#if USE_NHWC
            float8 s0, s1;
            for (int k = 0; k < 8; ++k)
                s0[k] = LOAD_DATA_1x16(&src[k * IC]);
            for (int k = 0; k < 8; ++k)
                s1[k] = LOAD_DATA_1x16(&src[(k + 8) * IC]);
#else
            float8 s0 = LOAD_DATA_8x16(&src[0]);
            float8 s1 = LOAD_DATA_8x16(&src[8 * 16]);
#endif
            for (int i = 0; i < 8; i++) {
                sum = summation(s0[i], sum);
                sum = summation(s1[i], sum);
                sum_sq = summation(s0[i] * s0[i], sum_sq);
                sum_sq = summation(s1[i] * s1[i], sum_sq);
            }
            src += 16 * IC_BLOCK_STRIDE;
        }
    }

#if FUSED_ATOMICS_REDUCTION
    __local SUM_DATA_T local_sum[2 * CALC_SLM_SIZE];
    __local SUM_DATA_T *local_sum_sq = local_sum + CALC_SLM_SIZE;
    gen9_mean_var_calc_fused_reduction(
            mean, variance, c, &sum, &sum_sq, local_sum, local_sum_sq);
#else
    STORE_FLOAT_1x16(&reduce_temp[group_c_offset + mb_sp_idx * 16], sum.s0);
    STORE_FLOAT_1x16(&reduce_temp[ver_offs + group_c_offset + mb_sp_idx * 16],
            sum_sq.s0);
#endif
}

#else // USE_STATS_ONE_PASS

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean(__global DATA_T *src, __global float *reduce_temp,
        volatile __global atomic_float *mean) {

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

#if USE_NHWC
    src += c + sp_block_idx * STAT_SP_BLOCK * IC;
#else
    src += (c & 15) + sp_block_idx * STAT_SP_BLOCK * 16 + (c & ~15) * SP
            + mb * SP * IC;
#endif

    float8 res0 = 0.0f, res1 = 0.0f;
    float v_mean = 0.0f;

#if HAS_STAT_SP_TAIL
    if (sp_block_idx == STAT_SP_TAIL) {
        int sp = SP - STAT_SP_TAIL * STAT_SP_BLOCK;
        while (sp >= 16) {
#if USE_NHWC
            float8 s0, s1;
#if IS_IC_EQ_8
            LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(8, s0, src, 0);
            LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(8, s1, src, 8);
            float8 t0 = intel_sub_group_shuffle_down(s0, s0, 8);
            float8 t1 = intel_sub_group_shuffle_down(s1, s1, 8);
            for (int k = 0; k < 7; k += 2) {
                s0[k + 1] = t0[k];
                s1[k + 1] = t1[k];
            }
#elif HAS_IC_TAIL
            const bool is_last_sp = sp == 16;
            if (is_last_sp && is_last_ic_block) {
                LOAD_DATA_Nx16_USING_LOOP_IDX(7, s0, src, 0);
                s0[7] = simd_id < 8 ? CONVERT_FLOAT_T(src[7 * IC + simd_id])
                                    : 0.0f;
                LOAD_DATA_Nx16_USING_LOOP_IDX(7, s1, src, 8);
                s1[7] = simd_id < 8 ? CONVERT_FLOAT_T(src[15 * IC + simd_id])
                                    : 0.0f;
            } else {
                LOAD_DATA_Nx16_USING_LOOP_IDX(8, s0, src, 0);
                LOAD_DATA_Nx16_USING_LOOP_IDX(8, s1, src, 8);
            }
#else
            LOAD_DATA_Nx16_USING_LOOP_IDX(8, s0, src, 0);
            LOAD_DATA_Nx16_USING_LOOP_IDX(8, s1, src, 8);
#endif // IS_IC_EQ_8
#else
            float8 s0 = LOAD_DATA_8x16(&src[0]);
            float8 s1 = LOAD_DATA_8x16(&src[8 * 16]);
#endif // USE_NHWC
            res0 += s0;
            res1 += s1;

            src += 16 * IC_BLOCK_STRIDE;
            sp -= 16;
        }
        while (sp >= 1) {
#if HAS_IC_TAIL
            float s0;
            if (sp == 1 && is_last_ic_block)
                s0 = simd_id < 8 ? CONVERT_FLOAT_T(src[simd_id]) : 0.0f;
            else
                s0 = LOAD_DATA_1x16(&src[0]);
#else
            float s0 = LOAD_DATA_1x16(&src[0]);
#endif
            v_mean += s0;
            src += IC_BLOCK_STRIDE;
            --sp;
        }
    } else
#endif // HAS_STAT_SP_TAIL
    {
        for (int sp = 0; sp < STAT_SP_BLOCK / 16; ++sp) {
#if USE_NHWC
            float8 s0, s1;
#if IS_IC_EQ_8
            LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(8, s0, src, 0);
            LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(8, s1, src, 8);
            float8 t0 = intel_sub_group_shuffle_down(s0, s0, 8);
            float8 t1 = intel_sub_group_shuffle_down(s1, s1, 8);
            for (int k = 0; k < 7; k += 2) {
                s0[k + 1] = t0[k];
                s1[k + 1] = t1[k];
            }
#elif HAS_IC_TAIL
            const bool is_last_sp = sp == STAT_SP_BLOCK / 16 - 1;
            if (is_last_sp && is_last_ic_block && is_last_sp_block) {
                LOAD_DATA_Nx16_USING_LOOP_IDX(7, s0, src, 0);
                s0[7] = simd_id < 8 ? CONVERT_FLOAT_T(src[7 * IC + simd_id])
                                    : 0.0f;
                LOAD_DATA_Nx16_USING_LOOP_IDX(7, s1, src, 8);
                s1[7] = simd_id < 8 ? CONVERT_FLOAT_T(src[15 * IC + simd_id])
                                    : 0.0f;
            } else {
                LOAD_DATA_Nx16_USING_LOOP_IDX(8, s0, src, 0);
                LOAD_DATA_Nx16_USING_LOOP_IDX(8, s1, src, 8);
            }
#else
            LOAD_DATA_Nx16_USING_LOOP_IDX(8, s0, src, 0);
            LOAD_DATA_Nx16_USING_LOOP_IDX(8, s1, src, 8);
#endif // IS_IC_EQ_8
#else
            float8 s0 = LOAD_DATA_8x16(&src[0]);
            float8 s1 = LOAD_DATA_8x16(&src[8 * 16]);
#endif // NHWC
            res0 += s0;
            res1 += s1;
            src += 16 * IC_BLOCK_STRIDE;
        }
    }

    for (int i = 0; i < 8; i++) {
        v_mean += res0[i] + res1[i];
    }

#if FUSED_ATOMICS_REDUCTION
    __local float local_sum[CALC_SLM_SIZE];
    gen9_calc_fused_reduction(mean, c, &v_mean, local_sum);
#else
    // reduce_temp is padded to PADDED_IC, no OOB writes
    STORE_FLOAT_1x16(&reduce_temp[group_c_offset + mb_sp_idx * 16], v_mean);
#endif
}

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_variance(__global DATA_T *src, __global float *mean,
        __global float *reduce_temp, __global atomic_float *variance) {

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
    reduce_temp += REDUCE_STAT_NBLOCKS * PADDED_IC;

#if USE_NHWC
    src += c + sp_block_idx * STAT_SP_BLOCK * IC;
#else
    src += (c & 15) + sp_block_idx * STAT_SP_BLOCK * 16 + (c & ~15) * SP
            + mb * SP * IC;
#endif

    float8 res0 = 0.0f, res1 = 0.0f;
    float v_var = 0.0f;

    float v_mean = MAYBE_LAST_IC_LOAD_FLOAT_1x16(mean, c);

#if HAS_STAT_SP_TAIL
    if (sp_block_idx == STAT_SP_TAIL) {
        int sp = SP - STAT_SP_TAIL * STAT_SP_BLOCK;
        while (sp >= 16) {
#if USE_NHWC
            float8 s0, s1;
#if IS_IC_EQ_8
            LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(8, s0, src, 0);
            LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(8, s1, src, 8);
            float8 t0 = intel_sub_group_shuffle_down(s0, s0, 8);
            float8 t1 = intel_sub_group_shuffle_down(s1, s1, 8);
            for (int k = 0; k < 7; k += 2) {
                s0[k + 1] = t0[k];
                s1[k + 1] = t1[k];
            }
#elif HAS_IC_TAIL
            const bool is_last_sp = sp == 16;
            if (is_last_sp && is_last_ic_block) {
                LOAD_DATA_Nx16_USING_LOOP_IDX(7, s0, src, 0);
                s0[7] = simd_id < 8 ? CONVERT_FLOAT_T(src[7 * IC + simd_id])
                                    : 0.0f;
                LOAD_DATA_Nx16_USING_LOOP_IDX(7, s1, src, 8);
                s1[7] = simd_id < 8 ? CONVERT_FLOAT_T(src[15 * IC + simd_id])
                                    : 0.0f;
            } else {
                LOAD_DATA_Nx16_USING_LOOP_IDX(8, s0, src, 0);
                LOAD_DATA_Nx16_USING_LOOP_IDX(8, s1, src, 8);
            }
#else
            LOAD_DATA_Nx16_USING_LOOP_IDX(8, s0, src, 0);
            LOAD_DATA_Nx16_USING_LOOP_IDX(8, s1, src, 8);
#endif
#else // USE_NHWC
            float8 s0 = LOAD_DATA_8x16(&src[0]);
            float8 s1 = LOAD_DATA_8x16(&src[8 * 16]);
#endif

            float8 v0 = s0 - v_mean;
            float8 v1 = s1 - v_mean;
            res0 = fma(v0, v0, res0);
            res1 = fma(v1, v1, res1);

            src += 16 * IC_BLOCK_STRIDE;
            sp -= 16;
        }

        while (sp >= 1) {
#if HAS_IC_TAIL
            float s0;
            if (sp == 1 && is_last_ic_block)
                s0 = simd_id < 8 ? CONVERT_FLOAT_T(src[simd_id]) : 0.0f;
            else
                s0 = LOAD_DATA_1x16(&src[0]);
#else
            float s0 = LOAD_DATA_1x16(&src[0]);
#endif
            float v0 = s0 - v_mean;
            v_var = fma(v0, v0, v_var);

            src += IC_BLOCK_STRIDE;
            --sp;
        }
    } else
#endif // HAS_STAT_SP_TAIL
    {
        for (int sp = 0; sp < STAT_SP_BLOCK / 16; ++sp) {
#if USE_NHWC
            float8 s0, s1;
#if IS_IC_EQ_8
            LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(8, s0, src, 0);
            LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(8, s1, src, 8);
            float8 t0 = intel_sub_group_shuffle_down(s0, s0, 8);
            float8 t1 = intel_sub_group_shuffle_down(s1, s1, 8);
            for (int k = 0; k < 7; k += 2) {
                s0[k + 1] = t0[k];
                s1[k + 1] = t1[k];
            }
#elif HAS_IC_TAIL
            const bool is_last_sp = sp == STAT_SP_BLOCK / 16 - 1;
            if (is_last_sp && is_last_ic_block && is_last_sp_block) {
                LOAD_DATA_Nx16_USING_LOOP_IDX(7, s0, src, 0);
                s0[7] = simd_id < 8 ? CONVERT_FLOAT_T(src[7 * IC + simd_id])
                                    : 0.0f;
                LOAD_DATA_Nx16_USING_LOOP_IDX(7, s1, src, 8);
                s1[7] = simd_id < 8 ? CONVERT_FLOAT_T(src[15 * IC + simd_id])
                                    : 0.0f;
            } else {
                LOAD_DATA_Nx16_USING_LOOP_IDX(8, s0, src, 0);
                LOAD_DATA_Nx16_USING_LOOP_IDX(8, s1, src, 8);
            }
#else
            LOAD_DATA_Nx16_USING_LOOP_IDX(8, s0, src, 0);
            LOAD_DATA_Nx16_USING_LOOP_IDX(8, s1, src, 8);
#endif // IS == 8
#else
            float8 s0 = LOAD_DATA_8x16(&src[0]);
            float8 s1 = LOAD_DATA_8x16(&src[8 * 16]);
#endif // USE_NHWC
            float8 v0 = s0 - v_mean;
            float8 v1 = s1 - v_mean;
            res0 = fma(v0, v0, res0);
            res1 = fma(v1, v1, res1);

            src += 16 * IC_BLOCK_STRIDE;
        }
    }

    for (int i = 0; i < 8; i++) {
        v_var += res0[i] + res1[i];
    }

#if FUSED_ATOMICS_REDUCTION
    __local float local_sum[CALC_SLM_SIZE];
    gen9_calc_fused_reduction(variance, c, &v_var, local_sum);
#else
    // reduce_temp is padded to PADDED_IC, no OOB writes
    STORE_FLOAT_1x16(&reduce_temp[group_c_offset + mb_sp_idx * 16], v_var);
#endif
}

#endif // USE_STATS_ONE_PASS

inline float8 read_src_block(__global DATA_T *src, int c, int sp) {
    float8 blockS0 = 0.0f;
    const int simd_id = get_sub_group_local_id();
#if HAS_IC_TAIL
    const bool is_last_ic_block = c + 16 > IC;
    const bool is_last_sp_block = sp >= SP - VECT_SIZE;
#endif
#if HAS_SP_TAIL
    if (sp == SP_TAIL) {
        for (int k = 0; k < SP - SP_TAIL; ++k)
#if HAS_IC_TAIL
            if (k == SP - SP_TAIL - 1 && is_last_ic_block)
                blockS0[k] = simd_id < 8
                        ? CONVERT_FLOAT_T(src[k * IC_BLOCK_STRIDE + simd_id])
                        : 0.0f;
            else
#endif
                blockS0[k] = LOAD_DATA_1x16(&src[k * IC_BLOCK_STRIDE]);
    } else
#endif // HAS_SP_TAIL
    {
#if USE_NHWC
#if IS_IC_EQ_8
        LOAD_DATA_Nx16_USING_LOOP_IDX_HALF(8, blockS0, src, 0);
        float8 t0 = intel_sub_group_shuffle_down(blockS0, blockS0, 8);
        for (int k = 0; k < 7; k += 2)
            blockS0[k + 1] = t0[k];
#elif HAS_IC_TAIL
        if (is_last_ic_block && is_last_sp_block) {
            LOAD_DATA_Nx16_USING_LOOP_IDX(7, blockS0, src, 0);
            blockS0[7] = simd_id < 8 ? CONVERT_FLOAT_T(src[7 * IC + simd_id])
                                     : 0.0f;
        } else {
            LOAD_DATA_Nx16_USING_LOOP_IDX(8, blockS0, src, 0);
        }
#else
        LOAD_DATA_Nx16_USING_LOOP_IDX(8, blockS0, src, 0);
#endif // IS_IC_EQ_8
#else
        blockS0 = LOAD_DATA_8x16(&src[0]);
#endif // USE_NHWC
    }
    return blockS0;
}

KERNEL_ATTR
__kernel void gen9_bnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst,
        __global float *scaleshift, __global float *shift, __global char *ws,
        float eps, __global DATA_T *src_add, float relu_alpha) {

    const int n = GWS_GET_MB();
    const int c = GWS_GET_IC();
    const int sp = GWS_GET_SP() * VECT_SIZE;

    const int simd_id = get_sub_group_local_id();
#if HAS_IC_TAIL
    const bool is_last_ic_block = c + 16 > IC;
    const bool is_last_sp_block = sp >= SP - VECT_SIZE;
#endif

#if USE_NHWC
    const uint d_off = sp * IC + c;
#else
    const uint d_off = (c & 15) + sp * 16 + (c & ~15) * SP + n * SP * IC;
#endif

    src += d_off;
    dst += d_off;

    float8 blockS0 = read_src_block(src, c, sp);
#if FUSE_BN_ADD_RELU
    src_add += d_off;
    float8 block_S0_Add = read_src_block(src_add, c, sp);
#endif

    float8 blockD0;

#if USE_SCALE == 1
    float sm = MAYBE_LAST_IC_LOAD_FLOAT_1x16(scaleshift, c);
#else
    float sm = 1.0f;
#endif
#if USE_SHIFT == 1
    float sv = MAYBE_LAST_IC_LOAD_FLOAT_1x16(shift, c);
#else
    float sv = 0.0f;
#endif

    float v_mean, v_variance;
#if HAS_IC_TAIL
    if (is_last_ic_block) {
        v_mean = simd_id < 8 ? mean[c + simd_id] : 0.0f;
        v_variance = simd_id < 8 ? variance[c + simd_id] : 0.0f;
    } else
#endif
    {
        v_mean = LOAD_FLOAT_1x16(&mean[c]);
        v_variance = LOAD_FLOAT_1x16(&variance[c]);
    }

    float sqrt_variance = sm / sqrt(v_variance + eps);

    blockD0 = fma(blockS0 - (float8)v_mean, (float8)sqrt_variance, (float8)sv);

#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
    blockD0 += block_S0_Add;
#endif
    int8 blockWS0 = isgreater(blockD0, (float8)0.0f);
    blockD0 = select((float8)0.0f, blockD0, blockWS0);
#if IS_TRAINING
    ws += d_off;
#if HAS_SP_TAIL
    if (sp == SP_TAIL) {
        for (int k = 0; k < SP - SP_TAIL; ++k) {
            STORE_CHAR_1x16(
                    &ws[k * IC_BLOCK_STRIDE], convert_char(blockWS0[k]));
        }
    } else
#endif // HAS_SP_TAIL
    {
#if USE_NHWC
        for (int k = 0; k < 8; ++k)
            STORE_CHAR_1x16(
                    &ws[k * IC_BLOCK_STRIDE], convert_char(blockWS0[k]));
#else
        STORE_CHAR_8x16(&ws[0], convert_char8(blockWS0));
#endif
    }
#endif // IS_TRAINING
#endif // FUSE_BN_RELU

#if WITH_RELU
#if WITH_LEAKY_RELU
    int8 blockWS0_2 = isless(blockD0, (float8)0.0f);
    blockD0 = select(blockD0, blockD0 * (float8)relu_alpha, blockWS0_2);
#else
    blockD0 = max(blockD0, (VECT_FLOAT_T)0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU

#if HAS_SP_TAIL
    if (sp == SP_TAIL) {
        for (int k = 0; k < SP - SP_TAIL; ++k) {
#if HAS_IC_TAIL
            if (is_last_ic_block) {
                if (simd_id < 8)
                    dst[k * IC_BLOCK_STRIDE + simd_id]
                            = CONVERT_DATA_T(blockD0[k]);
            } else
#endif
                STORE_DATA_1x16(&dst[k * IC_BLOCK_STRIDE], blockD0[k]);
        }
    } else
#endif // HAS_SP_TAIL
    {
#if USE_NHWC
        for (int k = 0; k < 8; ++k)
#if HAS_IC_TAIL
            if (is_last_ic_block) {
                if (simd_id < 8)
                    dst[k * IC_BLOCK_STRIDE + simd_id]
                            = CONVERT_DATA_T(blockD0[k]);
            } else
#endif
                STORE_DATA_1x16(&dst[k * IC_BLOCK_STRIDE], blockD0[k]);
#else
        STORE_DATA_8x16(&dst[0], blockD0);
#endif // USE_NHWC
    }
}
