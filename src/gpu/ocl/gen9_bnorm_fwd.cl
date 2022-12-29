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

// For regular and 1-pass (under USE_STATS_ONE_PASS) bnorm algorithms
// there are two sets of kernels:
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
#define ACCUM_DATA_T float
#define ACCUM_DATA8_T float8
#define ACCUM_DATA2_T float2
#define SUM_DATA_T ACCUM_DATA2_T

// Kahan summation algorithm. It's much more precise than simple sum and works
// just as fast, since kernel is still memory-bound.
SUM_DATA_T summation(ACCUM_DATA_T input, SUM_DATA_T state) {
    ACCUM_DATA2_T ret;
    ACCUM_DATA_T y = input - state.s1;
    ACCUM_DATA_T t = state.s0 + y;
    ret.s1 = (t - state.s0) - y;
    ret.s0 = t;
    return ret;
}
#endif // USE_STATS_ONE_PASS

#if FUSED_ATOMICS_REDUCTION
#if USE_STATS_ONE_PASS
void gen9_mean_var_calc_fused_reduction(volatile __global atomic_float *mean,
        volatile __global atomic_float *variance, int dst_offset,
        SUM_DATA_T *sum, SUM_DATA_T *sum_sq, __local SUM_DATA_T *local_sum,
        __local SUM_DATA_T *local_sum_sq) {
#else // regular alg
void gen9_calc_fused_reduction(volatile __global atomic_float *dst,
        int dst_offset, float *sum, __local float *local_sum) {
#endif
    const int simd_id = get_sub_group_local_id();

    const int group_size = GWS_LWS1_CALC * GWS_LWS2_CALC;
    const int sg_group_id = get_local_id(0) / 16;
    const int local_id = get_local_id(1);

    if (local_id > 0) {
        for (int sg = 0; sg < REDUCE_NUM_SGROUPS; ++sg) {
            const int slm_offset = CALC_SLM_LINE_SIZE * local_id
                    + REDUCE_NUM_SGROUPS * 16 * sg_group_id + sg * 16 + simd_id;
            local_sum[slm_offset] = sum[sg];
#if USE_STATS_ONE_PASS
            local_sum_sq[slm_offset] = sum_sq[sg];
#endif
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        for (int sg = 0; sg < REDUCE_NUM_SGROUPS; ++sg) {
            for (int gr_id = 1; gr_id < group_size; ++gr_id) {
                const int off_local = CALC_SLM_LINE_SIZE * gr_id
                        + REDUCE_NUM_SGROUPS * 16 * sg_group_id + sg * 16
                        + simd_id;
#if USE_STATS_ONE_PASS
                SUM_DATA_T tmp = local_sum[off_local];
                SUM_DATA_T tmp_sq = local_sum_sq[off_local];
                sum[sg] = summation(tmp.s1, sum[sg]);
                sum_sq[sg] = summation(tmp_sq.s1, sum_sq[sg]);
                sum[sg] = summation(tmp.s0, sum[sg]);
                sum_sq[sg] = summation(tmp_sq.s0, sum_sq[sg]);
#else // regular alg
                sum[sg] += local_sum[off_local];
#endif
            }
            const int offset = dst_offset + sg * 16 + simd_id;
#if HAS_IC_TAIL
            if (offset < IC) {
#endif
#if USE_STATS_ONE_PASS
                atomic_add_global(&mean[offset], sum[sg].s0);
                atomic_add_global(&variance[offset], sum_sq[sg].s0);
#else // regular alg
            atomic_add_global(&dst[offset], sum[sg]);
#endif
#if HAS_IC_TAIL
            }
#endif
        }
    }
    return;
}
#endif // FUSED_ATOMICS_REDUCTION

void gen9_reduce_common(__global float *reduce_temp, __local float *local_sum,
        __global float *dst) {

    const int ic_sub_group = get_global_id(0) / 16;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * 16 + simd_id;
    const bool is_last_ic_block = (IC - group_c * 16) < 16;
    float sum = 0.0f;

    reduce_temp
            += REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS * 16 * ic_sub_group
            + REDUCE_STAT_NBLOCKS * 16 * group_c + simd_id;
    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        sum += reduce_temp[i * 16];
    }

    if (ic_sub_group > 0) { local_sum[ic_sub_group * 16 + simd_id] = sum; }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ic_sub_group == 0) {
        for (int i = 1; i < REDUCE_IC_SUB_GROUPS; i++) {
            sum += local_sum[i * 16 + simd_id];
        }
#if HAS_IC_TAIL
        if (!is_last_ic_block || (is_last_ic_block && simd_id < 8))
#endif
            dst[c] = sum / (MB * ID * IH * IW);
    }
}

#if USE_STATS_ONE_PASS

// IC tail processing is not implemented for one pass algorithm
// Calculates partial sums of values and squares-of-values per channel

#if NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean_var(__global DATA_T *src,
        __global ACCUM_DATA_T *reduce_temp,
        volatile __global atomic_float *mean,
        volatile __global atomic_float *variance) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();

    const int ic_block_offset = (c / 16) * IC_BLOCK;
    const int group_c_offset
            = REDUCE_STAT_NBLOCKS * ic_block_offset + sp_block_idx * 16;
    const int ver_offs = REDUCE_STAT_NBLOCKS * IC;

    const int src_off = ic_block_offset + sp_block_idx * STAT_SP_BLOCK * IC;

    src += src_off;

    SUM_DATA_T sum[IC_BLOCK_SGROUPS] = {0.0f};
    SUM_DATA_T sum_sq[IC_BLOCK_SGROUPS] = {0.0f};

#if HAS_STAT_SP_BLOCK_TAIL
    for (int sp = 0; sp < min(STAT_SP_BLOCK, SP - sp_block_idx * STAT_SP_BLOCK);
            ++sp) {
#else
    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
#endif
        // vectorized part
        for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
            VECT_FLOAT_T s_vect = LOAD_VECT_DATA(&src[sg * 16 * VECT_SIZE]);

            for (int vect = 0; vect < VECT_SIZE; ++vect) {
                const int sum_idx = sg * VECT_SIZE + vect;
#if VECT_SIZE > 1
                sum[sum_idx] = summation(s_vect[vect], sum[sum_idx]);
                sum_sq[sum_idx] = summation(
                        s_vect[vect] * s_vect[vect], sum_sq[sum_idx]);
#else
                sum[sum_idx] = summation(s_vect, sum[sum_idx]);
                sum_sq[sum_idx] = summation(s_vect * s_vect, sum_sq[sum_idx]);
#endif
            }
        }
#if HAS_IC_VECT_TAIL
        // tails
        for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
            const int sg_idx = IC_VECT_SGROUPS + sg;
            float s_tail = LOAD_DATA_1x16(&src[(IC_VECT_SGROUPS + sg) * 16]);
            sum[sg_idx] = summation(s_tail, sum[sg_idx]);
            sum_sq[sg_idx] = summation(s_tail * s_tail, sum_sq[sg_idx]);
        }
#endif
        src += IC;
    }

#if FUSED_ATOMICS_REDUCTION
    __local SUM_DATA_T local_sum[2 * CALC_SLM_SIZE];
    __local SUM_DATA_T *local_sum_sq = local_sum + CALC_SLM_SIZE;
    gen9_mean_var_calc_fused_reduction(mean, variance, ic_block_offset, sum,
            sum_sq, local_sum, local_sum_sq);
#else
    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        const int reduce_off = group_c_offset + sg * 16 * REDUCE_STAT_NBLOCKS;
        STORE_FLOAT_1x16(&reduce_temp[reduce_off], sum[sg].s0);
        STORE_FLOAT_1x16(&reduce_temp[ver_offs + reduce_off], sum_sq[sg].s0);
    }
#endif
}

#else // NHWC_OPTIMIZED

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

#endif // NHWC_OPTIMIZED

// Calculates mean and variance by further reducing sum of values
// and sum of squares-of-values.
NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_mean_var(__global ACCUM_DATA_T *reduce_temp,
        __global float *mean, __global float *variance) {

    __local SUM_DATA_T local_sum[16 * REDUCE_IC_SUB_GROUPS];
    __local SUM_DATA_T local_sum_sq[16 * REDUCE_IC_SUB_GROUPS];

    const int ic_sub_group = get_global_id(0) / 16;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * 16 + simd_id;
    SUM_DATA_T sum;
    SUM_DATA_T sum_sq;
    sum.s0 = 0;
    sum.s1 = 0;
    sum_sq.s0 = 0;
    sum_sq.s1 = 0;

    int offs_sq = REDUCE_STAT_NBLOCKS * IC;
    int offs = REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS * 16 * ic_sub_group
            + REDUCE_STAT_NBLOCKS * 16 * group_c + simd_id;

    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        float tmp = reduce_temp[offs + i * 16];
        sum = summation(tmp, sum);
    }
    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        float tmp = reduce_temp[offs_sq + offs + i * 16];
        sum_sq = summation(tmp, sum_sq);
    }

    if (ic_sub_group > 0) {
        local_sum[ic_sub_group * 16 + simd_id] = sum;
        local_sum_sq[ic_sub_group * 16 + simd_id] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ic_sub_group == 0) {
        for (int i = 1; i < REDUCE_IC_SUB_GROUPS; i++) {
            SUM_DATA_T tmp = local_sum[i * 16 + simd_id];
            SUM_DATA_T tmp_sq = local_sum_sq[i * 16 + simd_id];
            sum = summation(tmp.s1, sum);
            sum_sq = summation(tmp_sq.s1, sum_sq);
            sum = summation(tmp.s0, sum);
            sum_sq = summation(tmp_sq.s0, sum_sq);
        }
        float tmp_mean = sum.s0 / (MB * ID * IH * IW);
        mean[c] = tmp_mean;
        float tmp_var = max(0.0f,
                (sum_sq.s0 / (MB * ID * IH * IW)) - (tmp_mean * tmp_mean));
        variance[c] = tmp_var;
    }
}
#else // USE_STATS_ONE_PASS

#if NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean(__global DATA_T *src, __global float *reduce_temp,
        volatile __global atomic_float *mean) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int ic_block_offset = (c / 16) * IC_BLOCK;
    const int group_c_offset
            = REDUCE_STAT_NBLOCKS * ic_block_offset + sp_block_idx * 16;

    const int src_off = ic_block_offset + sp_block_idx * STAT_SP_BLOCK * IC;
    src += src_off;

    float v_mean[IC_BLOCK_SGROUPS] = {0.0f};

#if HAS_STAT_SP_BLOCK_TAIL
    for (int sp = 0; sp < min(STAT_SP_BLOCK, SP - sp_block_idx * STAT_SP_BLOCK);
            ++sp) {
#else
    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
#endif
        // vectorized part
        for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
            VECT_FLOAT_T s_vect = LOAD_VECT_DATA(&src[sg * 16 * VECT_SIZE]);
            for (int vect = 0; vect < VECT_SIZE; ++vect) {
                v_mean[sg * VECT_SIZE + vect]
#if VECT_SIZE > 1
                        += s_vect[vect];
#else
                        += s_vect;
#endif
            }
        }
#if HAS_IC_VECT_TAIL
        // tails
        for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
            float s_tail = LOAD_DATA_1x16(&src[(IC_VECT_SGROUPS + sg) * 16]);
            v_mean[IC_VECT_SGROUPS + sg] += s_tail;
        }
#endif // HAS_IC_VECT_TAIL
        src += IC;
    }

#if FUSED_ATOMICS_REDUCTION
    __local float local_sum[CALC_SLM_SIZE];
    gen9_calc_fused_reduction(mean, ic_block_offset, v_mean, local_sum);
#else
    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        const int reduce_off = group_c_offset + sg * 16 * REDUCE_STAT_NBLOCKS;
        STORE_FLOAT_1x16(&reduce_temp[reduce_off], v_mean[sg]);
    }
#endif
}

#else // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean(__global DATA_T *src, __global float *reduce_temp,
        __global float *mean) {

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
    // reduce_temp is padded to IC16, no OOB writes
    STORE_FLOAT_1x16(&reduce_temp[group_c_offset + mb_sp_idx * 16], v_mean);
#endif
}

#endif // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_mean(
        __global float *reduce_temp, __global float *mean) {
    __local float local_sum[16 * REDUCE_IC_SUB_GROUPS];
    gen9_reduce_common(reduce_temp, local_sum, mean);
}

#if NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_variance(__global DATA_T *src, __global float *mean,
        __global float *reduce_temp, volatile __global atomic_float *variance) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int ic_block_offset = (c / 16) * IC_BLOCK;
    const int group_c_offset
            = REDUCE_STAT_NBLOCKS * ic_block_offset + sp_block_idx * 16;

    reduce_temp += REDUCE_STAT_NBLOCKS * IC16;
    mean += ic_block_offset;
    const int src_off = ic_block_offset + sp_block_idx * STAT_SP_BLOCK * IC;
    src += src_off;

    float v_mean[IC_BLOCK_SGROUPS];
    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        v_mean[sg] = as_float(intel_sub_group_block_read(
                (const __global uint *)(&mean[(sg * 16)])));
    }

    float v_var[IC_BLOCK_SGROUPS] = {0.0f};
    float v0[IC_BLOCK_SGROUPS] = {0.0f};

#if HAS_STAT_SP_BLOCK_TAIL
    for (int sp = 0; sp < min(STAT_SP_BLOCK, SP - sp_block_idx * STAT_SP_BLOCK);
            ++sp) {
#else
    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
#endif
        for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
            VECT_FLOAT_T s_vect = LOAD_VECT_DATA(&src[sg * 16 * VECT_SIZE]);

            for (int vect = 0; vect < VECT_SIZE; ++vect) {
                int sg_idx = sg * VECT_SIZE + vect;
#if VECT_SIZE > 1
                v0[sg_idx] = s_vect[vect] - v_mean[sg_idx];
#else
                v0[sg_idx] = s_vect - v_mean[sg_idx];
#endif
                v_var[sg_idx] = fma(v0[sg_idx], v0[sg_idx], v_var[sg_idx]);
            }
        }
#if HAS_IC_VECT_TAIL
        // tails
        for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
            const int sg_idx = IC_VECT_SGROUPS + sg;
            float s_tail = LOAD_DATA_1x16(&src[(IC_VECT_SGROUPS + sg) * 16]);
            v0[sg_idx] = s_tail - v_mean[sg_idx];
            v_var[sg_idx] = fma(v0[sg_idx], v0[sg_idx], v_var[sg_idx]);
        }
#endif // HAS_IC_VECT_TAIL
        src += IC;
    }

#if FUSED_ATOMICS_REDUCTION
    __local float local_sum[CALC_SLM_SIZE];
    gen9_calc_fused_reduction(variance, ic_block_offset, v_var, local_sum);
#else
    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        const int reduce_off = group_c_offset + sg * 16 * REDUCE_STAT_NBLOCKS;
        STORE_FLOAT_1x16(&reduce_temp[reduce_off], v_var[sg]);
    }
#endif
}

#else // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_variance(__global DATA_T *src, __global float *mean,
        __global float *reduce_temp, __global float *variance) {

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
    reduce_temp += REDUCE_STAT_NBLOCKS * IC16;

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
    // reduce_temp is padded to IC16, no OOB writes
    STORE_FLOAT_1x16(&reduce_temp[group_c_offset + mb_sp_idx * 16], v_var);
#endif
}

#endif // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_variance(
        __global float *reduce_temp, __global float *variance) {
    __local float local_sum[16 * REDUCE_IC_SUB_GROUPS];
    gen9_reduce_common(
            reduce_temp + REDUCE_STAT_NBLOCKS * IC16, local_sum, variance);
}

#endif // USE_STATS_ONE_PASS

#if NHWC_OPTIMIZED

KERNEL_ATTR
__kernel void gen9_bnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst,
        __global float *scaleshift, __global float *shift, __global char *ws,
        float eps, __global DATA_T *src_add) {

    const int n = GWS_GET_MB();
    const int c = GWS_GET_IC();
    const int sp = GWS_GET_SP() * STAT_SP_BLOCK;

    const int ic_block_offset = (c / 16) * IC_BLOCK;
    mean += ic_block_offset;
    variance += ic_block_offset;
    shift += ic_block_offset;
    scaleshift += ic_block_offset;
    const uint d_off = sp * IC + ic_block_offset;

    src += d_off;
#if FUSE_BN_ADD_RELU
    src_add += d_off;
#endif
    dst += d_off;
#if FUSE_BN_RELU && IS_TRAINING
    ws += d_off;
#endif

    VECT_FLOAT_T sm[IC_BLOCK_SGROUPS / VECT_SIZE],
            sv[IC_BLOCK_SGROUPS / VECT_SIZE],
            v_mean[IC_BLOCK_SGROUPS / VECT_SIZE],
            v_variance[IC_BLOCK_SGROUPS / VECT_SIZE],
            sqrt_variance[IC_BLOCK_SGROUPS / VECT_SIZE];
    for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
        const int sg_idx = sg * 16 * VECT_SIZE;
#if USE_SCALE == 1
        sm[sg] = LOAD_VECT_FLOAT(&scaleshift[sg_idx]);
#else
        sm[sg] = (VECT_FLOAT_T)1.0f;
#endif
#if USE_SHIFT == 1
        sv[sg] = LOAD_VECT_FLOAT(&shift[sg_idx]);
#else
        sv[sg] = (VECT_FLOAT_T)0.0f;
#endif
        v_mean[sg] = LOAD_VECT_FLOAT(&mean[sg_idx]);
        v_variance[sg] = LOAD_VECT_FLOAT(&variance[sg_idx]);
        sqrt_variance[sg] = sm[sg] / sqrt(v_variance[sg] + (VECT_FLOAT_T)eps);
    }

#if HAS_IC_VECT_TAIL
    float sm_tail[IC_TAIL_SGROUPS], sv_tail[IC_TAIL_SGROUPS],
            v_mean_tail[IC_TAIL_SGROUPS], v_variance_tail[IC_TAIL_SGROUPS],
            sqrt_variance_tail[IC_TAIL_SGROUPS];
    for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
        const int sg_idx = (IC_VECT_SGROUPS + sg) * 16;
#if USE_SCALE == 1
        sm_tail[sg] = LOAD_FLOAT_1x16(&scaleshift[sg_idx]);
#else
        sm_tail[sg] = 1.0f;
#endif
#if USE_SHIFT == 1
        sv_tail[sg] = LOAD_FLOAT_1x16(&shift[sg_idx]);
#else
        sv_tail[sg] = 0.0f;
#endif
        v_mean_tail[sg] = LOAD_FLOAT_1x16(&mean[sg_idx]);
        v_variance_tail[sg] = LOAD_FLOAT_1x16(&variance[sg_idx]);
        sqrt_variance_tail[sg] = sm_tail[sg] / sqrt(v_variance_tail[sg] + eps);
    }
#endif

#if HAS_STAT_SP_BLOCK_TAIL
    for (int sp_idx = 0; sp_idx < min(STAT_SP_BLOCK, SP - sp); ++sp_idx) {
#else
    for (int sp_idx = 0; sp_idx < STAT_SP_BLOCK; ++sp_idx) {
#endif
        // vectorized part
        for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
            const int sg_idx = sg * 16 * VECT_SIZE;
            VECT_FLOAT_T s_vect = LOAD_VECT_DATA(&src[sg_idx]);
            VECT_FLOAT_T d_vect
                    = fma(s_vect - v_mean[sg], sqrt_variance[sg], sv[sg]);

#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
            VECT_FLOAT_T s_add_vect = LOAD_VECT_DATA(&src_add[sg_idx]);
            d_vect += s_add_vect;
#endif
            VECT_INT_T ws_vect = isgreater(d_vect, (VECT_FLOAT_T)0.0f);
            d_vect = select((VECT_FLOAT_T)0.0f, d_vect, ws_vect);
#if IS_TRAINING
            STORE_VECT_CHAR(&ws[sg_idx], ws_vect);
#endif // IS_TRAINING
#endif // FUSE_BN_RELU
#if WITH_RELU
            d_vect = max(d_vect, (VECT_FLOAT_T)0.0f);
#endif
            STORE_VECT_DATA(&dst[sg_idx], d_vect);
        }

#if HAS_IC_VECT_TAIL
        // tails
        for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
            const int sg_idx = (IC_VECT_SGROUPS + sg) * 16;
            float s_tail = LOAD_DATA_1x16(&src[sg_idx]);
            float d_tail = fma(s_tail - v_mean_tail[sg], sqrt_variance_tail[sg],
                    sv_tail[sg]);
#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
            float s_add_tail = LOAD_DATA_1x16(&src_add[sg_idx]);
            d_tail += s_add_tail;
#endif
            int ws_tail = isgreater(d_tail, 0.0f);
            d_tail = select(0.0f, d_tail, ws_tail);
#if IS_TRAINING
            STORE_CHAR_1x16(&ws[sg_idx], convert_char(ws_tail));
#endif // IS_TRAINING
#endif // FUSE_BN_RELU
#if WITH_RELU
            d_tail = max(d_tail, 0.0f);
#endif
            STORE_DATA_1x16(&dst[sg_idx], d_tail);
        }
#endif
        src += IC;
#if FUSE_BN_ADD_RELU
        src_add += IC;
#endif
        dst += IC;
#if FUSE_BN_RELU && IS_TRAINING
        ws += IC;
#endif
    }
}

#else // NHWC_OPTIMIZED

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
        float eps, __global DATA_T *src_add) {

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
    blockD0 = max(blockD0, (VECT_FLOAT_T)0.0f);
#endif

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
#endif // NHWC_OPTIMIZED
