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

// Zeroing and finalization kernels are required for atomics-based
// reduction (FUSED_ATOMICS_REDUCTION definition).

NAMED_KERNEL_ATTR(AUX)
__kernel void gen9_fused_reduce_init(
#if IS_FWD
        __global float *mean, __global float *variance
#else
        __global float *diff_scale, __global float *diff_shift
#endif
) {
    const int c = GWS_GET_IC_AUX();
#if IS_FWD
    mean[c] = 0.0f;
    variance[c] = 0.0f;
#else
    diff_scale[c] = 0.0f;
#if DIFF_SHIFT == 1
    diff_shift[c] = 0.0f;
#else
    diff_shift[IC + IC * REDUCE_STAT_NBLOCKS + c] = 0.0f;
#endif
#endif
    return;
}

#if IS_FWD
NAMED_KERNEL_ATTR(AUX)
__kernel void gen9_fused_reduce_final(
#if USE_STATS_ONE_PASS
        __global float *mean, __global float *variance
#else
        __global float *data_reduce
#endif
) {
    const int c = GWS_GET_IC_AUX();
#if USE_STATS_ONE_PASS
    mean[c] = mean[c] / (MB * ID * IH * IW);
    float tmp_var = max(
            0.0f, (variance[c] / (MB * ID * IH * IW)) - (mean[c] * mean[c]));
    variance[c] = tmp_var;
#else
    data_reduce[c] /= (MB * ID * IH * IW);
#endif
    return;
}
#else
NAMED_KERNEL_ATTR(AUX)
__kernel void gen9_fused_reduce_final(
        __global float *diff_scale, __global float *variance, float eps) {
    const int c = GWS_GET_IC_AUX();
    diff_scale[c] *= 1.0f / sqrt(variance[c] + eps);
    return;
}
#endif // IS_FWD

// Reduction over scratchpad (reduce_temp) with SLM use.

#if IS_FWD
#if USE_STATS_ONE_PASS
// Calculates mean and variance by further reducing sum of values
// and sum of squares-of-values.
NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_mean_var(__global ACCUM_DATA_T *reduce_temp,
        __global float *mean, __global float *variance) {

    __local SUM_DATA_T local_sum[SG_SIZE * REDUCE_IC_SUB_GROUPS];
    __local SUM_DATA_T local_sum_sq[SG_SIZE * REDUCE_IC_SUB_GROUPS];

    const int ic_sub_group = get_global_id(0) / SG_SIZE;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * SG_SIZE + simd_id;
    SUM_DATA_T sum;
    SUM_DATA_T sum_sq;
    sum.s0 = 0;
    sum.s1 = 0;
    sum_sq.s0 = 0;
    sum_sq.s1 = 0;

    int offs_sq = REDUCE_STAT_NBLOCKS * IC;
    int offs = REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS * SG_SIZE
                    * ic_sub_group
            + REDUCE_STAT_NBLOCKS * SG_SIZE * group_c + simd_id;

    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        float tmp = reduce_temp[offs + i * SG_SIZE];
        sum = summation(tmp, sum);
    }
    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        float tmp = reduce_temp[offs_sq + offs + i * SG_SIZE];
        sum_sq = summation(tmp, sum_sq);
    }

    if (ic_sub_group > 0) {
        local_sum[ic_sub_group * SG_SIZE + simd_id] = sum;
        local_sum_sq[ic_sub_group * SG_SIZE + simd_id] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ic_sub_group == 0) {
        for (int i = 1; i < REDUCE_IC_SUB_GROUPS; i++) {
            SUM_DATA_T tmp = local_sum[i * SG_SIZE + simd_id];
            SUM_DATA_T tmp_sq = local_sum_sq[i * SG_SIZE + simd_id];
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

void gen9_reduce_common(__global float *reduce_temp, __local float *local_sum,
        __global float *dst) {

    const int ic_sub_group = get_global_id(0) / SG_SIZE;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * SG_SIZE + simd_id;
    const bool is_last_ic_block = (IC - group_c * SG_SIZE) < SG_SIZE;
    float sum = 0.0f;

    reduce_temp += REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS * SG_SIZE
                    * ic_sub_group
            + REDUCE_STAT_NBLOCKS * SG_SIZE * group_c + simd_id;
    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        sum += reduce_temp[i * SG_SIZE];
    }

    if (ic_sub_group > 0) { local_sum[ic_sub_group * SG_SIZE + simd_id] = sum; }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ic_sub_group == 0) {
        for (int i = 1; i < REDUCE_IC_SUB_GROUPS; i++) {
            sum += local_sum[i * SG_SIZE + simd_id];
        }
#if HAS_IC_TAIL
        if (!is_last_ic_block || (is_last_ic_block && simd_id < 8))
#endif
            dst[c] = sum / (MB * ID * IH * IW);
    }
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_mean(
        __global float *reduce_temp, __global float *mean) {
    __local float local_sum[SG_SIZE * REDUCE_IC_SUB_GROUPS];
    gen9_reduce_common(reduce_temp, local_sum, mean);
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_variance(
        __global float *reduce_temp, __global float *variance) {
    __local float local_sum[SG_SIZE * REDUCE_IC_SUB_GROUPS];
    gen9_reduce_common(
            reduce_temp + REDUCE_STAT_NBLOCKS * PADDED_IC, local_sum, variance);
}

#endif // USE_STATS_ONE_PASS
#endif // IS_FWD

#if IS_BWD
NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_stats(__global float *temp_reduce,
        __global float *diff_scale, __global float *diff_shift,
        __global float *variance, float eps) {

    __local float local_gamma[SG_SIZE * REDUCE_IC_SUB_GROUPS];
    __local float local_beta[SG_SIZE * REDUCE_IC_SUB_GROUPS];
    const int ic_sub_group = get_global_id(0) / SG_SIZE;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * SG_SIZE + simd_id;
    float diff_gamma = 0.0f;
    float diff_beta = 0.0f;

    temp_reduce += PADDED_IC + REDUCE_STAT_NBLOCKS * SG_SIZE * group_c
            + REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS * SG_SIZE
                    * ic_sub_group
            + simd_id;
    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        diff_gamma += temp_reduce[i * SG_SIZE];
    }
    temp_reduce += PADDED_IC + PADDED_IC * REDUCE_STAT_NBLOCKS;
    for (int i = 0; i < REDUCE_STAT_NBLOCKS / REDUCE_IC_SUB_GROUPS; i++) {
        diff_beta += temp_reduce[i * SG_SIZE];
    }

    if (ic_sub_group > 0) {
        local_gamma[ic_sub_group * SG_SIZE + simd_id] = diff_gamma;
        local_beta[ic_sub_group * SG_SIZE + simd_id] = diff_beta;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ic_sub_group == 0) {
        for (int i = 1; i < REDUCE_IC_SUB_GROUPS; i++) {
            diff_gamma += local_gamma[i * SG_SIZE + simd_id];
            diff_beta += local_beta[i * SG_SIZE + simd_id];
        }

        float sqrt_variance = 1.0f / sqrt(variance[c] + eps);

#if HAS_IC_TAIL
        const bool is_last_ic_block = group_c * SG_SIZE + SG_SIZE > IC;
        if (!is_last_ic_block || (is_last_ic_block && simd_id < 8))
#endif
        {
            diff_scale[c] = diff_gamma * sqrt_variance;
#if DIFF_SHIFT == 1
            diff_shift[c] = diff_beta;
#else
            diff_shift[PADDED_IC + PADDED_IC * REDUCE_STAT_NBLOCKS + c]
                    = diff_beta;
#endif // #if DIFF_SHIFT == 1
        }
    }
}
#endif
