/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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
#include "gpu/ocl/bnorm/nhwc_reusable.h"

__kernel void nhwc_reusable_reduce_aux(__global float *dst1,
        __global float *dst2, off_t sp_size, int use_stats_one_pass,
        int init_stage) {
    const int c = get_global_id(0);
    if (init_stage) {
        // initialization
        dst1[c] = 0.0f;
        dst2[c] = 0.0f;
    } else {
        // finalization
        if (use_stats_one_pass) {
            dst1[c] /= sp_size;
            float tmp_var
                    = max(0.0f, (dst2[c] / sp_size) - (dst1[c] * dst1[c]));
            dst2[c] = tmp_var;
        } else {
            dst1[c] /= sp_size;
        }
    }
}
__attribute__((intel_reqd_sub_group_size(16))) __kernel void
nhwc_reusable_reduce_fwd_reg(__global float *reduce_scratchpad,
        off_t scratchpad_off, __global float *dst, off_t ic_size,
        off_t reduce_ic_sub_groups, off_t reduce_stat_nblocks, off_t sp_size,
        __local float *local_sum) {
    const int ic_sub_group = get_global_id(0) / SG_SIZE;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * SG_SIZE + simd_id;
    float sum = 0.0f;

    const int reduce_chunk = reduce_stat_nblocks / reduce_ic_sub_groups;
    const int reduce_scratchpad_off
            = scratchpad_off + c + ic_sub_group * reduce_chunk * ic_size;
    reduce_scratchpad += reduce_scratchpad_off;

    unroll_16_for(int i = 0; i < reduce_chunk; i++) {
        sum += reduce_scratchpad[i * ic_size];
    }

    if (ic_sub_group > 0) { local_sum[ic_sub_group * SG_SIZE + simd_id] = sum; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ic_sub_group == 0) {
        unroll_16_for(int i = 1; i < reduce_ic_sub_groups; i++) {
            sum += local_sum[i * SG_SIZE + simd_id];
        }
        dst[c] = sum / sp_size;
    }
}

__attribute__((intel_reqd_sub_group_size(16))) __kernel void
nhwc_reusable_reduce_fwd_1pass(__global float *reduce_temp,
        __global float *mean, __global float *variance, off_t ic_size,
        off_t reduce_ic_sub_groups, off_t reduce_stat_nblocks, off_t sp_size,
        __local SUM_DATA_T *local_sum, __local SUM_DATA_T *local_sum_sq) {
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

    const int offs_sq = reduce_stat_nblocks * ic_size;
    const int reduce_chunk = reduce_stat_nblocks / reduce_ic_sub_groups;
    const int offs = c + ic_sub_group * reduce_chunk * ic_size;

    unroll_16_for(int i = 0; i < reduce_chunk; i++) {
        float tmp = reduce_temp[offs + i * ic_size];
        sum = summation(tmp, sum);
    }
    unroll_16_for(int i = 0; i < reduce_chunk; i++) {
        float tmp = reduce_temp[offs_sq + offs + i * ic_size];
        sum_sq = summation(tmp, sum_sq);
    }
    if (ic_sub_group > 0) {
        local_sum[ic_sub_group * SG_SIZE + simd_id] = sum;
        local_sum_sq[ic_sub_group * SG_SIZE + simd_id] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ic_sub_group == 0) {
        unroll_16_for(int i = 1; i < reduce_ic_sub_groups; i++) {
            SUM_DATA_T tmp = local_sum[i * SG_SIZE + simd_id];
            SUM_DATA_T tmp_sq = local_sum_sq[i * SG_SIZE + simd_id];
            sum = summation(tmp.s1, sum);
            sum_sq = summation(tmp_sq.s1, sum_sq);
            sum = summation(tmp.s0, sum);
            sum_sq = summation(tmp_sq.s0, sum_sq);
        }
        float tmp_mean = sum.s0 / sp_size;
        mean[c] = tmp_mean;
        float tmp_var
                = max(0.0f, (sum_sq.s0 / sp_size) - (tmp_mean * tmp_mean));
        variance[c] = tmp_var;
    }
}
