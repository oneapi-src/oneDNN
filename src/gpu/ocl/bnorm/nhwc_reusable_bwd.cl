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

void nhwc_reusable_bwd_calc_fused_reduction(
        volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift, off_t dst_offset,
        float *diff_gamma, float *diff_beta, __local float *local_sums,
        off_t vect_size, off_t calc_slm_size) {
    const int local_id = get_local_id(1);
    const int simd_id = get_sub_group_local_id();
    const int row_size = vect_size * SG_SIZE;
    const int group_size = get_local_size(1);

    __local float *local_gamma = local_sums;
    __local float *local_beta = local_sums + calc_slm_size / sizeof(float);

    if (local_id > 0) {
        unroll_4_for(int v_idx = 0; v_idx < vect_size; v_idx++) {
            const int slm_offset
                    = local_id * row_size + v_idx * SG_SIZE + simd_id;
            local_gamma[slm_offset] = diff_gamma[v_idx];
            local_beta[slm_offset] = diff_beta[v_idx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unroll_16_for(int l_id = 1; l_id < group_size; l_id++) {
            for (int v_idx = 0; v_idx < vect_size; v_idx++) {
                const int off = l_id * row_size + v_idx * SG_SIZE + simd_id;
                diff_gamma[v_idx] += local_gamma[off];
                diff_beta[v_idx] += local_beta[off];
            }
        }
        unroll_4_for(int v_idx = 0; v_idx < vect_size; v_idx++) {
            const int off = v_idx * SG_SIZE + simd_id;
            atomic_add_global(&diff_scale[dst_offset + off], diff_gamma[v_idx]);
            atomic_add_global(&diff_shift[dst_offset + off], diff_beta[v_idx]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
}

__attribute__((intel_reqd_sub_group_size(16))) __kernel void
nhwc_reusable_calc_stat(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global char *ws,
        __global float *temp_reduce, __global float *temp_reduce_shift,
        volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift, off_t ic_size,
        off_t ic_block, off_t sp_size, off_t stat_sp_block,
        off_t reduce_stat_nblocks, int use_fused_atomics_reduction,
        __local float *local_sums, off_t calc_slm_size) {
    const int c = get_global_id(0);
    const int sp_block_idx = get_global_id(1);
    const int ic_block_offset = (c / SG_SIZE) * ic_block;
    const int offset = ic_block_offset + sp_block_idx * stat_sp_block * ic_size;

    mean += ic_block_offset;
    src += offset;
    diff_dst += offset;
    ws += offset;

    // scratchpad layout: (reduce_stat_nblocks + 1) rows x ic columns
    const int reduce_off = ic_block_offset + (sp_block_idx + 1) * ic_size;

    temp_reduce += reduce_off;
    temp_reduce_shift += reduce_off;

    const bool has_sp_block_tail = sp_size % stat_sp_block;
    const int sp_idx_bnd = has_sp_block_tail
            ? min(stat_sp_block, sp_size - sp_block_idx * stat_sp_block)
            : stat_sp_block;
    const int ic_block_sgroups = ic_block / SG_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups - ic_tail_sgroups;

    // vectorized part

    for (int sg = 0; sg < ic_block_sgroups / VECT_SIZE; ++sg) {
        const int sg_idx = sg * SG_SIZE * VECT_SIZE;
        VECT_FLOAT_T diff_gamma = 0.0f;
        VECT_FLOAT_T diff_beta = 0.0f;
        const VECT_FLOAT_T v_mean = LOAD_VECT_FLOAT(&mean[(sg_idx)]);

        // reduce
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            const int tn_idx = sg_idx + sp * ic_size;
#if FUSE_BN_RELU
            const VECT_CHAR_T ws_vect = LOAD_VECT_CHAR(&ws[tn_idx]);
#endif
            const VECT_FLOAT_T src_vect = LOAD_VECT_DATA(&src[tn_idx]);
            VECT_FLOAT_T dd_vect = LOAD_VECT_DATA(&diff_dst[tn_idx]);
            const VECT_FLOAT_T v0 = src_vect - v_mean;
#if FUSE_BN_RELU
            dd_vect = select(
                    (VECT_FLOAT_T)0.0f, dd_vect, CONVERT_VECT_INT_T(ws_vect));
#endif
            diff_gamma = fma(v0, dd_vect, diff_gamma);
            diff_beta += dd_vect;
        } // sp loop

        // store results
        if (use_fused_atomics_reduction) {
            const int dst_off = ic_block_offset + sg * VECT_SIZE * SG_SIZE;
            nhwc_reusable_bwd_calc_fused_reduction(diff_scale, diff_shift,
                    dst_off, (float *)(&diff_gamma), (float *)(&diff_beta),
                    local_sums, VECT_SIZE, calc_slm_size);
        } else {
            // Two different scratchpads: for diff_gamma and diff_beta
            // scratchpad layout (elements):
            // ic_size - final reduced data,
            //           wrote by nhwc_reusable_reduce_stats kernel
            // reduce_stat_nblocks * ic_size - initialy reduced data,
            //           calculated by this kernel

            const int sg_off = sg * VECT_SIZE * SG_SIZE;
            for (int v_idx = 0; v_idx < VECT_SIZE; v_idx++) {
                STORE_FLOAT_1x16(&temp_reduce[sg_off + v_idx * SG_SIZE],
#if VECT_SIZE > 1
                        diff_gamma[v_idx]);
#else
                        diff_gamma);
#endif
                STORE_FLOAT_1x16(&temp_reduce_shift[sg_off + v_idx * SG_SIZE],
#if VECT_SIZE > 1
                        diff_beta[v_idx]);
#else
                        diff_beta);
#endif
            }
        }
    } // sg loop

    // tails
    for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
        const int sg_idx = (ic_vect_sgroups + sg) * SG_SIZE;
        float diff_gamma = 0.0f;
        float diff_beta = 0.0f;
        const float v_mean = LOAD_FLOAT_1x16(&mean[(sg_idx)]);

        // reduce
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            const int tn_idx = sg_idx + sp * ic_size;
#if FUSE_BN_RELU
            const char ws_vect = LOAD_CHAR_1x16(&ws[tn_idx]);
#endif
            const float src_vect = LOAD_DATA_1x16(&src[tn_idx]);
            float dd_vect = LOAD_DATA_1x16(&diff_dst[tn_idx]);
            const float v0 = src_vect - v_mean;
#if FUSE_BN_RELU
            dd_vect = select(0.0f, dd_vect, convert_int(ws_vect));
#endif
            diff_gamma = fma(v0, dd_vect, diff_gamma);
            diff_beta += dd_vect;
        } // sp loop

        // store results
        if (use_fused_atomics_reduction) {
            const int dst_off
                    = ic_block_offset + (ic_vect_sgroups + sg) * SG_SIZE;
            nhwc_reusable_bwd_calc_fused_reduction(diff_scale, diff_shift,
                    dst_off, (float *)(&diff_gamma), (float *)(&diff_beta),
                    local_sums, 1, calc_slm_size);
        } else {
            const int sg_off = (ic_vect_sgroups + sg) * SG_SIZE;
            STORE_FLOAT_1x16(&temp_reduce[sg_off], diff_gamma);
            STORE_FLOAT_1x16(&temp_reduce_shift[sg_off], diff_beta);
        }
    } // sg loop
}

__attribute__((intel_reqd_sub_group_size(16))) __kernel void
nhwc_reusable_update_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scaleshift, __global char *ws,
        __global DATA_T *diff_src, __global float *diff_scale,
        __global float *diff_shift, float eps, __global DATA_T *diff_src_add,
        off_t ic_size, off_t ic_block, off_t sp_size, off_t update_sp_block) {
    const int c = get_global_id(0);
    const int ic_block_offset = (c / SG_SIZE) * ic_block;

    variance += ic_block_offset;
    mean += ic_block_offset;
    diff_scale += ic_block_offset;
    diff_shift += ic_block_offset;
    scaleshift += ic_block_offset;

    const int sp_block_idx = get_global_id(1);
    const int offset
            = ic_block_offset + sp_block_idx * update_sp_block * ic_size;

    src += offset;
    diff_dst += offset;
    ws += offset;
    diff_src += offset;
#if FUSE_BN_ADD_RELU
    diff_src_add += offset;
#endif

    const bool has_sp_block_tail = sp_size % update_sp_block;
    const int sp_idx_bnd = has_sp_block_tail
            ? min(update_sp_block, sp_size - sp_block_idx * update_sp_block)
            : update_sp_block;
    const int ic_block_sgroups = ic_block / SG_SIZE;

    for (int sp = 0; sp < sp_idx_bnd; ++sp) {
        // vectorized part
        for (int sg = 0; sg < ic_block_sgroups / VECT_SIZE; ++sg) {
            const int sg_idx = sg * SG_SIZE * VECT_SIZE;

            const VECT_FLOAT_T v_variance = LOAD_VECT_FLOAT(&variance[sg_idx]);
            const VECT_FLOAT_T sqrt_variance
                    = (VECT_FLOAT_T)1.0f / sqrt(v_variance + (VECT_FLOAT_T)eps);
            const VECT_FLOAT_T gamma = USE_SCALE
                    ? LOAD_VECT_FLOAT(&scaleshift[sg_idx])
                    : (VECT_FLOAT_T)1.0f;
            const VECT_FLOAT_T src_vect = LOAD_VECT_DATA(&src[sg_idx]);
            VECT_FLOAT_T dd_vect = LOAD_VECT_DATA(&diff_dst[sg_idx]);

#if FUSE_BN_RELU
            const VECT_CHAR_T ws_vect = LOAD_VECT_CHAR(&ws[sg_idx]);
            dd_vect = select(
                    (VECT_FLOAT_T)0.0f, dd_vect, CONVERT_VECT_INT_T(ws_vect));
#if FUSE_BN_ADD_RELU
            STORE_VECT_DATA(&diff_src_add[sg_idx], dd_vect);
#endif
#endif
#if CALCULATE_STATS == 1
            const VECT_FLOAT_T v_mean = LOAD_VECT_FLOAT(&mean[sg_idx]);
            const VECT_FLOAT_T diff_gamma
                    = LOAD_VECT_FLOAT(&diff_scale[sg_idx]);
            const VECT_FLOAT_T diff_beta = LOAD_VECT_FLOAT(&diff_shift[sg_idx]);
            dd_vect -= (diff_beta
                               + (src_vect - v_mean) * diff_gamma
                                       * sqrt_variance)
                    / sp_size;
#endif
            dd_vect *= gamma * sqrt_variance;
            STORE_VECT_DATA(&diff_src[sg_idx], dd_vect);

        } // sg loop

        const int ic_tail_sgroups = (ic_block / SG_SIZE) % VECT_SIZE;
        const int ic_vect_sgroups = ic_block_sgroups - ic_tail_sgroups;

        // tails
        for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
            const int sg_idx = (ic_vect_sgroups + sg) * SG_SIZE;

            const float v_variance = LOAD_FLOAT_1x16(&variance[sg_idx]);
            const float sqrt_variance
                    = (float)1.0f / sqrt(v_variance + (float)eps);
            const float gamma
                    = USE_SCALE ? LOAD_FLOAT_1x16(&scaleshift[sg_idx]) : 1.0f;
            const float src_vect = LOAD_DATA_1x16(&src[sg_idx]);
            float dd_vect = LOAD_DATA_1x16(&diff_dst[sg_idx]);

#if FUSE_BN_RELU
            const char ws_vect = LOAD_CHAR_1x16(&ws[sg_idx]);
            dd_vect = select(0.0f, dd_vect, convert_int(ws_vect));
#if FUSE_BN_ADD_RELU
            STORE_DATA_1x16(&diff_src_add[sg_idx], dd_vect);
#endif
#endif

#if CALCULATE_STATS == 1
            const float v_mean = LOAD_FLOAT_1x16(&mean[sg_idx]);
            const float diff_gamma = LOAD_FLOAT_1x16(&diff_scale[sg_idx]);
            const float diff_beta = LOAD_FLOAT_1x16(&diff_shift[sg_idx]);
            dd_vect -= (diff_beta
                               + (src_vect - v_mean) * diff_gamma
                                       * sqrt_variance)
                    / sp_size;
#endif
            dd_vect *= gamma * sqrt_variance;
            STORE_DATA_1x16(&diff_src[sg_idx], dd_vect);
        }

        src += ic_size;
        diff_dst += ic_size;
        diff_src += ic_size;
#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
        diff_src_add += ic_size;
#endif
        ws += ic_size;
#endif
    } // sp loop
}
