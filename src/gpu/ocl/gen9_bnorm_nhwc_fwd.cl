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

// FWD kernels for regular and 1-pass (under USE_STATS_ONE_PASS) bnorm
// algorithms that are specially optimized for NHWC layout
// (NHWC_OPTIMIZED definition).
// These kernels not supported IC tail processing.
// For both algorithms, two types of reduction are implemented:
// 1) Reduction over scratchpad (reduce_temp) with SLM use, implemented by
//    gen9_reduce_* kernels.
// 2) Atomics-based reduction with SLM use (FUSED_ATOMICS_REDUCTION definition),
//    implemented as part of calc kernels, see gen9_*_fused_reduction()
//    functions in gen9_bnorm.h. This reduction implementation requires
//    zeroing and finalization steps, see gen9_fused_reduce_* kernels
//    in gen9_bnorm_reduce.cl

#if USE_STATS_ONE_PASS

// Calculates partial sums of values and squares-of-values per channel
NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean_var_nhwc(__global DATA_T *src,
        __global ACCUM_DATA_T *reduce_temp,
        volatile __global atomic_float *mean,
        volatile __global atomic_float *variance) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();

    const int ic_block_offset = (c / SG_SIZE) * IC_BLOCK;
    const int group_c_offset
            = REDUCE_STAT_NBLOCKS * ic_block_offset + sp_block_idx * SG_SIZE;
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
            VECT_FLOAT_T s_vect
                    = LOAD_VECT_DATA(&src[sg * SG_SIZE * VECT_SIZE]);

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
            float s_tail
                    = LOAD_DATA_1x16(&src[(IC_VECT_SGROUPS + sg) * SG_SIZE]);
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
        const int reduce_off
                = group_c_offset + sg * SG_SIZE * REDUCE_STAT_NBLOCKS;
        STORE_FLOAT_1x16(&reduce_temp[reduce_off], sum[sg].s0);
        STORE_FLOAT_1x16(&reduce_temp[ver_offs + reduce_off], sum_sq[sg].s0);
    }
#endif
}

#else // USE_STATS_ONE_PASS

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean_nhwc(__global DATA_T *src,
        __global float *reduce_temp, volatile __global atomic_float *mean) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int ic_block_offset = (c / SG_SIZE) * IC_BLOCK;
    const int group_c_offset
            = REDUCE_STAT_NBLOCKS * ic_block_offset + sp_block_idx * SG_SIZE;

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
            VECT_FLOAT_T s_vect
                    = LOAD_VECT_DATA(&src[sg * SG_SIZE * VECT_SIZE]);
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
            float s_tail
                    = LOAD_DATA_1x16(&src[(IC_VECT_SGROUPS + sg) * SG_SIZE]);
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
        const int reduce_off
                = group_c_offset + sg * SG_SIZE * REDUCE_STAT_NBLOCKS;
        STORE_FLOAT_1x16(&reduce_temp[reduce_off], v_mean[sg]);
    }
#endif
}

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_variance_nhwc(__global DATA_T *src,
        __global float *mean, __global float *reduce_temp,
        volatile __global atomic_float *variance) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int ic_block_offset = (c / SG_SIZE) * IC_BLOCK;
    const int group_c_offset
            = REDUCE_STAT_NBLOCKS * ic_block_offset + sp_block_idx * SG_SIZE;

    reduce_temp += REDUCE_STAT_NBLOCKS * PADDED_IC;
    mean += ic_block_offset;
    const int src_off = ic_block_offset + sp_block_idx * STAT_SP_BLOCK * IC;
    src += src_off;

    float v_mean[IC_BLOCK_SGROUPS];
    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        v_mean[sg] = as_float(intel_sub_group_block_read(
                (const __global uint *)(&mean[(sg * SG_SIZE)])));
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
            VECT_FLOAT_T s_vect
                    = LOAD_VECT_DATA(&src[sg * SG_SIZE * VECT_SIZE]);

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
            float s_tail
                    = LOAD_DATA_1x16(&src[(IC_VECT_SGROUPS + sg) * SG_SIZE]);
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
        const int reduce_off
                = group_c_offset + sg * SG_SIZE * REDUCE_STAT_NBLOCKS;
        STORE_FLOAT_1x16(&reduce_temp[reduce_off], v_var[sg]);
    }
#endif
}

#endif // USE_STATS_ONE_PASS

KERNEL_ATTR
__kernel void gen9_bnorm_fwd_nhwc(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst,
        __global float *scaleshift, __global float *shift, __global char *ws,
        float eps, __global DATA_T *src_add, float relu_alpha) {

    const int n = GWS_GET_MB();
    const int c = GWS_GET_IC();
    const int sp = GWS_GET_SP() * UPDATE_SP_BLOCK;

    const int ic_block_offset = (c / SG_SIZE) * IC_BLOCK;
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
        const int sg_idx = sg * SG_SIZE * VECT_SIZE;
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
        const int sg_idx = (IC_VECT_SGROUPS + sg) * SG_SIZE;
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

#if HAS_UPDATE_SP_BLOCK_TAIL
    for (int sp_idx = 0; sp_idx < min(UPDATE_SP_BLOCK, SP - sp);
            sp_idx += UPDATE_SP_UNROLL) {
#else
    for (int sp_idx = 0; sp_idx < UPDATE_SP_BLOCK; sp_idx += UPDATE_SP_UNROLL) {
#endif
        // vectorized part
        for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
            const int sg_idx = sg * SG_SIZE * VECT_SIZE;

            VECT_FLOAT_T s_vect[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                s_vect[i] = LOAD_VECT_DATA(&src[sg_idx + IC * i]);
            }

            VECT_FLOAT_T d_vect[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                d_vect[i] = fma(
                        s_vect[i] - v_mean[sg], sqrt_variance[sg], sv[sg]);
            }

#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
            VECT_FLOAT_T s_add_vect[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                s_add_vect[i] = LOAD_VECT_DATA(&src_add[sg_idx + IC * i]);
                d_vect[i] += s_add_vect[i];
            }
#endif
            VECT_INT_T ws_vect[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                ws_vect[i] = isgreater(d_vect[i], (VECT_FLOAT_T)0.0f);
                d_vect[i] = select((VECT_FLOAT_T)0.0f, d_vect[i], ws_vect[i]);
            }
#if IS_TRAINING
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                STORE_VECT_CHAR(&ws[sg_idx + IC * i], ws_vect[i]);
            }

#endif // IS_TRAINING
#endif // FUSE_BN_RELU

#if WITH_RELU && WITH_LEAKY_RELU
            VECT_INT_T l_vect[UPDATE_SP_UNROLL];
#endif //WITH_RELU && WITH_LEAKY_RELU
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
#if WITH_RELU
#if WITH_LEAKY_RELU
                l_vect[i] = isless(d_vect[i], 0.0f);
                d_vect[i]
                        = select(d_vect[i], d_vect[i] * relu_alpha, l_vect[i]);
#else
                d_vect[i] = max(d_vect[i], (VECT_FLOAT_T)0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU
                STORE_VECT_DATA(&dst[sg_idx + IC * i], d_vect[i]);
            }
        }

#if HAS_IC_VECT_TAIL
        // tails
        for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
            const int sg_idx = (IC_VECT_SGROUPS + sg) * SG_SIZE;
            float s_tail[UPDATE_SP_UNROLL], d_tail[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                s_tail[i] = LOAD_DATA_1x16(&src[sg_idx + IC * i]);
                d_tail[i] = fma(s_tail[i] - v_mean_tail[sg],
                        sqrt_variance_tail[sg], sv_tail[sg]);
            }
#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
            float s_add_tail[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                s_add_tail[i] = LOAD_DATA_1x16(&src_add[sg_idx + IC * i]);
                d_tail[i] += s_add_tail[i];
            }
#endif
            int ws_tail[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                ws_tail[i] = isgreater(d_tail[i], 0.0f);
                d_tail[i] = select(0.0f, d_tail[i], ws_tail[i]);
            }
#if IS_TRAINING
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                STORE_CHAR_1x16(&ws[sg_idx + IC * i], convert_char(ws_tail[i]));
            }
#endif // IS_TRAINING
#endif // FUSE_BN_RELU

#if WITH_RELU && WITH_LEAKY_RELU
            int l_tail[UPDATE_SP_UNROLL];
#endif //WITH_RELU && WITH_LEAKY_RELU
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
#if WITH_RELU
#if WITH_LEAKY_RELU
                l_tail[i] = isless(d_tail[i], 0.0f);
                d_tail[i]
                        = select(d_tail[i], d_tail[i] * relu_alpha, l_tail[i]);
#else
                d_tail[i] = max(d_tail[i], 0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU
                STORE_DATA_1x16(&dst[sg_idx + IC * i], d_tail[i]);
            }
        }
#endif
        src += IC * UPDATE_SP_UNROLL;
#if FUSE_BN_ADD_RELU
        src_add += IC * UPDATE_SP_UNROLL;
#endif
        dst += IC * UPDATE_SP_UNROLL;
#if FUSE_BN_RELU && IS_TRAINING
        ws += IC * UPDATE_SP_UNROLL;
#endif
    }
}
