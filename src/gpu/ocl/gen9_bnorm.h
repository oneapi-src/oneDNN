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
#ifndef GPU_OCL_GEN9_BNORM_H
#define GPU_OCL_GEN9_BNORM_H

#define VECT_DT_N VECT_SIZE
#include "gpu/ocl/ocl_types.h"

#define IS_IC_EQ_8 (IC == 8)
#define HAS_IC_TAIL (IC != PADDED_IC)
#define HAS_STAT_SP_BLOCK_TAIL (SP % STAT_SP_BLOCK)
#define HAS_UPDATE_SP_BLOCK_TAIL (SP % UPDATE_SP_BLOCK)

#if HAS_UPDATE_SP_BLOCK_TAIL % UPDATE_SP_UNROLL
#error "UPDATE_SP_UNROLL value not expected"
#endif

#if NHWC_OPTIMIZED
#if HAS_IC_TAIL
#error IC tail processing not supported
#endif
#else // NHWC_OPTIMIZED
#if HAS_IC_TAIL && !USE_NHWC
#error IC tail processing not supported
#endif
#define HAS_STAT_SP_TAIL (STAT_SP_TAIL != STAT_SP_NBLOCKS)
#define HAS_SP_TAIL (SP != SP_TAIL)
#endif // NHWC_OPTIMIZED

#define IC_BLOCK_SGROUPS (IC_BLOCK / SG_SIZE)
#define IC_TAIL_SGROUPS (IC_BLOCK_SGROUPS % VECT_SIZE)
#define IC_VECT_SGROUPS (IC_BLOCK_SGROUPS - IC_TAIL_SGROUPS)
#define HAS_IC_VECT_TAIL (IC_TAIL_SGROUPS > 0)

#define LOAD_FLOAT_1x16(ptr) \
    as_float(intel_sub_group_block_read((const __global uint *)(ptr)))

#define LOAD_UINT_1x16(ptr) \
    as_uint(intel_sub_group_block_read((const __global uint *)(ptr)))

#define LOAD_UINT_8x16(ptr) \
    convert_uint8(as_uint8( \
            intel_sub_group_block_read8((const __global uint *)(ptr))))

#define LOAD_CHAR_1x16(ptr) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(ptr)))

#define LOAD_CHAR_8x16(ptr) \
    convert_char8(as_char8( \
            intel_sub_group_block_read_uc8((const __global uchar *)(ptr))))

#define LOAD_DATA_1x16(ptr) \
    CONVERT_FLOAT_T(AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)(ptr))))

#define LOAD_DATA_8x16(ptr) \
    CONVERT_FLOAT8_T( \
            AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)(ptr))))

#define LOAD_VECT_DATA(ptr) \
    CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T( \
            VECT_BLOCK_READ((const __global BLOCK_DATA_T *)(ptr))))

#define LOAD_VECT_CHAR(ptr) \
    CONVERT_VECT_CHAR_T( \
            AS_VECT_CHAR_T(VECT_UCHAR_READ((const __global uchar *)(ptr))))

#define LOAD_VECT_FLOAT(ptr) \
    AS_VECT_FLOAT_T(VECT_UINT_READ((const __global uint *)(ptr)))

#define STORE_DATA_1x16(ptr, val) \
    BLOCK_WRITE((__global BLOCK_DATA_T *)(ptr), \
            AS_BLOCK_DATA_T(CONVERT_DATA_T(val)))

#define STORE_DATA_8x16(ptr, val) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)ptr, \
            AS_BLOCK_DATA8_T(CONVERT_DATA8_T(val)))

#define STORE_VECT_DATA(ptr, val) \
    VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)(ptr), \
            AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(val)))

#define STORE_FLOAT_1x16(ptr, val) \
    intel_sub_group_block_write((__global uint *)(ptr), as_uint(val))

#define STORE_FLOAT_8x16(ptr, val) \
    intel_sub_group_block_write8((__global uint *)(ptr), as_uint8(val))

#define STORE_CHAR_1x16(ptr, val) \
    intel_sub_group_block_write_uc((__global uchar *)(ptr), as_uchar(val))

#define STORE_CHAR_8x16(ptr, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(ptr), as_uchar8(val))

#define STORE_VECT_CHAR(ptr, val) \
    VECT_UCHAR_WRITE((__global uchar *)(ptr), \
            AS_VECT_UCHAR_T(CONVERT_VECT_CHAR_T(val)))

#if HAS_IC_TAIL
#define MAYBE_LAST_IC_LOAD_FLOAT_1x16(ptr, idx) \
    (is_last_ic_block ? (simd_id < 8 ? ptr[(idx) + simd_id] : 0.0f) \
                      : as_float(intel_sub_group_block_read( \
                              (const __global uint *)(&ptr[(idx)]))))
#else
#define MAYBE_LAST_IC_LOAD_FLOAT_1x16(ptr, idx) LOAD_FLOAT_1x16(&ptr[(idx)])
#endif

#if USE_NHWC
#define IC_BLOCK_STRIDE IC
#else
#define IC_BLOCK_STRIDE 16
#endif

#if NHWC_OPTIMIZED
#define REDUCE_NUM_SGROUPS IC_BLOCK_SGROUPS
#else
#define REDUCE_NUM_SGROUPS 1
#endif

#define CALC_SLM_LINE_SIZE (REDUCE_NUM_SGROUPS * GWS_LWS0_CALC)
#define CALC_SLM_SIZE (CALC_SLM_LINE_SIZE * GWS_LWS1_CALC * GWS_LWS2_CALC)

#if IS_FWD
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
#endif // IS_FWD

#if FUSED_ATOMICS_REDUCTION

#if NHWC_OPTIMIZED
#if VECT_SIZE > 1
#define GET_SCALAR_VAL(v, idx) v[idx / VECT_SIZE][idx % VECT_SIZE]
#else
#define GET_SCALAR_VAL(v, idx) v[idx]
#endif
#else
#define GET_SCALAR_VAL(v, idx) v[idx]
#endif

// Atomics-based reduction with SLM use, fused with calculating kernels.
#if IS_FWD
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
    const int sg_group_id = get_local_id(0) / SG_SIZE;
    const int local_id = get_local_id(1);

    if (local_id > 0) {
        for (int sg = 0; sg < REDUCE_NUM_SGROUPS; ++sg) {
            const int slm_offset = CALC_SLM_LINE_SIZE * local_id
                    + REDUCE_NUM_SGROUPS * SG_SIZE * sg_group_id + sg * SG_SIZE
                    + simd_id;
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
                        + REDUCE_NUM_SGROUPS * SG_SIZE * sg_group_id
                        + sg * SG_SIZE + simd_id;
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
            const int offset = dst_offset + sg * SG_SIZE + simd_id;
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
#endif // IS_FWD

#if IS_BWD
void gen9_calc_fused_reduction(volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift, int dst_offset,
#if NHWC_OPTIMIZED
        VECT_FLOAT_T *diff_gamma, VECT_FLOAT_T *diff_beta,
#else
        float *diff_gamma, float *diff_beta,
#endif
        float *diff_gamma_tail, float *diff_beta_tail,
        __local float *local_gamma, __local float *local_beta) {

    const int simd_id = get_sub_group_local_id();
    const int group_size = GWS_LWS1_CALC * GWS_LWS2_CALC;
    const int sg_group_id = get_local_id(0) / SG_SIZE;
    const int local_id = get_local_id(1);

    for (int sg = 0; sg < REDUCE_NUM_SGROUPS; ++sg) {
        const int slm_offset = CALC_SLM_LINE_SIZE * local_id
                + REDUCE_NUM_SGROUPS * SG_SIZE * sg_group_id + sg * SG_SIZE
                + simd_id;
#if HAS_IC_VECT_TAIL && NHWC_OPTIMIZED
        if (sg >= IC_VECT_SGROUPS) {
            local_gamma[slm_offset] = diff_gamma_tail[sg - IC_VECT_SGROUPS];
            local_beta[slm_offset] = diff_beta_tail[sg - IC_VECT_SGROUPS];
        } else
#endif
        {
            local_gamma[slm_offset] = GET_SCALAR_VAL(diff_gamma, sg);
            local_beta[slm_offset] = GET_SCALAR_VAL(diff_beta, sg);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        for (int sg = 0; sg < REDUCE_NUM_SGROUPS; ++sg) {
            float d_gamma = 0.f;
            float d_beta = 0.f;

            for (int gr_id = 0; gr_id < group_size; ++gr_id) {
                const int off_local = CALC_SLM_LINE_SIZE * gr_id
                        + REDUCE_NUM_SGROUPS * SG_SIZE * sg_group_id
                        + sg * SG_SIZE + simd_id;
                d_gamma += local_gamma[off_local];
                d_beta += local_beta[off_local];
            }
            const int offset = dst_offset + sg * SG_SIZE + simd_id;
#if HAS_IC_TAIL
            if (offset < IC)
#endif
            {
                atomic_add_global(&diff_scale[offset], d_gamma);
#if DIFF_SHIFT == 1
                atomic_add_global(&diff_shift[offset], d_beta);
#else
                atomic_add_global(
                        &diff_shift[IC + IC * REDUCE_STAT_NBLOCKS + offset],
                        d_beta);
#endif
            }
        }
    }
    return;
}
#endif // IS_BWD
#endif // FUSED_ATOMICS_REDUCTION
#endif // GPU_OCL_GEN9_BNORM_H
