/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#define VECT_DT_N VECT_SIZE

#include "gpu/ocl/ocl_types.h"

// For regular and 1-pass (under USE_STATS_ONE_PASS) bnorm algorithms
// there are two sets of kernels:
// 1) The kernels that support both blocked and NHWC layouts (USE_NHWC).
//    These kernels perform IC tail processing for NHWC and for ic % 8 == 0
//    cases only.
// 2) Specially optimized for NHWC kernels (under NHWC_OPTIMIZED).
//    Not supported IC tail processing.

#define IS_IC_EQ_8 (IC == 8)
#define HAS_IC_TAIL (IC != IC16)

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

#define IC_BLOCK_SGROUPS (IC_BLOCK / 16)
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

#if IS_FWD

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

// Calculates partial sums of values and squares-of-values per channel
#if NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean_var(
        __global DATA_T *src, __global ACCUM_DATA_T *reduce_temp) {

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

    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
        if (sp_block_idx * STAT_SP_BLOCK + sp >= SP) break;
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
        for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
            const int sg_idx = IC_VECT_SGROUPS + sg;
            float s_tail = LOAD_DATA_1x16(&src[(IC_VECT_SGROUPS + sg) * 16]);
            sum[sg_idx] = summation(s_tail, sum[sg_idx]);
            sum_sq[sg_idx] = summation(s_tail * s_tail, sum_sq[sg_idx]);
        }
#endif
        src += IC;
    }

    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        const int reduce_off = group_c_offset + sg * 16 * REDUCE_STAT_NBLOCKS;
        STORE_FLOAT_1x16(&reduce_temp[reduce_off], sum[sg].s0);
        STORE_FLOAT_1x16(&reduce_temp[ver_offs + reduce_off], sum_sq[sg].s0);
    }
}

#else // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean_var(
        __global DATA_T *src, __global ACCUM_DATA_T *reduce_temp) {

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

    STORE_FLOAT_1x16(&reduce_temp[group_c_offset + mb_sp_idx * 16], sum.s0);
    STORE_FLOAT_1x16(&reduce_temp[ver_offs + group_c_offset + mb_sp_idx * 16],
            sum_sq.s0);
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
#endif // USE_STATS_ONE_PASS

#if NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean(
        __global DATA_T *src, __global float *reduce_temp) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int ic_block_offset = (c / 16) * IC_BLOCK;
    const int group_c_offset
            = REDUCE_STAT_NBLOCKS * ic_block_offset + sp_block_idx * 16;

    const int src_off = ic_block_offset + sp_block_idx * STAT_SP_BLOCK * IC;
    src += src_off;

    float v_mean[IC_BLOCK_SGROUPS] = {0.0f};

    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
        if (sp_block_idx * STAT_SP_BLOCK + sp >= SP) break;
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
    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        const int reduce_off = group_c_offset + sg * 16 * REDUCE_STAT_NBLOCKS;
        STORE_FLOAT_1x16(&reduce_temp[reduce_off], v_mean[sg]);
    }
}

#else // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean(
        __global DATA_T *src, __global float *reduce_temp) {

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

    // reduce_temp is padded to IC16, no OOB writes
    STORE_FLOAT_1x16(&reduce_temp[group_c_offset + mb_sp_idx * 16], v_mean);
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
        __global float *reduce_temp) {

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
    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
        if (sp_block_idx * STAT_SP_BLOCK + sp >= SP) break;
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

    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        const int reduce_off = group_c_offset + sg * 16 * REDUCE_STAT_NBLOCKS;
        STORE_FLOAT_1x16(&reduce_temp[reduce_off], v_var[sg]);
    }
}

#else // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_variance(__global DATA_T *src, __global float *mean,
        __global float *reduce_temp) {
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
    // reduce_temp is padded to IC16, no OOB writes
    STORE_FLOAT_1x16(&reduce_temp[group_c_offset + mb_sp_idx * 16], v_var);
}

#endif // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_variance(
        __global float *reduce_temp, __global float *variance) {
    __local float local_sum[16 * REDUCE_IC_SUB_GROUPS];
    gen9_reduce_common(
            reduce_temp + REDUCE_STAT_NBLOCKS * IC16, local_sum, variance);
}

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

    for (int sp_idx = 0; sp_idx < STAT_SP_BLOCK; sp_idx++) {
        if (sp_idx + sp >= SP) break;

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
#endif // IS_FWD

#if IS_BWD == 1

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
        __global float *temp_reduce) {

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
#endif

    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
        if (sp_block_idx * STAT_SP_BLOCK + sp >= SP) break;

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

    // any reduction is not needed due to we run only 1 raw  ???

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
}

#else // NHWC_OPTIMIZED

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calculate_stats(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global char *ws,
        __global float *temp_reduce) {

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

    // scratchpad layout:
    // IC16 - diff_gamma reduction, wrote by gen9_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * IC16 - diff_gamma stats calculated by this kernel
    // IC16 - diff_beta reduction, wrote by gen9_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * IC16 - diff_beta stats calculated by this kernel
    STORE_FLOAT_1x16(&temp_reduce[IC16 + mb_sp_idx * 16], diff_gamma[0]);
    STORE_FLOAT_1x16(&temp_reduce[2 * IC16 + REDUCE_STAT_NBLOCKS * IC16
                             + mb_sp_idx * 16],
            diff_beta[0]);
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

    for (int sp = 0; sp < STAT_SP_BLOCK; sp++) {
        if (sp_block_idx * STAT_SP_BLOCK + sp >= SP) break;
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
#endif // IS_BWD
