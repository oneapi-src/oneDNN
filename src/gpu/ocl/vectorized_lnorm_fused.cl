/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#ifdef VECT_DT_N
#undef VECT_DT_N
#endif
#define VECT_DT_N VECT_SIZE_FUSED

#include "gpu/ocl/ocl_types.h"

#undef SRC_OFF
#undef DST_OFF
#define SRC_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#define STAT_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(STAT, x0, x1, x2, x3, x4, x5)

#if NDIMS > 3
#error "NDIMS > 3 not supported"
#endif

#if NDIMS == 2
#define SRC_PLAIN_OFF(n, c) SRC_OFF(n, c, 0, 0, 0, 0)
#define DST_PLAIN_OFF(n, c) DST_OFF(n, c, 0, 0, 0, 0)
#define STAT_PLAIN_OFF(n) STAT_OFF(n, 0, 0, 0, 0, 0)
#else
#define SRC_PLAIN_OFF(n, c) SRC_OFF(0, n, c, 0, 0, 0)
#define DST_PLAIN_OFF(n, c) DST_OFF(0, n, c, 0, 0, 0)
#define STAT_PLAIN_OFF(n) STAT_OFF(0, n, 0, 0, 0, 0)
#endif

#define VLEN_C (C / (SUB_GROUP_SIZE * VECT_DT_N))
#define VLEN_C_BLOCK \
    ((C / NUM_NORM_BLOCKS_FUSED) / (SUB_GROUP_SIZE * VECT_DT_N))
#define C_BLOCK (C / NUM_NORM_BLOCKS_FUSED)

#define LOAD_VECT_FLOAT(ptr) \
    AS_VECT_FLOAT_T(VECT_UINT_READ((const __global uint *)(ptr)))

#define STORE_FLOAT_SGx1(ptr, val) \
    intel_sub_group_block_write((__global uint *)(ptr), as_uint(val))
#define STORE_FLOAT_SGx2(ptr, val) \
    intel_sub_group_block_write2((__global uint *)(ptr), as_uint2(val))
#define STORE_FLOAT_SGx4(ptr, val) \
    intel_sub_group_block_write4((__global uint *)(ptr), as_uint4(val))
#define STORE_FLOAT_SGx8(ptr, val) \
    intel_sub_group_block_write8((__global uint *)(ptr), as_uint8(val))
#define STORE_VECT_FLOAT(ptr, val) CONCAT2(STORE_FLOAT_SGx, VECT_DT_N)(ptr, val)

#define STORE_LOCAL_FLOAT_SGx1(ptr, val) \
    intel_sub_group_block_write((__local uint *)(ptr), as_uint(val))
#define STORE_LOCAL_FLOAT_SGx2(ptr, val) \
    intel_sub_group_block_write2((__local uint *)(ptr), as_uint2(val))
#define STORE_LOCAL_FLOAT_SGx4(ptr, val) \
    intel_sub_group_block_write4((__local uint *)(ptr), as_uint4(val))
#define STORE_LOCAL_FLOAT_SGx8(ptr, val) \
    intel_sub_group_block_write8((__local uint *)(ptr), as_uint8(val))
#define STORE_VECT_LOCAL_FLOAT(ptr, val) \
    CONCAT2(STORE_LOCAL_FLOAT_SGx, VECT_DT_N)(ptr, val)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(a, b) ((a) + ((b)-1)) / (b)

#define MAX_CHUNKS MAX(NUM_ACROSS_BLOCKS, NUM_NORM_BLOCKS_FUSED)

NAMED_KERNEL_ATTR(FUSED)
__kernel void vectorized_lnorm_bwd_fused(__global DATA_T *src,
        __global float *mean, __global float *variance,
        __global DATA_T *diff_dst, __global float *diff_scale,
        __global float *diff_shift, __global float *scale,
        __global DATA_T *diff_src, float eps) {

    // Dispatching
    // scale/shift reduction requires:
    //      LWS = SG, number of across blocks, 1
    //      GWS = SG * num_norm_blocks, number of across blocks, 1
    // diff_gamma reduction and update part requires:
    //      LWS = SG, number of norm_blocks, 1
    //      GWS = SG, across dim * num_norm_blocks, 1
    // final dispatching as max:
    //      LWS = SG, max(n_chunks,num_norm_blocks), 1
    //      GWS = SG * num_norm_blocks, N * LWS1, 1
    // Since more work items have been started than required at each stage,
    // there are idle work items, which however must also execute the barrier
    // like any other member of the work group

#if USE_SCALE || USE_SHIFT

    const int c_uid = GWS_GET_C_fused();
    const int n_uid = GWS_GET_N_fused();

    __local float local_reduce_mem[2 * NUM_ACROSS_BLOCKS * NORM_BLOCK_FUSED];

    if (n_uid < NUM_ACROSS_BLOCKS) {
        const int c_block_off = (c_uid / SUB_GROUP_SIZE) * NORM_BLOCK_FUSED;
        const int n_start = ACROSS_BLOCK * n_uid;
        const int n_end = MIN(n_start + ACROSS_BLOCK, N);

        // local scratchpad
        __local float *tmp_diff_scale
                = local_reduce_mem + n_uid * NORM_BLOCK_FUSED;
        __local float *tmp_diff_shift
                = tmp_diff_scale + NUM_ACROSS_BLOCKS * NORM_BLOCK_FUSED;

        for (int c = 0; c < VLEN_C_BLOCK; c++) {
            const int c_idx = c * SUB_GROUP_SIZE * VECT_DT_N + c_block_off;
            const int c_slm_idx = c * SUB_GROUP_SIZE * VECT_DT_N;

            VECT_FLOAT_T diff_gamma_vect = 0;
            VECT_FLOAT_T diff_beta_vect = 0;
            for (int n_idx = n_start; n_idx < n_end; n_idx++) {
                const float mean_vect = mean[n_idx];
                const float variance_vect = variance[n_idx];
                const float inv_sqrt_variance = rsqrt(variance_vect + eps);

                const int src_off = SRC_PLAIN_OFF(n_idx, c_idx);
                const int dst_off = DST_PLAIN_OFF(n_idx, c_idx);

                const VECT_FLOAT_T src_vect = CONVERT_VECT_FLOAT_T(
                        AS_VECT_DATA_T(VECT_BLOCK_READ((const __global
                                        BLOCK_DATA_T *)(&src[src_off]))));
                const VECT_FLOAT_T diff_dst_vect = CONVERT_VECT_FLOAT_T(
                        AS_VECT_DATA_T(VECT_BLOCK_READ((const __global
                                        BLOCK_DATA_T *)(&diff_dst[dst_off]))));

                diff_gamma_vect += (src_vect - mean_vect) * diff_dst_vect
                        * inv_sqrt_variance;
                diff_beta_vect += diff_dst_vect;
            }

            if (USE_SCALE)
                STORE_VECT_LOCAL_FLOAT(
                        &tmp_diff_scale[c_slm_idx], diff_gamma_vect);
            if (USE_SHIFT)
                STORE_VECT_LOCAL_FLOAT(
                        &tmp_diff_shift[c_slm_idx], diff_beta_vect);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (n_uid == 0) {
            for (int c = 0; c < VLEN_C_BLOCK; c++) {
                const int c_idx = c * SUB_GROUP_SIZE * VECT_DT_N + c_block_off;
                const int c_slm_idx = c * SUB_GROUP_SIZE * VECT_DT_N;

                VECT_FLOAT_T diff_gamma_vect = 0;
                VECT_FLOAT_T diff_beta_vect = 0;
                for (int n_idx = 0; n_idx < NUM_ACROSS_BLOCKS; n_idx++) {
                    if (USE_SCALE) {
                        diff_gamma_vect += AS_VECT_FLOAT_T(VECT_UINT_READ(
                                (const __local uint *)&tmp_diff_scale[c_slm_idx
                                        + n_idx * NORM_BLOCK_FUSED]));
                    }
                    if (USE_SHIFT) {
                        diff_beta_vect += AS_VECT_FLOAT_T(VECT_UINT_READ(
                                (const __local uint *)&tmp_diff_shift[c_slm_idx
                                        + n_idx * NORM_BLOCK_FUSED]));
                    }
                }

                if (USE_SCALE)
                    STORE_VECT_FLOAT(&diff_scale[c_idx], diff_gamma_vect);
                if (USE_SHIFT)
                    STORE_VECT_FLOAT(&diff_shift[c_idx], diff_beta_vect);
            }
        }
    } else if (n_uid < MAX_CHUNKS) {
        // idle wi must perform barrier too
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif // USE_SCALE || USE_SHIFT

#define SRC_BUF_SIZE (NORM_BLOCK_FUSED / (SUB_GROUP_SIZE * VECT_DT_N))
#define GAMMA_SLM_SIZE (SUB_GROUP_SIZE * NUM_NORM_BLOCKS_FUSED)

    if (GWS_GET_C_fused() >= SUB_GROUP_SIZE) return;

    const int c_uni_id = GWS_GET_C_fused();
    const int n_uni_id = GWS_GET_N_fused() / MAX_CHUNKS;
    const int c_block_id = GWS_GET_N_fused() % MAX_CHUNKS;
    const int simd_id = get_sub_group_local_id();
    const int local_id = get_local_id(1);

    float dd_gamma = 0, dd_gamma_x = 0;

#if CALCULATE_STATS
    __local float dd_gamma_slm[GAMMA_SLM_SIZE];
    __local float dd_gamma_x_slm[GAMMA_SLM_SIZE];
#endif

    if (local_id >= NUM_NORM_BLOCKS_FUSED) {
#if CALCULATE_STATS
        // idle wi must perform barrier too
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        return;
    }

    const float rC = 1.0 / C;
    const int s_off = STAT_PLAIN_OFF(n_uni_id);
    const int c_block_off = c_block_id * NORM_BLOCK_FUSED;

    VECT_FLOAT_T dd_gamma_vect = 0, dd_gamma_x_vect = 0;
    VECT_FLOAT_T v_src[SRC_BUF_SIZE];
    VECT_FLOAT_T v_diff_dst[SRC_BUF_SIZE];

    const float mean_val = mean[s_off];
    const float inv_sqrt_variance = rsqrt(variance[s_off] + eps);

    for (int c = 0; c < SRC_BUF_SIZE; c++) {
        const int c_idx = c * SUB_GROUP_SIZE * VECT_DT_N + c_block_off;
        const int src_off = SRC_PLAIN_OFF(n_uni_id, c_idx);
        const int dst_off = DST_PLAIN_OFF(n_uni_id, c_idx);

        v_src[c] = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off])));
        v_diff_dst[c] = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                (const __global BLOCK_DATA_T *)&diff_dst[dst_off])));
    }

#if CALCULATE_STATS
    for (int c = 0; c < SRC_BUF_SIZE; c++) {
        const int c_idx = c * SUB_GROUP_SIZE * VECT_DT_N + c_block_off;
        VECT_FLOAT_T gamma = 1.0f;
        if (scale) {
            gamma = AS_VECT_FLOAT_T(
                    VECT_UINT_READ((const __global uint *)&scale[c_idx]));
        }
        const VECT_FLOAT_T src_vect = v_src[c];
        const VECT_FLOAT_T dst_vect = v_diff_dst[c];
        dd_gamma_vect += dst_vect * gamma;
        dd_gamma_x_vect += dst_vect * gamma * (src_vect - mean_val);
    }

    const int slm_off = SUB_GROUP_SIZE * local_id + simd_id;
#if VECT_DT_N == 1
    dd_gamma_slm[slm_off] = dd_gamma_vect;
    dd_gamma_x_slm[slm_off] = dd_gamma_x_vect;
#else
    dd_gamma_slm[slm_off] = 0;
    dd_gamma_x_slm[slm_off] = 0;
    for (int i = 0; i < VECT_DT_N; ++i) {
        dd_gamma_slm[slm_off] += dd_gamma_vect[i];
        dd_gamma_x_slm[slm_off] += dd_gamma_x_vect[i];
    }
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < GAMMA_SLM_SIZE; i++) {
        dd_gamma += dd_gamma_slm[i];
        dd_gamma_x += dd_gamma_x_slm[i];
    }
    dd_gamma_x *= inv_sqrt_variance;
#endif // CALCULATE_STATS

    for (int c = 0; c < SRC_BUF_SIZE; c++) {
        VECT_FLOAT_T gamma = 1.0f;
        if (scale) {
            gamma = AS_VECT_FLOAT_T(VECT_UINT_READ(
                    (const __global uint *)&scale[c * SUB_GROUP_SIZE * VECT_DT_N
                            + c_block_off]));
        }
        const VECT_FLOAT_T src_vect = v_src[c];
        VECT_FLOAT_T v_diff_src_vect = v_diff_dst[c];
        v_diff_src_vect *= gamma;

#if CALCULATE_STATS
        v_diff_src_vect -= dd_gamma * rC
                + (src_vect - mean_val) * dd_gamma_x * inv_sqrt_variance * rC;
#endif
        v_diff_src_vect *= inv_sqrt_variance;

        const int c_idx = c * SUB_GROUP_SIZE * VECT_DT_N + c_block_off;
        const int src_off = SRC_PLAIN_OFF(n_uni_id, c_idx);

        VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)&diff_src[src_off],
                AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(v_diff_src_vect)));
    }
}
