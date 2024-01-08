/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/ocl/dispatch.h"
#include "gpu/ocl/ocl_types.h"

#undef SRC_OFF
#undef DST_OFF
#define SRC_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#define STAT_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(STAT, x0, x1, x2, x3, x4, x5)

#define VLEN_C (C / (SUB_GROUP_SIZE * VECT_DT_N))
#define VLEN_C_BLOCK ((C / NUM_NORM_BLOCKS) / (SUB_GROUP_SIZE * VECT_DT_N))
#define C_BLOCK (C / NUM_NORM_BLOCKS)

#if (GWS_LWS0_DEFAULT * GWS_LWS1_DEFAULT * GWS_LWS2_DEFAULT) == GWS_SGS_DEFAULT
#define GROUP_REDUCE_ADD sub_group_reduce_add
#else
#define GROUP_REDUCE_ADD work_group_reduce_add
#endif

#define LOAD_VECT_FLOAT(ptr) \
    AS_VECT_FLOAT_T(VECT_UINT_READ((const __global uint *)(ptr)))

#if IS_FWD
#if VECT_DT_N == 1
#define CALC_V_STAT(v, acc) v = acc;
#else
#define CALC_V_STAT(v, acc) \
    v = 0; \
    for (int i = 0; i < VECT_DT_N; ++i) { \
        v += acc[i]; \
    }
#endif

#define STORE_VECT_DATA(ptr, val) \
    VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)(ptr), \
            AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(val)))

KERNEL_ATTR
__kernel void vectorized_lnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst,
        __global WEI_DATA_T *scale, __global WEI_DATA_T *shift, float eps,
        __global float *src_scale, __global float *dst_scale) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

    float v_mean = CALCULATE_STATS ? 0 : mean[s_off];
    float v_variance = CALCULATE_STATS ? 0 : variance[s_off];

    const float rC = 1.f / C;
    const int c_block_off = (x[NDIMS - 1] / SUB_GROUP_SIZE) * NORM_BLOCK;
#if USE_SRC_BUFFER
    // Key feature of this version is single reading src data, keeping it in
    // v_src buffer and reusing this buffer for stats calculation.
    // Targeted for PVC+.
    VECT_FLOAT_T v_src[VLEN_C_BLOCK];
    for (int c = 0; c < VLEN_C_BLOCK; c++) {
        x[NDIMS - 1] = c * SUB_GROUP_SIZE * VECT_DT_N + c_block_off;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        v_src[c] = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off])));
    }
#if CALCULATE_STATS
    VECT_FLOAT_T v_acc = 0;
    for (int c = 0; c < VLEN_C_BLOCK; c++) {
        v_acc += v_src[c];
    }
    CALC_V_STAT(v_mean, v_acc);
    v_mean = GROUP_REDUCE_ADD(v_mean) * rC;
    v_acc = 0;
    VECT_FLOAT_T m = 0;
    for (int c = 0; c < VLEN_C_BLOCK; c++) {
        m = v_src[c] - v_mean;
        v_acc += m * m;
    }
    CALC_V_STAT(v_variance, v_acc);
    v_variance = GROUP_REDUCE_ADD(v_variance) * rC;
#endif // CALCULATE_STATS
    const float rsqrt_variance = rsqrt(v_variance + eps);

    for (int c = 0; c < VLEN_C_BLOCK; c++) {
        const VECT_FLOAT_T sm
#if USE_SCALE
                = LOAD_VECT_WEI(
                          &scale[c * SUB_GROUP_SIZE * VECT_DT_N + c_block_off])
                * rsqrt_variance;
#else
                = rsqrt_variance;
#endif
        const VECT_FLOAT_T sv
#if USE_SHIFT
                = LOAD_VECT_WEI(
                        &shift[c * SUB_GROUP_SIZE * VECT_DT_N + c_block_off]);
#else
                = 0.0f;
#endif
        x[NDIMS - 1] = c * SUB_GROUP_SIZE * VECT_DT_N + c_block_off;
        int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        VECT_FLOAT_T v_dst = sm * (v_src[c] - v_mean) + sv;

#if WITH_SRC_SCALES
        v_dst *= src_scale[0];
#endif
#if WITH_DST_SCALES
        v_dst /= dst_scale[0];
#endif

        STORE_VECT_DATA(&dst[dst_off], v_dst);
    }
#else // USE_SRC_BUFFER
    // Key feature of this version is only vectorized block read/write
    // without using GRF src buffer.
    // Targeted for ATSM/DG2.
#if CALCULATE_STATS
    VECT_FLOAT_T v_acc = 0;
    for (int c = 0; c < C_BLOCK; c += SUB_GROUP_SIZE * VECT_DT_N) {
        x[NDIMS - 1] = c + c_block_off;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        v_acc += CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off])));
    }
    CALC_V_STAT(v_mean, v_acc);
    v_mean = GROUP_REDUCE_ADD(v_mean) * rC;
    v_acc = 0;
    VECT_FLOAT_T m = 0;
    for (int c = 0; c < C_BLOCK; c += SUB_GROUP_SIZE * VECT_DT_N) {
        x[NDIMS - 1] = c + c_block_off;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        m = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off])));
        m -= v_mean;
        v_acc += m * m;
    }
    CALC_V_STAT(v_variance, v_acc);
    v_variance = GROUP_REDUCE_ADD(v_variance) * rC;
#endif // CALCULATE_STATS
    const float r_sqrt_variance = rsqrt(v_variance + eps);

    for (int c = 0; c < C_BLOCK; c += SUB_GROUP_SIZE * VECT_DT_N) {
        const VECT_FLOAT_T sm
#if USE_SCALE
                = LOAD_VECT_WEI(&scale[c + c_block_off]) * r_sqrt_variance;
#else
                = r_sqrt_variance;
#endif
        const VECT_FLOAT_T sv
#if USE_SHIFT
                = LOAD_VECT_WEI(&shift[c + c_block_off]);
#else
                = 0.0f;
#endif
        x[NDIMS - 1] = c + c_block_off;
        const int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        const int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        const VECT_FLOAT_T v_src = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off])));
        VECT_FLOAT_T v_dst = sm * (v_src - v_mean) + sv;

#if WITH_SRC_SCALES
        v_dst *= src_scale[0];
#endif
#if WITH_DST_SCALES
        v_dst /= dst_scale[0];
#endif

        STORE_VECT_DATA(&dst[dst_off], v_dst);
    }
#endif // USE_SRC_BUFFER

#if CALCULATE_STATS && SAVE_STATS
    if (get_local_linear_id() == 0) {
        mean[s_off] = v_mean;
        variance[s_off] = v_variance;
    }
#endif
}
#endif // IS_FWD

#if IS_BWD

#define STORE_FLOAT_SGx1(ptr, val) \
    intel_sub_group_block_write((__global uint *)(ptr), as_uint(val))
#define STORE_FLOAT_SGx2(ptr, val) \
    intel_sub_group_block_write2((__global uint *)(ptr), as_uint2(val))
#define STORE_FLOAT_SGx4(ptr, val) \
    intel_sub_group_block_write4((__global uint *)(ptr), as_uint4(val))
#define STORE_FLOAT_SGx8(ptr, val) \
    intel_sub_group_block_write8((__global uint *)(ptr), as_uint8(val))

#define STORE_VECT_FLOAT(ptr, val) CONCAT2(STORE_FLOAT_SGx, VECT_DT_N)(ptr, val)

#if USE_SCALE || USE_SHIFT

NAMED_KERNEL_ATTR(SCALESHIFT)
__kernel void vectorized_lnorm_bwd_scaleshift(__global DATA_T *src,
        __global float *mean, __global float *variance,
        __global DATA_T *diff_dst, __global float *diff_scale,
        __global float *diff_shift, float eps) {

    const int c = GWS_GET_C() * VECT_DT_N;
    const int n_chunk_idx = GWS_GET_N();
    const int n_start = n_chunk_idx * N_CHUNK_SIZE;
    const int n_end = min(n_start + N_CHUNK_SIZE, N);

    // diff_scale and diff_shift use the same tensor in scratchpad
    const int shift_off = N_CHUNKS * C;
    diff_shift += shift_off;

    VECT_FLOAT_T diff_gamma_vect = 0;
    VECT_FLOAT_T diff_beta_vect = 0;

    for (int n_off = n_start; n_off < n_end; n_off++) {
        const float mean_vect = mean[n_off];
        const float variance_vect = variance[n_off];
        const float inv_sqrt_variance = rsqrt(variance_vect + eps);
#if NDIMS == 2
        const int src_off = SRC_OFF(n_off, c, 0, 0, 0, 0);
        const int dst_off = DST_OFF(n_off, c, 0, 0, 0, 0);
#else
        const int src_off = SRC_OFF(0, n_off, c, 0, 0, 0);
        const int dst_off = DST_OFF(0, n_off, c, 0, 0, 0);
#endif

        const VECT_FLOAT_T src_vect
                = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                        (const __global BLOCK_DATA_T *)(&src[src_off]))));
        const VECT_FLOAT_T diff_dst_vect
                = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                        (const __global BLOCK_DATA_T *)(&diff_dst[dst_off]))));

        diff_gamma_vect
                += (src_vect - mean_vect) * diff_dst_vect * inv_sqrt_variance;
        diff_beta_vect += diff_dst_vect;
    }

    const int result_offset = n_chunk_idx * C + c;
    if (USE_SCALE)
        STORE_VECT_FLOAT(&diff_scale[result_offset], diff_gamma_vect);
    if (USE_SHIFT) STORE_VECT_FLOAT(&diff_shift[result_offset], diff_beta_vect);
}

NAMED_KERNEL_ATTR(SCALESHIFT_FINALIZE)
__kernel void vectorized_lnorm_bwd_scaleshift_final(
        __global float *tmp_reduce_mem, __global WEI_DATA_T *diff_scale,
        __global WEI_DATA_T *diff_shift) {

    const int c = GWS_GET_C_finalize();
    const int n_chunk = div_up(N_CHUNKS, FINALIZE_N_CHUNKS);
    const int n = GWS_GET_N_finalize() * n_chunk;
    const int lid = get_local_id(0);

    // diff_scale and diff_shift use the same tensor in scratchpad
    const int diff_shift_off = N_CHUNKS * C;
    __global float *tmp_diff_scale = tmp_reduce_mem;
    __global float *tmp_diff_shift = tmp_reduce_mem + diff_shift_off;

    float diff_gamma = 0;
    float diff_beta = 0;

    for (int n_idx = n; n_idx < min(n + n_chunk, N_CHUNKS); n_idx++) {
        const int result_off = n_idx * C + c;
        diff_gamma += tmp_diff_scale[result_off];
        diff_beta += tmp_diff_shift[result_off];
    }

    diff_gamma = work_group_reduce_add(diff_gamma);
    diff_beta = work_group_reduce_add(diff_beta);

    if (diff_scale && lid == 0) diff_scale[c] = CONVERT_WEI_DATA_T(diff_gamma);
    if (diff_shift && lid == 0) diff_shift[c] = CONVERT_WEI_DATA_T(diff_beta);
}
#endif // USE_SCALE || USE_SHIFT

KERNEL_ATTR
__kernel void vectorized_lnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global WEI_DATA_T *scale, __global DATA_T *diff_src, float eps) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    const int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
    const float mean_val = mean[s_off];
    const float inv_sqrt_variance = rsqrt(variance[s_off] + eps);

    float dd_gamma = 0, dd_gamma_x = 0;
    VECT_FLOAT_T dd_gamma_vect = 0;
    VECT_FLOAT_T dd_gamma_x_vect = 0;
    if (CALCULATE_STATS) {
        for (int c = 0; c < C; c += VECT_DT_N * SUB_GROUP_SIZE) {
            VECT_FLOAT_T gamma = 1.0f;
            if (scale) { gamma = LOAD_VECT_WEI(&scale[c]); }
            x[NDIMS - 1] = c;
            const int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            const int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            const VECT_FLOAT_T src_vect
                    = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                            (const __global BLOCK_DATA_T *)&src[src_off])));
            const VECT_FLOAT_T dst_vect
                    = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ((
                            const __global BLOCK_DATA_T *)&diff_dst[dst_off])));

            dd_gamma_vect += dst_vect * gamma;
            dd_gamma_x_vect += dst_vect * gamma * (src_vect - mean_val);
        }
#if VECT_DT_N == 1
        dd_gamma = dd_gamma_vect;
        dd_gamma_x = dd_gamma_x_vect;
#else
        for (int i = 0; i < VECT_DT_N; ++i) {
            dd_gamma += dd_gamma_vect[i];
            dd_gamma_x += dd_gamma_x_vect[i];
        }
#endif
        dd_gamma = sub_group_reduce_add(dd_gamma);
        dd_gamma_x = sub_group_reduce_add(dd_gamma_x);
        dd_gamma_x *= inv_sqrt_variance;
    }

    for (int c = 0; c < C; c += VECT_DT_N * SUB_GROUP_SIZE) {
        VECT_FLOAT_T gamma = 1.0f;
        if (scale) { gamma = LOAD_VECT_WEI(&scale[c]); }
        x[NDIMS - 1] = c;
        const int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        const int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        const VECT_FLOAT_T src_vect = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off])));
        VECT_FLOAT_T v_diff_src_vect
                = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                        (const __global BLOCK_DATA_T *)&diff_dst[dst_off])));
        v_diff_src_vect *= gamma;
        if (CALCULATE_STATS) {
            v_diff_src_vect -= dd_gamma / C
                    + (src_vect - mean_val) * dd_gamma_x * inv_sqrt_variance
                            / C;
        }
        v_diff_src_vect *= inv_sqrt_variance;
        VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)&diff_src[src_off],
                AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(v_diff_src_vect)));
    }
}
#endif // IS_BWD
