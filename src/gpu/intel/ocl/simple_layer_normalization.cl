/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/ocl/layer_norm_common.h"

#if IS_FWD
KERNEL_ATTR
__kernel void simple_lnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DST_DATA_T *dst,
        __global WEI_DATA_T *scale, __global WEI_DATA_T *shift, float eps,
        __global float *src_scale, __global float *dst_scale) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    if (x[0] >= DST_D0 || x[1] >= DST_D1 || x[2] >= DST_D2 || x[3] >= DST_D3) {
        int local_id = get_sub_group_local_id();
        for (int c = 0; c < C; c += SUB_GROUP_SIZE) {
            x[NDIMS - 1] = c + local_id;
            int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            dst[dst_off] = TO_DST(CONVERT_DATA_T(0.f));
        }
        return;
    }

    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

    float v_mean = CALCULATE_STATS ? 0 : mean[s_off];
    float v_variance = CALCULATE_STATS ? 0 : variance[s_off];

    if (CALCULATE_STATS) {
        VECT_FLOAT_T v_acc = 0;
        for (int c = 0; c < C; c += SUB_GROUP_SIZE * VECT_DT_N) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            v_acc += CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                    (const __global BLOCK_DATA_T *)&src[src_off])));
        }
#if VECT_DT_N == 1
        v_mean = v_acc;
#else // VECT_DT_N == 1
        v_mean = 0;
        for (int i = 0; i < VECT_DT_N; ++i) {
            v_mean += v_acc[i];
        }
#endif // VECT_DT_N == 1

        float total_sum = sub_group_reduce_add(v_mean);
        v_mean = total_sum / C;

        v_acc = 0;
        VECT_FLOAT_T m = 0;

        for (int c = 0; c < C; c += SUB_GROUP_SIZE * VECT_DT_N) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            m = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                    (const __global BLOCK_DATA_T *)&src[src_off])));
            m -= v_mean;
            v_acc += m * m;
        }
#if VECT_DT_N == 1
        v_variance = v_acc;
#else // VECT_DT_N == 1
        v_variance = 0;
        for (int i = 0; i < VECT_DT_N; ++i) {
            v_variance += v_acc[i];
        }
#endif // VECT_DT_N == 1

        total_sum = sub_group_reduce_add(v_variance);
        v_variance = total_sum / C;
    }
    const float rsqrt_variance = rsqrt(v_variance + eps);

    int local_id = get_sub_group_local_id();
    for (int c = 0; c < C; c += SUB_GROUP_SIZE) {
        float sm = (scale ? CONVERT_WEI_FLOAT_T(scale[c + local_id]) : 1.0f)
                * rsqrt_variance;
        float sv = shift ? CONVERT_WEI_FLOAT_T(shift[c + local_id]) : 0.0f;

        x[NDIMS - 1] = c + local_id;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        float d = (sm * (SRC_TO_REF(src[src_off]) - v_mean) + sv);
#if WITH_SRC_SCALES
        d *= src_scale[0];
#endif
#if WITH_DST_SCALES
        d /= dst_scale[0];
#endif
        dst[dst_off] = TO_DST(d);
    }

    if (CALCULATE_STATS) {
        if (SAVE_STATS) {
            mean[s_off] = v_mean;
            variance[s_off] = v_variance;
        }
    }
}

#else
#if USE_SCALE || USE_SHIFT
NAMED_KERNEL_ATTR(SCALESHIFT)
__kernel void simple_lnorm_bwd_scaleshift(__global SRC_DATA_T *src,
        __global float *mean, __global float *variance,
        __global DATA_T *diff_dst, __global float *diff_scale,
        __global float *diff_shift, float eps) {

    const int c = GWS_GET_C();
    const int n_chunk_idx = GWS_GET_N();
    const int n_start = n_chunk_idx * N_CHUNK_SIZE;
    const int n_end = n_start + N_CHUNK_SIZE;

    // diff_scale and diff_shift use the same tensor in scratchpad
    const int shift_off = N_CHUNKS * C;
    diff_shift += shift_off;

    vector_float diff_gamma_vect = 0;
    vector_float diff_beta_vect = 0;

    for (int n_off = n_start; n_off < n_end; n_off += VECTOR_SIZE_SCALESHIFT) {
        const vector_float mean_vect = vector_load(mean[n_off]);
        const vector_float variance_vect = vector_load(variance[n_off]);
        const vector_float inv_sqrt_variance = rsqrt(variance_vect + eps);
#if NDIMS == 2
        const int src_off = SRC_OFF(n_off, c, 0, 0, 0, 0);
        const int dst_off = DST_OFF(n_off, c, 0, 0, 0, 0);
#else
        const int src_off = SRC_OFF(0, n_off, c, 0, 0, 0);
        const int dst_off = DST_OFF(0, n_off, c, 0, 0, 0);
#endif
        const vector_float src_vect = convert_vector_src_to_float(
                as_vector_src_data_t(sub_group_read(
                        (const __global SRC_BLOCK_DATA_T *)&src[src_off])));
        const vector_float diff_dst_vect
                = convert_vector_to_float(as_vector_data_t(sub_group_read(
                        (const __global BLOCK_DATA_T *)&diff_dst[dst_off])));

        diff_gamma_vect
                += (src_vect - mean_vect) * diff_dst_vect * inv_sqrt_variance;
        diff_beta_vect += diff_dst_vect;
    }

    float diff_gamma = 0, diff_beta = 0;
#if VECTOR_SIZE_SCALESHIFT == 1
    diff_gamma = diff_gamma_vect;
    diff_beta = diff_beta_vect;
#else
    for (int elem_idx = 0; elem_idx < VECTOR_SIZE_SCALESHIFT; elem_idx++) {
        diff_gamma += diff_gamma_vect[elem_idx];
        diff_beta += diff_beta_vect[elem_idx];
    }
#endif

    const int result_offset = n_chunk_idx * C + c;
    if (USE_SCALE)
        intel_sub_group_block_write((__global uint *)&diff_scale[result_offset],
                as_uint(diff_gamma));
    if (USE_SHIFT)
        intel_sub_group_block_write((__global uint *)&diff_shift[result_offset],
                as_uint(diff_beta));
}

NAMED_KERNEL_ATTR(SCALESHIFT_FINALIZE)
__kernel void simple_lnorm_bwd_scaleshift_final(__global float *tmp_reduce_mem,
        __global WEI_DATA_T *diff_scale, __global WEI_DATA_T *diff_shift) {
    const int c = GWS_GET_C_finalize();
    const int diff_shift_off = N_CHUNKS * C;
    __global float *tmp_diff_scale = tmp_reduce_mem;
    // diff_scale and diff_shift use the same tensor in scratchpad
    __global float *tmp_diff_shift = tmp_reduce_mem + diff_shift_off;

    float diff_gamma = 0;
    float diff_beta = 0;

    for (int n_chunk_idx = 0; n_chunk_idx < N_CHUNKS; n_chunk_idx++) {
        const int result_off = n_chunk_idx * C + c;
        diff_gamma += tmp_diff_scale[result_off];
        diff_beta += tmp_diff_shift[result_off];
    }

    if (diff_scale) diff_scale[c] = CONVERT_WEI_DATA_T(diff_gamma);
    if (diff_shift) diff_shift[c] = CONVERT_WEI_DATA_T(diff_beta);
}
#endif //USE_SCALE

KERNEL_ATTR
__kernel void simple_lnorm_bwd(__global DATA_T *src, __global float *mean,
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

#endif // IS_FWD
