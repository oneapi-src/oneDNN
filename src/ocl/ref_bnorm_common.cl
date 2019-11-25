/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#define IC_BLOCK 16

#if MB_BLOCK == 16
#define MB16
#define VECT_DT_N 8
#else
#define VECT_DT_N 1
#endif

#include "ocl/ocl_types.h"

#if BNORM_FWD == 1

#if USE_16MB_UNROLL == 1

__attribute__((reqd_work_group_size(1, 1, 16))) // attr:no-format
__attribute__((intel_reqd_sub_group_size(16))) // attr:no-format
__kernel void
calculate_mean(__global DATA_T *src, __global float *mean) {
    const int mb = get_global_id(1);
    const int sp_chunk = get_global_id(0);
    const int c = get_group_id(2) * 16;
    const uint sp_str = (sp_chunk * ID * IH * IW) / SP_CHUNK;
    const uint sp_end = ((sp_chunk + 1) * ID * IH * IW) / SP_CHUNK;
    const int chunk = mb * SP_CHUNK + sp_chunk;

    src += c * ID * IH * IW * MB_BLOCK + mb * IC * ID * IH * IW * MB_BLOCK
            + sp_str * MB_BLOCK * IC_BLOCK;

    VECT_FLOAT_T sum0 = 0.0f, sum1 = 0.0f;
    for (int sp = 0; sp < sp_end - sp_str; sp++) {
        sum0 += CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[0])));
#ifdef MB16
        sum1 += CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[8 * 16])));
#endif
        src += MB_BLOCK * IC_BLOCK;
    }
#ifdef MB16
    float v_mean = 0.0;
    for (int i = 0; i < 8; i++) {
        v_mean += sum0[i] + sum1[i];
    }
#else
    float v_mean = sum0;
#endif
    intel_sub_group_block_write(
            (__global uint *)&mean[chunk * IC + c], as_uint(v_mean));
}
__attribute__((reqd_work_group_size(1, 1, 1))) // attr:no-format
__kernel void
reduce_mean(__global float *reduce_temp, __global float *mean) {
    const int c = get_global_id(0);
    reduce_temp += c;
    float sum = 0.0f;
    for (int i = 0; i < SP_CHUNK * MB_CHUNK; i++)
        sum += reduce_temp[i * IC];

    mean[c] = sum / (MB * ID * IH * IW);
}
__attribute__((reqd_work_group_size(1, 1, 16))) // attr:no-format
__attribute__((intel_reqd_sub_group_size(16))) // attr:no-format
__kernel void
calculate_variance(
        __global DATA_T *src, __global float *mean, __global float *variance) {
    const int mb = get_global_id(1);
    const int sp_chunk = get_global_id(0);
    const int c = get_group_id(2) * 16;
    const uint sp_str = (sp_chunk * ID * IH * IW) / SP_CHUNK;
    const uint sp_end = ((sp_chunk + 1) * ID * IH * IW) / SP_CHUNK;
    const int chunk = mb * SP_CHUNK + sp_chunk;

    src += c * ID * IH * IW * MB_BLOCK + mb * IC * ID * IH * IW * MB_BLOCK
            + sp_str * MB_BLOCK * IC_BLOCK;

    VECT_FLOAT_T sum0 = 0.0, sum1 = 0.0f;
    float v_mean = as_float(
            intel_sub_group_block_read((const __global uint *)&mean[c]));

    for (int sp = 0; sp < sp_end - sp_str; sp++) {
        VECT_FLOAT_T v0 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                                  (const __global BLOCK_DATA_T *)&src[0])))
                - (VECT_FLOAT_T)v_mean;
        sum0 = fma(v0, v0, sum0);
#ifdef MB16
        VECT_FLOAT_T v1 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                                  (const __global BLOCK_DATA_T *)&src[8 * 16])))
                - (VECT_FLOAT_T)v_mean;
        sum1 = fma(v1, v1, sum1);
#endif
        src += MB_BLOCK * IC_BLOCK;
    }
#ifdef MB16
    float v_variance = 0.0;
    for (int i = 0; i < 8; i++) {
        v_variance += sum0[i] + sum1[i];
    }
#else
    float v_variance = sum0;
#endif
    intel_sub_group_block_write(
            (__global uint *)&variance[SP_CHUNK * (MB / MB_BLOCK) * IC
                    + chunk * IC + c],
            as_uint(v_variance));
}
__attribute__((reqd_work_group_size(1, 1, 1))) // attr:no-format
__kernel void
reduce_variance(__global float *reduce_temp, __global float *variance) {
    const int c = get_global_id(0);
    reduce_temp += SP_CHUNK * MB_CHUNK * IC + c;
#if SAVE_STATS == 0
    variance += IC;
#endif
    float sum = 0.0f;
    for (int i = 0; i < SP_CHUNK * MB_CHUNK; i++)
        sum += reduce_temp[i * IC];

    variance[c] = sum / (MB * ID * IH * IW);
}
#endif

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if USE_16MB_UNROLL == 1
__attribute__((intel_reqd_sub_group_size(LWS_1))) // attr:no-format
#endif
__kernel void
ref_bnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst,
        __global float *scaleshift, __global int *ws, float eps) {

#if USE_16MB_UNROLL == 1
    const int n = get_global_id(0) * MB_BLOCK;
    const int c = get_group_id(1) * IC_BLOCK;
    const int sp = get_global_id(2);
    const int d = sp / (IW * IH);
    const int hw = sp % (IW * IH);
    const int h = hw / IW;
    const int w = hw % IW;

#if USE_SCALESHIFT == 1
    float sm = as_float(
            intel_sub_group_block_read((const __global uint *)&scaleshift[c]));
    float sv = as_float(intel_sub_group_block_read(
            (const __global uint *)&scaleshift[IC + c]));
#else
    float sm = 1.0f;
    float sv = 0.0f;
#endif
#if SAVE_STATS == 0 && CALCULATE_STATS == 1
    variance += IC;
#endif
    float v_mean = as_float(
            intel_sub_group_block_read((const __global uint *)&mean[c]));
    float v_variance = as_float(
            intel_sub_group_block_read((const __global uint *)&variance[c]));
    float sqrt_variance = sm / sqrt(v_variance + eps);

    const uint d_off = SRC_OFF(n, c, d, h, w);
    src += d_off;
    dst += d_off;

    VECT_FLOAT_T blockS0 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
            VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[0])));
    VECT_FLOAT_T blockD0 = fma(blockS0 - (VECT_FLOAT_T)v_mean,
            (VECT_FLOAT_T)sqrt_variance, (VECT_FLOAT_T)sv);
#ifdef MB16
    VECT_FLOAT_T blockS1 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
            (const __global BLOCK_DATA_T *)&src[8 * IC_BLOCK])));
    VECT_FLOAT_T blockD1 = fma(blockS1 - (VECT_FLOAT_T)v_mean,
            (VECT_FLOAT_T)sqrt_variance, (VECT_FLOAT_T)sv);
#endif

#if FUSE_BN_RELU == 1
    VECT_INT_T blockWS0 = isgreater(blockD0, (VECT_FLOAT_T)0.0f);
    blockD0 = select((VECT_FLOAT_T)0.0f, blockD0, blockWS0);
#ifdef MB16
    VECT_INT_T blockWS1 = isgreater(blockD1, (VECT_FLOAT_T)0.0f);
    blockD1 = select((VECT_FLOAT_T)0.0f, blockD1, blockWS1);
#endif
#if IS_TRAINING == 1
    ws += d_off;
    VECT_UINT_WRITE((__global uint *)&ws[0], AS_VECT_UINT_T(blockWS0));
#ifdef MB16
    VECT_UINT_WRITE((__global uint *)&ws[8 * 16], AS_VECT_UINT_T(blockWS1));
#endif
#endif
#endif

#if WITH_RELU
    blockD0 = max(blockD0, (VECT_FLOAT_T)0.0f);
#ifdef MB16
    blockD1 = max(blockD1, (VECT_FLOAT_T)0.0f);
#endif
#endif

    VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)&dst[0],
            AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(blockD0)));
#ifdef MB16
    VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)&dst[8 * 16],
            AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(blockD1)));
#endif
#else
    const int c = get_global_id(0);

#if USE_SCALESHIFT == 1
    float sm = scaleshift[c];
    float sv = scaleshift[IC + c];
#else
    float sm = 1;
    float sv = 0;
#endif

#if CALCULATE_STATS == 1
    float v_mean = 0.0f;
    float v_variance = 0.0f;

    for (int n = 0; n < MB; ++n) {
        for (int d = 0; d < ID; ++d)
            for (int h = 0; h < IH; ++h)
                for (int w = 0; w < IW; ++w) {
                    uint d_off = SRC_OFF(n, c, d, h, w);
                    v_mean += TO_DEF_ACC_DATA_T(src[d_off]);
                }
    }
    v_mean /= MB * ID * IH * IW;

    for (int n = 0; n < MB; ++n) {
        for (int d = 0; d < ID; ++d)
            for (int h = 0; h < IH; ++h)
                for (int w = 0; w < IW; ++w) {
                    uint d_off = SRC_OFF(n, c, d, h, w);
                    float m = TO_DEF_ACC_DATA_T(src[d_off]) - v_mean;
                    v_variance += m * m;
                }
    }
    v_variance /= MB * ID * IH * IW;
#else
    float v_mean = mean[c];
    float v_variance = variance[c];
#endif

    float sqrt_variance = 1.0f / sqrt(v_variance + eps);

    for (int n = 0; n < MB; ++n) {
        for (int d = 0; d < ID; ++d)
            for (int h = 0; h < IH; ++h)
                for (int w = 0; w < IW; ++w) {
                    uint d_off = SRC_OFF(n, c, d, h, w);
                    float bn_res = sm * (TO_DEF_ACC_DATA_T(src[d_off]) - v_mean)
                                    * sqrt_variance
                            + sv;
#if FUSE_BN_RELU == 1
                    if (bn_res <= 0) {
                        bn_res = 0;
#if IS_TRAINING == 1
                        ws[d_off] = 0;
#endif
                    } else {
#if IS_TRAINING == 1
                        ws[d_off] = 1;
#endif
                    }
#endif
#if WITH_RELU
                    dst[d_off] = TO_DATA_T(max(bn_res, 0.0f));
#else
                    dst[d_off] = TO_DATA_T(bn_res);
#endif
                }
    }

#if CALCULATE_STATS == 1 && SAVE_STATS == 1
    mean[c] = v_mean;
    variance[c] = v_variance;
#endif
#endif
}
#endif

#if BNORM_BWD == 1

#if USE_16MB_UNROLL == 1
__attribute__((reqd_work_group_size(1, 1, 16))) // attr:no-format
__attribute__((intel_reqd_sub_group_size(16))) // attr:no-format
__kernel void
calculate_stats(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global int *ws,
        __global float *diff_scaleshift) {
    const int mb = get_global_id(1);
    const int sp_chunk = get_global_id(0);
    const int c = get_group_id(2) * 16;
    const uint sp_str = (sp_chunk * ID * IH * IW) / SP_CHUNK;
    const uint sp_end = ((sp_chunk + 1) * ID * IH * IW) / SP_CHUNK;
    const int chunk = mb * SP_CHUNK + sp_chunk;
    const int s_off = c * ID * IH * IW * MB_BLOCK
            + mb * IC * ID * IH * IW * MB_BLOCK + sp_str * MB_BLOCK * IC_BLOCK;
    src += s_off;
    diff_dst += s_off;
#if FUSE_BN_RELU == 1
    ws += s_off;
#endif
    VECT_FLOAT_T diff_gamma0 = 0.0f, diff_beta0 = 0.0f;
    VECT_FLOAT_T diff_gamma1 = 0.0f, diff_beta1 = 0.0f;
    float v_mean = as_float(
            intel_sub_group_block_read((const __global uint *)&mean[c]));

    for (int sp = 0; sp < sp_end - sp_str; sp++) {
        VECT_FLOAT_T dd0 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&diff_dst[0])));
        VECT_FLOAT_T ss0 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[0])));
#ifdef MB16
        VECT_FLOAT_T dd1 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                (const __global BLOCK_DATA_T *)&diff_dst[8 * 16])));
        VECT_FLOAT_T ss1 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[8 * 16])));
#endif
#if FUSE_BN_RELU == 1
        VECT_INT_T ws0
                = AS_VECT_INT_T(VECT_UINT_READ((const __global uint *)&ws[0]));
        dd0 = select((VECT_FLOAT_T)0.0f, dd0, ws0);
#ifdef MB16
        VECT_INT_T ws1 = AS_VECT_INT_T(
                VECT_UINT_READ((const __global uint *)&ws[8 * 16]));
        dd1 = select((VECT_FLOAT_T)0.0f, dd1, ws1);
#endif
        ws += MB_BLOCK * IC_BLOCK;
#endif
        diff_gamma0 = fma((ss0 - (VECT_FLOAT_T)v_mean), dd0, diff_gamma0);
        diff_beta0 += dd0;
#ifdef MB16
        diff_gamma1 = fma((ss1 - (VECT_FLOAT_T)v_mean), dd1, diff_gamma1);
        diff_beta1 += dd1;
#endif

        src += MB_BLOCK * IC_BLOCK;
        diff_dst += MB_BLOCK * IC_BLOCK;
    }
#ifdef MB16
    float v_diff_gamma = 0.0f, v_diff_beta = 0.0;
    for (int i = 0; i < 8; i++) {
        v_diff_gamma += diff_gamma0[i] + diff_gamma1[i];
        v_diff_beta += diff_beta0[i] + diff_beta1[i];
    }
#else
    float v_diff_gamma = diff_gamma0, v_diff_beta = diff_beta0;
#endif
    intel_sub_group_block_write(
            (__global uint *)&diff_scaleshift[chunk * IC + c],
            as_uint(v_diff_gamma));
    intel_sub_group_block_write(
            (__global uint *)&diff_scaleshift[MB_CHUNK * SP_CHUNK * IC
                    + chunk * IC + c],
            as_uint(v_diff_beta));
}

__attribute__((reqd_work_group_size(1, 1, 1))) // attr:no-format
__kernel void
reduce_stats(__global float *reduce_temp, __global float *diff_scaleshift,
        __global float *variance, float eps) {
    const int c = get_global_id(0);
    reduce_temp += c;
    float diff_gamma = 0.0f, diff_beta = 0.0f;
    for (int i = 0; i < MB_CHUNK * SP_CHUNK; i++) {
        diff_gamma += reduce_temp[i * IC];
        diff_beta += reduce_temp[MB_CHUNK * SP_CHUNK * IC + i * IC];
    }

    float sqrt_variance = 1.0f / sqrt(variance[c] + eps);

    diff_scaleshift[c] = diff_gamma * sqrt_variance;
#if DIFF_SCALESHIFT == 1
    diff_scaleshift[IC + c] = diff_beta;
#else
    diff_scaleshift[MB_CHUNK * SP_CHUNK * IC + c] = diff_beta;
#endif
}
#endif

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if USE_16MB_UNROLL == 1
__attribute__((intel_reqd_sub_group_size(LWS_1))) // attr:no-format
#endif
__kernel void
ref_bnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scaleshift, __global int *ws, __global DATA_T *diff_src,
        __global float *diff_scaleshift, float eps) {

#if USE_16MB_UNROLL == 1
    const int n = get_global_id(0) * MB_BLOCK;
    const int c = get_group_id(1) * IC_BLOCK;
    const int sp = get_global_id(2);
    const int d = sp / (IW * IH);
    const int hw = sp % (IW * IH);
    const int h = hw / IW;
    const int w = hw % IW;

#if USE_SCALESHIFT == 1
    float gamma = as_float(
            intel_sub_group_block_read((const __global uint *)&scaleshift[c]));
#else
    float gamma = 1.0f;
#endif

    float v_mean = as_float(
            intel_sub_group_block_read((const __global uint *)&mean[c]));
    float v_variance = as_float(
            intel_sub_group_block_read((const __global uint *)&variance[c]));
    float sqrt_variance = 1.0f / sqrt(v_variance + eps);

    float diff_gamma = as_float(intel_sub_group_block_read(
            (const __global uint *)&diff_scaleshift[c]));
#if DIFF_SCALESHIFT == 1
    float diff_beta = as_float(intel_sub_group_block_read(
            (const __global uint *)&diff_scaleshift[IC + c]));
#else
    float diff_beta = as_float(intel_sub_group_block_read((const __global uint
                    *)&diff_scaleshift[MB_CHUNK * SP_CHUNK * IC + c]));
#endif

    const uint d_off = SRC_OFF(n, c, d, h, w);
    diff_src += d_off;
    diff_dst += d_off;
    src += d_off;

    VECT_FLOAT_T blockD0 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
            VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&diff_dst[0])));
#ifdef MB16
    VECT_FLOAT_T blockD1 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
            (const __global BLOCK_DATA_T *)&diff_dst[8 * IC_BLOCK])));
#endif
#if FUSE_BN_RELU == 1
    ws += d_off;
    VECT_INT_T blockWS0
            = AS_VECT_INT_T(VECT_UINT_READ((const __global uint *)&ws[0]));
    blockD0 = select((VECT_FLOAT_T)0.0f, blockD0, blockWS0);
#ifdef MB16
    VECT_INT_T blockWS1 = AS_VECT_INT_T(
            VECT_UINT_READ((const __global uint *)&ws[8 * IC_BLOCK]));
    blockD1 = select((VECT_FLOAT_T)0.0f, blockD1, blockWS1);
#endif
#endif

    gamma *= sqrt_variance;

#if CALCULATE_DIFF_STATS == 1
    diff_gamma *= sqrt_variance;
    diff_gamma /= (MB * ID * IH * IW);
    diff_beta /= (MB * ID * IH * IW);

    VECT_FLOAT_T blockS0 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
            VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[0])));
    blockD0 -= fma((VECT_FLOAT_T)diff_gamma, (blockS0 - (VECT_FLOAT_T)v_mean),
            (VECT_FLOAT_T)diff_beta);
#ifdef MB16
    VECT_FLOAT_T blockS1 = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
            (const __global BLOCK_DATA_T *)&src[8 * IC_BLOCK])));
    blockD1 -= fma((VECT_FLOAT_T)diff_gamma, (blockS1 - (VECT_FLOAT_T)v_mean),
            (VECT_FLOAT_T)diff_beta);
#endif
#endif
    blockD0 *= gamma;
    VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)&diff_src[0],
            AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(blockD0)));
#ifdef MB16
    blockD1 *= gamma;
    VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)&diff_src[8 * 16],
            AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(blockD1)));
#endif
#else

    const int c = get_global_id(0);

    float v_mean = mean[c];
    float v_variance = variance[c];
    float sqrt_variance = 1.0f / sqrt(v_variance + eps);
#if USE_SCALESHIFT == 1
    float gamma = scaleshift[c];
#else
    float gamma = 1;
#endif
    float diff_gamma = 0.0f;
    float diff_beta = 0.0f;

    for (int n = 0; n < MB; ++n) {
        for (int d = 0; d < ID; ++d)
            for (int h = 0; h < IH; ++h)
                for (int w = 0; w < IW; ++w) {
                    uint s_off = SRC_OFF(n, c, d, h, w);
                    float dd = CONVERT_FLOAT_T(diff_dst[s_off]);
#if FUSE_BN_RELU == 1
                    if (!ws[s_off]) dd = 0;
#endif
                    diff_gamma += (CONVERT_FLOAT_T(src[s_off]) - v_mean) * dd;
                    diff_beta += dd;
                }
    }

    diff_gamma *= sqrt_variance;

#if DIFF_SCALESHIFT == 1
    diff_scaleshift[c] = diff_gamma;
    diff_scaleshift[IC + c] = diff_beta;
#endif

    for (int n = 0; n < MB; ++n) {
        for (int d = 0; d < ID; ++d)
            for (int h = 0; h < IH; ++h)
                for (int w = 0; w < IW; ++w) {
                    uint s_off = SRC_OFF(n, c, d, h, w);
                    float dd = CONVERT_FLOAT_T(diff_dst[s_off]);
#if FUSE_BN_RELU == 1
                    if (!ws[s_off]) dd = 0;
#endif

                    float v_diff_src = dd;
#if CALCULATE_DIFF_STATS == 1
                    v_diff_src -= diff_beta / (MB * ID * IH * IW)
                            + (CONVERT_FLOAT_T(src[s_off]) - v_mean)
                                    * diff_gamma * sqrt_variance
                                    / (MB * ID * IH * IW);
#endif
                    v_diff_src *= gamma * sqrt_variance;
                    diff_src[s_off] = TO_DATA_T(v_diff_src);
                }
    }
#endif
}
#endif
