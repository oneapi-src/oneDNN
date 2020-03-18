/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#if IS_FWD

#define HAS_STAT_SP_TAIL (STAT_SP_TAIL != STAT_SP_NBLOCKS)
#define HAS_SP_TAIL (SP != SP_TAIL)

#define LOAD_DATA_8x16(ptr) \
    CONVERT_FLOAT8_T( \
            AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)(ptr))))

#define LOAD_DATA_1x16(ptr) \
    CONVERT_FLOAT_T(AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)(ptr))))

#define STORE_DATA_1x16(ptr, val) \
    BLOCK_WRITE((__global BLOCK_DATA_T *)(ptr), \
            AS_BLOCK_DATA_T(CONVERT_DATA_T(val)));

#define STORE_DATA_8x16(ptr, val) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)&dst[0], \
            AS_BLOCK_DATA8_T(CONVERT_DATA8_T(blockD0)))

#define LOAD_FLOAT_1x16(ptr) \
    as_float(intel_sub_group_block_read((const __global uint *)(ptr)));

#define STORE_FLOAT_1x16(ptr, val) \
    intel_sub_group_block_write((__global uint *)(ptr), as_uint(val));

#define STORE_FLOAT_8x16(ptr, val) \
    intel_sub_group_block_write8((__global uint *)(ptr), as_uint8(val));

#if USE_NHWC
#define IC_BLOCK_STRIDE IC
#else
#define IC_BLOCK_STRIDE 16
#endif

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_mean(__global DATA_T *src, __global float *mean) {
    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();

#if USE_NHWC
    src += c + sp_block_idx * STAT_SP_BLOCK * IC;
#else
    src += (c & 15) + sp_block_idx * STAT_SP_BLOCK * 16 + (c & ~15) * SP
            + mb * SP * IC;
#endif

    const int mb_sp_idx = mb * STAT_SP_NBLOCKS + sp_block_idx;

    float8 res0 = 0.0f, res1 = 0.0f;
    float v_mean = 0.0f;

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
            res0 += s0;
            res1 += s1;

            src += 16 * IC_BLOCK_STRIDE;
            sp -= 16;
        }
        while (sp >= 1) {
            float s0 = LOAD_DATA_1x16(&src[0]);
            v_mean += s0;
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
            res0 += s0;
            res1 += s1;
            src += 16 * IC_BLOCK_STRIDE;
        }
    }

    for (int i = 0; i < 8; i++) {
        v_mean += res0[i] + res1[i];
    }

    STORE_FLOAT_1x16(&mean[mb_sp_idx * IC + c], v_mean);
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_mean(
        __global float *reduce_temp, __global float *mean) {
    const int c = GWS_GET_REDUCE_STAT_IC();
    reduce_temp += c;
    float sum = 0.0f;
    for (int i = 0; i < REDUCE_STAT_NBLOCKS; i++)
        sum += reduce_temp[i * IC];

    mean[c] = sum / (MB * ID * IH * IW);
}

NAMED_KERNEL_ATTR(CALC)
__kernel void gen9_calc_variance(
        __global DATA_T *src, __global float *mean, __global float *variance) {
    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();

#if USE_NHWC
    src += c + sp_block_idx * STAT_SP_BLOCK * IC;
#else
    src += (c & 15) + sp_block_idx * STAT_SP_BLOCK * 16 + (c & ~15) * SP
            + mb * SP * IC;
#endif

    const int mb_sp_idx = mb * STAT_SP_NBLOCKS + sp_block_idx;

    float8 res0 = 0.0f, res1 = 0.0f;
    float v_var = 0.0f;

    float v_mean = LOAD_FLOAT_1x16(&mean[c]);

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
            float8 v0 = s0 - v_mean;
            float8 v1 = s1 - v_mean;
            res0 = fma(v0, v0, res0);
            res1 = fma(v1, v1, res1);

            src += 16 * IC_BLOCK_STRIDE;
            sp -= 16;
        }
        while (sp >= 1) {
            float s0 = LOAD_DATA_1x16(&src[0]);
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
            for (int k = 0; k < 8; ++k)
                s0[k] = LOAD_DATA_1x16(&src[k * IC]);
            for (int k = 0; k < 8; ++k)
                s1[k] = LOAD_DATA_1x16(&src[(k + 8) * IC]);
#else
            float8 s0 = LOAD_DATA_8x16(&src[0]);
            float8 s1 = LOAD_DATA_8x16(&src[8 * 16]);
#endif
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

    STORE_FLOAT_1x16(
            &variance[REDUCE_STAT_NBLOCKS * IC + mb_sp_idx * IC + c], v_var);
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void gen9_reduce_variance(
        __global float *reduce_temp, __global float *variance) {
    const int c = GWS_GET_REDUCE_STAT_IC();
    reduce_temp += REDUCE_STAT_NBLOCKS * IC + c;
#if SAVE_STATS == 0
    variance += IC;
#endif
    float sum = 0.0f;
    for (int i = 0; i < REDUCE_STAT_NBLOCKS; i++)
        sum += reduce_temp[i * IC];

    variance[c] = sum / (MB * ID * IH * IW);
}

KERNEL_ATTR
__kernel void gen9_bnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst,
        __global float *scaleshift, __global int *ws, float eps) {
    const int n = GWS_GET_MB();
    const int c = GWS_GET_IC();
    const int sp = GWS_GET_SP() * VECT_SIZE;

#if SAVE_STATS == 0 && CALCULATE_STATS == 1
    variance += IC;
#endif

#if USE_NHWC
    const uint d_off = sp * IC + c;
#else
    const uint d_off = (c & 15) + sp * 16 + (c & ~15) * SP + n * SP * IC;
#endif
    src += d_off;
    dst += d_off;

    float8 blockS0 = 0.0f, blockD0;
#if HAS_SP_TAIL
    if (sp == SP_TAIL) {
        for (int k = 0; k < SP - SP_TAIL; ++k)
            blockS0[k] = LOAD_DATA_1x16(&src[k * IC_BLOCK_STRIDE]);
    } else
#endif // HAS_SP_TAIL
    {
#if USE_NHWC
        for (int k = 0; k < 8; ++k)
            blockS0[k] = LOAD_DATA_1x16(&src[k * IC]);
#else
        blockS0 = LOAD_DATA_8x16(&src[0]);
#endif
    }

#if USE_SCALESHIFT == 1
    float sm = LOAD_FLOAT_1x16(&scaleshift[c]);
    float sv = LOAD_FLOAT_1x16(&scaleshift[IC + c]);
#else
    float sm = 1.0f;
    float sv = 0.0f;
#endif

    float v_mean = LOAD_FLOAT_1x16(&mean[c]);
    float v_variance = LOAD_FLOAT_1x16(&variance[c]);

    float sqrt_variance = sm / sqrt(v_variance + eps);

    blockD0 = fma(blockS0 - (float8)v_mean, (float8)sqrt_variance, (float8)sv);

#if FUSE_BN_RELU
    int8 blockWS0 = isgreater(blockD0, (float8)0.0f);
    blockD0 = select((float8)0.0f, blockD0, blockWS0);
#if IS_TRAINING
    ws += d_off;
#if HAS_SP_TAIL
    if (sp == SP_TAIL) {
        for (int k = 0; k < SP - SP_TAIL; ++k) {
            STORE_FLOAT_1x16(&ws[k * IC_BLOCK_STRIDE], blockWS0[k]);
        }
    } else
#endif // HAS_SP_TAIL
    {
#if USE_NHWC
        for (int k = 0; k < 8; ++k)
            STORE_FLOAT_1x16(&ws[k * IC_BLOCK_STRIDE], blockWS0[k]);
#else
        STORE_FLOAT_8x16(&ws[0], blockWS0);
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
            STORE_DATA_1x16(&dst[k * IC_BLOCK_STRIDE], blockD0[k]);
        }
    } else
#endif // HAS_SP_TAIL
    {
#if USE_NHWC
        for (int k = 0; k < 8; ++k)
            STORE_DATA_1x16(&dst[k * IC_BLOCK_STRIDE], blockD0[k]);
#else
        STORE_DATA_8x16(&dst[0], blockD0);
#endif
    }
}

#endif // IS_FWD
