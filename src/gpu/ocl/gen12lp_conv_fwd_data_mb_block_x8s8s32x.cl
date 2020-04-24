/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_types.h"
#if WITH_ELTWISE == 1 || WITH_POST_SUM_ELTWISE == 1
#include "gpu/ocl/ocl_post_ops.h"
#endif

#define SRC_DATA_BLOCK_T MMAD_DATA8_T
#define AS_SRC_DATA_BLOCK_T AS_MMAD_DATA8_T

#define BLOCK_READ_SRC(data, idx) \
    data = AS_SRC_DATA_BLOCK_T( \
            intel_sub_group_block_read8((__global uint *)&src[idx]));

#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

#define BLOCK_READ_SCALES(data, idx) \
    data = as_float4(intel_sub_group_block_read4( \
            (__global uint *)&scales_per_oc[idx]));

#if SCALES_PER_OC
#define SCALE scales
#elif SCALES_COMMON
#define SCALE scale
#else
#define SCALE 1
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_fwd_mb_block_x8s8s32x(const __global uchar *src, const __global char *wei,
        const __global float *bias, __global DATA_T *dst, float eltwise_alpha,
        float eltwise_beta, float eltwise_scale, float sum_scale, float scale,
        const __global float *scales_per_oc) {
#ifdef MB_FULL_BLOCK
    const int mb_blocks = 1;
#else // MB_FULL_BLOCK
    const int mb_blocks = 2;
#endif // MB_FULL_BLOCK

    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP / mb_blocks;
    const int group_sp = get_group_id(1) * SP_GROUP;

    // XXX: Each work-group calculates 16 mb instead of 32
    const int mb = get_group_id(2) % mb_blocks;
    const int sub_group_id = get_sub_group_id();
    const int oc = (sub_group_id % OC_GROUP);
    const int sp = (sub_group_id / OC_GROUP);

    const int g = (group_oc + oc) / OC_NCHUNK;
    const int group_ic = IC_NCHUNK * g;

    const int god = group_sp / (OW_PADDED * OH);
    const int gohw = group_sp % (OW_PADDED * OH);
    const int goh = gohw / OW_PADDED;
    const int gow = gohw % OW_PADDED;

    const int gid = god * SD;
    const int gih = goh * SH;
    const int giw = gow * SW;

    const int local_oh = sp / OW_PADDED;
    const int local_ow = sp % OW_PADDED;
    const int local_ih = local_oh * SH;
    const int local_iw = local_ow * SW;

    const int od = god;
    const int ow = gow + local_ow;
    const int oh = goh + local_oh;
    const int id = gid - PD;
    const int iw = giw + local_iw - PW;
    const int ih = gih + local_ih - PH;

    if (ow >= OW) return;

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK / mb_blocks * mb; // XXX: Why this offset?
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);

    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * group_ic;
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK / mb_blocks * mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw);

    wei += IC_BLOCK * KD * KH * KW * OC_BLOCK * (group_oc + oc) * IC_NCHUNK;

    int8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    int8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    int8 C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    int8 C30 = 0, C31 = 0, C32 = 0, C33 = 0;

    __attribute__((opencl_unroll_hint)) for (int ic_chunk = 0;
                                             ic_chunk < IC_NCHUNK; ic_chunk++) {
        SRC_DATA_BLOCK_T S0, S1, S2, S3;
        int8 W0, W1, W2, W3;
        for (int kd = 0; kd < KD; kd++) {
            if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
                src += IC_BLOCK * MB_BLOCK * IH * IW * (1 + DD);
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            for (int kh = 0; kh < KH; kh++) {
                if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                    src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }
                __attribute__((opencl_unroll_hint)) for (int kw = 0; kw < KW;
                                                         kw++) {
                    if (kw * (1 + DW) + iw >= 0 && kw * (1 + DW) + iw < IW) {
                        BLOCK_READ_SRC(S0, 0);
#if MB > 8
                        BLOCK_READ_SRC(S1, 8 * IC_BLOCK);
#ifdef MB_FULL_BLOCK
                        BLOCK_READ_SRC(S2, 16 * IC_BLOCK);
                        BLOCK_READ_SRC(S3, 24 * IC_BLOCK);
#endif // MB_FULL_BLOCK
#endif // MB > 8
                        BLOCK_READ_WHT(W0, 0);
                        BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
                        BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
                        BLOCK_READ_WHT(W3, 24 * IC_BLOCK);
                        C00 = mmad8x8(S0, W0, C00);
                        C01 = mmad8x8(S0, W1, C01);
                        C02 = mmad8x8(S0, W2, C02);
                        C03 = mmad8x8(S0, W3, C03);
#if MB > 8
                        C10 = mmad8x8(S1, W0, C10);
                        C11 = mmad8x8(S1, W1, C11);
                        C12 = mmad8x8(S1, W2, C12);
                        C13 = mmad8x8(S1, W3, C13);
#ifdef MB_FULL_BLOCK
                        C20 = mmad8x8(S2, W0, C20);
                        C21 = mmad8x8(S2, W1, C21);
                        C22 = mmad8x8(S2, W2, C22);
                        C23 = mmad8x8(S2, W3, C23);
                        C30 = mmad8x8(S3, W0, C30);
                        C31 = mmad8x8(S3, W1, C31);
                        C32 = mmad8x8(S3, W2, C32);
                        C33 = mmad8x8(S3, W3, C33);
#endif // MB_FULL_BLOCK
#endif // MB > 8
                    }
                    src += IC_BLOCK * MB_BLOCK * (1 + DW);
                    wei += IC_BLOCK * OC_BLOCK;
                } // kw loop
                src += IC_BLOCK * MB_BLOCK * (IW * (1 + DH) - KW * (1 + DW));
            } // kh loop
            src += IC_BLOCK * MB_BLOCK * (IH * (1 + DD) - KH * (1 + DH)) * IW;
        } // kd loop
        src += IC_BLOCK * MB_BLOCK * (ID - KD * (1 + DD)) * IH * IW;
    } // IC_NCHUNK loop

    float4 tmp;
    uint8 dst_pack;
    uint8 D0, D1, D2, D3;

#if SCALES_PER_OC
    float4 scales;
    BLOCK_READ_SCALES(scales, (group_oc + oc) * OC_BLOCK);
#endif

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, (group_oc + oc) * OC_BLOCK);
    bia *= SCALE;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)SCALE, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= SCALE;
#endif

#if WITH_SUM
    D0.s0123 = as_uint4(intel_sub_group_block_read_uc16((__global uchar *)dst));
    D0.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[4 * OC_BLOCK]));
#if MB > 8
    D1.s0123 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[8 * OC_BLOCK]));
    D1.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[12 * OC_BLOCK]));
#ifdef MB_FULL_BLOCK
    D2.s0123 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[16 * OC_BLOCK]));
    D2.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[20 * OC_BLOCK]));
    D3.s0123 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[24 * OC_BLOCK]));
    D3.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[28 * OC_BLOCK]));
#endif // MB_FULL_BLOCK
#endif // MB > 8

#define DO_SUM(d_pack) \
    do { \
        DATA4_T d = AS_DATA4_T(d_pack); \
        float4 df = convert_float4(d); \
        tmp = fma(df, (float4)sum_scale, tmp); \
    } while (0)

#else
#define DO_SUM(d) ;
#endif // with_sum

#define ELTWISE() \
    do { \
        tmp[0] = fwd_eltwise( \
                tmp[0], eltwise_alpha, eltwise_beta, eltwise_scale); \
        tmp[1] = fwd_eltwise( \
                tmp[1], eltwise_alpha, eltwise_beta, eltwise_scale); \
        tmp[2] = fwd_eltwise( \
                tmp[2], eltwise_alpha, eltwise_beta, eltwise_scale); \
        tmp[3] = fwd_eltwise( \
                tmp[3], eltwise_alpha, eltwise_beta, eltwise_scale); \
    } while (0)

#if WITH_ELTWISE
#define DO_ELTWISE() ELTWISE();
#else
#define DO_ELTWISE() ;
#endif

#if WITH_POST_SUM_ELTWISE
#define DO_POST_SUM_ELTWISE() ELTWISE();
#else
#define DO_POST_SUM_ELTWISE() ;
#endif

#define PACK(C0, C1, C2, C3, idx) \
    do { \
        tmp[0] = C0[idx]; \
        tmp[1] = C1[idx]; \
        tmp[2] = C2[idx]; \
        tmp[3] = C3[idx]; \
    } while (0)

#define CONVERT_PACK(idx) \
    do { \
        DATA4_T tmp_cvt \
                = (DATA4_T)(CONVERT_DATA_T(tmp.s0), CONVERT_DATA_T(tmp.s1), \
                        CONVERT_DATA_T(tmp.s2), CONVERT_DATA_T(tmp.s3)); \
        dst_pack[idx] = as_uint(tmp_cvt); \
    } while (0)

#define STORE_DST(C0, C1, C2, C3, D, mb_stride) \
    do { \
        for (int n_i = 0; n_i < 8; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            DO_ELTWISE(); \
            DO_SUM(D[n_i]); \
            DO_POST_SUM_ELTWISE(); \
            CONVERT_PACK(n_i); \
        } \
        intel_sub_group_block_write_uc16( \
                &dst[mb_stride * OC_BLOCK], as_uchar16(dst_pack.s0123)); \
        intel_sub_group_block_write_uc16(&dst[mb_stride * OC_BLOCK + 16 * 8], \
                as_uchar16(dst_pack.s4567)); \
    } while (0)

    STORE_DST(C00, C01, C02, C03, D0, 0);
#if MB > 8
    STORE_DST(C10, C11, C12, C13, D1, 8);
#ifdef MB_FULL_BLOCK
    STORE_DST(C20, C21, C22, C23, D2, 16);
    STORE_DST(C30, C31, C32, C33, D3, 24);
#endif // MB_FULL_BLOCK
#endif // MB > 8
}
