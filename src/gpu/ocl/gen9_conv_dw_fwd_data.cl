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

#if WITH_ELTWISE == 1
#include "gpu/ocl/ocl_post_ops.h"
#endif
#include "gpu/ocl/ocl_types.h"

#if IS_DW != 1
#error "Kernel supports depth-wise convolutions only"
#endif

#define DO_ELTWISE(blockC, nelems, alpha, beta, scale) \
    do { \
        for (uint i = 0; i < nelems; i++) \
            blockC[i] = fwd_eltwise(blockC[i], alpha, beta, scale); \
    } while (0)

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if SUB_GROUP_SIZE != 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
#endif
__kernel void
gen9_conv_dw_fwd(const __global DATA_T *src, const __global DATA_T *wei,
        const __global DATA_T *bias, __global DATA_T *dst, float eltwise_alpha,
        float eltwise_beta, float eltwise_scale, float sum_scale) {

#ifdef VER_8OW16C
    const int osp = get_global_id(1);
    const int od = osp / (OWB * OH);
    const int ohw = osp % (OWB * OH);
    const int ow = (ohw % OWB) * OW_BLOCK;
    const int oh = ohw / OWB;
    const int g = get_group_id(0) * OC_BLOCK;
    const int mb = get_global_id(2) * MB_BLOCK;

    const int id = od * SD - PD;
    const int ih = oh * SH - PH;
    const int iw = ow * SW - PW;

    dst += mb * G * OD * OH * OW + g * OD * OH * OW * MB_BLOCK
            + (od * OH * OW + oh * OW + ow) * MB_BLOCK * OC_BLOCK;
    src += mb * G * ID * IH * IW + g * ID * IH * IW * MB_BLOCK
            + (id * IH * IW + ih * IW + iw) * MB_BLOCK * IC_BLOCK;
    wei += g * KD * KH * KW;

#if WITH_BIAS
    DATA_T S00[OW_BLOCK];
    DATA_T B = AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)&bias[g]));
    __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
    for (int k = 0; k < OW_BLOCK; k++) {
        S00[k] = B;
    }
#else
    DATA_T S00[OW_BLOCK] = {DATA_ZERO};
#endif

#if KH != 1 || KW != 1 || KD != 1
    for (int kd = 0; kd < KD; kd++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {

                if (id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID)
                    continue;
                if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH)
                    continue;

                const __global DATA_T *wei1
                        = wei + (kd * KH * KW + kh * KW + kw) * OC_BLOCK;
                const __global DATA_T *src1 = src
                        + (kd * (1 + DD) * IH * IW + kh * (1 + DH) * IW
                                  + kw * (1 + DW))
                                * MB_BLOCK * IC_BLOCK;
#else
    const int kw = 0;
    const __global DATA_T *wei1 = wei;
    const __global DATA_T *src1 = src;
#endif
                DATA_T B0 = AS_DATA_T(
                        BLOCK_READ((const __global BLOCK_DATA_T *)(wei1)));
                DATA_T A0;

                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int k = 0; k < OW_BLOCK; k++) {

                    if (iw + kw * (1 + DW) + k * SW < 0
                            || iw + kw * (1 + DW) + k * SW >= IW)
                        A0 = DATA_ZERO;
                    else
                        A0 = AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T
                                        *)(&src1[k * SW * IC_BLOCK])));

                    S00[k] = fma(A0, (DATA_T)B0, S00[k]);
                }
#if KH != 1 || KW != 1 || KD != 1
            }
#endif

#if WITH_SUM == 1 || WITH_ELTWISE == 1
    DATA_T D00[OW_BLOCK];
    __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
    for (int k = 0; k < OW_BLOCK; k++) {
#if WITH_SUM == 1
        D00[k] = AS_DATA_T(
                BLOCK_READ((const __global BLOCK_DATA_T *)&dst[k * OC_BLOCK]));
#if SUM_SCALE == 1
        S00[k] += D00[k];
#else
        S00[k] = fma(D00[k], (DATA_T)sum_scale, S00[k]);
#endif
#endif
#if WITH_ELTWISE == 1
        S00[k] = fwd_eltwise(
                S00[k], eltwise_alpha, eltwise_beta, eltwise_scale);
#endif
    }
#endif

    __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
    for (int k = 0; k < OW_BLOCK; k++) {
        BLOCK_WRITE(
                (__global BLOCK_DATA_T *)&dst[k * OC_BLOCK], AS_UINT_T(S00[k]));
    }

#endif

#ifdef VER_16MB16C
    const int osp = get_global_id(1);
    const int od = osp / (OWB * OH);
    const int ohw = osp % (OWB * OH);
    const int ow = (ohw % OWB) * OW_BLOCK;
    const int oh = ohw / OWB;
    const int g = get_group_id(0) * OC_BLOCK;
    const int mb = get_global_id(2) * MB_BLOCK;

    const int id = od * SD - PD;
    const int ih = oh * SH - PH;
    const int iw = ow * SW - PW;

    dst += mb * G * OD * OH * OW + g * OD * OH * OW * MB_BLOCK
            + (od * OH * OW + oh * OW + ow) * MB_BLOCK * OC_BLOCK;
    src += mb * G * ID * IH * IW + g * ID * IH * IW * MB_BLOCK
            + (id * IH * IW + ih * IW + iw) * MB_BLOCK * IC_BLOCK;
    wei += g * KD * KH * KW;

#if WITH_BIAS
    DATA8_T S00, S01;
    DATA_T B = AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)&bias[g]));
    __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
    for (int k = 0; k < 8; k++) {
        S00[k] = B;
        S01[k] = B;
    }
#else
    DATA8_T S00 = DATA_ZERO;
    DATA8_T S01 = DATA_ZERO;
#endif

#if KH != 1 || KW != 1 || KD != 1
    for (int kd = 0; kd < KD; kd++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                if (id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID)
                    continue;
                if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH)
                    continue;
                if (iw + kw * (1 + DW) < 0 || iw + kw * (1 + DW) >= IW)
                    continue;

                const __global DATA_T *wei1
                        = wei + (kd * KH * KW + kh * KW + kw) * OC_BLOCK;
                const __global DATA_T *src1 = src
                        + (kd * (1 + DD) * IH * IW + kh * (1 + DH) * IW
                                  + kw * (1 + DW))
                                * MB_BLOCK * IC_BLOCK;
#else
    const __global DATA_T *wei1 = wei;
    const __global DATA_T *src1 = src;
#endif
                DATA8_T A0 = AS_DATA8_T(
                        BLOCK_READ8((const __global BLOCK_DATA_T *)(src1)));
                DATA8_T A1 = AS_DATA8_T(BLOCK_READ8(
                        (const __global BLOCK_DATA_T *)&src1[8 * IC_BLOCK]));

                DATA_T B0 = AS_DATA_T(
                        BLOCK_READ((const __global BLOCK_DATA_T *)(wei1)));

                S00 = fma(A0, (DATA8_T)B0, S00);
                S01 = fma(A1, (DATA8_T)B0, S01);
#if KH != 1 || KW != 1 || KD != 1
            }
#endif

#if WITH_SUM == 1
    DATA8_T D00 = AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)dst));
    DATA8_T D01 = AS_DATA8_T(
            BLOCK_READ8((const __global BLOCK_DATA_T *)&dst[8 * OC_BLOCK]));

#if SUM_SCALE == 1
    S00 += D00;
    S01 += D01;
#else
    S00 = fma(D00, (DATA8_T)sum_scale, S00);
    S01 = fma(D01, (DATA8_T)sum_scale, S01);
#endif
#endif
#if WITH_ELTWISE == 1
    DO_ELTWISE(S00, 8, eltwise_alpha, eltwise_beta, eltwise_scale);
    DO_ELTWISE(S01, 8, eltwise_alpha, eltwise_beta, eltwise_scale);
#endif

    BLOCK_WRITE8((__global BLOCK_DATA_T *)&dst[0], AS_UINT8_T(S00));
    BLOCK_WRITE8((__global BLOCK_DATA_T *)&dst[8 * OC_BLOCK], AS_UINT8_T(S01));

#endif
    return;
}
