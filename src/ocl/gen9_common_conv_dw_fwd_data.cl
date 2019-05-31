/*******************************************************************************
* copyright 2019 intel corporation
*
* licensed under the apache license, version 2.0 (the "license");
* you may not use this file except in compliance with the license.
* you may obtain a copy of the license at
*
*     http://www.apache.org/licenses/license-2.0
*
* unless required by applicable law or agreed to in writing, software
* distributed under the license is distributed on an "as is" basis,
* without warranties or conditions of any kind, either express or implied.
* see the license for the specific language governing permissions and
* limitations under the license.
*******************************************************************************/

#include "ocl/ocl_types.h"

#if IS_DW != 1
#    error "Kernel supports depth-wise convolutions only"
#endif

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
#if SUB_GROUP_SIZE != 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#endif
__kernel void gen9_common_conv_fwd_kernel(const __global DATA_T *src,
        const __global DATA_T *wei, const __global DATA_T *bias,
        __global DATA_T *dst,
#if DT_F32
        float relu_negative_slope, float sum_scale) {
#endif
#if DT_F16
    float relu_negative_slope_, float sum_scale_) {
        half relu_negative_slope = relu_negative_slope_;
        half sum_scale = sum_scale_;
#endif

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

#    if WITH_BIAS
        DATA_T S00[OW_BLOCK];
        DATA_T B = AS_DATA_T(BLOCK_READ((const __global uint *)&bias[g]));
        __attribute__((opencl_unroll_hint(OW_BLOCK)))
        for (int k = 0; k < OW_BLOCK; k++) {
            S00[k] = B;
        }
#    else
        DATA_T S00[OW_BLOCK] = { 0.0 };
#    endif

#    if KH != 1 || KW != 1 || KD != 1
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
                            + kw * (1 + DW)) * MB_BLOCK * IC_BLOCK;
#    else
        const int kw = 0;
        const __global DATA_T *wei1 = wei;
        const __global DATA_T *src1 = src;
#    endif
                    DATA_T B0 = AS_DATA_T(
                            BLOCK_READ((const __global uint *)(wei1)));
                    DATA_T A0;

                    __attribute__((opencl_unroll_hint(OW_BLOCK)))
                    for (int k = 0; k < OW_BLOCK; k++) {

                        if (iw + kw * (1 + DW) + k * SW < 0
                            || iw + kw * (1 + DW) + k * SW >= IW)
                            A0 = 0.0;
                        else
                            A0 = AS_DATA_T(BLOCK_READ((const __global uint
                                            *)(&src1[k * SW * IC_BLOCK])));

                        S00[k] = fma(A0, (DATA_T)B0, S00[k]);
                    }
#    if KH != 1 || KW != 1 || KD != 1
                }
#    endif

#    if WITH_SUM_RELU == 1
        DATA_T D00[OW_BLOCK];
        __attribute__((opencl_unroll_hint(OW_BLOCK)))
        for (int k = 0; k < OW_BLOCK; k++) {
            D00[k] = AS_DATA_T(
                    BLOCK_READ((const __global uint *)&dst[k * OC_BLOCK]));
#        if SUM_SCALE == 1
            S00[k] += D00[k];
#        else
            S00[k] = fma(D00[k], (DATA_T)sum_scale, S00[k]);
#        endif
            if (S00[k] < 0)
                S00[k] *= relu_negative_slope;
        }
#    else
#        if WITH_RELU == 1
        for (uint i = 0; i < OW_BLOCK; i++) {
            if (S00[i] < 0)
                S00[i] *= relu_negative_slope;
        }
#        endif
#        if WITH_SUM == 1
        DATA_T D00[OW_BLOCK];
        __attribute__((opencl_unroll_hint(OW_BLOCK)))
        for (int k = 0; k < OW_BLOCK; k++) {
            D00[k] = AS_DATA_T(
                    BLOCK_READ((const __global uint *)&dst[k * OC_BLOCK]));
#            if SUM_SCALE == 1
            S00[k] += D00[k];
#            else
            S00[k] = fma(D00[k], (DATA_T)sum_scale, S00[k]);
#            endif
        }
#        endif
#    endif

        __attribute__((opencl_unroll_hint(OW_BLOCK)))
        for (int k = 0; k < OW_BLOCK; k++) {
            BLOCK_WRITE((__global unsigned int *)&dst[k * OC_BLOCK],
                    AS_UINT_T(S00[k]));
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

#    if WITH_BIAS
        DATA8_T S00, S01;
        DATA_T B = AS_DATA_T(BLOCK_READ((const __global uint *)&bias[g]));
        __attribute__((opencl_unroll_hint(OW_BLOCK)))
        for (int k = 0; k < 8; k++) {
            S00[k] = B;
            S01[k] = B;
        }
#    else
        DATA8_T S00 = 0.0;
        DATA8_T S01 = 0.0;
#    endif

#    if KH != 1 || KW != 1 || KD != 1
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
                            + kw * (1 + DW)) * MB_BLOCK * IC_BLOCK;
#    else
        const __global DATA_T *wei1 = wei;
        const __global DATA_T *src1 = src;
#    endif
                    DATA8_T A0 = AS_DATA8_T(
                            BLOCK_READ8((const __global uint *)(src1)));
                    DATA8_T A1 = AS_DATA8_T(BLOCK_READ8(
                            (const __global uint *)&src1[8 * IC_BLOCK]));

                    DATA_T B0 = AS_DATA_T(
                            BLOCK_READ((const __global uint *)(wei1)));

                    S00 = fma(A0, (DATA8_T)B0, S00);
                    S01 = fma(A1, (DATA8_T)B0, S01);
#    if KH != 1 || KW != 1 || KD != 1
                }
#    endif

#    if WITH_SUM_RELU == 1
        DATA8_T D00 = AS_DATA8_T(BLOCK_READ8((const __global uint *)dst));
        DATA8_T D01 = AS_DATA8_T(
                BLOCK_READ8((const __global uint *)&dst[8 * OC_BLOCK]));

#        if SUM_SCALE == 1
        S00 += D00;
        S01 += D01;
#        else
        S00 = fma(D00, (DATA8_T)sum_scale, S00);
        S01 = fma(D01, (DATA8_T)sum_scale, S01);
#        endif
        for (uint i = 0; i < 8; i++) {
            if (S00[i] < 0)
                S00[i] *= relu_negative_slope;
            if (S01[i] < 0)
                S01[i] *= relu_negative_slope;
        }
#    else
#        if WITH_RELU == 1
        for (uint i = 0; i < 8; i++) {
            if (S00[i] < 0)
                S00[i] *= relu_negative_slope;
            if (S01[i] < 0)
                S01[i] *= relu_negative_slope;
        }
#        endif
#        if WITH_SUM == 1
        DATA8_T D00 = AS_DATA8_T(BLOCK_READ8((const __global uint *)dst));
        DATA8_T D01 = AS_DATA8_T(
                BLOCK_READ8((const __global uint *)&dst[8 * OC_BLOCK]));
#            if SUM_SCALE == 1
        S00 += D00;
        S01 += D01;
#            else
        S00 = fma(D00, (DATA8_T)sum_scale, S00);
        S01 = fma(D01, (DATA8_T)sum_scale, S01);
#            endif
#        endif
#    endif

        BLOCK_WRITE8((__global unsigned int *)&dst[0], AS_UINT8_T(S00));
        BLOCK_WRITE8(
                (__global unsigned int *)&dst[8 * OC_BLOCK], AS_UINT8_T(S01));

#endif
        return;
    }
