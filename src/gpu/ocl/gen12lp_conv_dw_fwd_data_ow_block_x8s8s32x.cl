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

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

inline ushort16 read_block16(const __global ushort *p)
        __attribute__((overloadable)) {
    ushort16 __builtin_IB_simd_block_read_16_global_h(const __global ushort *p)
            __attribute__((const));
    return __builtin_IB_simd_block_read_16_global_h(p);
}

#if SCALES_PER_OC
#define SCALE scales.s0101010101010101
#elif SCALES_COMMON
#define SCALE scale
#else
#define SCALE 1
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_dw_fwd_ow_block_x8s8s32x(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global DATA_T *dst, float eltwise_alpha, float eltwise_beta,
        float eltwise_scale, float sum_scale, float scale,
        const __global float *scales_per_oc) {
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

    const __global uchar *src_original = src;
    dst += mb * G * OD * OH * OW + g * OD * OH * OW * MB_BLOCK
            + (od * OH * OW + oh * OW + ow) * MB_BLOCK * OC_BLOCK;
    src += mb * G * ID * IH * IW + g * ID * IH * IW * MB_BLOCK
            + (id * IH * IW + ih * IW + iw) * MB_BLOCK * IC_BLOCK;
    wei += g * KD * KH * KW;
    int16 S0 = 0;
    int16 S1 = 0;

#if SCALES_PER_OC
    // TODO: use block read after fix from compiler
    // float2 scales = as_float2(intel_sub_group_block_read_ul((const __global ulong *)&scales_per_oc[g]));
    float2 scales;
    scales.s0 = scales_per_oc[g + 2 * get_sub_group_local_id()];
    scales.s1 = scales_per_oc[g + 2 * get_sub_group_local_id() + 1];
#endif

#if WITH_BIAS
    // change after fix from compiler
    // float2 B = as_float2(intel_sub_group_block_read_ul((const __global ulong *)&bias[g]));
    float2 B;
    B.s0 = bias[g + 2 * get_sub_group_local_id()];
    B.s1 = bias[g + 2 * get_sub_group_local_id() + 1];
    S0 = convert_int16(B.s0101010101010101);
    S1 = convert_int16(B.s0101010101010101);
#endif
    for (int kd = 0; kd < KD; kd++) {
        if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
            src += IC_BLOCK * MB_BLOCK * IH * IW * (1 + DD);
            wei += OC_BLOCK * KH * KW;
            continue;
        }
        __attribute__((opencl_unroll_hint)) for (int kh = 0; kh < KH; kh++) {
            if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
                wei += OC_BLOCK * KW;
                continue;
            }
#if SW == 2
            ushort16 AAA = 0;
#else
            ushort AAA;
#endif
            ushort16 AA = 0;

            /* get main block */
#if IW < OW_NCHUNK * OW_BLOCK * SW
            if (iw + SW * OW_BLOCK >= IW) {
                if (iw >= 0) {
                    AA.s0 = intel_sub_group_block_read_us(
                            (const __global ushort *)(&src[0 * IC_BLOCK]));
                }
#if IW_TAIL > 15
                AA.s12345678 = intel_sub_group_block_read_us8(
                        (const __global ushort *)(&src[PW * IC_BLOCK]));
                AA.s9abc = intel_sub_group_block_read_us4(
                        (const __global ushort *)(&src[(PW + 8) * IC_BLOCK]));
                AA.sde = intel_sub_group_block_read_us2(
                        (const __global ushort *)(&src[(PW + 12) * IC_BLOCK]));
                AA.sf = intel_sub_group_block_read_us(
                        (const __global ushort *)(&src[(PW + 14) * IC_BLOCK]));
                for (int i = 0; i < IW_TAIL - 15; ++i) {
                    AAA[i] = intel_sub_group_block_read_us((const __global
                                    ushort *)(&src[(i + PW + 15) * IC_BLOCK]));
                }
#else
                for (int i = PW; i < IW_TAIL + PW; ++i) {
                    AA[i] = intel_sub_group_block_read_us(
                            (const __global ushort *)(&src[i * IC_BLOCK]));
                }
#endif
            } else {
#endif
#if SW == 1
#if OW_NCHUNK > 1
                if (iw >= 0) {
                    AA.s0 = intel_sub_group_block_read_us(
                            (const __global ushort *)(&src[0 * IC_BLOCK]));
                }
#endif
#if (OW_BLOCK & 0b1000) == 0b1000
                AA.s12345678 = intel_sub_group_block_read_us8(
                        (const __global ushort *)(&src[(PW)*IC_BLOCK]));
#endif
#if (OW_BLOCK & 0b100) == 0b100
                *((ushort4 *)(((ushort *)&AA) + PW + (OW_BLOCK & 0b1000)))
                        = intel_sub_group_block_read_us4((const __global ushort
                                        *)(&src[(PW + (OW_BLOCK & 0b1000))
                                * IC_BLOCK]));
#endif
#if (OW_BLOCK & 0b10) == 0b10
                *((ushort2 *)(((ushort *)&AA) + PW + (OW_BLOCK & 0b1100)))
                        = intel_sub_group_block_read_us2((const __global ushort
                                        *)(&src[(PW + (OW_BLOCK & 0b1100))
                                * IC_BLOCK]));
#endif
#if (OW_BLOCK & 0b1) == 0b1
                *(((ushort *)&AA) + PW + (OW_BLOCK & 0b1110))
                        = intel_sub_group_block_read_us((const __global ushort
                                        *)(&src[(PW + (OW_BLOCK & 0b1110))
                                * IC_BLOCK]));
#endif
#if OW_NCHUNK > 1
                if (iw + OW_BLOCK + PW < IW) {
                    AA[OW_BLOCK + PW] = intel_sub_group_block_read_us((
                            const __global ushort
                                    *)(&src[(OW_BLOCK + PW) * IC_BLOCK]));
                }
#endif

#elif SW == 2
#define READ_BLOCK (2 * (OW_BLOCK - 1) + 2)
#if OW_BLOCK >= 8
#if OW_NCHUNK > 1
            if (iw >= 0) {
                AA = read_block16(
                        (const __global ushort *)(&src[0 * IC_BLOCK]));
            } else {
#endif
                AA.s12345678 = intel_sub_group_block_read_us8(
                        (const __global ushort *)(&src[PW * IC_BLOCK]));
                AA.s9abc = intel_sub_group_block_read_us4(
                        (const __global ushort *)(&src[(PW + 8) * IC_BLOCK]));
                AA.sde = intel_sub_group_block_read_us2(
                        (const __global ushort *)(&src[(PW + 12) * IC_BLOCK]));
                AA.sf = intel_sub_group_block_read_us(
                        (const __global ushort *)(&src[(PW + 14) * IC_BLOCK]));
#if OW_NCHUNK > 1
            }
#endif
#if ((READ_BLOCK - 15) & 0b1000) == 0b1000
            AAA.s01234567 = intel_sub_group_block_read_us8(
                    (const __global ushort *)(&src[(PW + 15) * IC_BLOCK]));
#endif
#if ((READ_BLOCK - 15) & 0b100) == 0b100
            *((ushort4 *)(((ushort *)&AAA) + ((READ_BLOCK - 15) & 0b1000)))
                    = intel_sub_group_block_read_us4((const __global ushort
                                    *)(&src[(PW + 15
                                                    + ((READ_BLOCK - 15)
                                                            & 0b1000))
                            * IC_BLOCK]));
#endif
#if ((READ_BLOCK - 15) & 0b10) == 0b10
            *((ushort2 *)(((ushort *)&AAA) + ((READ_BLOCK - 15) & 0b1100)))
                    = intel_sub_group_block_read_us2((const __global ushort
                                    *)(&src[(PW + 15
                                                    + ((READ_BLOCK - 15)
                                                            & 0b1100))
                            * IC_BLOCK]));
#endif
#if ((READ_BLOCK - 15) & 0b1) == 0b1
            *(((ushort *)&AAA) + ((READ_BLOCK - 15) & 0b1110))
                    = intel_sub_group_block_read_us((const __global ushort
                                    *)(&src[(PW + 15
                                                    + ((READ_BLOCK - 15)
                                                            & 0b1110))
                            * IC_BLOCK]));
#endif
#else // OW_BLOCK >= 8
#if OW_NCHUNK > 1
            if (iw >= 0) {
                AA.s0 = intel_sub_group_block_read_us(
                        (const __global ushort *)(&src[0 * IC_BLOCK]));
            }
#endif
#if (READ_BLOCK & 0b1000) == 0b1000
            AA.s12345678 = intel_sub_group_block_read_us8(
                    (const __global ushort *)(&src[(PW)*IC_BLOCK]));
#endif
#if (READ_BLOCK & 0b100) == 0b100
            *((ushort4 *)(((ushort *)&AA) + PW + (READ_BLOCK & 0b1000)))
                    = intel_sub_group_block_read_us4((const __global ushort
                                    *)(&src[(PW + (READ_BLOCK & 0b1000))
                            * IC_BLOCK]));
#endif
#if (READ_BLOCK & 0b10) == 0b10
            *((ushort2 *)(((ushort *)&AA) + PW + (READ_BLOCK & 0b1100)))
                    = intel_sub_group_block_read_us2((const __global ushort
                                    *)(&src[(PW + (READ_BLOCK & 0b1100))
                            * IC_BLOCK]));
#endif
#if (READ_BLOCK & 0b1) == 0b1
            *(((ushort *)&AA) + PW + (READ_BLOCK & 0b1110))
                    = intel_sub_group_block_read_us((const __global ushort
                                    *)(&src[(PW + (READ_BLOCK & 0b1110))
                            * IC_BLOCK]));
#endif
#endif

#endif

#if IW < OW_NCHUNK * OW_BLOCK * SW
            }
#endif
            ushort4 WW = 0;
            WW.s01 = intel_sub_group_block_read_us2(
                    (const __global ushort *)wei);
            WW.s2 = intel_sub_group_block_read_us(
                    (const __global ushort *)wei + OC_BLOCK);

            SRC_DATA16_T A0 = 0, A1 = 0;
            A0.s01234567 = AS_SRC_DATA16_T(AA.s01234567).s02468ace;
            A0.s89abcdef = AS_SRC_DATA16_T(AA.s89abcdef).s02468ace;
            A1.s01234567 = AS_SRC_DATA16_T(AA.s01234567).s13579bdf;
            A1.s89abcdef = AS_SRC_DATA16_T(AA.s89abcdef).s13579bdf;
#if SW == 2
            SRC_DATA16_T right0, right1;
            right0.s01234567 = AS_SRC_DATA16_T(AAA.s01234567).s02468ace;
            right0.s89abcdef = AS_SRC_DATA16_T(AAA.s89abcdef).s02468ace;
            right1.s01234567 = AS_SRC_DATA16_T(AAA.s01234567).s13579bdf;
            right1.s89abcdef = AS_SRC_DATA16_T(AAA.s89abcdef).s13579bdf;
#else
#if OW_BLOCK == 15
            SRC_DATA_T right0, right1;
            right0 = AS_SRC_DATA2_T(AAA).s0;
            right1 = AS_SRC_DATA2_T(AAA).s1;
#endif
#endif
            char8 W = as_char8(WW);
#if SW == 1
            S0.s0 = IMAD(A0.s0123, W.s0246, S0.s0);
#if OW_BLOCK >= 2
            S0.s2 = IMAD(A0.s1234, W.s0246, S0.s2);
#endif
#if OW_BLOCK >= 3
            S0.s4 = IMAD(A0.s2345, W.s0246, S0.s4);
#endif
#if OW_BLOCK >= 4
            S0.s6 = IMAD(A0.s3456, W.s0246, S0.s6);
#endif
#if OW_BLOCK >= 5
            S0.s8 = IMAD(A0.s4567, W.s0246, S0.s8);
#endif
#if OC_BLOCK >= 6
            S0.sa = IMAD(A0.s5678, W.s0246, S0.sa);
#endif
#if OW_BLOCK >= 7
            S0.sc = IMAD(A0.s6789, W.s0246, S0.sc);
#endif
#if OW_BLOCK >= 8
            S0.se = IMAD(A0.s789a, W.s0246, S0.se);
#endif
#if OW_BLOCK >= 9
            S1.s0 = IMAD(A0.s89ab, W.s0246, S1.s0);
#endif
#if OW_BLOCK >= 10
            S1.s2 = IMAD(A0.s9abc, W.s0246, S1.s2);
#endif
#if OW_BLOCK >= 11
            S1.s4 = IMAD(A0.sabcd, W.s0246, S1.s4);
#endif
#if OW_BLOCK >= 12
            S1.s6 = IMAD(A0.sbcde, W.s0246, S1.s6);
#endif
#if OW_BLOCK >= 13
            S1.s8 = IMAD(A0.scdef, W.s0246, S1.s8);
#endif
#if OW_BLOCK >= 14
            S1.sa = IMAD((SRC_DATA4_T)(A0.sde, A0.sf, 0), W.s0246, S1.sa);
#endif
#if OW_BLOCK >= 15
            S1.sc = IMAD((SRC_DATA4_T)(A0.sef, right0, 0), W.s0246, S1.sc);
#endif

            S0.s1 = IMAD(A1.s0123, W.s1357, S0.s1);
#if OW_BLOCK >= 2
            S0.s3 = IMAD(A1.s1234, W.s1357, S0.s3);
#endif
#if OW_BLOCK >= 3
            S0.s5 = IMAD(A1.s2345, W.s1357, S0.s5);
#endif
#if OW_BLOCK >= 4
            S0.s7 = IMAD(A1.s3456, W.s1357, S0.s7);
#endif
#if OW_BLOCK >= 5
            S0.s9 = IMAD(A1.s4567, W.s1357, S0.s9);
#endif
#if OW_BLOCK >= 6
            S0.sb = IMAD(A1.s5678, W.s1357, S0.sb);
#endif
#if OW_BLOCK >= 7
            S0.sd = IMAD(A1.s6789, W.s1357, S0.sd);
#endif
#if OW_BLOCK >= 8
            S0.sf = IMAD(A1.s789a, W.s1357, S0.sf);
#endif
#if OW_BLOCK >= 9
            S1.s1 = IMAD(A1.s89ab, W.s1357, S1.s1);
#endif
#if OW_BLOCK >= 10
            S1.s3 = IMAD(A1.s9abc, W.s1357, S1.s3);
#endif
#if OW_BLOCK >= 11
            S1.s5 = IMAD(A1.sabcd, W.s1357, S1.s5);
#endif
#if OW_BLOCK >= 12
            S1.s7 = IMAD(A1.sbcde, W.s1357, S1.s7);
#endif
#if OW_BLOCK >= 13
            S1.s9 = IMAD(A1.scdef, W.s1357, S1.s9);
#endif
#if OW_BLOCK >= 14
            S1.sb = IMAD((SRC_DATA4_T)(A1.sde, A1.sf, 0), W.s1357, S1.sb);
#endif
#if OW_BLOCK >= 15
            S1.sd = IMAD((SRC_DATA4_T)(A1.sef, right1, 0), W.s1357, S1.sd);
#endif

#elif SW == 2
            S0.s0 = IMAD(A0.s0123, W.s0246, S0.s0);
#if OW_BLOCK >= 2
            S0.s2 = IMAD(A0.s2345, W.s0246, S0.s2);
#endif
#if OW_BLOCK >= 3
            S0.s4 = IMAD(A0.s4567, W.s0246, S0.s4);
#endif
#if OW_BLOCK >= 4
            S0.s6 = IMAD(A0.s6789, W.s0246, S0.s6);
#endif
#if OW_BLOCK >= 5
            S0.s8 = IMAD(A0.s89ab, W.s0246, S0.s8);
#endif
#if OW_BLOCK >= 6
            S0.sa = IMAD(A0.sabcd, W.s0246, S0.sa);
#endif
#if OW_BLOCK >= 7
            S0.sc = IMAD(A0.scdef, W.s0246, S0.sc);
#endif
#if OW_BLOCK >= 8
            S0.se = IMAD((SRC_DATA4_T)(A0.sef, right0.s0, 0), W.s0246, S0.se);
#endif
#if OW_BLOCK >= 9
            S1.s0 = IMAD(right0.s0123, W.s0246, S1.s0);
#endif
#if OW_BLOCK >= 10
            S1.s2 = IMAD(right0.s2345, W.s0246, S1.s2);
#endif
#if OW_BLOCK >= 11
            S1.s4 = IMAD(right0.s4567, W.s0246, S1.s4);
#endif
#if OW_BLOCK >= 12
            S1.s6 = IMAD(right0.s6789, W.s0246, S1.s6);
#endif
#if OW_BLOCK >= 13
            S1.s8 = IMAD(right0.s89ab, W.s0246, S1.s8);
#endif
#if OW_BLOCK >= 14
            S1.sa = IMAD(right0.sabcd, W.s0246, S1.sa);
#endif
#if OW_BLOCK >= 15
            S1.sc = IMAD(right0.scdef, W.s0246, S1.sc);
#endif

            S0.s1 = IMAD(A1.s0123, W.s1357, S0.s1);
#if OW_BLOCK >= 2
            S0.s3 = IMAD(A1.s2345, W.s1357, S0.s3);
#endif
#if OW_BLOCK >= 3
            S0.s5 = IMAD(A1.s4567, W.s1357, S0.s5);
#endif
#if OW_BLOCK >= 4
            S0.s7 = IMAD(A1.s6789, W.s1357, S0.s7);
#endif
#if OW_BLOCK >= 5
            S0.s9 = IMAD(A1.s89ab, W.s1357, S0.s9);
#endif
#if OW_BLOCK >= 6
            S0.sb = IMAD(A1.sabcd, W.s1357, S0.sb);
#endif
#if OW_BLOCK >= 7
            S0.sd = IMAD(A1.scdef, W.s1357, S0.sd);
#endif
#if OW_BLOCK >= 8
            S0.sf = IMAD((SRC_DATA4_T)(A1.sef, right1.s0, 0), W.s1357, S0.sf);
#endif
#if OW_BLOCK >= 9
            S1.s1 = IMAD(right1.s0123, W.s1357, S1.s1);
#endif
#if OW_BLOCK >= 10
            S1.s3 = IMAD(right1.s2345, W.s1357, S1.s3);
#endif
#if OW_BLOCK >= 11
            S1.s5 = IMAD(right1.s4567, W.s1357, S1.s5);
#endif
#if OW_BLOCK >= 12
            S1.s7 = IMAD(right1.s6789, W.s1357, S1.s7);
#endif
#if OW_BLOCK >= 13
            S1.s9 = IMAD(right1.s89ab, W.s1357, S1.s9);
#endif
#if OW_BLOCK >= 14
            S1.sb = IMAD(right1.sabcd, W.s1357, S1.sb);
#endif
#if OW_BLOCK >= 15
            S1.sd = IMAD(right1.scdef, W.s1357, S1.sd);
#endif
#else
#error // SW > 2
#endif
            src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
            wei += OC_BLOCK * KW;
        }
        src += IC_BLOCK * MB_BLOCK * IW * (IH * (1 + DD) - KH * (1 + DH));
    }

#if WITH_ELTWISE || WITH_POST_SUM_ELTWISE || WITH_SUM && !SUM_SCALE \
        || SCALES_PER_OC || SCALES_COMMON
    float16 tmp00 = convert_float16(S0) * SCALE;
    float16 tmp01 = convert_float16(S1) * SCALE;
#define CONVERT_TO_ACC convert_float16
#define ACC0 tmp00
#define ACC1 tmp01
#define DO_ELTWISE() \
    do { \
        for (uint i = 0; i < 16; i++) { \
            tmp00[i] = fwd_eltwise( \
                    tmp00[i], eltwise_alpha, eltwise_beta, eltwise_scale); \
            tmp01[i] = fwd_eltwise( \
                    tmp01[i], eltwise_alpha, eltwise_beta, eltwise_scale); \
        } \
    } while (0)
#else
#define CONVERT_TO_ACC convert_int16
#define ACC0 S0
#define ACC1 S1
#endif

#if WITH_ELTWISE && !WITH_POST_SUM_ELTWISE
    DO_ELTWISE();
#endif

#if WITH_SUM
    DST_DATA16_T D0 = 0;
    DST_DATA16_T D1 = 0;

#if OW_TAIL != 0
    if (ow + OW_BLOCK >= OW) {
#if OW_TAIL > 8
        D0 = AS_DST_DATA16_T(
                intel_sub_group_block_read_us8((__global ushort *)dst));
        for (int i = 0; i < OW_TAIL - 8; ++i) {
            ushort t = intel_sub_group_block_read_us(
                    (__global ushort *)dst + (8 + i) * OC_BLOCK / 2);
            D1[i * 2] = AS_DST_DATA2_T(t).s0;
            D1[i * 2 + 1] = AS_DST_DATA2_T(t).s1;
        }
#else
        for (int i = 0; i < OW_TAIL; ++i) {
            ushort t = intel_sub_group_block_read_us(
                    (__global ushort *)dst + (8 + i) * OC_BLOCK / 2);
            D0[i * 2] = AS_DST_DATA2_T(t).s0;
            D0[i * 2 + 1] = AS_DST_DATA2_T(t).s1;
        }
#endif
    } else {
#endif
#if OW_BLOCK >= 8
        D0 = AS_DST_DATA16_T(
                intel_sub_group_block_read_us8((__global ushort *)dst));
        __global DATA_T *dst1 = dst + 8 * OC_BLOCK;
#define DST_P dst1
#define SUM_BLOCK D1
#define SEND_TAIL (OW_BLOCK - 8)

#else
#define DST_P dst
#define SUM_BLOCK D0
#define SEND_TAIL OW_BLOCK
#endif

#if SEND_TAIL == 7
        SUM_BLOCK.s01234567 = AS_DST_DATA8_T(
                intel_sub_group_block_read_us4((__global ushort *)DST_P));
        SUM_BLOCK.s89ab = AS_DST_DATA4_T(intel_sub_group_block_read_us2(
                (__global ushort *)DST_P + 2 * OC_BLOCK));
        SUM_BLOCK.scd = AS_DST_DATA2_T(intel_sub_group_block_read_us(
                (__global ushort *)DST_P + 3 * OC_BLOCK));
#elif SEND_TAIL == 6
    SUM_BLOCK.s01234567 = AS_DST_DATA8_T(
            intel_sub_group_block_read_us4((__global ushort *)DST_P));
    SUM_BLOCK.s89ab = AS_DST_DATA4_T(intel_sub_group_block_read_us2(
            (__global ushort *)DST_P + 2 * OC_BLOCK));
#elif SEND_TAIL == 5
    SUM_BLOCK.s01234567 = AS_DST_DATA8_T(
            intel_sub_group_block_read_us4((__global ushort *)DST_P));
    SUM_BLOCK.s89 = AS_DST_DATA2_T(intel_sub_group_block_read_us(
            (__global ushort *)DST_P + 2 * OC_BLOCK));
#elif SEND_TAIL == 4
    SUM_BLOCK.s01234567 = AS_DST_DATA8_T(
            intel_sub_group_block_read_us4((__global ushort *)DST_P));
#elif SEND_TAIL == 3
    SUM_BLOCK.s0123 = AS_DST_DATA4_T(
            intel_sub_group_block_read_us2((__global ushort *)DST_P));
    SUM_BLOCK.s45 = AS_DST_DATA2_T(
            intel_sub_group_block_read_us((__global ushort *)DST_P + OC_BLOCK));
#elif SEND_TAIL == 2
    SUM_BLOCK.s0123 = AS_DST_DATA4_T(
            intel_sub_group_block_read_us2((__global ushort *)DST_P));
#elif SEND_TAIL == 1
    SUM_BLOCK.s01 = AS_DST_DATA2_T(
            intel_sub_group_block_read_us((__global ushort *)DST_P));
#endif
#if OW % OW_BLOCK != 0
    }
#endif
#if SUM_SCALE
    ACC0 += CONVERT_TO_ACC(D0);
    ACC1 += CONVERT_TO_ACC(D1);
#else // SUM_SCALE
    ACC0 += CONVERT_TO_ACC(D0) * sum_scale;
    ACC1 += CONVERT_TO_ACC(D1) * sum_scale;
#endif // SUM_SCALE
#endif // WITH_SUM

#if WITH_POST_SUM_ELTWISE
    DO_ELTWISE();
#endif

    DST_DATA16_T R0 = TO_DST16(ACC0);
    DST_DATA16_T R1 = TO_DST16(ACC1);

#if OW_TAIL != 0
    if (ow + OW_BLOCK >= OW) {
#if OW_TAIL > 8
        intel_sub_group_block_write_us8((__global ushort *)dst, as_ushort8(R0));
        for (int i = 0; i < OW_TAIL - 8; ++i) {
            intel_sub_group_block_write_us(
                    (__global ushort *)dst + (8 + i) * OC_BLOCK / 2,
                    as_ushort((DST_DATA2_T)(R1[i * 2], R1[i * 2 + 1])));
        }
#else
        for (int i = 0; i < OW_TAIL; ++i) {
            intel_sub_group_block_write_us(
                    (__global ushort *)dst + i * OC_BLOCK / 2,
                    as_ushort((DST_DATA2_T)(R0[i * 2], R0[i * 2 + 1])));
        }
#endif
    } else {
#endif

#if OW_BLOCK >= 8
        intel_sub_group_block_write_us8((__global ushort *)dst, as_ushort8(R0));
        dst += 8 * OC_BLOCK;
#define SEND_BLOCK R1
#define SEND_TAIL (OW_BLOCK - 8)

#else
#define SEND_BLOCK R0
#define SEND_TAIL OW_BLOCK
#endif

#if SEND_TAIL == 7
        intel_sub_group_block_write_us4(
                (__global ushort *)dst, as_ushort4(SEND_BLOCK.s01234567));
        intel_sub_group_block_write_us2((__global ushort *)dst + 2 * OC_BLOCK,
                as_ushort2(SEND_BLOCK.s89ab));
        intel_sub_group_block_write_us((__global ushort *)dst + 3 * OC_BLOCK,
                as_ushort(SEND_BLOCK.scd));
#elif SEND_TAIL == 6
    intel_sub_group_block_write_us4(
            (__global ushort *)dst, as_ushort4(SEND_BLOCK.s01234567));
    intel_sub_group_block_write_us2((__global ushort *)dst + 2 * OC_BLOCK,
            as_ushort2(SEND_BLOCK.s89ab));
#elif SEND_TAIL == 5
    intel_sub_group_block_write_us4(
            (__global ushort *)dst, as_ushort4(SEND_BLOCK.s01234567));
    intel_sub_group_block_write_us(
            (__global ushort *)dst + 2 * OC_BLOCK, as_ushort(SEND_BLOCK.s89));
#elif SEND_TAIL == 4
    intel_sub_group_block_write_us4(
            (__global ushort *)dst, as_ushort4(SEND_BLOCK.s01234567));
#elif SEND_TAIL == 3
    intel_sub_group_block_write_us2(
            (__global ushort *)dst, as_ushort2(SEND_BLOCK.s0123));
    intel_sub_group_block_write_us(
            (__global ushort *)dst + OC_BLOCK, as_ushort(SEND_BLOCK.s45));
#elif SEND_TAIL == 2
    intel_sub_group_block_write_us2(
            (__global ushort *)dst, as_ushort2(SEND_BLOCK.s0123));
#elif SEND_TAIL == 1
    intel_sub_group_block_write_us(
            (__global ushort *)dst, as_ushort(SEND_BLOCK.s01));
#endif

#if OW % OW_BLOCK != 0
    }
#endif
}
