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

void block_read_dst(int n, DST_DATA_T *d, const __global DST_DATA_T *dst);
void block_write_dst(int n, const DST_DATA_T *d, __global DST_DATA_T *dst);

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_dw_fwd_ow_block_x8s8s32x(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global DST_DATA_T *dst, float eltwise_alpha, float eltwise_beta,
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

#if WITH_ELTWISE || WITH_SUM && !SUM_SCALE1 || SCALES_PER_OC || SCALES_COMMON
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

#if ELTWISE_IDX == 0
    DO_ELTWISE();
#endif

#if WITH_SUM
    DST_DATA16_T D0 = 0;
    DST_DATA16_T D1 = 0;

    if (OW_TAIL != 0 && ow + OW_BLOCK >= OW) {
        block_read_dst(min(8, OW_TAIL), &D0, dst);
        block_read_dst(OW_TAIL - 8, &D1, dst + 8 * OC_BLOCK);
    } else {
        block_read_dst(min(8, OW_BLOCK), &D0, dst);
        block_read_dst(OW_BLOCK - 8, &D1, dst + 8 * OC_BLOCK);
    }
#if SUM_SCALE1
    ACC0 += CONVERT_TO_ACC(D0);
    ACC1 += CONVERT_TO_ACC(D1);
#else // SUM_SCALE == 1
    ACC0 += CONVERT_TO_ACC(D0) * sum_scale;
    ACC1 += CONVERT_TO_ACC(D1) * sum_scale;
#endif // SUM_SCALE == 1
#endif // WITH_SUM

#if ELTWISE_IDX == 1
    DO_ELTWISE();
#endif

    DST_DATA16_T R0 = CONVERT_DST_DATA16_T(ACC0);
    DST_DATA16_T R1 = CONVERT_DST_DATA16_T(ACC1);

    if (OW_TAIL != 0 && ow + OW_BLOCK >= OW) {
        block_write_dst(min(8, OW_TAIL), &R0, dst);
        block_write_dst(OW_TAIL - 8, &R1, dst + 8 * OC_BLOCK);
    } else {
        block_write_dst(min(8, OW_BLOCK), &R0, dst);
        block_write_dst(OW_BLOCK - 8, &R1, dst + 8 * OC_BLOCK);
    }
}

// Shuffled read.
void block_read_dst(int n, DST_DATA_T *d, const __global DST_DATA_T *dst) {
#if DST_DT_S8 || DST_DT_U8
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < n / 8 * 8; i += 8) {
        ushort8 block = intel_sub_group_block_read_us8(
                (const __global ushort *)(dst + i * SUB_GROUP_SIZE * 2));
        *(DST_DATA16_T *)(&d[i * 2]) = AS_DST_DATA16_T(block);
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = n / 8 * 8; i < n; i++) {
        ushort block = intel_sub_group_block_read_us(
                (const __global ushort *)(dst + i * SUB_GROUP_SIZE * 2));
        *(DST_DATA2_T *)(&d[i * 2]) = AS_DST_DATA2_T(block);
    }
    return;
#elif DST_DT_S32 || DST_DT_F32
    int sglid = get_sub_group_local_id();
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < n; i++) {
        uint block0 = intel_sub_group_block_read(
                (const __global uint *)(dst + i * SUB_GROUP_SIZE * 2));
        uint block1 = intel_sub_group_block_read((const __global uint *)(dst
                + i * SUB_GROUP_SIZE * 2 + SUB_GROUP_SIZE));

        int from00 = min(2 * sglid, SUB_GROUP_SIZE - 1);
        int from01 = max(2 * sglid - 16, 0);
        int from10 = min(2 * sglid + 1, SUB_GROUP_SIZE - 1);
        int from11 = max(2 * sglid + 1 - 16, 0);

        uint block00 = intel_sub_group_shuffle(block0, from00);
        uint block01 = intel_sub_group_shuffle(block1, from01);
        uint block10 = intel_sub_group_shuffle(block0, from10);
        uint block11 = intel_sub_group_shuffle(block1, from11);

        block0 = (2 * sglid < SUB_GROUP_SIZE) ? block00 : block01;
        block1 = (2 * sglid + 1 < SUB_GROUP_SIZE) ? block10 : block11;

        *(DST_DATA2_T *)(&d[i * 2]) = AS_DST_DATA2_T((uint2)(block0, block1));
    }
    return;
#else
#error "Not expected"
#endif
}

// Shuffled write.
void block_write_dst(int n, const DST_DATA_T *d, __global DST_DATA_T *dst) {
#if DST_DT_S8 || DST_DT_U8
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < n / 8 * 8; i += 8) {
        intel_sub_group_block_write_us8(
                (__global ushort *)(dst + i * SUB_GROUP_SIZE * 2),
                as_ushort8(*(DST_DATA16_T *)(&d[i * 2])));
    }
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = n / 8 * 8; i < n; i++) {
        intel_sub_group_block_write_us(
                (__global ushort *)(dst + i * SUB_GROUP_SIZE * 2),
                as_ushort(*(DST_DATA2_T *)(&d[i * 2])));
    }
    return;
#elif DST_DT_S32 || DST_DT_F32
    int sglid = get_sub_group_local_id();
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < n; i++) {
        uint2 block = as_uint2(*(DST_DATA2_T *)(&d[i * 2]));

        int from0 = sglid / 2;
        int from1 = (16 + sglid) / 2;

        uint2 block0_ = intel_sub_group_shuffle(block, from0);
        uint2 block1_ = intel_sub_group_shuffle(block, from1);

        uint block0 = (sglid % 2 == 0) ? block0_.s0 : block0_.s1;
        uint block1 = (sglid % 2 == 0) ? block1_.s0 : block1_.s1;

        intel_sub_group_block_write(
                (__global uint *)(dst + i * SUB_GROUP_SIZE * 2), block0);
        intel_sub_group_block_write(
                (__global uint *)(dst + i * SUB_GROUP_SIZE * 2
                        + SUB_GROUP_SIZE),
                block1);
    }
    return;
#else
#error "Not expected"
#endif
}
