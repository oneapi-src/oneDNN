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
#include "gpu/ocl/ocl_types.h"
#if WITH_ELTWISE
#include "gpu/ocl/ocl_post_ops.h"
#endif

//   FWD fp32 Convolution Kernel with NHWC data layout support
//  Features:
//   - based on 8ow16c version of blocked implementation
//   - weights are blocked and padded: OIhw16i16o
//   - explicit tail processing for src and dst channels
//   - due to 16-bytes alignment requred, intel_sub_groups_block_write usage
//     is limited

#if ID > 1
#define CASE_3D 1
#else
#define CASE_3D 0
#endif

#define KDHW_SIZE (KD * KH * KW)

#define HAS_PAD_D (PD != 0 || PD_R != 0)
#define HAS_PAD_H (PH != 0 || PH_R != 0)
#define HAS_PAD_W (PW != 0 || PW_R != 0)

#define IS_1x1x1 (KH == 1 && KW == 1 && KD == 1)
#define IC_TAIL (IC_WO_PADDING % IC_BLOCK)
#define OC_TAIL (OC_WO_PADDING % OC_BLOCK)

#define DO_ELTWISE(blockC, nelems, alpha, beta, scale) \
    do { \
        for (uint i = 0; i < nelems; i++) \
            blockC[i] = fwd_eltwise(blockC[i], alpha, beta, scale); \
    } while (0)

#define TRANSPOSE_1(_block, _col) (float)(intel_sub_group_shuffle(_block, _col))

#define FMA8(a, b, c) fma((float)(a), (float)b, (float)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, _blockB1) \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_1(_blockA, 0), _result); \
        _result = FMA8(_blockB.s1, TRANSPOSE_1(_blockA, 1), _result); \
        _result = FMA8(_blockB.s2, TRANSPOSE_1(_blockA, 2), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_1(_blockA, 3), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_1(_blockA, 4), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_1(_blockA, 5), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_1(_blockA, 6), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_1(_blockA, 7), _result); \
        _result = FMA8(_blockB1.s0, TRANSPOSE_1(_blockA, 8), _result); \
        _result = FMA8(_blockB1.s1, TRANSPOSE_1(_blockA, 9), _result); \
        _result = FMA8(_blockB1.s2, TRANSPOSE_1(_blockA, 10), _result); \
        _result = FMA8(_blockB1.s3, TRANSPOSE_1(_blockA, 11), _result); \
        _result = FMA8(_blockB1.s4, TRANSPOSE_1(_blockA, 12), _result); \
        _result = FMA8(_blockB1.s5, TRANSPOSE_1(_blockA, 13), _result); \
        _result = FMA8(_blockB1.s6, TRANSPOSE_1(_blockA, 14), _result); \
        _result = FMA8(_blockB1.s7, TRANSPOSE_1(_blockA, 15), _result); \
    }

#define IS_ALIGNED(addr, b) (!(ulong)(addr) & (b))

void multiply_blocks_8x8_ic_tail(float *res, __local float *blockA, int i,
        float8 blockB00, float8 blockB01) {
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < min(8, IC_TAIL);
                                             j++) {
        *res = FMA8(blockB00[j], blockA[j * OW_BLOCK + i], *res);
    }
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < min(8, IC_TAIL);
                                             j++) {
        *res = FMA8(blockB01[j], blockA[(8 + j) * OW_BLOCK + i], *res);
    }
}

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
#if SUB_GROUP_SIZE != 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#endif
__kernel void
gen9_conv_nhwc_fwd_f32(const __global float *src, const __global float *wei,
        const __global float *bias, __global float *dst, float eltwise_alpha,
        float eltwise_beta, float eltwise_scale, float sum_scale) {

    const int sp = get_group_id(1);
    const int local_id = get_sub_group_local_id();
    const int ocb_mb = get_group_id(2);
    const int ocb = ocb_mb / (MB);
    const int mb = ocb_mb % (MB);
    const int oc = (ocb * OCB) / OC_BLOCK + get_group_id(0);

    const int g = oc / (OC / OC_BLOCK);
    const int goc = oc % (OC / OC_BLOCK);

    bool is_oc_tail = OC != OC_WO_PADDING && oc == (OC / OC_BLOCK - 1);

#if CASE_3D
    const int od = sp / (OWB * OHB);
    const int ohw = sp % (OWB * OHB);
    const int id = od * SD - PD;
#else
    const int od = 0;
    const int ohw = sp;
    const int id = 0;
#endif
    const int oh = (ohw / OWB) * OH_BLOCK;
    const int ow = (ohw % OWB) * OW_BLOCK;

#if WITH_BIAS
    float blockC00[OW_BLOCK];
    for (int i = 0; i < OW_BLOCK; i++)
        blockC00[i] = bias[oc * OC_BLOCK + local_id];
#else
    float blockC00[OW_BLOCK] = {0.0f};
#endif

    int ih = oh * SH - PH;
    int iw = ow * SW - PW;

    __local float blockA_tail[SUB_GROUP_SIZE * OW_BLOCK];

    src += mb * ID * IH * IW * G * IC_WO_PADDING
            + id * IH * IW * G * IC_WO_PADDING + ih * IW * G * IC_WO_PADDING
            + iw * G * IC_WO_PADDING + g * IC_WO_PADDING;
    wei += goc * KDHW_SIZE * OC_BLOCK * IC + g * IC * OC * KDHW_SIZE;

    const bool do_if = iw < 0 || iw + SW * OW_BLOCK + KW * (1 + DW) >= IW;

#if IS_1x1x1 && (HAS_PAD_D || HAS_PAD_H || HAS_PAD_W)
    if (!(id < 0 || id >= ID || ih < 0 || ih >= IH)) {
#endif

        //      main loop by input channels

        for (int icb = 0; icb < (IC_TAIL ? IC - IC_BLOCK : IC);
                icb += IC_BLOCK) {
#if !IS_1x1x1
            __attribute__((opencl_unroll_hint(1))) for (int kd = 0; kd < KD;
                                                        ++kd)
                    __attribute__((opencl_unroll_hint(1))) for (int kh = 0;
                                                                kh < KH; ++kh) {
#if HAS_PAD_D || HAS_PAD_H || HAS_PAD_W
                if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH
#if CASE_3D
                        || id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID) {
#else
                ) {
#endif
                    continue;
                }
#endif
                const __global float *src1 = src
                        + kd * (1 + DD) * IH * IW * G * IC_WO_PADDING
                        + kh * (1 + DH) * IW * G * IC_WO_PADDING;

                float tempA[SW * OW_BLOCK + KW * (1 + DW)];
                int k = iw;
                if (do_if) {
                    __attribute__((opencl_unroll_hint(
                            SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                    for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                        if (k >= 0 && k < IW)
                            tempA[i] = as_float(intel_sub_group_block_read((
                                    const __global uint
                                            *)(&src1[i * G * IC_WO_PADDING])));
                        else
                            tempA[i] = 0.0f;
                        k++;
                    }
                } else {
                    __attribute__((opencl_unroll_hint(
                            SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                    for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                        tempA[i] = as_float(intel_sub_group_block_read((
                                const __global uint
                                        *)(&src1[i * G * IC_WO_PADDING])));
                    }
                }
                __attribute__((opencl_unroll_hint(KW))) // attr:no-format
                for (int kw = 0; kw < KW; ++kw) {

                    const __global float *wei1 = wei
                            + kd * KH * KW * OC_BLOCK * IC_BLOCK
                            + kh * KW * OC_BLOCK * IC_BLOCK
                            + kw * OC_BLOCK * IC_BLOCK;

#else
        const __global float *src1 = src;
        const __global float *wei1 = wei;
#endif
                    float8 blockB00 = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)wei1));
                    float8 blockB01 = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(wei1 + 8 * IC_BLOCK)));

#if !IS_1x1x1
                    float blockA[OW_BLOCK];
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockA[i] = tempA[kw * (1 + DW) + SW * i];
                    }
#else
#if OW_BLOCK != 8 || HAS_PAD_W
        float blockA[OW_BLOCK];
#else
        float8 blockA;
#endif
#if OW % OW_BLOCK != 0 || HAS_PAD_W
        if (ow == OW_LAST) {
            for (int i = 0; i < OW - OW_LAST; i++) {
#if HAS_PAD_W
                if (iw + i * SW < 0 || iw + i * SW >= IW) {
                    blockA[i] = 0.0f;
                } else {
#endif
                    blockA[i] = as_float(intel_sub_group_block_read((
                            const __global uint
                                    *)(&src1[i * G * IC_WO_PADDING * SW])));
#if HAS_PAD_W
                }
#endif
            }
            for (int i = OW - OW_LAST; i < OW_BLOCK; i++)
                blockA[i] = 0.0f;
        } else {
#endif
#if SW != 1 || OW_BLOCK != 8 || HAS_PAD_W
            __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
            for (int i = 0; i < OW_BLOCK; i++) {
#if HAS_PAD_W
                if (iw + i * SW < 0 || iw + i * SW >= IW) {
                    blockA[i] = 0.0f;
                } else {
#endif
                    blockA[i] = as_float(intel_sub_group_block_read((
                            const __global uint
                                    *)(&src1[i * G * IC_WO_PADDING * SW])));
#if HAS_PAD_W
                }
#endif
            }
#else

        for (int i = 0; i < 8; i++) {
            blockA[i] = as_float(intel_sub_group_block_read(
                    (const __global uint *)(&src1[i * G * IC_WO_PADDING])));
        }

#endif
#if OW % OW_BLOCK != 0 || HAS_PAD_W
        }
#endif
#endif

#if OW_BLOCK != 16
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        MULTIPLY_BLOCKS_8x8(
                                blockC00[i], blockA[i], blockB00, blockB01);
                    }
#else
        __attribute__((opencl_unroll_hint(8))) // attr:no-format
        for (int i = 0; i < 8; i++) {
            MULTIPLY_BLOCKS_8x8(blockC00[i], blockA[i], blockB00, blockB01);
            MULTIPLY_BLOCKS_8x8(
                    blockC00[8 + i], blockA[i + 8], blockB00, blockB01);
        }
#endif

#if !IS_1x1x1
                }
            }
#endif
            src += IC_BLOCK;
            wei += OC_BLOCK * KDHW_SIZE * IC_BLOCK;
        }

#if IC_TAIL

        // input channels tails processing

#if !IS_1x1x1
        __attribute__((opencl_unroll_hint(1))) for (int kd = 0; kd < KD;
                                                    ++kd) // attr:no-format
                __attribute__((opencl_unroll_hint(1))) // attr:no-format
                for (int kh = 0; kh < KH; ++kh) {

            if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH
#if CASE_3D
                    || id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID) {
#else
            ) {
#endif
                continue;
            }
            const __global float *src1 = src
                    + kd * (1 + DD) * IH * IW * G * IC_WO_PADDING
                    + kh * (1 + DH) * IW * G * IC_WO_PADDING;

            float tempA[SW * OW_BLOCK + KW * (1 + DW)];
            int k = iw;
            if (do_if) {
                __attribute__((opencl_unroll_hint(
                        SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                    if (k >= 0 && k < IW)
                        tempA[i] = src1[i * G * IC_WO_PADDING + local_id];
                    else
                        tempA[i] = 0.0f;
                    k++;
                }
            } else {
                __attribute__((opencl_unroll_hint(
                        SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                    tempA[i] = src1[i * G * IC_WO_PADDING + local_id];
                }
            }
            __attribute__((opencl_unroll_hint(KW))) // attr:no-format
            for (int kw = 0; kw < KW; ++kw) {

                const __global float *wei1 = wei
                        + kd * KH * KW * OC_BLOCK * IC_BLOCK
                        + kh * KW * OC_BLOCK * IC_BLOCK
                        + kw * OC_BLOCK * IC_BLOCK;

#else
        const __global float *src1 = src;
        const __global float *wei1 = wei;
#endif
                float8 blockB00 = as_float8(intel_sub_group_block_read8(
                        (const __global uint *)wei1));
                float8 blockB01 = as_float8(intel_sub_group_block_read8(
                        (const __global uint *)(wei1 + 8 * IC_BLOCK)));

#if !IS_1x1x1
                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int i = 0; i < OW_BLOCK; i++) {
                    blockA_tail[local_id * OW_BLOCK + i]
                            = tempA[kw * (1 + DW) + SW * i];
                }
#else

#if OW % OW_BLOCK != 0 || HAS_PAD_W
        if (ow == OW_LAST) {
            for (int i = 0; i < OW - OW_LAST; i++) {
#if HAS_PAD_W
                if (iw + i * SW < 0 || iw + i * SW >= IW) {
                    blockA_tail[local_id * OW_BLOCK + i] = 0.0f;
                } else {
#endif
                    blockA_tail[local_id * OW_BLOCK + i]
                            = src1[i * G * IC_WO_PADDING * SW + local_id];
#if HAS_PAD_W
                }
#endif
            }
            for (int i = OW - OW_LAST; i < OW_BLOCK; i++)
                blockA_tail[local_id * OW_BLOCK + i] = 0.0f;
        } else {
#endif
#if SW != 1 || OW_BLOCK != 8 || HAS_PAD_W
            __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
            for (int i = 0; i < OW_BLOCK; i++) {
#if HAS_PAD_W
                if (iw + i * SW < 0 || iw + i * SW >= IW) {
                    blockA_tail[local_id * OW_BLOCK + i] = 0.0f;
                } else {
#endif
                    blockA_tail[local_id * OW_BLOCK + i]
                            = src1[i * G * IC_WO_PADDING * SW + local_id];
#if HAS_PAD_W
                }
#endif
            }
#else
        for (int i = 0; i < 8; i++) {
            blockA_tail[local_id * OW_BLOCK + i]
                    = src1[i * G * IC_WO_PADDING + local_id];
        }
#endif
#if OW % OW_BLOCK != 0 || HAS_PAD_W
        }
#endif
#endif

#if OW_BLOCK != 16
                __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                for (int i = 0; i < OW_BLOCK; i++) {
                    multiply_blocks_8x8_ic_tail(
                            &blockC00[i], blockA_tail, i, blockB00, blockB01);
                }
#else
        __attribute__((opencl_unroll_hint(8))) // attr:no-format
        for (int i = 0; i < 8; i++) {
            multiply_blocks_8x8_ic_tail(
                    &blockC00[i], blockA_tail, i, blockB00, blockB01);
            multiply_blocks_8x8_ic_tail(
                    &blockC00[i + 8], blockA_tail, i + 8, blockB00, blockB01);
        }
#endif
#if !IS_1x1x1
            }
        }
#endif

#endif // IC_TAIL

#if IS_1x1x1 & (HAS_PAD_D || HAS_PAD_H || HAS_PAD_W)
    }
#endif

    __global float *dst_write0 = dst + mb * OD * OH * OW * G * OC_WO_PADDING
            + od * OH * OW * G * OC_WO_PADDING + oh * OW * G * OC_WO_PADDING
            + ow * G * OC_WO_PADDING + g * OC_WO_PADDING + goc * OC_BLOCK;

    // Apply postops

#if WITH_SUM == 1
    float blockS00[OW_BLOCK];

#define READ_DST_CH_BLOCK \
    do { \
        if (!is_oc_tail) \
            blockS00[i] = as_float(intel_sub_group_block_read((const __global \
                            uint *)&dst_write0[i * G * OC_WO_PADDING])); \
        else if (local_id < OC_TAIL) \
            blockS00[i] = dst_write0[i * G * OC_WO_PADDING + local_id]; \
    } while (0)

#if OW % OW_BLOCK != 0
    if (ow == OW_LAST) {
        for (int i = 0; i < OW - OW_LAST; i++) {
            READ_DST_CH_BLOCK;
        }
    } else {
#endif
        for (int i = 0; i < OW_BLOCK; i++) {
            READ_DST_CH_BLOCK;
        }
#if OW % OW_BLOCK != 0
    }
#endif

    for (int i = 0; i < OW_BLOCK; i++) {
#if SUM_SCALE == 1
        blockC00[i] += blockS00[i];
#else
        blockC00[i] = fma(blockS00[i], (float)sum_scale, blockC00[i]);
#endif
    }
#endif // WITH_SUM

#if WITH_ELTWISE == 1
    DO_ELTWISE(blockC00, OW_BLOCK, eltwise_alpha, eltwise_beta, eltwise_scale);
#endif

    // Save

#if OC_WO_PADDING % 4 == 0
#define WRITE_DST_CH_BLOCK \
    do { \
        if (!is_oc_tail) \
            intel_sub_group_block_write( \
                    (__global unsigned int \
                                    *)(&dst_write0[i * G * OC_WO_PADDING]), \
                    as_uint(blockC00[i])); \
        else if (local_id < OC_TAIL) \
            dst_write0[i * G * OC_WO_PADDING + local_id] = blockC00[i]; \
    } while (0)
#else
#define WRITE_DST_CH_BLOCK \
    do { \
        if (!is_oc_tail) \
            dst_write0[i * G * OC_WO_PADDING + local_id] = blockC00[i]; \
        else if (local_id < OC_TAIL) \
            dst_write0[i * G * OC_WO_PADDING + local_id] = blockC00[i]; \
    } while (0)
#endif

#if OW % OW_BLOCK != 0
    if (ow + OW_BLOCK > OW) {
        for (int i = 0; i < OW - OW_LAST; i++) {
            WRITE_DST_CH_BLOCK;
        }
    } else {
#endif
        __attribute__((opencl_unroll_hint(OW_BLOCK))) // attr:no-format
        for (int i = 0; i < OW_BLOCK; i++) {
            WRITE_DST_CH_BLOCK;
        }

#if OW % OW_BLOCK != 0
    }
#endif
    return;
}
#undef READ_DST_CH_BLOCK
#undef WRITE_DST_CH_BLOCK
