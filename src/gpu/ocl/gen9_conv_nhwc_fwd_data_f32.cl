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

// FWD fp32 Convolution Kernel with NHWC data layout support
// Features:
// - based on 8ow16c version of blocked implementation
// - weights are blocked and padded: OIhw16i16o
// - explicit tail processing for src and dst channels
// - due to 16-bytes alignment requred, intel_sub_groups_block_write usage
//   is limited

#define ENABLE_KW_BUF (KW >= 5)

#define CASE_3D (ID > 1)
#define KDHW_SIZE (KD * KH * KW)
#define HAS_PAD_D (PD != 0 || PD_R != 0)
#define HAS_PAD_H (PH != 0 || PH_R != 0)

inline float read_ic_block(const __global float *ptr, int off) {
    const int local_id = get_local_id(0);
#if IC == 3
    return (local_id < IC) ? *ptr : 0.0f;
#else
#if (IS_DW ? G_WO_PADDING : IC_WO_PADDING) % IC_BLOCK != 0
    int tail = (IS_DW ? G_WO_PADDING : IC_WO_PADDING) - off;
    if (tail < IC_BLOCK) { return (local_id < tail) ? ptr[local_id] : 0.0f; }
#endif
    return as_float(intel_sub_group_block_read((const __global uint *)ptr));
#endif
}

inline float read_oc_block(const __global float *ptr, int off) {
#if (IS_DW ? G_WO_PADDING : OC_WO_PADDING) % OC_BLOCK != 0
    int tail = (IS_DW ? G_WO_PADDING : OC_WO_PADDING) - off;
    if (tail < OC_BLOCK) {
        const int local_id = get_local_id(0);
        return (local_id < tail) ? ptr[local_id] : 0.0f;
    }
#endif
    return as_float(intel_sub_group_block_read((const __global uint *)ptr));
}

inline void write_oc_block(__global float *ptr, int off, float value) {
    const int local_id = get_local_id(0);
#if (IS_DW ? G_WO_PADDING : OC_WO_PADDING) % OC_BLOCK != 0
    int tail = (IS_DW ? G_WO_PADDING : OC_WO_PADDING) - off;
    if (tail < OC_BLOCK) {
        if (local_id < tail) ptr[local_id] = value;
        return;
    }
#endif
    if ((IS_DW ? G_WO_PADDING : OC_WO_PADDING) % 4 != 0) {
        ptr[local_id] = value;
        return;
    }
    return intel_sub_group_block_write((__global uint *)ptr, as_uint(value));
}

void multiply_blocks_8x8_ic3(float *res, float blockA, const float *blockB) {
    *res = fma(blockB[0], intel_sub_group_shuffle(blockA, 0), *res);
    *res = fma(blockB[1], intel_sub_group_shuffle(blockA, 1), *res);
    *res = fma(blockB[2], intel_sub_group_shuffle(blockA, 2), *res);
}

void multiply_blocks_8x8(
        float *res, float blockA, float8 blockB0, float8 blockB1) {
    for (int i = 0; i < 8; i++) {
        *res = fma(blockB0[i], intel_sub_group_shuffle(blockA, i), *res);
    }
    for (int i = 0; i < 8; i++) {
        *res = fma(blockB1[i], intel_sub_group_shuffle(blockA, 8 + i), *res);
    }
}

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_conv_nhwc_fwd_f32(const __global float *src, const __global float *wei,
        const __global float *bias, __global float *dst, float eltwise_alpha,
        float eltwise_beta, float eltwise_scale, float sum_scale) {

    const int sp = get_group_id(1);
    const int local_id = get_sub_group_local_id();
    const int ocb_mb = get_group_id(2);
    const int ocb = ocb_mb / (MB);
    const int mb = ocb_mb % (MB);

#if IS_DW
    const int oc = get_group_id(0);
    const int g = 0;
    const int goc = oc;
#else
    const int oc = (ocb * OCB) / OC_BLOCK + get_group_id(0);
    const int g = oc / (OC / OC_BLOCK);
    const int goc = oc % (OC / OC_BLOCK);
#endif

    const int od = CASE_3D ? sp / (OWB * OHB) : 0;
    const int ohw = CASE_3D ? sp % (OWB * OHB) : sp;
    const int id = CASE_3D ? od * SD - PD : 0;

    const int oh = (ohw / OWB) * OH_BLOCK;
    const int ow = (ohw % OWB) * OW_BLOCK;

    float blockC00[OW_BLOCK];
    for (int i = 0; i < OW_BLOCK; i++)
        blockC00[i] = WITH_BIAS ? bias[oc * OC_BLOCK + local_id] : 0.0f;

    int ih = oh * SH - PH;
    int iw = ow * SW - PW;

    src += mb * ID * IH * IW * G * IC_WO_PADDING;
    src += (id * IH * IW + ih * IW + iw) * G * IC_WO_PADDING;
    src += g * IC_WO_PADDING;
    src += (IS_DW ? oc * OC_BLOCK : 0);

    wei += goc * KDHW_SIZE * OC_BLOCK * IC + g * IC * OC * KDHW_SIZE;

#if (KD == 1 && KH == 1) && (HAS_PAD_D || HAS_PAD_H)
    const bool dh_out_of_range = (id < 0 || id >= ID || ih < 0 || ih >= IH);
#else
    const bool dh_out_of_range = false;
#endif

#if IS_DW
    const int icb_min = goc * OC_BLOCK;
    const int icb_max = icb_min + OC_BLOCK;
#else
    const int icb_min = 0;
    const int icb_max = dh_out_of_range ? 0 : (IC == 3 ? 1 : IC);
#endif

    for (int icb = icb_min; icb < icb_max; icb += IC_BLOCK) {
        __attribute__((opencl_unroll_hint(1))) // attr:no-format
        for (int kd = 0; kd < KD; ++kd) {
#if HAS_PAD_D && (KD > 1)
            if (id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID) continue;
#endif
            __attribute__((opencl_unroll_hint(1))) // attr:no-format
            for (int kh = 0; kh < KH; ++kh) {
#if HAS_PAD_H && (KH > 1)
                if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH)
                    continue;
#endif
                const __global float *src1 = src
                        + kd * (1 + DD) * IH * IW * G * IC_WO_PADDING
                        + kh * (1 + DH) * IW * G * IC_WO_PADDING;
                if (IC == 3) src1 += local_id;
#if ENABLE_KW_BUF
                float tempA[SW * OW_BLOCK + KW * (1 + DW)] = {0.0f};
                __attribute__((opencl_unroll_hint(
                        SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                    if ((i + iw) >= 0 && (i + iw) < IW) {
                        tempA[i] = read_ic_block(
                                &src1[i * G * IC_WO_PADDING], icb);
                    }
                }
#endif
                __attribute__((opencl_unroll_hint(KW))) // attr:no-format
                for (int kw = 0; kw < KW; ++kw) {
#if IC == 3
                    const __global float *wei1 = wei
                            + (kd * KH * KW + kh * KW + kw) * IC * OC_BLOCK;
#elif IS_DW
                    const __global float *wei1
                            = wei + (kd * KH * KW + kh * KW + kw) * OC_BLOCK;
#else
                    const __global float *wei1 = wei
                            + (kd * KH * KW + kh * KW + kw) * IC_BLOCK
                                    * OC_BLOCK;
#endif

                    float blockA[OW_BLOCK] = {0.0f};
#if ENABLE_KW_BUF
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockA[i] = tempA[i * SW + kw * (1 + DW)];
                    }
#else
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        int iw_off = i * SW + kw * (1 + DW);
                        if (iw + iw_off >= 0 && iw + iw_off < IW) {
                            blockA[i] = read_ic_block(
                                    &src1[iw_off * G * IC_WO_PADDING], icb);
                        }
                    }
#endif

#if IC == 3
                    float blockB[IC];
                    __attribute__((opencl_unroll_hint(IC))) // attr:no-format
                    for (int i = 0; i < IC; i++) {
                        blockB[i] = as_float(intel_sub_group_block_read(
                                (const __global uint *)wei1 + i * OC_BLOCK));
                    }

                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        multiply_blocks_8x8_ic3(
                                &blockC00[i], blockA[i], blockB);
                    }
#elif IS_DW
                    float blockB = as_float(intel_sub_group_block_read(
                            (const __global uint *)wei1));
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockC00[i] = fma(blockA[i], blockB, blockC00[i]);
                    }
#else
                    float8 blockB00 = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)wei1));
                    float8 blockB01 = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(wei1 + 8 * OC_BLOCK)));

                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        multiply_blocks_8x8(
                                &blockC00[i], blockA[i], blockB00, blockB01);
                    }
#endif
                }
            }
        }
        src += IC_BLOCK;
        wei += KDHW_SIZE * IC_BLOCK * OC_BLOCK;
    }

    __global float *dst_write0 = dst + mb * OD * OH * OW * G * OC_WO_PADDING;
    dst_write0 += (od * OH * OW + oh * OW + ow) * G * OC_WO_PADDING;
    dst_write0 += g * OC_WO_PADDING + goc * OC_BLOCK;

    // Apply postops
#if WITH_SUM == 1
    float blockS00[OW_BLOCK];
    for (int i = 0; i < min(OW_BLOCK, OW - ow); i++) {
        blockS00[i] = read_oc_block(
                &dst_write0[i * G * OC_WO_PADDING], goc * OC_BLOCK);
    }

    for (int i = 0; i < OW_BLOCK; i++) {
#if SUM_SCALE == 1
        blockC00[i] += blockS00[i];
#else
        blockC00[i] = fma(blockS00[i], (float)sum_scale, blockC00[i]);
#endif
    }
#endif // WITH_SUM

#if WITH_ELTWISE == 1
    for (int i = 0; i < OW_BLOCK; i++) {
        blockC00[i] = fwd_eltwise(
                blockC00[i], eltwise_alpha, eltwise_beta, eltwise_scale);
    }
#endif

    // Save
    for (int i = 0; i < min(OW_BLOCK, OW - ow); i++) {
        write_oc_block(&dst_write0[i * G * OC_WO_PADDING], goc * OC_BLOCK,
                blockC00[i]);
    }
}
