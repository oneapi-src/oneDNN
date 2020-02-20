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

#if ID > 1
#define CASE_3D 1
#else
#define CASE_3D 0
#endif

#define HAS_PAD_D (PD != 0 || PD_R != 0)
#define HAS_PAD_H (PH != 0 || PH_R != 0)
#define HAS_PAD_W (PW != 0 || PW_R != 0)

#if BWD_WEIGHTS == 1

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1))) // attr:no-format
#if VER_16MB16C == 1 || VER_8OW16C == 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
#endif
__kernel void
gen9_reduce_bwd_weights(__global float *diff_wei, __global float *wht_work,
        __global float *diff_bias, __global float *bias_work) {

    const uint g_ic_oc = get_global_id(0);

#if IS_DW
    const uint g = 0;
    const uint oc = get_group_id(0);
#else
    const uint g = g_ic_oc / (OC * (IC / IC_BLOCK));
    const uint io = g_ic_oc % (OC * (IC / IC_BLOCK));

    const uint oc = (io % OC) / OC_BLOCK;
#endif

    const uint ksp = get_global_id(1);
#if CASE_3D
    const uint kd = ksp / (KW * KH);
    const uint khw = ksp % (KW * KH);
#else
    const uint khw = ksp;
    const uint kd = 0;
#endif
    const uint kh = khw / KW;
    const uint kw = khw % KW;
    const uint local_x = get_local_id(0);
#if IC == 3
    const uint ic = 0;
    diff_wei += oc * KD * KH * KW * IC * OC_BLOCK + g * IC * OC * KD * KH * KW
            + kd * KH * KW * IC * OC_BLOCK + kh * KW * IC * OC_BLOCK
            + kw * IC * OC_BLOCK;
    wht_work += oc * KD * KH * KW * IC * OC_BLOCK + g * IC * OC * KD * KH * KW
            + kd * KH * KW * IC * OC_BLOCK + kh * KW * IC * OC_BLOCK
            + kw * IC * OC_BLOCK;
#elif IS_DW
    const uint ic = oc;
    diff_wei += oc * KD * KH * KW * IC * OC_BLOCK + g * IC * OC * KD * KH * KW
            + kd * KH * KW * IC * OC_BLOCK + kh * KW * IC * OC_BLOCK
            + kw * IC * OC_BLOCK;
    wht_work += oc * KD * KH * KW * IC * OC_BLOCK + g * IC * OC * KD * KH * KW
            + kd * KH * KW * IC * OC_BLOCK + kh * KW * IC * OC_BLOCK
            + kw * IC * OC_BLOCK;

#else
    const uint ic = io / OC;
    diff_wei += ic * OC * KD * KH * KW * IC_BLOCK
            + oc * KD * KH * KW * IC_BLOCK * OC_BLOCK
            + kd * KH * KW * OC_BLOCK * IC_BLOCK + kh * KW * OC_BLOCK * IC_BLOCK
            + kw * IC_BLOCK * OC_BLOCK + g * IC * OC * KD * KH * KW;
    wht_work += ic * OC * KD * KH * KW * IC_BLOCK
            + oc * KD * KH * KW * IC_BLOCK * OC_BLOCK
            + kd * KH * KW * OC_BLOCK * IC_BLOCK + kh * KW * IC_BLOCK * OC_BLOCK
            + kw * IC_BLOCK * OC_BLOCK + g * IC * OC * KD * KH * KW;
#endif

#if WITH_BIAS == 1
    diff_bias += g * OC + oc * OC_BLOCK + local_x;
    bias_work += g * OC + oc * OC_BLOCK + local_x;
    if ((ic == 0 || IS_DW) && kh == 0 && kw == 0 && kd == 0)
        diff_bias[0] = 0.0f;
#endif

#if IC == 3
    float C0 = 0.0f, C1 = 0.0f, C2 = 0.0f;
#elif IS_DW
    float blockC00 = 0.0f;
#else
    float8 blockC00 = 0.0f;
    float8 blockC01 = 0.0f;
#endif

    for (uint i = 0; i < NCHUNK; i++) {
#if IC == 3
        float blockA0 = as_float(intel_sub_group_block_read(
                (const __global uint *)(&wht_work[0])));
        float blockA1 = as_float(intel_sub_group_block_read(
                (const __global uint *)(&wht_work[OC_BLOCK])));
        float blockA2 = as_float(intel_sub_group_block_read(
                (const __global uint *)(&wht_work[2 * OC_BLOCK])));

        C0 += blockA0;
        C1 += blockA1;
        C2 += blockA2;

#elif IS_DW
        float blockA0 = as_float(intel_sub_group_block_read(
                (const __global uint *)(&wht_work[0])));
        blockC00 += blockA0;
#else
        float8 blockA0 = as_float8(intel_sub_group_block_read8(
                (const __global uint *)(&wht_work[0])));
        float8 blockA1 = as_float8(intel_sub_group_block_read8(
                (const __global uint *)(&wht_work[8 * OC_BLOCK])));
        blockC00 += blockA0;
        blockC01 += blockA1;
#endif

#if WITH_BIAS == 1
        if ((ic == 0 || IS_DW) && kh == 0 && kw == 0 && kd == 0)
            diff_bias[0] += bias_work[i * G * OC];
#endif
        wht_work += G * OC * IC * KD * KH * KW;
    }
#if IC == 3
    intel_sub_group_block_write(
            (__global unsigned int *)(&diff_wei[0]), as_uint(C0));
    intel_sub_group_block_write(
            (__global unsigned int *)(&diff_wei[OC_BLOCK]), as_uint(C1));
    intel_sub_group_block_write(
            (__global unsigned int *)(&diff_wei[2 * OC_BLOCK]), as_uint(C2));
#elif IS_DW
    intel_sub_group_block_write(
            (__global unsigned int *)(&diff_wei[0]), as_uint(blockC00));
#else
    intel_sub_group_block_write8(
            (__global unsigned int *)(&diff_wei[0]), as_uint8(blockC00));
    intel_sub_group_block_write8(
            (__global unsigned int *)(&diff_wei[8 * OC_BLOCK]),
            as_uint8(blockC01));
#endif
}

__attribute__((reqd_work_group_size(1, 1, 1))) // attr:no-format
__kernel void
gen9_load_tails_bwd_weights(__global float *src, __global float *tails) {

    for (int j = 0; j < PW; j++)
        for (int i = 0; i < 16; i++) {
            tails[0] = 0.0f;
            tails++;
        }
    for (int j = 0; j < IW; j++)
        for (int i = 0; i < 16; i++) {
            tails[0] = src[j * 16 + i];
            tails++;
        }
    for (int j = 0; j < PW + KW; j++)
        for (int i = 0; i < 16; i++) {
            tails[0] = 0.0f;
            tails++;
        }

    src += (MB * IC * G * ID * IH * IW - 16 * IW);
    for (int j = 0; j < PW; j++)
        for (int i = 0; i < 16; i++) {
            tails[0] = 0.0f;
            tails++;
        }
    for (int j = 0; j < IW; j++)
        for (int i = 0; i < 16; i++) {
            tails[0] = src[j * 16 + i];
            tails++;
        }
    for (int j = 0; j < PW + KW + 8; j++)
        for (int i = 0; i < 16; i++) {
            tails[0] = 0.0f;
            tails++;
        }
}

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if VER_16MB16C == 1 || VER_8OW16C == 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
#endif
__kernel void
gen9_common_conv_bwd_weights(__global float *src, __global float *diff_wei,
        __global float *diff_bias, __global float *diff_dst
#if VER_8OW16C == 1 && (IC % 16 == 0 || IS_DW)
        ,
        __global float *tails
#endif
) {

#if VER_16MB16C == 1
    const uint g_ic_oc = get_global_id(0);

#if IS_DW
    const uint g = 0;
    const uint oc = get_group_id(0);
    const uint ic = oc;
#else
    const uint g = g_ic_oc / (OC * (IC / IC_BLOCK));
    const uint io = g_ic_oc % (OC * (IC / IC_BLOCK));

    const uint oc = (io % OC) / OC_BLOCK;
    const uint ic = io / OC;
#endif
    const uint ksp = get_global_id(1);
#if CASE_3D
    const uint kd = ksp / (KW * KH);
    const uint khw = ksp % (KW * KH);
#else
    const uint khw = ksp;
    const uint kd = 0;
#endif
    const uint kh = khw / KW;
    const uint kw = khw % KW;
    const uint local_x = get_local_id(0);

    const uint chunk = get_global_id(2);
    const uint oh_chunk = chunk % OH_CHUNK;
    const uint mb_chunk = chunk / OH_CHUNK;

    const uint str = oh_chunk * OH_BLOCK;
    uint end = (oh_chunk + 1) * OH_BLOCK;
    if (end > OD * OH * OW) end = OD * OH * OW;
    const uint mb = mb_chunk * (MB_CHUNK_SIZE);
    const uint mb_end = min((mb_chunk + 1) * (MB_CHUNK_SIZE), (uint)MB);

    const bool do_bias = (ic == 0 || IS_DW) && kh == 0 && kw == 0 && kd == 0;

    src += ic * ID * IH * IW * IC_BLOCK * MB_BLOCK + mb * IC * G * ID * IH * IW
            + g * IC * ID * IH * IW * MB_BLOCK;
    diff_dst += oc * OD * OH * OW * OC_BLOCK * MB_BLOCK
            + g * OC * OD * OH * OW * MB_BLOCK;

#if WITH_BIAS == 1
    diff_bias += g * OC + oc * OC_BLOCK + local_x + chunk * OC * G;
    float bias_loc = 0.0f;
#endif

#if IS_DW
    float blockC00 = 0.0f;
#else
    float8 blockC00 = 0.0f;
    float8 blockC01 = 0.0f;
#endif

#if MB != (MB_CHUNK * MB_BLOCK)
    uint omb = mb;
    do {
        const __global float *diff_dst1 = diff_dst + str * OC_BLOCK * MB_BLOCK
                + omb * OC * G * OD * OH * OW;
#else
    const __global float *diff_dst1
            = diff_dst + str * OC_BLOCK * MB_BLOCK + mb * OC * G * OD * OH * OW;
#endif
        for (uint k = str; k < end; k++) {
#if CASE_3D
            uint od = k / (OW * OH);
            uint k_ = k % (OW * OH);
#else
        uint k_ = k;
#endif
            uint oh = k_ / OW;
            uint ow = k_ % OW;

            const uint ih = oh * SH - PH + kh * (1 + DH);
            const uint iw = ow * SW - PW + kw * (1 + DW);
#if CASE_3D
            const uint id = od * SD - PD + kd * (1 + DD);
#endif

            if (iw < 0 || ih < 0 || iw >= IW || ih >= IH
#if CASE_3D
                    || id < 0 || id >= ID
#endif
            ) {
#if WITH_BIAS == 1
                if (do_bias) {
                    float8 blockB = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(diff_dst1)));
                    for (int i = 0; i < 8; i++)
                        bias_loc += blockB[i];
                    blockB = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(diff_dst1 + 8 * OC_BLOCK)));
                    for (int i = 0; i < 8; i++)
                        bias_loc += blockB[i];
                }
#endif
                diff_dst1 += OC_BLOCK * MB_BLOCK;
                continue;
            }

            const __global float *src1 = src + ih * IW * IC_BLOCK * MB_BLOCK
                    + iw * IC_BLOCK * MB_BLOCK;
#if CASE_3D
            src1 += id * IH * IW * IC_BLOCK * MB_BLOCK;
#endif
#define TRANSPOSE_8(_block, _row, _col) \
    (float8)(intel_sub_group_shuffle(_block[_row], 0 + _col), \
            intel_sub_group_shuffle(_block[_row], 1 + _col), \
            intel_sub_group_shuffle(_block[_row], 2 + _col), \
            intel_sub_group_shuffle(_block[_row], 3 + _col), \
            intel_sub_group_shuffle(_block[_row], 4 + _col), \
            intel_sub_group_shuffle(_block[_row], 5 + _col), \
            intel_sub_group_shuffle(_block[_row], 6 + _col), \
            intel_sub_group_shuffle(_block[_row], 7 + _col))

#define FMA8(a, b, c) fma((float8)(a), (float8)b, (float8)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, col) \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_8(_blockA, 0, col), _result); \
        _result = FMA8(_blockB.s1, TRANSPOSE_8(_blockA, 1, col), _result); \
        _result = FMA8(_blockB.s2, TRANSPOSE_8(_blockA, 2, col), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_8(_blockA, 3, col), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_8(_blockA, 4, col), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_8(_blockA, 5, col), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_8(_blockA, 6, col), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_8(_blockA, 7, col), _result); \
    }

#if IS_DW
            float8 blockA = as_float8(
                    intel_sub_group_block_read8((const __global uint *)(src1)));
            float8 blockA1 = as_float8(intel_sub_group_block_read8(
                    (const __global uint *)(src1 + 8 * IC_BLOCK)));

            float8 blockB = as_float8(intel_sub_group_block_read8(
                    (const __global uint *)(diff_dst1)));
            float8 blockB1 = as_float8(intel_sub_group_block_read8(
                    (const __global uint *)(diff_dst1 + 8 * OC_BLOCK)));

            for (int i = 0; i < 8; i++) {
                blockC00 = fma(blockA[i], blockB[i], blockC00);
            }

#if WITH_BIAS == 1
            for (int i = 0; i < 8; i++)
                bias_loc += blockB[i];
#endif

            for (int i = 0; i < 8; i++) {
                blockC00 = fma(blockA1[i], blockB1[i], blockC00);
            }

#if WITH_BIAS == 1
            for (int i = 0; i < 8; i++)
                bias_loc += blockB1[i];
#endif
#else
        float8 blockA = as_float8(
                intel_sub_group_block_read8((const __global uint *)(src1)));
        float8 blockB = as_float8(intel_sub_group_block_read8(
                (const __global uint *)(diff_dst1)));

        MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB, 0);
        MULTIPLY_BLOCKS_8x8(blockC01, blockA, blockB, 8);

#if WITH_BIAS == 1
        for (int i = 0; i < 8; i++)
            bias_loc += blockB[i];
#endif
        blockA = as_float8(intel_sub_group_block_read8(
                (const __global uint *)(src1 + 8 * IC_BLOCK)));
        blockB = as_float8(intel_sub_group_block_read8(
                (const __global uint *)(diff_dst1 + 8 * OC_BLOCK)));
        MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB, 0);
        MULTIPLY_BLOCKS_8x8(blockC01, blockA, blockB, 8);
#if WITH_BIAS == 1
        for (int i = 0; i < 8; i++)
            bias_loc += blockB[i];
#endif
#endif
            diff_dst1 += OC_BLOCK * MB_BLOCK;
        }
#if MB != (MB_CHUNK * MB_BLOCK)
        omb += MB_BLOCK;
        src += IC * G * ID * IH * IW * MB_BLOCK;
    } while (omb < mb_end);
#endif

#if WITH_BIAS == 1
    if (do_bias) diff_bias[0] = bias_loc;
#endif

#if IS_DW
    diff_wei += oc * KD * KH * KW * OC_BLOCK + kd * KH * KW * OC_BLOCK
            + kh * KW * OC_BLOCK + kw * OC_BLOCK
            + chunk * OC * IC * G * KD * KH * KW;
    intel_sub_group_block_write(
            (__global unsigned int *)(&diff_wei[0]), as_uint(blockC00));
#else
    diff_wei += ic * OC * KD * KH * KW * IC_BLOCK
            + oc * KD * KH * KW * IC_BLOCK * OC_BLOCK
            + kd * KH * KW * IC_BLOCK * OC_BLOCK + kh * KW * IC_BLOCK * OC_BLOCK
            + kw * IC_BLOCK * OC_BLOCK + chunk * OC * IC * G * KD * KH * KW
            + g * OC * IC * KD * KH * KW;
    intel_sub_group_block_write8(
            (__global unsigned int *)(&diff_wei[0]), as_uint8(blockC00));
    intel_sub_group_block_write8(
            (__global unsigned int *)(&diff_wei[8 * OC_BLOCK]),
            as_uint8(blockC01));
#endif

#endif
#if VER_8OW16C == 1
    const uint g_ic_oc = get_global_id(0);
#if IS_DW
    const uint g = 0;
    const uint oc = get_group_id(0);
#else
    const uint g = g_ic_oc / (OC * (IC / IC_BLOCK));
    const uint io = g_ic_oc % (OC * (IC / IC_BLOCK));
    const uint oc = (io % OC) / OC_BLOCK;
#endif

#if IS_DW
    const uint ic = oc;
#elif IC == 3
    const int ic = 0;
#else
    const uint ic = io / OC;
#endif
    const int ksp = get_global_id(1);
#if CASE_3D
    const int kd = ksp / (KW * KH);
    const int khw = ksp % (KW * KH);
#else
    const int khw = ksp;
    const int kd = 0;
#endif
    const int kh = khw / KW;
    const int kw = khw % KW;
    const int local_x = get_local_id(0);

    const int mb_chunk = get_global_id(2);

    const int mb = mb_chunk * MB_CHUNK_SIZE;
    const int mb_end = min((mb_chunk + 1) * MB_CHUNK_SIZE, (int)MB);

    const bool do_bias = (ic == 0 || IS_DW) && kh == 0 && kw == 0 && kd == 0;

    src += ic * ID * IH * IW * IC_BLOCK * MB_BLOCK + mb * IC * G * ID * IH * IW
            + g * IC * ID * IH * IW * MB_BLOCK;
    diff_dst += oc * OD * OH * OW * OC_BLOCK * MB_BLOCK
            + g * OC * OD * OH * OW * MB_BLOCK;

    int dst_bound = MB * OC * G * OD * OH * OW - 16 * 8;
    int src_bound = MB * IC * G * ID * IH * IW - SW * 16 * 8;

#if WITH_BIAS == 1
    diff_bias += g * OC + oc * OC_BLOCK + local_x + mb_chunk * OC * G;
    float bias_loc = 0.0f;
#endif

#if IC == 3
    float8 blockC00 = 0.0f;
#elif IS_DW
    float blockC00 = 0.0f;
#else
    float8 blockC00 = 0.0f;
    float8 blockC01 = 0.0f;
#endif

    int dst_ptr
            = oc * OD * OH * OW * OC_BLOCK * MB_BLOCK + g * OC * OD * OH * OW;
    int src_ptr
            = ic * ID * IH * IW * IC_BLOCK * MB_BLOCK + g * IC * ID * IH * IW;

    int omb = mb;
    do {
        const __global float *diff_dst1
                = diff_dst + omb * OC * G * OD * OH * OW;
        int dst_ptr1 = dst_ptr + omb * OC * G * OD * OH * OW;
        int src_ptr1 = src_ptr + omb * IC * G * ID * IH * IW;

        for (int od = 0; od < OD; od++)
            for (int oh = 0; oh < OH; oh++) {

                if (oh * SH + kh * (1 + DH) < PH
                        || oh * SH + kh * (1 + DH) >= IH + PH
#if CASE_3D
                        || od * SD + kd * (1 + DD) < PD
                        || od * SD + kd * (1 + DD) >= ID + PD
#endif
                ) {
#if WITH_BIAS == 1
                    if (do_bias) {
                        for (int ow = 0; ow < OW; ow += OW_BLOCK) {
                            float8 blockB;
#if OW != OW_LAST
                            for (int i = 0; i < 8; i++) {
                                if (ow + i >= OW) {
                                    blockB[i] = 0.0;
                                } else {
                                    blockB[i] = as_float(
                                            intel_sub_group_block_read((
                                                    const __global uint
                                                            *)(&diff_dst1[0])));
                                    diff_dst1 += OC_BLOCK;
                                }
                            }
#else
                            blockB = as_float8(intel_sub_group_block_read8(
                                    (const __global uint *)(diff_dst1)));
                            diff_dst1 += OC_BLOCK * OW_BLOCK;
#endif

                            for (int i = 0; i < OW_BLOCK; i++)
                                bias_loc += blockB[i];
                        }
                    } else {
                        diff_dst1 += OC_BLOCK * MB_BLOCK * OW;
                    }
#else
                    diff_dst1 += OC_BLOCK * MB_BLOCK * OW;
#endif
                    continue;
                }

                for (int ow = 0; ow < OW; ow += OW_BLOCK) {
                    const int id = od * SD - PD + kd * (1 + DD);
                    const int ih = oh * SH - PH + kh * (1 + DH);
                    const int iw = ow * SW - PW + kw * (1 + DW);
                    __global float *src1;
                    int src_ptr2 = src_ptr1 + id * IH * IW * IC_BLOCK
                            + ih * IW * IC_BLOCK + iw * IC_BLOCK;

#if IC % 16 == 0 || IS_DW
                    if (src_ptr2 > src_bound) {
                        src1 = tails + IC_BLOCK * (2 * PW + KW + IW)
                                + (iw + PW) * IC_BLOCK;
                    } else if (src_ptr2 < 0) {
                        src1 = tails + kw * (1 + DW) * IC_BLOCK;
                    } else {
                        src1 = src + id * IH * IW * IC_BLOCK
                                + ih * IW * IC_BLOCK + iw * IC_BLOCK;
                    }
#else
                    src1 = src + id * IH * IW * IC_BLOCK + ih * IW * IC_BLOCK
                            + iw * IC_BLOCK;
#endif

#define TRANSPOSE_8(_block, _row, _col) \
    { \
        (float8)(intel_sub_group_shuffle(_block[_row], 0 + _col), \
                intel_sub_group_shuffle(_block[_row], 1 + _col), \
                intel_sub_group_shuffle(_block[_row], 2 + _col), \
                intel_sub_group_shuffle(_block[_row], 3 + _col), \
                intel_sub_group_shuffle(_block[_row], 4 + _col), \
                intel_sub_group_shuffle(_block[_row], 5 + _col), \
                intel_sub_group_shuffle(_block[_row], 6 + _col), \
                intel_sub_group_shuffle(_block[_row], 7 + _col)) \
    }

#define FMA8(a, b, c) fma((float8)(a), (float8)b, (float8)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, col) \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_8(_blockA, 0, col), _result); \
        _result = FMA8(_blockB.s1, TRANSPOSE_8(_blockA, 1, col), _result); \
        _result = FMA8(_blockB.s2, TRANSPOSE_8(_blockA, 2, col), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_8(_blockA, 3, col), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_8(_blockA, 4, col), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_8(_blockA, 5, col), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_8(_blockA, 6, col), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_8(_blockA, 7, col), _result); \
    }

                    float8 blockA, blockB;
#if IC == 3
                    if (local_x < IC) {
                        for (int i = 0; i < OW_BLOCK; i++) {
                            if (iw + i * SW < 0 || iw + i * SW >= IW)
                                blockA[i] = 0;
                            else
                                blockA[i]
                                        = src1[local_x * ID * IH * IW + i * SW];
                        }
                    } else {
                        blockA = 0.0f;
                    }
#else
#if SW == 1
                    blockA = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(src1)));
#else
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockA[i] = as_float(
                                intel_sub_group_block_read((const __global uint
                                                *)(&src1[i * IC_BLOCK * SW])));
                    }
#endif
#if HAS_PAD_W || KW != 1
                    if (iw < 0 || iw + (OW_BLOCK)*SW >= IW) {
                        for (int i = 0; i < OW_BLOCK; i++) {
                            if (iw + i * SW < 0 || iw + i * SW >= IW)
                                blockA[i] = 0.0f;
                        }
                    }
#endif
#endif
#if OW != OW_LAST
                    for (int i = 0; i < 8; i++) {
                        if (ow + i >= OW) {
                            blockB[i] = 0.0;
                        } else {
                            blockB[i] = as_float(intel_sub_group_block_read(
                                    (const __global uint *)(&diff_dst1[0])));
                            diff_dst1 += OC_BLOCK;
                        }
                    }
#else
                    blockB = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(diff_dst1)));
                    diff_dst1 += OC_BLOCK * OW_BLOCK;
#endif

#if IC == 3
                    MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB, 0);
#elif IS_DW
                    for (int i = 0; i < 8; i++) {
                        blockC00 = fma(blockA[i], blockB[i], blockC00);
                    }
#else
                    MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB, 0);
                    MULTIPLY_BLOCKS_8x8(blockC01, blockA, blockB, 8);
#endif
#if WITH_BIAS == 1
                    for (int i = 0; i < 8; i++)
                        bias_loc += blockB[i];
#endif
                }
            }
        omb += MB_BLOCK;
        src += G * IC * ID * IH * IW * MB_BLOCK;
    } while (omb < mb_end);

#if WITH_BIAS == 1
    if (do_bias) diff_bias[0] = bias_loc;
#endif

#if IC == 3
    diff_wei += oc * KD * KH * KW * IC * OC_BLOCK + g * OC * IC * KD * KH * KW
            + kd * KH * KW * IC * OC_BLOCK + kh * KW * IC * OC_BLOCK
            + kw * IC * OC_BLOCK + mb_chunk * G * OC * IC * KD * KH * KW;
    intel_sub_group_block_write(
            (__global uint *)(&diff_wei[0]), as_uint(blockC00[0]));
    intel_sub_group_block_write(
            (__global uint *)(&diff_wei[OC_BLOCK]), as_uint(blockC00[1]));
    intel_sub_group_block_write(
            (__global uint *)(&diff_wei[2 * OC_BLOCK]), as_uint(blockC00[2]));
#elif IS_DW
    diff_wei += oc * KD * KH * KW * OC_BLOCK + kd * KH * KW * OC_BLOCK
            + kh * KW * OC_BLOCK + kw * OC_BLOCK
            + mb_chunk * G * OC * IC * KD * KH * KW;
    intel_sub_group_block_write(
            (__global unsigned int *)(&diff_wei[0]), as_uint(blockC00));
#else
    diff_wei += ic * OC * KD * KH * KW * IC_BLOCK
            + oc * KD * KH * KW * IC_BLOCK * OC_BLOCK
            + kd * KH * KW * IC_BLOCK * OC_BLOCK + kh * KW * IC_BLOCK * OC_BLOCK
            + kw * IC_BLOCK * OC_BLOCK + g * OC * IC * KD * KH * KW
            + mb_chunk * G * OC * IC * KD * KH * KW;
    intel_sub_group_block_write8(
            (__global unsigned int *)(&diff_wei[0]), as_uint8(blockC00));
    intel_sub_group_block_write8(
            (__global unsigned int *)(&diff_wei[8 * OC_BLOCK]),
            as_uint8(blockC01));
#endif
#endif
}
#endif
