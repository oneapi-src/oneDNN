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

#if ID > 1
#    define CASE_3D 1
#else
#    define CASE_3D 0
#endif

#if BWD_DATA == 1

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
#    if VER_16MB16C == 1 || VER_8OW16C == 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#    endif
__kernel void gen9_common_conv_bwd_data_kernel(__global float *diff_src,
        __global float *wei, __global float *diff_dst) {

#    if VER_16MB16C == 1 || VER_8OW16C == 1
    const int mb_unroll = 16;

    const int ic = get_group_id(0);
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);
    int mb = get_group_id(2) * mb_unroll;

#        if IS_DW
    const int g = ic * IC_BLOCK;
    const int gic = 0;
#        else
    const int g = ic / (IC / IC_BLOCK);
    const int gic = ic % (IC / IC_BLOCK);
#        endif

#        if CASE_3D
    const int id = sp / (IW * IH);
    const int ihw = sp % (IW * IH);
#        else
    const int id = 0;
    const int ihw = sp;
#        endif
    const int ih = ihw / IW;
    const int iw = ihw % IW;

    diff_dst += mb * OC * G * OD * OH * OW + g * OC * OD * OH * OW * MB_BLOCK;

    float8 blockC00 = 0.0f;
    float8 blockC01 = 0.0f;

    wei += gic * KD * KH * KW * OC_BLOCK * IC_BLOCK
            + g * IC * OC * KD * KH * KW;

    int ocb = 0;
    do {
#        if KH != 1 || KW != 1 || KD != 1
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {

                    if (iw + PW < kw * (1 + DW) || ih + PH < kh * (1 + DH))
                        continue;
#            if CASE_3D
                    if (id + PD < kd * (1 + DD))
                        continue;
                    int od = id - kd * (1 + DD) + PD;
                    if (od % SD != 0)
                        continue;
                    od /= SD;
                    if (od >= OD)
                        continue;
#            endif

                    int ow = iw - kw * (1 + DW) + PW;
                    int oh = ih - kh * (1 + DH) + PH;
                    if (ow % SW != 0 || oh % SH != 0)
                        continue;
                    ow /= SW;
                    oh /= SH;
                    if (oh >= OH || ow >= OW)
                        continue;

                    const __global float *diff_dst1 = diff_dst
                            + ow * OC_BLOCK * MB_BLOCK
                            + oh * OW * OC_BLOCK * MB_BLOCK;
#            if CASE_3D
                    diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
#            endif
#            if IS_DW
                    const __global float *wei1 = wei
#                if CASE_3D
                            + kd * KH * KW * OC_BLOCK
#                endif
                            + kh * KW * OC_BLOCK + kw * OC_BLOCK;
#            else
                    const __global float *wei1 = wei
#                if CASE_3D
                            + kd * KH * KW * OC_BLOCK * IC_BLOCK
#                endif
                            + kh * KW * OC_BLOCK * IC_BLOCK
                            + kw * OC_BLOCK * IC_BLOCK;
#            endif
#        else
        int ow = (iw + PW);
        int oh = (ih + PH);
#            if CASE_3D
        int od = (id + PD);
#            endif
        bool do_ker = true;
#            if SW != 1 || SH != 1 || SD != 1
        do_ker = ow % SW == 0 && oh % SH == 0;
        ow /= SW;
        oh /= SH;
#                if CASE_3D
        do_ker = do_ker && od % SD == 0;
        od /= SD;
#                endif
#            endif
#            if PH != 0 || PW != 0 || PD != 0
        do_ker = do_ker && (oh < OH && ow < OW);
#                if CASE_3D
        do_ker = do_ker && (od < OD);
#                endif
#            endif
#            if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
        if (do_ker) {
#            endif
            const __global float *diff_dst1 = diff_dst
                    + ow * OC_BLOCK * MB_BLOCK + oh * OW * OC_BLOCK * MB_BLOCK;
#            if CASE_3D
            diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
#            endif
            const __global float *wei1 = wei;
#        endif

#        if MB == MB_LAST
#            define LOAD_DIFF_DST(_block, _diff_dst, mb_chunk)        \
                {                                                     \
                    (_block) = as_float8(intel_sub_group_block_read8( \
                            (const __global uint *)((_diff_dst)       \
                                    + (mb_chunk)*OC_BLOCK)));         \
                }
#        else
#            define LOAD_DIFF_DST(_block, _diff_dst, mb_chunk)                 \
                {                                                              \
                    if (mb == MB_LAST) {                                       \
                        for (int i = 0; i < min(8, MB - MB_LAST - (mb_chunk)); \
                                i++)                                           \
                            (_block)[i] = as_float(intel_sub_group_block_read( \
                                    (const __global uint *)(&(                 \
                                            _diff_dst)[((mb_chunk) + i) * OC   \
                                            * G * OD * OH * OW])));            \
                    } else {                                                   \
                        for (int i = 0; i < 8; i++)                            \
                            (_block)[i] = as_float(intel_sub_group_block_read( \
                                    (const __global uint *)(&(                 \
                                            _diff_dst)[((mb_chunk) + i) * OC   \
                                            * G * OD * OH * OW])));            \
                    }                                                          \
                }
#        endif

#        if MB == MB_LAST
#            define SAVE_SRC_DIFF(_block, _diff_src, mb_chunk)        \
                {                                                     \
                    intel_sub_group_block_write8(                     \
                            (__global unsigned int *)(&(              \
                                    _diff_src)[(mb_chunk)*IC_BLOCK]), \
                            as_uint8((_block)));                      \
                }
#        else
#            define SAVE_SRC_DIFF(_block, _diff_src, mb_chunk)                 \
                {                                                              \
                    if (mb == MB_LAST) {                                       \
                        for (int i = 0; i < min(8, MB - MB_LAST - (mb_chunk)); \
                                i++) {                                         \
                            intel_sub_group_block_write(                       \
                                    (__global unsigned int *)(&(               \
                                            _diff_src)[((mb_chunk) + i) * IC   \
                                            * G * ID * IH * IW]),              \
                                    as_uint((_block)[i]));                     \
                        }                                                      \
                    } else {                                                   \
                        for (int i = 0; i < 8; i++) {                          \
                            intel_sub_group_block_write(                       \
                                    (__global unsigned int *)(&(               \
                                            _diff_src)[((mb_chunk) + i) * IC   \
                                            * G * ID * IH * IW]),              \
                                    as_uint((_block)[i]));                     \
                        }                                                      \
                    }                                                          \
                }
#        endif

#        define TRANSPOSE_8(_block, _col) \
            (float8)(intel_sub_group_shuffle(_block, _col))

#        define FMA8(a, b, c) fma((float8)(a), (float8)b, (float8)c)

#        define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, _blockB1)       \
            {                                                                  \
                _result = FMA8(_blockB.s0, TRANSPOSE_8(_blockA, 0), _result);  \
                _result = FMA8(_blockB.s1, TRANSPOSE_8(_blockA, 1), _result);  \
                _result = FMA8(_blockB.s2, TRANSPOSE_8(_blockA, 2), _result);  \
                _result = FMA8(_blockB.s3, TRANSPOSE_8(_blockA, 3), _result);  \
                _result = FMA8(_blockB.s4, TRANSPOSE_8(_blockA, 4), _result);  \
                _result = FMA8(_blockB.s5, TRANSPOSE_8(_blockA, 5), _result);  \
                _result = FMA8(_blockB.s6, TRANSPOSE_8(_blockA, 6), _result);  \
                _result = FMA8(_blockB.s7, TRANSPOSE_8(_blockA, 7), _result);  \
                _result = FMA8(_blockB1.s0, TRANSPOSE_8(_blockA, 8), _result); \
                _result = FMA8(_blockB1.s1, TRANSPOSE_8(_blockA, 9), _result); \
                _result = FMA8(                                                \
                        _blockB1.s2, TRANSPOSE_8(_blockA, 10), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s3, TRANSPOSE_8(_blockA, 11), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s4, TRANSPOSE_8(_blockA, 12), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s5, TRANSPOSE_8(_blockA, 13), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s6, TRANSPOSE_8(_blockA, 14), _result);       \
                _result = FMA8(                                                \
                        _blockB1.s7, TRANSPOSE_8(_blockA, 15), _result);       \
            }

#        if IS_DW
                    float blockB00 = as_float(intel_sub_group_block_read(
                            (const __global uint *)wei1));
#        else
            float8 blockB00 = as_float8(
                    intel_sub_group_block_read8((const __global uint *)wei1));
            float8 blockB01 = as_float8(intel_sub_group_block_read8(
                    (const __global uint *)(wei1 + 8 * IC_BLOCK)));
#        endif
                    float8 blockA;

                    LOAD_DIFF_DST(blockA, diff_dst1, 0);
#        if IS_DW
                    blockC00 = fma(blockA, (float8)blockB00, blockC00);
#        else
            MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB00, blockB01);
#        endif

                    LOAD_DIFF_DST(blockA, diff_dst1, 8);
                    if ((mb != MB_LAST) || (MB % 16 > 8)) {
#        if IS_DW
                        blockC01 = fma(blockA, (float8)blockB00, blockC01);
#        else
                MULTIPLY_BLOCKS_8x8(blockC01, blockA, blockB00, blockB01);
#        endif
                    }

#        undef TRANSPOSE_BLOCK_8
#        undef MULTIPLY_BLOCKS_8x8
#        if KH != 1 || KW != 1 || KD != 1
                }
#        else
#            if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
        }
#            endif
#        endif
        diff_dst += OC_BLOCK * OD * OH * OW * MB_BLOCK;
        wei += IC * KD * KH * KW * OC_BLOCK;
        ocb += OC_BLOCK;
    } while (ocb < OC);

    __global float *src_write0 = diff_src + mb * IC * G * ID * IH * IW
            + gic * ID * IH * IW * IC_BLOCK * MB_BLOCK
            + g * IC * ID * IH * IW * MB_BLOCK
            + id * IH * IW * IC_BLOCK * MB_BLOCK + ih * IW * IC_BLOCK * MB_BLOCK
            + iw * IC_BLOCK * MB_BLOCK;

    SAVE_SRC_DIFF(blockC00, src_write0, 0);
    SAVE_SRC_DIFF(blockC01, src_write0, 8);

#    endif

#    if VER_REF == 1
    const uint iw = get_global_id(0) * IW_BLOCK;
    const uint ihd = get_global_id(1);
    const uint id = ihd / IH;
    const uint ih = ihd % IH;
    const uint mb = get_global_id(2) / (G * IC);
    const uint ic = get_global_id(2) % (G * IC);

    const int g = ic / (IC / IC_BLOCK);
    const int gic = ic % (IC / IC_BLOCK);

    diff_dst += g * OC * OD * OH * OW;
    wei += g * IC * OC * KD * KH * KW;

    float d = 0.0;
    const uint diff_src_off = mb * IC * G * ID * IH * IW + gic * ID * IH * IW
            + g * IC * ID * IH * IW + id * IH * IW + ih * IW + iw;
    for (int oc = 0; oc < OC; ++oc)
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {
                    if (iw + PW < kw * (1 + DW) || ih + PH < kh * (1 + DH)
                            || id + PD < kd * (1 + DD))
                        continue;
                    int ow = iw - kw * (1 + DW) + PW;
                    int oh = ih - kh * (1 + DH) + PH;
                    int od = id - kd * (1 + DD) + PD;
                    if (ow % SW != 0 || oh % SH != 0 || od % SD != 0)
                        continue;

                    ow /= SW;
                    oh /= SH;
                    od /= SD;
                    if (oh < OH && ow < OW && od < OD) {
                        const uint diff_dst_off = mb * OC * G * OD * OH * OW
                                + oc * OD * OH * OW + od * OH * OW + oh * OW
                                + ow;
                        const uint wei_off = oc * IC * KD * KH * KW
                                + gic * KD * KH * KW + kd * KH * KW + kh * KW
                                + kw;
                        d += diff_dst[diff_dst_off] * wei[wei_off];
                    }
                }
    diff_src[diff_src_off] = d;
#    endif
}
#endif
