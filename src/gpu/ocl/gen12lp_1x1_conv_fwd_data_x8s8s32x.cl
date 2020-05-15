/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#if IC % IC_BLOCK != 0
#define IC_NBLOCKS_TAIL ((IC - (IC & ~(IC_BLOCK - 1)) + 3) / 4)
#else
#define IC_NBLOCKS_TAIL 8
#endif

#if SP_BLOCK == 4
#define ACC_DATA_BLOCK int4
#define SRC_DATA_BLOCK_T MMAD_DATA4_T
#define DST_DATA_BLOCK_T uint4
#define BLOCK_READ_SRC(data, idx) \
    data = AS_MMAD_DATA4_T( \
            intel_sub_group_block_read4((__global uint *)&src[idx]));
DECLARE_MMAD(_full0, SRC_DATA_BLOCK_T, ACC_DATA_BLOCK, MMAD_DATA8_T, 4, 8)
DECLARE_MMAD(_tail0, SRC_DATA_BLOCK_T, ACC_DATA_BLOCK, MMAD_DATA8_T, 4,
        IC_NBLOCKS_TAIL)
#define MMAD_FULL0 mmad_full0
#define MMAD_TAIL0 mmad_tail0
#else
#define ACC_DATA_BLOCK int8
#define SRC_DATA_BLOCK_T MMAD_DATA8_T
#define DST_DATA_BLOCK_T uint8
#define BLOCK_READ_SRC(data, idx) \
    data = AS_MMAD_DATA8_T( \
            intel_sub_group_block_read8((__global uint *)&src[idx]));
DECLARE_MMAD(_full0, SRC_DATA_BLOCK_T, ACC_DATA_BLOCK, MMAD_DATA8_T, 8, 8)
DECLARE_MMAD(_tail0, SRC_DATA_BLOCK_T, ACC_DATA_BLOCK, MMAD_DATA8_T, 8,
        IC_NBLOCKS_TAIL)
#define MMAD_FULL0 mmad_full0
#define MMAD_TAIL0 mmad_tail0
#endif

#if SP_BLOCK == 12
#define ACC_DATA_BLOCK1 int4
#define SRC_DATA_BLOCK_T1 MMAD_DATA4_T
#define DST_DATA_BLOCK_T1 uint4
#define BLOCK_READ_SRC1(data, idx) \
    data = AS_MMAD_DATA4_T( \
            intel_sub_group_block_read4((__global uint *)&src[idx]));
DECLARE_MMAD(_full1, SRC_DATA_BLOCK_T1, ACC_DATA_BLOCK1, MMAD_DATA8_T, 4, 8)
DECLARE_MMAD(_tail1, SRC_DATA_BLOCK_T1, ACC_DATA_BLOCK1, MMAD_DATA8_T, 4,
        IC_NBLOCKS_TAIL)
#define MMAD_FULL1 mmad_full1
#define MMAD_TAIL1 mmad_tail1
#else
#define ACC_DATA_BLOCK1 int8
#define SRC_DATA_BLOCK_T1 MMAD_DATA8_T
#define DST_DATA_BLOCK_T1 uint8
#define BLOCK_READ_SRC1(data, idx) \
    data = AS_MMAD_DATA8_T( \
            intel_sub_group_block_read8((__global uint *)&src[idx]));
DECLARE_MMAD(_full1, SRC_DATA_BLOCK_T1, ACC_DATA_BLOCK1, MMAD_DATA8_T, 8, 8)
DECLARE_MMAD(_tail1, SRC_DATA_BLOCK_T1, ACC_DATA_BLOCK1, MMAD_DATA8_T, 8,
        IC_NBLOCKS_TAIL)
#define MMAD_FULL1 mmad_full1
#define MMAD_TAIL1 mmad_tail1
#endif

#if INT8_WEI_SLM
#define BLOCK_READ_WHT_1x32(data, idx) \
    data = as_int(READ_LOCAL_1((__local uint *)&wei_tmp[idx]));
#define BLOCK_READ_WHT_8x32(data, idx) \
    data = as_int8(READ_LOCAL_8((__local uint *)&wei_tmp[idx]));
#else
#define BLOCK_READ_WHT_1x32(data, idx) \
    data = as_int(intel_sub_group_block_read((__global uint *)&wei[idx]));
#define BLOCK_READ_WHT_8x32(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));
#endif

#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

#define BLOCK_READ_SCALES(data, idx) \
    data = as_float4(intel_sub_group_block_read4( \
            (__global uint *)&scales_per_oc[idx]));

#define CHANNEL_OFFSET 1
#define MB_OFFSET IC_BLOCK
#define PIXEL_WIDTH_OFFSET (MB_OFFSET * MB_BLOCK)
#define PIXEL_HEIGHT_OFFSET (PIXEL_WIDTH_OFFSET * IW)
#define CHANNEL_BLOCK_OFFSET (PIXEL_HEIGHT_OFFSET * IH)
#define DST_PIXEL_HEIGHT_OFFSET (PIXEL_WIDTH_OFFSET * OW)
#define DST_CHANNEL_BLOCK_OFFSET (DST_PIXEL_HEIGHT_OFFSET * OH)

// Weights offsets
#define WEIGHTS_WIDTH_OFFSET (4 * 8 * 8 * 4)
#define WEIGHTS_HEIGHT_OFFSET (WEIGHTS_WIDTH_OFFSET * 1)
#define KERNEL_BLOCK_OFFSET (WEIGHTS_HEIGHT_OFFSET * 1)

#if SCALES_PER_OC
#define SCALE scales
#elif SCALES_COMMON
#define SCALE scale
#else
#define SCALE 1
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
gen12lp_1x1_conv_fwd_x8s8s32x(const __global SRC_DATA_T *src,
        const __global char *wei, const __global float *bias,
        __global DATA_T *dst, float eltwise_alpha, float eltwise_beta,
        float eltwise_scale, float sum_scale, float scale,
        const __global float *scales_per_oc) {

    // Groups:
    const uint oc_group_id = get_group_id(0);
    const uint sp_group_id = get_group_id(1);
    const uint mb_group_id = get_group_id(2);
    const uint ic_group_id = oc_group_id / OC_NCHUNK * IC_NCHUNK;

    // SIMD
    const uint sg_local_id = get_sub_group_local_id();
    const uint sg_id = get_sub_group_id();

    // Spatial
    const uint sp = get_global_id(1) * SP_BLOCK;
    const int sp_local_id = get_local_id(1);
    const uint ow = sp % OW;
    const uint oh = sp / OW;
    const uint iw = SW * ow;
    const uint ih = SH * oh;

    // Source (At ic = 0)
#if MB_BLOCK == 32
    src += (mb_group_id % 2) * MB_BLOCK / 2 * MB_OFFSET; // MB block offset
    src += (mb_group_id / 2) * CHANNEL_BLOCK_OFFSET * IC_NCHUNK
            * G; // MB offset
#else
    src += mb_group_id * CHANNEL_BLOCK_OFFSET * IC_NCHUNK * G; // MB offset
#endif
    src += CHANNEL_BLOCK_OFFSET * ic_group_id; // IC offset
    src += ih * PIXEL_HEIGHT_OFFSET; // height offset
    src += iw * PIXEL_WIDTH_OFFSET; // width offset

    // Destination
#if MB_BLOCK == 32
    dst += (mb_group_id % 2) * MB_BLOCK / 2 * MB_OFFSET; // MB block offset
    dst += (mb_group_id / 2) * DST_CHANNEL_BLOCK_OFFSET * OC_NCHUNK
            * G; // MB offset
#else
    dst += mb_group_id * DST_CHANNEL_BLOCK_OFFSET * OC_NCHUNK * G; // MB offset
#endif
    dst += DST_CHANNEL_BLOCK_OFFSET * oc_group_id; // OC offset
    dst += oh * DST_PIXEL_HEIGHT_OFFSET;
    dst += ow * PIXEL_WIDTH_OFFSET;

    // Weights
    wei += oc_group_id * KERNEL_BLOCK_OFFSET * IC_NCHUNK;
    // Output accumulators:

    // 8 MB (0-7) x 4 Kernels  (32 8bit ints)
    ACC_DATA_BLOCK C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    // 8 MB (8-15) x 4 Kernels  (32 8bit ints)
    ACC_DATA_BLOCK1 C10 = 0, C11 = 0, C12 = 0, C13 = 0;

#if INT8_WEI_SLM
#define READ_SLM() \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const __global char *wei_copy_from \
            = wei + sp_local_id * KERNEL_BLOCK_OFFSET / LWS_1; \
    __local char *wei_copy_to \
            = wei_slm + sp_local_id * KERNEL_BLOCK_OFFSET / LWS_1; \
    WRITE_LOCAL_4((__local uint *)wei_copy_to, \
            intel_sub_group_block_read4((__global uint *)wei_copy_from)); \
    __local char *wei_tmp = wei_slm; \
    barrier(CLK_LOCAL_MEM_FENCE);

    __local char wei_slm[KERNEL_BLOCK_OFFSET];
#endif // INT8_WEI_SLM
    for (uint ic_block_id = 0; ic_block_id < IC_NCHUNK; ++ic_block_id) {
#if INT8_WEI_SLM
        READ_SLM()
#if SP_TAIL
        if (sp < OW * OH) {
#endif
#endif

            SRC_DATA_BLOCK_T S0;
            SRC_DATA_BLOCK_T1 S1;

#if OUT_SP_TAIL
            if (sp + SP_BLOCK > OW * OH) {
#if OUT_SP_TAIL < 8
                S0 = 0;
                for (int i = 0; i < OUT_SP_TAIL; ++i) {
                    S0[i] = AS_MMAD_DATA_T(intel_sub_group_block_read(
                            (__global uint *)&src[i * IC_BLOCK]));
                }
#else
            BLOCK_READ_SRC(S0, 0 * IC_BLOCK);
            S1 = 0;
            for (int i = 8; i < OUT_SP_TAIL; ++i) {
                S1[i - 8] = AS_MMAD_DATA_T(intel_sub_group_block_read(
                        (__global uint *)&src[i * IC_BLOCK]));
            }
#endif
            } else {
#endif
                BLOCK_READ_SRC(S0, 0 * IC_BLOCK);
#if (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
                BLOCK_READ_SRC1(S1, 8 * IC_BLOCK);
#endif
#if OUT_SP_TAIL
            }
#endif

            int8 W0 = 0, W1 = 0, W2 = 0, W3 = 0;

#if IC % IC_BLOCK != 0
            if (ic_block_id == IC_NCHUNK - 1) {
                unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                        BLOCK_READ_WHT_1x32(W0[i], (i + 0) * IC_BLOCK);
                if (OC > 8)
                    unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                            BLOCK_READ_WHT_1x32(W1[i], (i + 8) * IC_BLOCK);
                if (OC > 16)
                    unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                            BLOCK_READ_WHT_1x32(W2[i], (i + 16) * IC_BLOCK);
                if (OC > 24)
                    unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                            BLOCK_READ_WHT_1x32(W3[i], (i + 24) * IC_BLOCK);

                C00 = MMAD_TAIL0(S0, W0, C00);
                if (OC > 8) C01 = MMAD_TAIL0(S0, W1, C01);
                if (OC > 16) C02 = MMAD_TAIL0(S0, W2, C02);
                if (OC > 24) C03 = MMAD_TAIL0(S0, W3, C03);
#if (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
                C10 = MMAD_TAIL1(S1, W0, C10);
                if (OC > 8) C11 = MMAD_TAIL1(S1, W1, C11);
                if (OC > 16) C12 = MMAD_TAIL1(S1, W2, C12);
                if (OC > 24) C13 = MMAD_TAIL1(S1, W3, C13);
#endif
            } else
#endif // IC % IC_BLOCK != 0
            {
                BLOCK_READ_WHT_8x32(W0, 0);
                if (OC > 8) BLOCK_READ_WHT_8x32(W1, 8 * IC_BLOCK);
                if (OC > 16) BLOCK_READ_WHT_8x32(W2, 16 * IC_BLOCK);
                if (OC > 24) BLOCK_READ_WHT_8x32(W3, 24 * IC_BLOCK);

                C00 = MMAD_FULL0(S0, W0, C00);
                if (OC > 8) C01 = MMAD_FULL0(S0, W1, C01);
                if (OC > 16) C02 = MMAD_FULL0(S0, W2, C02);
                if (OC > 24) C03 = MMAD_FULL0(S0, W3, C03);
#if (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
                C10 = MMAD_FULL1(S1, W0, C10);
                if (OC > 8) C11 = MMAD_FULL1(S1, W1, C11);
                if (OC > 16) C12 = MMAD_FULL1(S1, W2, C12);
                if (OC > 24) C13 = MMAD_FULL1(S1, W3, C13);
#endif
            }
#if INT8_WEI_SLM && SP_TAIL
        }
#endif

        src += CHANNEL_BLOCK_OFFSET;
        wei += KERNEL_BLOCK_OFFSET;
    }

    float4 tmp;
    uint8 dst_pack;
    DST_DATA_BLOCK_T D0, D1;

#if SCALES_PER_OC
    float4 scales;
    BLOCK_READ_SCALES(scales, oc_group_id * OC_BLOCK);
#endif

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, oc_group_id * OC_BLOCK);
    bia *= SCALE;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)SCALE, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= SCALE;
#endif

#if WITH_SUM
#if MB_BLOCK == 32
    D0.s0123 = as_uint4(intel_sub_group_block_read_uc16((__global uchar *)dst));
    D0.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[4 * OC_BLOCK]));
#if MB > 8
    D1.s0123 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[8 * OC_BLOCK]));
    D1.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[12 * OC_BLOCK]));
#endif
#else
#if OUT_SP_TAIL
if (sp + SP_BLOCK > OW * OH) {
#if OUT_SP_TAIL < 8
    D0 = 0;
    for (int i = 0; i < OUT_SP_TAIL; ++i) {
        D0[i] = as_uint(intel_sub_group_block_read_uc4(
                (__global uchar *)&dst[i * OC_BLOCK]));
    }
#else
    D0.s0123 = as_uint4(intel_sub_group_block_read_uc16((__global uchar *)dst));
    D0.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[4 * OC_BLOCK]));
    D1 = 0;
    for (int i = 8; i < OUT_SP_TAIL; ++i) {
        D1[i - 8] = as_uint(intel_sub_group_block_read_uc4(
                (__global uchar *)&dst[i * OC_BLOCK]));
    }
#endif
} else {
#endif
    D0.s0123 = as_uint4(intel_sub_group_block_read_uc16((__global uchar *)dst));
#if SP_BLOCK > 4
    D0.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[4 * OC_BLOCK]));
#endif
#if SP_BLOCK > 8
    D1.s0123 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[8 * OC_BLOCK]));
#endif
#if SP_BLOCK > 12
    D1.s4567 = as_uint4(intel_sub_group_block_read_uc16(
            (__global uchar *)&dst[12 * OC_BLOCK]));
#endif
#if OUT_SP_TAIL
}
#endif
#endif

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

#if WITH_ELTWISE && !WITH_POST_SUM_ELTWISE
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

#if SP_BLOCK == 4
#define STORE_DST(C0, C1, C2, C3, D, mb_stride) \
    do { \
        for (int n_i = 0; n_i < 4; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            DO_ELTWISE(); \
            DO_SUM(D[n_i]); \
            DO_POST_SUM_ELTWISE(); \
            CONVERT_PACK(n_i); \
        } \
        intel_sub_group_block_write_uc16( \
                &dst[mb_stride * OC_BLOCK], as_uchar16(dst_pack.s0123)); \
    } while (0)
#else
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
#endif
#if SP_BLOCK == 12
#define STORE_DST1(C0, C1, C2, C3, D, mb_stride) \
    do { \
        for (int n_i = 0; n_i < 4; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            DO_ELTWISE(); \
            DO_SUM(D[n_i]); \
            DO_POST_SUM_ELTWISE(); \
            CONVERT_PACK(n_i); \
        } \
        intel_sub_group_block_write_uc16( \
                &dst[mb_stride * OC_BLOCK], as_uchar16(dst_pack.s0123)); \
    } while (0)
#else
#define STORE_DST1(C0, C1, C2, C3, D, mb_stride) \
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
#endif

#define STORE_DST_TAIL(C0, C1, C2, C3, D, mb_stride, tail) \
    do { \
        for (int n_i = 0; n_i < tail; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            DO_ELTWISE(); \
            DO_SUM(D[n_i]); \
            DO_POST_SUM_ELTWISE(); \
            CONVERT_PACK(n_i); \
            intel_sub_group_block_write_uc4( \
                    &dst[mb_stride * OC_BLOCK + n_i * 4 * 8], \
                    as_uchar4(dst_pack[n_i])); \
        } \
    } while (0)
#if INT8_WEI_SLM && SP_TAIL
    if (sp < OW * OH) {
#endif

#if OUT_SP_TAIL
        if (sp + SP_BLOCK > OW * OH) {
#if OUT_SP_TAIL < 8
            STORE_DST_TAIL(C00, C01, C02, C03, D0, 0, OUT_SP_TAIL);
#else
        STORE_DST(C00, C01, C02, C03, D0, 0);
        STORE_DST_TAIL(C10, C11, C12, C13, D1, 8, OUT_SP_TAIL - 8);
#endif
        } else {
#endif

            STORE_DST(C00, C01, C02, C03, D0, 0);
#if (MB_BLOCK == 32 && MB > 8) || SP_BLOCK > 8
            STORE_DST1(C10, C11, C12, C13, D1, 8);
#endif
#if OUT_SP_TAIL
        }
#endif
#if INT8_WEI_SLM && SP_TAIL
    }
#endif
}
