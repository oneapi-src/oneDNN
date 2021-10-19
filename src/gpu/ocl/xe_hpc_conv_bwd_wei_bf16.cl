/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#define USHORT_PER_READ (16 * SUB_GROUP_SIZE)
#define INT_PER_READ (USHORT_PER_READ / 2)

#define WORKGROUP_SIZE (LWS_0 / SUB_GROUP_SIZE)

#define MAX_SGID_IC (MB_BLK_WORKGROUP * IC_BLK_WORKGROUP)
#define MAX_SGID_OC (MB_BLK_WORKGROUP * OC_BLK_WORKGROUP)
#define MAX_SGID_COMPUTE \
    ((OC_BLK_WORKGROUP / OC_BLK_SUBGROUP) \
            * (IC_BLK_WORKGROUP / IC_BLK_SUBGROUP))

// Using hard-code strides instead of SRC_OFF/DST_OFF/WEI_OFF
// because compiler generates ugly code for SRC_OFF
#define SRC_W_STRIDE \
    (2 * MB_BLOCK * IC_BLOCK) // factor of 2 is because src fmt is NChw32n16c
#define SRC_H_STRIDE (IW * SRC_W_STRIDE)
#define SRC_D_STRIDE (IH * SRC_H_STRIDE)
#define SRC_C_STRIDE (ID * SRC_D_STRIDE)
#define SRC_MB_STRIDE (G * IC / IC_BLOCK * SRC_C_STRIDE)

// DST fmt is NChw32n16c
#define DST_W_STRIDE (2 * MB_BLOCK * OC_BLOCK)
#define DST_H_STRIDE (OW * DST_W_STRIDE)
#define DST_D_STRIDE (OH * DST_H_STRIDE)
#define DST_C_STRIDE (OD * DST_D_STRIDE)
#define DST_MB_STRIDE (G * OC / OC_BLOCK * DST_C_STRIDE)

#define GEMM_IC_blk(o, i) \
    do { \
        ACC[o][2 * i] = mmad8x8_bf16( \
                as_ushort8(D[o].s0123), as_int8(S[i][0]), ACC[o][2 * i]); \
        ACC[o][2 * i + 1] = mmad8x8_bf16( \
                as_ushort8(D[o].s4567), as_int8(S[i][0]), ACC[o][2 * i + 1]); \
    } while (0)

#if OC_BLK_SUBGROUP == 2
#define READ_DST_GLOBAL() \
    do { \
        dst_off = (size_t)(n_block / MAX_SGID_OC) * DST_MB_STRIDE \
                + od * DST_D_STRIDE + oh * DST_H_STRIDE + ow * DST_W_STRIDE \
                + n_block_inner * MB_BLOCK * OC_BLOCK + ((sg_loc_id)*16); \
        Dt[0] = as_ushort16(vload8(0, (__global uint *)&diff_dst[dst_off])); \
        Dt[1] = as_ushort16(vload8( \
                0, (__global uint *)&diff_dst[dst_off + DST_C_STRIDE])); \
    } while (0)

#else // OC_BLK_SUBGROUP == 1
#define READ_DST_GLOBAL() \
    do { \
        dst_off = (size_t)(n_block / MAX_SGID_OC) * DST_MB_STRIDE \
                + od * DST_D_STRIDE + oh * DST_H_STRIDE + ow * DST_W_STRIDE \
                + (n_block_inner / MAX_SGID_OC) * MB_BLOCK * OC_BLOCK \
                + ((sg_loc_id)*16); \
        Dt[0] = as_ushort16(vload8(0, (__global uint *)&diff_dst[dst_off])); \
    } while (0)

#endif

#if WITH_BIAS
#define CONVERT_TO_F32(x) cvt_bf16_to_f32(x)

#if OC_BLK_SUBGROUP == 2
#define WRITE_DST() \
    do { \
        BIAS_ACC[0][0] += CONVERT_TO_F32(Dt[0].s0); \
        BIAS_ACC[0][1] += CONVERT_TO_F32(Dt[0].s1); \
        BIAS_ACC[0][2] += CONVERT_TO_F32(Dt[0].s2); \
        BIAS_ACC[0][3] += CONVERT_TO_F32(Dt[0].s3); \
        BIAS_ACC[0][4] += CONVERT_TO_F32(Dt[0].s4); \
        BIAS_ACC[0][5] += CONVERT_TO_F32(Dt[0].s5); \
        BIAS_ACC[0][6] += CONVERT_TO_F32(Dt[0].s6); \
        BIAS_ACC[0][7] += CONVERT_TO_F32(Dt[0].s7); \
        BIAS_ACC[0][8] += CONVERT_TO_F32(Dt[0].s8); \
        BIAS_ACC[0][9] += CONVERT_TO_F32(Dt[0].s9); \
        BIAS_ACC[0][10] += CONVERT_TO_F32(Dt[0].sa); \
        BIAS_ACC[0][11] += CONVERT_TO_F32(Dt[0].sb); \
        BIAS_ACC[0][12] += CONVERT_TO_F32(Dt[0].sc); \
        BIAS_ACC[0][13] += CONVERT_TO_F32(Dt[0].sd); \
        BIAS_ACC[0][14] += CONVERT_TO_F32(Dt[0].se); \
        BIAS_ACC[0][15] += CONVERT_TO_F32(Dt[0].sf); \
        BIAS_ACC[1][0] += CONVERT_TO_F32(Dt[1].s0); \
        BIAS_ACC[1][1] += CONVERT_TO_F32(Dt[1].s1); \
        BIAS_ACC[1][2] += CONVERT_TO_F32(Dt[1].s2); \
        BIAS_ACC[1][3] += CONVERT_TO_F32(Dt[1].s3); \
        BIAS_ACC[1][4] += CONVERT_TO_F32(Dt[1].s4); \
        BIAS_ACC[1][5] += CONVERT_TO_F32(Dt[1].s5); \
        BIAS_ACC[1][6] += CONVERT_TO_F32(Dt[1].s6); \
        BIAS_ACC[1][7] += CONVERT_TO_F32(Dt[1].s7); \
        BIAS_ACC[1][8] += CONVERT_TO_F32(Dt[1].s8); \
        BIAS_ACC[1][9] += CONVERT_TO_F32(Dt[1].s9); \
        BIAS_ACC[1][10] += CONVERT_TO_F32(Dt[1].sa); \
        BIAS_ACC[1][11] += CONVERT_TO_F32(Dt[1].sb); \
        BIAS_ACC[1][12] += CONVERT_TO_F32(Dt[1].sc); \
        BIAS_ACC[1][13] += CONVERT_TO_F32(Dt[1].sd); \
        BIAS_ACC[1][14] += CONVERT_TO_F32(Dt[1].se); \
        BIAS_ACC[1][15] += CONVERT_TO_F32(Dt[1].sf); \
        D[0] = as_uint8((ushort16)(Dt[0])); \
        D[2] = as_uint8((ushort16)(Dt[1])); \
    } while (0)

#else
#define WRITE_DST() \
    do { \
        BIAS_ACC[0][0] += CONVERT_TO_F32(Dt[0].s0); \
        BIAS_ACC[0][1] += CONVERT_TO_F32(Dt[0].s1); \
        BIAS_ACC[0][2] += CONVERT_TO_F32(Dt[0].s2); \
        BIAS_ACC[0][3] += CONVERT_TO_F32(Dt[0].s3); \
        BIAS_ACC[0][4] += CONVERT_TO_F32(Dt[0].s4); \
        BIAS_ACC[0][5] += CONVERT_TO_F32(Dt[0].s5); \
        BIAS_ACC[0][6] += CONVERT_TO_F32(Dt[0].s6); \
        BIAS_ACC[0][7] += CONVERT_TO_F32(Dt[0].s7); \
        BIAS_ACC[0][8] += CONVERT_TO_F32(Dt[0].s8); \
        BIAS_ACC[0][9] += CONVERT_TO_F32(Dt[0].s9); \
        BIAS_ACC[0][10] += CONVERT_TO_F32(Dt[0].sa); \
        BIAS_ACC[0][11] += CONVERT_TO_F32(Dt[0].sb); \
        BIAS_ACC[0][12] += CONVERT_TO_F32(Dt[0].sc); \
        BIAS_ACC[0][13] += CONVERT_TO_F32(Dt[0].sd); \
        BIAS_ACC[0][14] += CONVERT_TO_F32(Dt[0].se); \
        BIAS_ACC[0][15] += CONVERT_TO_F32(Dt[0].sf); \
        D[0] = as_uint8((ushort16)(Dt[0])); \
    } while (0)

#endif
#else //WITHOUT  BIAS

#if OC_BLK_SUBGROUP == 2
#define WRITE_DST() \
    do { \
        D[0] = as_uint8((ushort16)(Dt[0])); \
        D[2] = as_uint8((ushort16)(Dt[1])); \
    } while (0)
#else
#define WRITE_DST() \
    do { \
        D[0] = as_uint8((ushort16)(Dt[0])); \
    } while (0)

#endif // OC_BLK_SUBGROUP == 2
#endif // WITH_BIAS

#if IC_BLK_SUBGROUP == 2
#define READ_SRC_GLOBAL() \
    do { \
        src_off = (size_t)(n_block / MAX_SGID_IC) * SRC_MB_STRIDE \
                + id * SRC_D_STRIDE + ih * SRC_H_STRIDE + iw * SRC_W_STRIDE \
                + n_block_inner * MB_BLOCK * IC_BLOCK; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h(&src[src_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                &src[src_off + SRC_C_STRIDE]); \
    } while (0)
#else
#define READ_SRC_GLOBAL() \
    do { \
        src_off = (size_t)(n_block / MAX_SGID_IC) * SRC_MB_STRIDE \
                + id * SRC_D_STRIDE + ih * SRC_H_STRIDE + iw * SRC_W_STRIDE \
                + (n_block_inner / MAX_SGID_IC) * MB_BLOCK * IC_BLOCK; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h(&src[src_off]); \
    } while (0)
#endif
#if IC_BLK_SUBGROUP == 2
#define WRITE_SRC() \
    do { \
        S[0][0] = (uint8)(as_uint(Dt[0].s01), as_uint(Dt[0].s23), \
                as_uint(Dt[0].s45), as_uint(Dt[0].s67), as_uint(Dt[0].s89), \
                as_uint(Dt[0].sAB), as_uint(Dt[0].sCD), as_uint(Dt[0].sEF)); \
        S[1][0] = (uint8)(as_uint(Dt[1].s01), as_uint(Dt[1].s23), \
                as_uint(Dt[1].s45), as_uint(Dt[1].s67), as_uint(Dt[1].s89), \
                as_uint(Dt[1].sAB), as_uint(Dt[1].sCD), as_uint(Dt[1].sEF)); \
    } while (0)

#else
#define WRITE_SRC() \
    do { \
        S[0][0] = (uint8)(as_uint(Dt[0].s01), as_uint(Dt[0].s23), \
                as_uint(Dt[0].s45), as_uint(Dt[0].s67), as_uint(Dt[0].s89), \
                as_uint(Dt[0].sAB), as_uint(Dt[0].sCD), as_uint(Dt[0].sEF)); \
    } while (0)
#endif
// READ_SRC reads 16n block of src (block layout: 2c8n8c2n) from SLM
#if OC_BLK_SUBGROUP == 2
#define COMPUTE(i_c) \
    do { \
        GEMM_IC_blk(0, i_c); \
        GEMM_IC_blk(2, i_c); \
    } while (0)
#elif OC_BLK_SUBGROUP == 1
#define COMPUTE(i_c) \
    do { \
        GEMM_IC_blk(0, i_c); \
    } while (0)
#else
#error UNEXPECTED OC_BLK_SUBGROUP
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
xe_hpc_conv_bwd_wei_bf16(const __global ushort *src, __global float *diff_wei,
        __global float *diff_bias, const __global ushort *diff_dst) {

    const int gid[2] = {get_group_id(0), get_group_id(1)};
    const int sg_id = get_sub_group_id();
    const int sg_loc_id = get_sub_group_local_id();

    // blocks which subgroup will read from global memory
    // e.g. threads TO,T1 read the same oc_block but different mb_block
    const int sgid_n_block = sg_id % MB_BLK_WORKGROUP;
    const int sgid_c_block = sg_id / MB_BLK_WORKGROUP;

    // compute blocks
    // threads T0, T1 compute the same oc_block but different ic_block
    const int sg_oc_blk = sg_id / (IC_BLK_WORKGROUP / IC_BLK_SUBGROUP);
    const int sg_ic_blk = sg_id % (IC_BLK_WORKGROUP / IC_BLK_SUBGROUP);

    const int workgroup_id = gid[0];
    const int group_ic = (workgroup_id % (IC / (IC_BLK_WORKGROUP * IC_BLOCK)))
            * IC_BLK_WORKGROUP;
    const int group_oc = (workgroup_id / (IC / (IC_BLK_WORKGROUP * IC_BLOCK)))
            * OC_BLK_WORKGROUP;

    const int group_g = (gid[1] / K_WORKGROUPS) / (KD * KH * KW);
    const int group_k_block = (gid[1] % K_WORKGROUPS) * K_BLOCKS;
    const int kd = (gid[1] / K_WORKGROUPS / KH / KW) % KD;
    const int kh = (gid[1] / K_WORKGROUPS / KW) % KH;
    const int kw = (gid[1] / K_WORKGROUPS) % KW;

    const int od_start = max((PD - kd * (1 + DD) + SD - 1) / SD, 0);
    const int oh_start = max((PH - kh * (1 + DH) + SH - 1) / SH, 0);
    const int ow_start = max((PW - kw * (1 + DW) + SW - 1) / SW, 0);

    const int od_end
            = OD - max(0, (PD_R - (KD - 1 - kd) * (1 + DD) + SD - 1) / SD) - 1;
    const int oh_end
            = OH - max(0, (PH_R - (KH - 1 - kh) * (1 + DH) + SH - 1) / SH) - 1;
    const int ow_end
            = OW - max(0, (PW_R - (KW - 1 - kw) * (1 + DW) + SW - 1) / SW) - 1;

    // total accumulation dimension for given (kd,kh,kw)
    const int total_od = od_end - od_start + 1;
    const int total_oh = oh_end - oh_start + 1;
    const int total_ow = ow_end - ow_start + 1;
    const int mb_blk_rnd_up = (MB + MB_BLK_WORKGROUP * MB_BLOCK - 1)
            / (MB_BLK_WORKGROUP * MB_BLOCK);
    const int total_k_blocks = mb_blk_rnd_up * total_od * total_oh * total_ow;

    // last thread might do extra work if total_k_blocks % K_BLOCKS != 0
    const int max_k_blocks = ((gid[1] % K_WORKGROUPS) == K_WORKGROUPS - 1)
            ? max(0, total_k_blocks - group_k_block)
            : min(max(0, total_k_blocks - group_k_block), K_BLOCKS);

#if MB_BLK_WORKGROUP == 1 && MB > 16
    int n_block_inner = group_k_block;
    int od = od_start + ((group_k_block / 2 / total_ow / total_oh) % total_od);
    int oh = oh_start + ((group_k_block / 2 / total_ow) % total_oh);
    int ow = ow_start + ((group_k_block / 2) % total_ow);

    int n_block = group_k_block / 2 / (total_od * total_oh * total_ow);
#else
    int n_block_inner = 0;
    int od = od_start + ((group_k_block / total_ow / total_oh) % total_od);
    int oh = oh_start + ((group_k_block / total_ow) % total_oh);
    int ow = ow_start + (group_k_block % total_ow);

    int n_block = group_k_block / (total_od * total_oh * total_ow);
#endif

    const int group_id = od * SD - PD + kd * ((1 + DD) % ID);
    const int group_ih = oh * SH - PH + kh * ((1 + DH) % IH);
    const int group_iw = ow * SW - PW + kw * ((1 + DW) % IW);
    int id = group_id;
    int ih = group_ih;
    int iw = group_iw;

    // each subgroup may read (SRC:MB_BLOCK * IC_BLOCK + DST:MB_BLOCK * OC_BLOCK)
    // elements from global memory
#if MB_BLK_WORKGROUP == 2
#if IC_BLK_SUBGROUP == 2
    src += (group_g * IC / IC_BLOCK + group_ic
                   + (sg_id % IC_BLK_WORKGROUP)
                           % (IC_BLK_WORKGROUP / IC_BLK_SUBGROUP))
            * (SRC_C_STRIDE * IC_BLK_SUBGROUP);
#else
    src += (group_g * IC / IC_BLOCK + group_ic + sg_ic_blk)
            * (SRC_C_STRIDE * IC_BLK_SUBGROUP);
#endif
#else
    src += sgid_n_block * MB_BLOCK * IC_BLOCK
            + (group_g * IC / IC_BLOCK + group_ic
                      + (sg_ic_blk * IC_BLK_SUBGROUP))
                    * SRC_C_STRIDE;
#endif
#if MB_BLK_WORKGROUP == 2
    diff_dst += (group_g * OC / OC_BLOCK + group_oc + sg_oc_blk)
            * (DST_C_STRIDE * OC_BLK_SUBGROUP);
#else
    diff_dst += sgid_n_block * MB_BLOCK * OC_BLOCK
            + (group_g * OC / OC_BLOCK + group_oc
                      + (sg_oc_blk * OC_BLK_SUBGROUP))
                    * DST_C_STRIDE;
#endif
    diff_wei += WEI_OFF(group_g,
            (group_oc + (sg_oc_blk)*OC_BLK_SUBGROUP) * OC_BLOCK,
            (group_ic + (sg_ic_blk)*IC_BLK_SUBGROUP) * IC_BLOCK, kd, kh, kw);

#if WITH_BIAS
    float16 BIAS_ACC[2] = {0.0f, 0.0f};
    bool compute_bias = group_ic == 0 && kd == min(PD, KD - 1)
            && kh == min(PH, KH - 1) && kw == min(PW, KW - 1)
            && sg_id == sg_oc_blk * (IC_BLK_WORKGROUP / IC_BLK_SUBGROUP);
    size_t bia_off;
    volatile __global atomic_float *dbias;
    bia_off = group_g * OC
            + (group_oc + (sg_oc_blk * OC_BLK_SUBGROUP)) * OC_BLOCK;
    dbias = (volatile __global atomic_float *)&diff_bias[bia_off];
#endif // WITH_BIAS

    uint8 S[2][2];
    uint8 D[4];
    ushort16 Dt[2];

    float8 ACC[4][4] = {0.0f};

    int src_mb_offset = MB_BLOCK * IC_BLOCK;
    int dst_mb_offset = MB_BLOCK * OC_BLOCK;

    const int loc_src_compute_blk_offset
            = sg_ic_blk * MB_BLOCK * IC_BLK_SUBGROUP * IC_BLOCK / 2;

    int k_blk_iter = 0;

    size_t src_off, dst_off;
    int buf_num = 0;

    for (; buf_num < min(max_k_blocks, NUM_BUF - 1); ++buf_num) {
        // Each subgroups reads block of 16n16c from global memory
        READ_SRC_GLOBAL();
        WRITE_SRC();

        // Each subgroups reads block of 16n16c from global memory
        READ_DST_GLOBAL();
        WRITE_DST();

#if MB_BLK_WORKGROUP == 1 && MB > 16
        n_block_inner++;
        if (n_block_inner == 2) {
            n_block_inner = 0;
            ow++;
            iw += SW;
        }
#else
        ow++;
        iw += SW;
#endif

        if (ow == ow_end + 1) {
            ow = max((PW - kw * (1 + DW) + SW - 1) / SW, 0);
            oh++;
            iw = ow * SW - PW + kw * (1 + DW);
            ih += SH;
        }
        if (oh == oh_end + 1) {
            oh = max((PH - kh * (1 + DH) + SH - 1) / SH, 0);
            od++;
            ih = oh * SH - PH + kh * (1 + DH);
            id += SD;
        }
        if (od == od_end + 1) {
            od = max((PD - kd * (1 + DD) + SD - 1) / SD, 0);
            id = od * SD - PD + kd * (1 + DD);
            n_block++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(1))) // attr:no-format
    for (int k_blk = 0; k_blk < max_k_blocks; ++k_blk) {

        buf_num = ((k_blk_iter % NUM_BUF) + NUM_BUF - 1) % NUM_BUF;

        k_blk_iter++;

        COMPUTE(0);
#if IC_BLK_SUBGROUP == 2
        COMPUTE(1);
#endif
#if MB_BLK_WORKGROUP == 2
        src += src_mb_offset;
        diff_dst += dst_mb_offset;

        READ_DST_GLOBAL();
        WRITE_DST();
        READ_SRC_GLOBAL();
        WRITE_SRC();
        // Reduce on the same block(32o32i) with reduction on second block of 16n
        COMPUTE(0);
#if IC_BLK_SUBGROUP == 2
        COMPUTE(1);
#endif
#endif

        if (k_blk < max_k_blocks - (NUM_BUF - 1)) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    volatile __global atomic_float *diff_wei_write1;
    volatile __global atomic_float *diff_wei_write;

#define WRITE_WEI(i_o, i_i) \
    do { \
        diff_wei_write1 = (volatile __global atomic_float *)&diff_wei[WEI_OFF( \
                0, i_o * 8, i_i * IC_BLOCK + sg_loc_id, 0, 0, 0)]; \
        atomic_add_global(&diff_wei_write1[0], ACC[i_o][2 * i_i].s0); \
        \ 
        atomic_add_global(&diff_wei_write1[16], ACC[i_o][2 * i_i].s1); \
        \ 
        atomic_add_global(&diff_wei_write1[32], ACC[i_o][2 * i_i].s2); \
        \ 
        atomic_add_global(&diff_wei_write1[48], ACC[i_o][2 * i_i].s3); \
        \ 
        atomic_add_global(&diff_wei_write1[64], ACC[i_o][2 * i_i].s4); \
        \ 
        atomic_add_global(&diff_wei_write1[80], ACC[i_o][2 * i_i].s5); \
        \ 
        atomic_add_global(&diff_wei_write1[96], ACC[i_o][2 * i_i].s6); \
        \ 
        atomic_add_global(&diff_wei_write1[112], ACC[i_o][2 * i_i].s7); \
        diff_wei_write = (volatile __global atomic_float *)&diff_wei[WEI_OFF( \
                0, (i_o + 1) * 8, i_i * IC_BLOCK + sg_loc_id, 0, 0, 0)]; \
        atomic_add_global(&diff_wei_write[0], ACC[i_o][2 * i_i + 1].s0); \
        atomic_add_global(&diff_wei_write[16], ACC[i_o][2 * i_i + 1].s1); \
        \ 
        atomic_add_global(&diff_wei_write[32], ACC[i_o][2 * i_i + 1].s2); \
        \ 
        atomic_add_global(&diff_wei_write[48], ACC[i_o][2 * i_i + 1].s3); \
        \ 
        atomic_add_global(&diff_wei_write[64], ACC[i_o][2 * i_i + 1].s4); \
        \ 
        atomic_add_global(&diff_wei_write[80], ACC[i_o][2 * i_i + 1].s5); \
        \ 
        atomic_add_global(&diff_wei_write[96], ACC[i_o][2 * i_i + 1].s6); \
        \ 
        atomic_add_global(&diff_wei_write[112], ACC[i_o][2 * i_i + 1].s7); \
    } while (0)

    if (max_k_blocks > 0) {
        WRITE_WEI(0, 0);
#if OC_BLK_SUBGROUP == 2
        WRITE_WEI(2, 0);
#endif
#if IC_BLK_SUBGROUP == 2
        WRITE_WEI(0, 1);
#if OC_BLK_SUBGROUP == 2
        WRITE_WEI(2, 1);
#endif
#endif
    }

#if WITH_BIAS
#if OC_BLK_SUBGROUP == 2
#define COMPUTE_BIAS(nblk) \
    do { \
        dst_off = n * DST_MB_STRIDE + od * DST_D_STRIDE + oh * DST_H_STRIDE \
                + ow * DST_W_STRIDE + nblk * MB_BLOCK * OC_BLOCK; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h( \
                (__global ushort *)&diff_dst[dst_off]); \
        Dt[1] = __builtin_IB_simd_block_read_16_global_h( \
                (__global ushort *)&diff_dst[dst_off + DST_W_STRIDE]); \
        BIAS_ACC[0] += (CONVERT_TO_F32(Dt[0].s0) + CONVERT_TO_F32(Dt[0].s1) \
                + CONVERT_TO_F32(Dt[0].s2) + CONVERT_TO_F32(Dt[0].s3) \
                + CONVERT_TO_F32(Dt[0].s4) + CONVERT_TO_F32(Dt[0].s5) \
                + CONVERT_TO_F32(Dt[0].s6) + CONVERT_TO_F32(Dt[0].s7) \
                + CONVERT_TO_F32(Dt[0].s8) + CONVERT_TO_F32(Dt[0].s9) \
                + CONVERT_TO_F32(Dt[0].sa) + CONVERT_TO_F32(Dt[0].sb) \
                + CONVERT_TO_F32(Dt[0].sc) + CONVERT_TO_F32(Dt[0].sd) \
                + CONVERT_TO_F32(Dt[0].se) + CONVERT_TO_F32(Dt[0].sf)); \
        BIAS_ACC[1] += (CONVERT_TO_F32(Dt[1].s0) + CONVERT_TO_F32(Dt[1].s1) \
                + CONVERT_TO_F32(Dt[1].s2) + CONVERT_TO_F32(Dt[1].s3) \
                + CONVERT_TO_F32(Dt[1].s4) + CONVERT_TO_F32(Dt[1].s5) \
                + CONVERT_TO_F32(Dt[1].s6) + CONVERT_TO_F32(Dt[1].s7) \
                + CONVERT_TO_F32(Dt[1].s8) + CONVERT_TO_F32(Dt[1].s9) \
                + CONVERT_TO_F32(Dt[1].sa) + CONVERT_TO_F32(Dt[1].sb) \
                + CONVERT_TO_F32(Dt[1].sc) + CONVERT_TO_F32(Dt[1].sd) \
                + CONVERT_TO_F32(Dt[1].se) + CONVERT_TO_F32(Dt[1].sf)); \
    } while (0)

#else
#define COMPUTE_BIAS(nblk) \
    do { \
        dst_off = n * DST_MB_STRIDE + od * DST_D_STRIDE + oh * DST_H_STRIDE \
                + ow * DST_W_STRIDE + nblk * MB_BLOCK * OC_BLOCK; \
        Dt[0] = __builtin_IB_simd_block_read_16_global_h( \
                (__global ushort *)&diff_dst[dst_off]); \
        BIAS_ACC[0] += (CONVERT_TO_F32(Dt[0].s0) + CONVERT_TO_F32(Dt[0].s1) \
                + CONVERT_TO_F32(Dt[0].s2) + CONVERT_TO_F32(Dt[0].s3) \
                + CONVERT_TO_F32(Dt[0].s4) + CONVERT_TO_F32(Dt[0].s5) \
                + CONVERT_TO_F32(Dt[0].s6) + CONVERT_TO_F32(Dt[0].s7) \
                + CONVERT_TO_F32(Dt[0].s8) + CONVERT_TO_F32(Dt[0].s9) \
                + CONVERT_TO_F32(Dt[0].sa) + CONVERT_TO_F32(Dt[0].sb) \
                + CONVERT_TO_F32(Dt[0].sc) + CONVERT_TO_F32(Dt[0].sd) \
                + CONVERT_TO_F32(Dt[0].se) + CONVERT_TO_F32(Dt[0].sf)); \
    } while (0)

#endif
    // handle padded region for bias computation
    // first thread in spatial gws dimension, handles the left padding
    if (compute_bias) {
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = 0; od < od_start; ++od) {
                for (oh = 0; oh < OH; ++oh) {
                    for (ow = 0; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_start; od < OD; ++od) {
                for (oh = 0; oh < oh_start; ++oh) {
                    for (ow = 0; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_start; od < OD; ++od) {
                for (oh = oh_start; oh < OH; ++oh) {
                    for (ow = 0; ow < ow_start; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
    }

    // last thread handles the right padding
    if (compute_bias && gid[1] % K_WORKGROUPS == K_WORKGROUPS - 1) {
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_start; od < OD; ++od) {
                for (oh = oh_end + 1; oh < OH; ++oh) {
                    for (ow = ow_start; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_end + 1; od < OD; ++od) {
                for (oh = oh_start; oh < oh_end + 1; ++oh) {
                    for (ow = ow_start; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
        for (int n = 0; n < mb_blk_rnd_up; ++n) {
            for (od = od_start; od < od_end + 1; ++od) {
                for (oh = oh_start; oh < oh_end + 1; ++oh) {
                    for (ow = ow_end + 1; ow < OW; ++ow) {
                        COMPUTE_BIAS(0);
#if MB_BLK_WORKGROUP == 1 && MB > 16
                        COMPUTE_BIAS(1);
#endif
                    }
                }
            }
        }
    }
    if (compute_bias) {
        atomic_add_global(&dbias[0], BIAS_ACC[0].s0);
        atomic_add_global(&dbias[1], BIAS_ACC[0].s1);
        atomic_add_global(&dbias[2], BIAS_ACC[0].s2);
        atomic_add_global(&dbias[3], BIAS_ACC[0].s3);
        atomic_add_global(&dbias[4], BIAS_ACC[0].s4);
        atomic_add_global(&dbias[5], BIAS_ACC[0].s5);
        atomic_add_global(&dbias[6], BIAS_ACC[0].s6);
        atomic_add_global(&dbias[7], BIAS_ACC[0].s7);
        atomic_add_global(&dbias[8], BIAS_ACC[0].s8);
        atomic_add_global(&dbias[9], BIAS_ACC[0].s9);
        atomic_add_global(&dbias[10], BIAS_ACC[0].sa);
        atomic_add_global(&dbias[11], BIAS_ACC[0].sb);
        atomic_add_global(&dbias[12], BIAS_ACC[0].sc);
        atomic_add_global(&dbias[13], BIAS_ACC[0].sd);
        atomic_add_global(&dbias[14], BIAS_ACC[0].se);
        atomic_add_global(&dbias[15], BIAS_ACC[0].sf);
#if OC_BLK_SUBGROUP == 2
        atomic_add_global(&dbias[16], BIAS_ACC[1].s0);
        atomic_add_global(&dbias[17], BIAS_ACC[1].s1);
        atomic_add_global(&dbias[18], BIAS_ACC[1].s2);
        atomic_add_global(&dbias[19], BIAS_ACC[1].s3);
        atomic_add_global(&dbias[20], BIAS_ACC[1].s4);
        atomic_add_global(&dbias[21], BIAS_ACC[1].s5);
        atomic_add_global(&dbias[22], BIAS_ACC[1].s6);
        atomic_add_global(&dbias[23], BIAS_ACC[1].s7);
        atomic_add_global(&dbias[24], BIAS_ACC[1].s8);
        atomic_add_global(&dbias[25], BIAS_ACC[1].s9);
        atomic_add_global(&dbias[26], BIAS_ACC[1].sa);
        atomic_add_global(&dbias[27], BIAS_ACC[1].sb);
        atomic_add_global(&dbias[28], BIAS_ACC[1].sc);
        atomic_add_global(&dbias[29], BIAS_ACC[1].sd);
        atomic_add_global(&dbias[30], BIAS_ACC[1].se);
        atomic_add_global(&dbias[31], BIAS_ACC[1].sf);
#endif
    }
#endif
}
