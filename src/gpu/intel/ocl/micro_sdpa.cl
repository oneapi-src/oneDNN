/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/ocl/ocl_types.h"
#include "gpu/intel/ocl/sdpa_utils.h"
#include "gpu/intel/ocl/tile_ops.h"

/* Microkernel headers -- generated at runtime */
#include "gemm_kq.h"
#include "gemm_vs.h"

DECLARE_2D_TILE(ugemm_vs_c_type_half, half, SUBGROUP_SIZE,
        ugemm_vs_c_type_block0, ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
micro_sdpa(const global half *K, const global half *Q, const global half *V,
        global half *A, global SCALE_DATA_T *scale_ptr, global void *attn_mask,
        int d, int k, int q, local uint *slm) {
    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);
    uint b0 = get_group_id(1);
    uint b1 = get_group_id(2);

    uint ldk = KEY_S3;
    uint ldq = QRY_S2;
    uint ldv = VAL_S2;
    uint lda = DST_S2;

    float scale = convert_float(*scale_ptr);

    /* Locate K/Q/V/A matrices within batch */
    K += KEY_OFF(b1, b0, 0, 0);
    Q += QRY_OFF(b1, b0, 0, 0);
    V += VAL_OFF(b1, b0, 0, 0);
    A += DST_OFF(b1, b0, 0, 0, 0);

    /* Calculate S = (K^T) * Q */
    uint wg_i0 = 0;
    uint wg_j0 = get_group_id(0) * ugemm_kq_wg_tile_n;
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    ugemm_kq_c_type S_tile = ugemm_kq(
            K, ldk, Q, ldq, k, q, d, wg_i0, wg_j0, 0, sg_i_kq, sg_j_kq);

    if (ugemm_kq_slm_size > 0) barrier(CLK_LOCAL_MEM_FENCE);

/* Store tile to SLM */
#define chunk_m (4 * SUBGROUP_SIZE)
    uint slm_stride = max(chunk_m, ugemm_kq_wg_tile_m);
    uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;
    tile_store_full(S_tile, (local float *)slm, slm_stride, sg_i0_kq, sg_j0_kq);
    barrier(CLK_LOCAL_MEM_FENCE);

/* Read back full column(s) */
#define sg_per_wg (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#define cols_per_sg ((ugemm_kq_wg_tile_n + sg_per_wg - 1) / sg_per_wg)
#define chunks_per_col ((ugemm_kq_wg_tile_m + chunk_m - 1) / chunk_m)

    float4 sdata[cols_per_sg][chunks_per_col];
#pragma unroll
    for (int jj = 0; jj < cols_per_sg; jj++) {
        int j = jj * sg_per_wg + sg_ij;
#pragma unroll
        for (int ii = 0; ii < chunks_per_col; ii++) {
            sdata[jj][ii] = as_float4(intel_sub_group_block_read4(
                    slm + ii * chunk_m + j * slm_stride));
        }
    }

    /* Scale and apply softmax to each column. */
    /* 1) Scale + exp                          */
#if INVERT_SCALE
    scale = native_recip(scale);
#endif
    scale *= 1.442695f; // log2(e)
#pragma unroll
    for (int jj = 0; jj < cols_per_sg; jj++) {
#pragma unroll
        for (int ii = 0; ii < chunks_per_col; ii++)
            sdata[jj][ii] = native_exp2(scale * sdata[jj][ii]);
    }

    /* 2) Mask out-of-bounds elements */
    int mask0 = k - get_sub_group_local_id();
    int4 mask = {mask0, mask0 - SUBGROUP_SIZE, mask0 - 2 * SUBGROUP_SIZE,
            mask0 - 3 * SUBGROUP_SIZE};

#pragma unroll
    for (int ii = 0; ii < chunks_per_col; ii++) {
#pragma unroll
        for (int jj = 0; jj < cols_per_sg; jj++)
            sdata[jj][ii] = select(sdata[jj][ii], 0, mask);
        mask -= chunk_m;
    }

    /* 3) Sum columns */
    float ssums[cols_per_sg];

#pragma unroll
    for (int jj = 0; jj < cols_per_sg; jj++)
        ssums[jj] = 0.0f;

#pragma unroll
    for (int ii = 0; ii < chunks_per_col; ii++) {
#pragma unroll
        for (int jj = 0; jj < cols_per_sg; jj++) {
            ssums[jj] += sdata[jj][ii].s0;
            ssums[jj] += sdata[jj][ii].s1;
            ssums[jj] += sdata[jj][ii].s2;
            ssums[jj] += sdata[jj][ii].s3;
        }
    }

#pragma unroll
    for (int jj = 0; jj < cols_per_sg; jj++)
        ssums[jj] = sub_group_reduce_add(ssums[jj]);

/* 4) Normalize */
#pragma unroll
    for (int jj = 0; jj < cols_per_sg; jj++) {
        ssums[jj] = native_recip(sub_group_broadcast(ssums[jj], 0));
#pragma unroll
        for (int ii = 0; ii < chunks_per_col; ii++)
            sdata[jj][ii] *= ssums[jj];
    }

    /* Convert to half precision and write back full column(s) to SLM.
        Stride between columns is same as original f32 data. */
    uint slm_stride_half = slm_stride * 2;

#pragma unroll
    for (int jj = 0; jj < cols_per_sg; jj++) {
        int j = jj * sg_per_wg + sg_ij;
#pragma unroll
        for (int ii = 0; ii < chunks_per_col; ii++) {
            half4 sdata_half = convert_half4(sdata[jj][ii]);
            intel_sub_group_block_write_us4(
                    (local ushort *)slm + ii * chunk_m + j * slm_stride_half,
                    as_ushort4(sdata_half));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Calculate A = V*S */
    uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
    uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;
    ugemm_vs_c_type A_tile = ugemm_vs(V, ldv, (local half *)slm,
            slm_stride_half, d, q, k, 0, 0, 0, sg_i_vs, sg_j_vs);

    /* Convert to half precision and store */
    ugemm_vs_c_type_half A_tile_half;
    tile_copy(A_tile, A_tile_half);

    uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
    uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n + wg_j0;

    tile_store(A_tile_half, A, lda, d, q, sg_i0_vs, sg_j0_vs);
}
