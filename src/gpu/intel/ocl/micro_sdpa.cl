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

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

#define sg_per_wg (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#define q_tile_sg_n DIV_UP(ugemm_kq_wg_tile_n, sg_per_wg)

/* Instantiate tile types and operations */
typedef ugemm_kq_c_type s_tile_type;
typedef ugemm_vs_c_type a_tile_type;

DECLARE_2D_TILE(q_tile_type, uint, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)

#ifdef BLOCK_Q
DECLARE_2D_TILE_BLOCK_OPS(
        q_tile_type, uint, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)
#elif Q_ALIGN < 4
DECLARE_2D_TILE_LOAD_PACKED_HALF(
        q_tile_type, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)
#endif

#ifdef BLOCK_A
DECLARE_2D_TILE(a_tile_type_half, half, SUBGROUP_SIZE, ugemm_vs_sg_tile_m, 1, 1,
        ugemm_vs_sg_tile_n)
#else
DECLARE_2D_TILE(a_tile_type_half, half, SUBGROUP_SIZE, ugemm_vs_sg_tile_m, 8, 1,
        ugemm_vs_sg_tile_n / 8)
#endif

DECLARE_2D_TILE(s_tile_type_half2, uint, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1 / 2, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)

DECLARE_2D_TILE(
        s_sum_tile_type, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE(
        a_scale_tile_type, float, SUBGROUP_SIZE, ugemm_vs_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE(
        mask_tile_type, half, SUBGROUP_SIZE, ugemm_kq_sg_tile_m, 1, 1, 1)

DECLARE_2D_TILE(
        mask_tile_type_float, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_m, 1, 1, 1)

DECLARE_2D_TILE_BLOCK_OPS(
        mask_tile_type, half, SUBGROUP_SIZE, ugemm_kq_sg_tile_m, 1, 1, 1)

#ifdef BLOCK_A
DECLARE_2D_TILE_BLOCK_OPS(a_tile_type_half, half, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n)
#endif
#ifdef BLOCK_2D_A
DECLARE_2D_TILE_BLOCK2D_OPS(a_tile_type_half, half, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
#endif

#ifdef BLOCK_A
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_half, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n)
#else
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_half, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
#endif

DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, mask_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_scale_tile_type, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, 1, 1)

#if ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && (ugemm_kq_sg_tile_n % ugemm_vs_sg_tile_n) == 0
DECLARE_2D_TILE_RSELECT(a_scale_tile_type, SUBGROUP_SIZE, ugemm_vs_sg_tile_n, 1,
        1, 1, s_sum_tile_type, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
micro_sdpa(const global half *K, const global half *Q, const global half *V,
        global half *A, global SCALE_DATA_T *scale_ptr, const global half *msk,
        int d, int k, int q) {
    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);
    uint b0 = get_group_id(1);
    uint b1 = get_group_id(2);

    uint wg_j0 = get_group_id(0) * ugemm_kq_wg_tile_n;

    /* Leading dimension for matrices */
    uint ldk = KEY_S3;
    uint ldq = QRY_S2;
    uint ldv = VAL_S2;
    uint lda = DST_S2;

    /* Subgroup IDs for each GEMM */
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
    uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;

    /* SLM allocations */
    local half Q_slm[D_MAX * ugemm_kq_wg_tile_n];
    local half S_slm[ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n];
    local float S_sum_slm[ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m];
    local float S_max_slm[ugemm_kq_wg_tile_n];

#if ugemm_kq_slm_size + ugemm_vs_slm_size > 0
    local uint
            ugemm_slm[MAX(ugemm_kq_slm_size, ugemm_vs_slm_size) / sizeof(uint)];
#else
    local uint *ugemm_slm = NULL;
#endif

    const bool need_sum_barrier
            = (ugemm_kq_barrier_count == 0) && (ugemm_vs_barrier_count == 0);

    /* Locate K/Q/V/A matrices within batch */
    K += KEY_OFF(b1, b0, 0, 0);
    Q += QRY_OFF(b1, b0, 0, 0);
    V += VAL_OFF(b1, b0, 0, 0);
    A += DST_OFF(b1, b0, 0, 0, 0);

    __builtin_assume_aligned(K, K_ALIGN);
    __builtin_assume_aligned(Q, Q_ALIGN);
    __builtin_assume_aligned(V, V_ALIGN);
    __builtin_assume_aligned(A, A_ALIGN);

    /* Load Q tile, destined for SLM */
    q_tile_type Q_tile;
    uint q0_copy = q_tile_sg_n * sg_ij;
#ifdef BLOCK_Q
    tile_load_block(&Q_tile, (global uint *)Q, ldq >> 1, 0, wg_j0 + q0_copy);
#elif Q_ALIGN >= 4
    tile_load(&Q_tile, (global uint *)Q, (d + 1) >> 1, q, ldq >> 1, 0,
            wg_j0 + q0_copy);
#else
    tile_load_packed_half(&Q_tile, Q, d, q, ldq, 0, wg_j0 + q0_copy);
#endif

    /* Load scale */
#if INVERT_SCALE
    float iscale = convert_float(*scale_ptr);
    float scale = native_recip(iscale);
#else
    float scale = convert_float(*scale_ptr);
    float iscale = native_recip(scale);
#endif
    scale *= 1.442695f; // log2(e)

#ifdef PREFETCH_K0
    /* Prefetch first K tile. No remainder handling yet. */
    cooperative_prefetch_2d(K, D_MAX, ugemm_kq_wg_tile_m, ldk, sg_ij, sg_per_wg,
            SUBGROUP_SIZE, LSC_LDCC_L1C_L3C);
#endif

    /* Initialize S column sums in SLM to -inf */
    const uint n_col_sg = DIV_UP(ugemm_kq_wg_tile_n, SUBGROUP_SIZE * sg_per_wg);
    const float neg_inf = -INFINITY;

#pragma unroll
    for (int q = 0; q < n_col_sg; q++)
        intel_sub_group_block_write(
                (local uint *)&S_max_slm[q + sg_ij * n_col_sg * SUBGROUP_SIZE],
                as_uint(neg_inf));

    /* Clear accumulator */
    a_tile_type A_tile;
    tile_fill(A_tile, 0.0f);

    /* Store Q tile to SLM */
    tile_store_t_sys_src1(
            Q_tile, (local uint *)&Q_slm[0], D_MAX / 2, q0_copy, 0);

    /* Clear S column sums/maxes */
    s_sum_tile_type S_sum_tile;
    s_sum_tile_type S_max_tile, S_max_tile_old;
    tile_fill(S_sum_tile, 0.0f);
    tile_fill(S_max_tile, -INFINITY);

    /* Wait for Q data to reach SLM */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Main loop over k blocks */
    for (int k0 = 0; k0 < k; k0 += ugemm_kq_wg_tile_m) {
        bool first = (k0 == 0);
        bool last = (k0 + ugemm_kq_wg_tile_m >= k);

        uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
        uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;

#if WITH_ATTN_MASK
        /* Load mask. No remainder handling needed assuming k block size is a power of 2. */
        mask_tile_type mask_tile;
        tile_load_block(&mask_tile, msk, 0, k0 + sg_i0_kq, 0);
#endif

        /* Prepare k mask: 0 in bounds, -inf out of bounds */
        mask_tile_type_float k_mask;
#pragma unroll
        for (int ii = 0; ii < ugemm_kq_sg_tile_m / SUBGROUP_SIZE; ii++)
            k_mask.x[0][ii] = (k0 + sg_i0_kq + ii * SUBGROUP_SIZE
                                              + get_sub_group_local_id()
                                      < k)
                    ? 0
                    : -INFINITY;

        /* Calculate S = (K^T) * Q */
        s_tile_type S_tile = ugemm_kq(K, ldk, Q_slm, D_MAX, k,
                ugemm_kq_wg_tile_n, d, k0, 0, 0, sg_i_kq, sg_j_kq, ugemm_slm);

#if WITH_ATTN_MASK
/* Apply mask, manually masking in k dimension */
#define unscale_mask(x, y) ((x)*iscale + (y))
        mask_tile_type_float mask_tile_float;
        tile_copy(mask_tile, mask_tile_float);
        tile_binary(mask_tile_float, k_mask, unscale_mask);
        tile_hbroadcast_add(&S_tile, mask_tile_float);
#else
        tile_hbroadcast_add(&S_tile, k_mask);
#endif

        /* Before softmax, we will need to scale columns by maximum values to avoid overflow. */

        /* Compute our maxima and reduce across SLM */
        tile_vreduce_max(S_tile, &S_max_tile);
        tile_atomic_max_full(
                S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

#ifdef PREFETCH_V
        /* Prefetch V tile. No remainder handling yet. */
        cooperative_prefetch_2d(V, D_MAX, ugemm_kq_wg_tile_m, ldv, sg_ij,
                sg_per_wg, SUBGROUP_SIZE, LSC_LDCC_L1C_L3C);
#endif

#ifndef ALT_MAX
        /* Read back WG-wide maxima */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        tile_load_full(&S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);
#endif

        tile_vbroadcast_sub(&S_tile, S_max_tile);

/* Scale + exponentiate */
#define scaled_exp(x) native_exp2(x *scale)
        tile_elementwise(S_tile, scaled_exp);

#ifdef ALT_MAX
        /* Read back WG-wide maxima and adjust S to match */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        s_sum_tile_type S_max_tile1;
        tile_copy(S_max_tile, S_max_tile1);
        tile_load_full(&S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);

#define binary_exp_neg(x, y) native_exp2(scale *((x) - (y)))
        tile_binary(S_max_tile1, S_max_tile, binary_exp_neg);
        tile_vbroadcast_mul(&S_tile, S_max_tile1);
#endif

        /* Accumulate sums. S tile is transposed for easy summation. */
        s_sum_tile_type S_sum_tile1;
        tile_fill(S_sum_tile1, 0.0f);
        tile_vreduce_add(S_tile, &S_sum_tile1);

        /* Convert to half, VNNI format */
        s_tile_type_half2 S_tile_half2;
        tile_copy_to_half2(S_tile, S_tile_half2);

        /* Store to SLM, in packed format */
        tile_store_t_sys_src2(S_tile_half2, (local uint *)S_slm,
                ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2, sg_i0_kq / 2,
                sg_j0_kq);
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        /* Rescale existing accumulator and sums to match new maxima */
        if (!first) {
#define binary_exp_sub(x, y) native_exp2(scale *((x) - (y)))
#define binary_mul(x, y) ((x) * (y))
            tile_binary(S_max_tile_old, S_max_tile, binary_exp_sub);
            tile_binary(S_sum_tile, S_max_tile_old, binary_mul);

            /* Find the subset of sums that applies to the accumulation tile */
            a_scale_tile_type A_scale_tile;
#if ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && ugemm_kq_sg_tile_n == ugemm_vs_sg_tile_n
            tile_copy(S_max_tile_old, A_scale_tile);
#elif ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && (ugemm_kq_sg_tile_n % ugemm_vs_sg_tile_n) == 0
            tile_rselect(&A_scale_tile, S_max_tile_old,
                    sg_j_vs % (ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n));
#else
#error unimplemented
#endif
            tile_hbroadcast_mul(&A_tile, A_scale_tile);
        }

/* Accumulate sums */
#define binary_add(x, y) ((x) + (y))
        tile_binary(S_sum_tile, S_sum_tile1, binary_add);

        /* Save maxima */
        tile_copy(S_max_tile, S_max_tile_old);

        /* Last iteration: store column sums in SLM */
        if (last)
            tile_store_full(S_sum_tile, S_sum_slm, ugemm_kq_wg_tile_n, sg_j0_kq,
                    sg_i_kq);

#ifdef PREFETCH_K
        /* Prefetch next K tile. No remainder handling yet. */
        if (!last)
            cooperative_prefetch_2d(K + (k0 + ugemm_kq_wg_tile_m) * ldk, D_MAX,
                    ugemm_kq_wg_tile_m, ldk, sg_ij, sg_per_wg, SUBGROUP_SIZE,
                    LSC_LDCC_L1C_L3C);
#endif
#ifdef PREFETCH_MASK
#if WITH_ATTN_MASK
        /* Prefetch next mask tile. */
        if (!last)
            cooperative_prefetch_2d(msk + k0 + ugemm_kq_wg_tile_m + sg_i0_kq,
                    ugemm_kq_sg_tile_m, 1, 0, 0, 1, SUBGROUP_SIZE,
                    LSC_LDCC_L1UC_L3C);
#endif
#endif

        /* Wait for S stores */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        /* Last iteration: signal column sums are ready */
        if (last && need_sum_barrier)
            intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        /* Accumulate A += V * S */
        int k_chunk = min(k - k0, ugemm_kq_wg_tile_m);
        a_tile_type A_tile1 = ugemm_vs(V, ldv, S_slm, ugemm_kq_wg_tile_m, d,
                ugemm_kq_wg_tile_n, k_chunk, 0, 0, 0, sg_i_vs, sg_j_vs,
                ugemm_slm);
        V += ldv * ugemm_kq_wg_tile_m;

        tile_binary(A_tile, A_tile1, binary_add);
    }

    /* Wait for column sums to be ready */
    if (need_sum_barrier) intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

    /* Load column sums from SLM + reduce in registers */
    a_scale_tile_type A_scale_tile, A_scale_tile_load;
    tile_fill(A_scale_tile, 0.0f);

#pragma unroll
    for (uint sg1 = 0; sg1 < ugemm_kq_sg_per_wg_m; sg1++) {
        tile_load_full(&A_scale_tile_load, S_sum_slm, ugemm_kq_wg_tile_n,
                ugemm_vs_sg_tile_n * sg_j_vs, sg1);
        tile_binary(A_scale_tile, A_scale_tile_load, binary_add);
    }

    /* Rescale by 1 / (column sums) */
    tile_elementwise_s(A_scale_tile, native_recip);
    tile_hbroadcast_mul(&A_tile, A_scale_tile);

    /* Convert to half precision and store */
    a_tile_type_half A_tile_half;
    tile_copy_reblock(A_tile, &A_tile_half);

    uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
    uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n + wg_j0;

#ifdef BLOCK_2D_A
    tile_store_block2d(A_tile_half, A, d, q, lda, sg_i0_vs, sg_j0_vs);
#elif defined(BLOCK_A)
    tile_store_block(A_tile_half, A, lda, sg_i0_vs, sg_j0_vs);
#else
    tile_store(A_tile_half, A, d, q, lda, sg_i0_vs, sg_j0_vs);
#endif
}
