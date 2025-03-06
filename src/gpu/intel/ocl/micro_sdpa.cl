/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/ocl/sdpa_utils.h"
#include "gpu/intel/ocl/tile_ops.h"

/* Microkernel headers -- generated at runtime */
#include "gemm_kq.h"
#include "gemm_vs.h"

/* The quantization parameter may be unique for each token/element */
#define QUANTIZE_2D 2

/* The quantization parameter shares the same value across the work-group */
#define QUANTIZE_COMMON 3

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

#define sg_per_wg (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#define q_tile_sg_n DIV_UP(ugemm_kq_wg_tile_n, sg_per_wg)

/* Instantiate tile types and operations */
typedef ugemm_kq_c_type s_tile_type;
typedef ugemm_vs_c_type a_tile_type;

#ifdef QRY_DT_F16
#define VEC_TYPE2 half2
#elif defined(QRY_DT_BF16)
#define VEC_TYPE2 ushort2
#else
#error "Data type not supported for VEC_TYPE2"
#endif

#ifdef SCALE_DT_BF16
#define SCALES_TO_FLOAT cvt_bf16_to_f32
#else
#define SCALES_TO_FLOAT convert_float
#endif

#ifdef VAL_ATTR_SCALES_DT_BF16
#define VAL_SCALES_TO_FLOAT cvt_bf16_to_f32
#else
#define VAL_SCALES_TO_FLOAT convert_float
#endif

#if KEY_ATTR_SCALES_DT_BF16
#define KEY_SCALES_TO_FLOAT cvt_bf16_to_f32
#else
#define KEY_SCALES_TO_FLOAT convert_float
#endif

DECLARE_2D_TILE(q_tile_type, uint, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)

#ifdef BLOCK_Q
DECLARE_2D_TILE_BLOCK_OPS(
        q_tile_type, uint, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)
#elif Q_ALIGN < 4
DECLARE_2D_TILE_LOAD_PACKED_VEC(q_tile_type, QRY_DATA_T, VEC_TYPE2,
        SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)
#endif

#ifdef BLOCK_A
DECLARE_2D_TILE(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE, ugemm_vs_sg_tile_m,
        1, 1, ugemm_vs_sg_tile_n)
#else
DECLARE_2D_TILE(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE, ugemm_vs_sg_tile_m,
        8, 1, ugemm_vs_sg_tile_n / 8)
#endif

DECLARE_2D_TILE(s_tile_type_packed, uint, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1 / 2, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)

DECLARE_2D_TILE(
        s_sum_tile_type, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE(
        a_scale_tile_type, float, SUBGROUP_SIZE, ugemm_vs_sg_tile_n, 1, 1, 1)

#if BROADCAST_MASK_Q
#define mask_br ugemm_kq_sg_tile_m
#define mask_bc 1
#define mask_nbr 1
#define mask_nbc 1
#else
#define mask_br ugemm_kq_c_type_block0
#define mask_bc ugemm_kq_c_type_block1
#define mask_nbr ugemm_kq_c_type_nblock0
#define mask_nbc ugemm_kq_c_type_nblock1
#endif

DECLARE_2D_TILE(kmask_tile_type_float, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_m,
        1, 1, 1)

#if WITH_ATTN_MASK
DECLARE_2D_TILE(mask_tile_type, MSK_DATA_T, SUBGROUP_SIZE, mask_br, mask_bc,
        mask_nbr, mask_nbc)

#if BROADCAST_MASK_Q
DECLARE_2D_TILE_BLOCK_OPS(mask_tile_type, MSK_DATA_T, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc)
#endif
DECLARE_2D_TILE(mask_tile_type_float, float, SUBGROUP_SIZE, mask_br, mask_bc,
        mask_nbr, mask_nbc)
DECLARE_2D_TILE_COPY_REBLOCK(mask_tile_type, SUBGROUP_SIZE, mask_br, mask_bc,
        mask_nbr, mask_nbc, mask_tile_type_float, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc, CONVERT_FLOAT_T)
#endif

#ifdef BLOCK_A
DECLARE_2D_TILE_BLOCK_OPS(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n)
#endif
#ifdef BLOCK_2D_A
DECLARE_2D_TILE_BLOCK2D_OPS(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
#endif

#ifdef BLOCK_A
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_dst, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n, CONVERT_DATA_T)
#else
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_dst, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8, CONVERT_DATA_T)
#endif

DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, kmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, 1)
#if WITH_ATTN_MASK
DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, mask_tile_type_float, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc)
#endif

DECLARE_2D_TILE_HREDUCE(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_scale_tile_type, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, 1, 1)

#if ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && (ugemm_kq_sg_tile_n % ugemm_vs_sg_tile_n) == 0
DECLARE_2D_TILE_RSELECT(a_scale_tile_type, SUBGROUP_SIZE, ugemm_vs_sg_tile_n, 1,
        1, 1, s_sum_tile_type, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)
#endif

#if PREFETCH_REMAINDER
#define cooperative_prefetch_2d_maybe_rem cooperative_prefetch_2d_rem
#else
#define cooperative_prefetch_2d_maybe_rem( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d(ptr, rmax, cmax, ld, sg_id, n_sg, sg_size, caching)
#endif

#if TRANSPOSE_K
#define cooperative_prefetch_2d_k( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d_maybe_rem( \
            ptr, c, r, cmax, rmax, ld, sg_id, n_sg, sg_size, caching)
#else
#define cooperative_prefetch_2d_k cooperative_prefetch_2d_maybe_rem
#endif

#if REMAINDER_Q
#define tile_load_block_rem_q tile_load_block
#define tile_store_block_rem_q tile_store_block
#else
#define tile_load_block_rem_q(t, ptr, n, ld, off_r, off_c) \
    tile_load_block(t, ptr, ld, off_r, off_c)
#define tile_store_block_rem_q(t, ptr, n, ld, off_r, off_c) \
    tile_store_block(t, ptr, ld, off_r, off_c)
#endif

#define binary_add(x, y) ((x) + (y))

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
micro_sdpa(const global KEY_DATA_T *K, const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V, global DST_DATA_T *A,
        const global SCALE_DATA_T *scale_ptr, int d, int k, int q,
        const global KEY_ATTR_SCALES_DATA_T *K_scales,
        const global KEY_ATTR_ZP_DATA_T *K_zp,
        const global VAL_ATTR_SCALES_DATA_T *V_scales,
        const global VAL_ATTR_ZP_DATA_T *V_zp
#if WITH_ATTN_MASK
        ,
        const global MSK_DATA_T *msk
#endif
#if WITH_PAGED_ATTN
        ,
        const global INDEX_DATA_T *prompt_lens,
        const global INDEX_DATA_T *subsequence_begins,
        const global INDEX_DATA_T *block_indices,
        const global INDEX_DATA_T *block_indices_begins, const int context_len
#endif
) {
        printf("hello world\n");    

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);
    uint b0 = get_group_id(1);
    uint b1 = get_group_id(2);
    uint b0_kv = b0 / KV_GROUP_SIZE;

    const global VAL_DATA_T *V_backup = V;

    const uint subsequence_no = b0;
    const uint start_query_no = subsequence_begins[subsequence_no];
    const uint stop_query_no = subsequence_begins[subsequence_no + 1];
    q = prompt_lens[subsequence_no];
    printf("subsequence_no = %d, start_query_no = %d, stop_query_no = %d, q = %d\n", subsequence_no, start_query_no, stop_query_no, q);    
    for (int i=0; i<2; i++) {
        printf("subsequence_begins[%d] = %d\n", i, subsequence_begins[i]);
    }
    const uint block_start_index = block_indices_begins[subsequence_no];
    const uint block_stop_index = block_indices_begins[subsequence_no + 1];
    for (int i=0; i<2; i++) {
        printf("block_indices_begins[%d] = %d\n", i, block_indices_begins[i]);
    }

    const uint num_blocks = block_stop_index - block_start_index;
    k = PAGE_SIZE * num_blocks;
    printf("num_blocks = %d, k = %d\n, page_size %d\n", num_blocks, k, PAGE_SIZE);
    uint wg_j0 = get_group_id(0) * ugemm_kq_wg_tile_n;

    /* Leading dimension for matrices */
    uint ldk = TRANSPOSE_K ? KEY_S3 : KEY_S2;
    uint ldq = QRY_S2;
    uint ldv = VAL_S2;
    uint lda = DST_S2;

#if KEY_SCALES || KEY_ZERO_POINTS
    uint ldkq = KEY_D3;
    uint num_key_groups = d / KEY_GROUP_SIZE;
#endif
#if VAL_SCALES || VAL_ZERO_POINTS
    uint ldvq = div_up(d, VAL_GROUP_SIZE);
    uint num_val_groups = d / VAL_GROUP_SIZE;
#endif

    /* Subgroup IDs for each GEMM */
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
    uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;

    /* SLM allocations -- place in one array to work around compiler bug */
#define Q_slm_size (D_MAX * ugemm_kq_wg_tile_n * sizeof(QRY_DATA_T))
#define S_slm_size \
    (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n * sizeof(QRY_DATA_T))
#define S_sum_slm_size \
    (ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m * sizeof(float))
#define S_max_slm_size (ugemm_kq_wg_tile_n * sizeof(float))
#define ugemm_slm_size MAX(ugemm_kq_slm_size, ugemm_vs_slm_size)
    printf("total_size = %d \n", Q_slm_size + S_slm_size + S_sum_slm_size + S_max_slm_size + ugemm_slm_size);
    local char slm[Q_slm_size + S_slm_size + S_sum_slm_size + S_max_slm_size
            + ugemm_slm_size];
    auto total_size = Q_slm_size + S_slm_size + S_sum_slm_size + S_max_slm_size
    + ugemm_slm_size;

    printf("total_size = %d, Q_slm_size = %d, S_slm_size = %d, S_sum_slm_size = %d, S_max_slm_size = %d, ugemm_slm_size = %d\n", total_size, Q_slm_size, S_slm_size, S_sum_slm_size, S_max_slm_size, ugemm_slm_size);
    local QRY_DATA_T *Q_slm = (local QRY_DATA_T *)&slm[0];
    local QRY_DATA_T *S_slm = (local QRY_DATA_T *)&slm[Q_slm_size];
    local float *S_sum_slm = (local float *)&slm[Q_slm_size + S_slm_size];
    local float *S_max_slm
            = (local float *)&slm[Q_slm_size + S_slm_size + S_sum_slm_size];
    local uint *ugemm_slm = (local uint *)&slm[Q_slm_size + S_slm_size
            + S_sum_slm_size + S_max_slm_size];

    const bool need_sum_barrier = (ugemm_vs_barrier_count == 0);

    /* Locate K/Q/V/A matrices within batch */
    K += KEY_OFF(0, 0, 0, 0) / KEY_ELEMENTS_PER_BYTE;
    Q += QRY_OFF(0, 0, start_query_no, 0);
    V += VAL_OFF(0, 0, 0, 0) / VAL_ELEMENTS_PER_BYTE;
    A += DST_OFF(0, 0, 0, start_query_no, 0);
#if WITH_ATTN_MASK
    uint ldmsk = MSK_S2;
    msk += MSK_OFF(b1 % MSK_D0, b0 % MSK_D1, 0, 0);
#ifndef BLOCK_MSK
    int mask_aligned = (((size_t)msk) % 4) == 0;
#endif
#endif

#if KEY_SCALES
    K_scales += KEY_OFF(b1, b0_kv, 0, 0) / KEY_GROUP_SIZE;
#endif
#if KEY_SCALES == QUANTIZE_COMMON
    float k_scale = KEY_SCALES_TO_FLOAT(*K_scales);
#endif
#if KEY_ZERO_POINTS
    K_zp += KEY_OFF(b1, b0_kv, 0, 0) / KEY_GROUP_SIZE
            / KEY_ZP_ELEMENTS_PER_BYTE;
#endif
#if VAL_SCALES
    V_scales += VAL_OFF(b1, b0_kv, 0, 0) / VAL_GROUP_SIZE;
#endif
#if VAL_SCALES == QUANTIZE_COMMON
    float v_scale = VAL_SCALES_TO_FLOAT(*V_scales);
#endif
#if VAL_ZERO_POINTS
    V_zp += VAL_OFF(b1, b0_kv, 0, 0) / VAL_GROUP_SIZE
            / VAL_ZP_ELEMENTS_PER_BYTE;
#endif

    /* Load Q tile, destined for SLM */
    q_tile_type Q_tile;
    uint q0_copy = q_tile_sg_n * sg_ij;
#ifdef BLOCK_Q
    tile_load_block_rem_q(
            &Q_tile, (global uint *)Q, q, ldq >> 1, 0, wg_j0 + q0_copy);
#elif Q_ALIGN >= 4
    tile_load(&Q_tile, (global uint *)Q, (d + 1) >> 1, q, ldq >> 1, 0,
            wg_j0 + q0_copy);
#else
    tile_load_packed_vec2(&Q_tile, Q, d, q, ldq, 0, wg_j0 + q0_copy);
#endif

    /* Load scale */
#if WITH_ATTN_SCALE
#if INVERT_SCALE
    float iscale = SCALES_TO_FLOAT(*scale_ptr);
    float scale = native_recip(iscale);
#else
    float scale = SCALES_TO_FLOAT(*scale_ptr);
    float iscale = native_recip(scale);
#endif
#else
    float scale = 1.0;
    float iscale = 1.0;
#endif
    scale *= 1.442695f; // log2(e)

#ifdef PREFETCH_K0
    /* Prefetch first K tile. */
    cooperative_prefetch_2d_k(
            /* ptr */ K,
            /* r */ k,
            /* c */ d,
            /* rmax */ ugemm_kq_wg_tile_m,
            /* cmax */ PREFETCH_D_MAX,
            /* ld */ ldk,
            /* sg_id */ sg_ij,
            /* n_sg */ sg_per_wg,
            /* sg_size */ SUBGROUP_SIZE,
            /* cache */ LSC_LDCC_L1C_L3C);

#if KEY_SCALES == QUANTIZE_2D
    cooperative_prefetch_2d_maybe_rem(
            /* ptr */ K_scales,
            /* r */ k,
            /* c */ num_key_groups,
            /* rmax */ ugemm_kq_wg_tile_m,
            /* cmax */ D_MAX / KEY_GROUP_SIZE,
            /* ld */ ldkq,
            /* sg_id */ sg_ij,
            /* n_sg */ sg_per_wg,
            /* sg_size */ SUBGROUP_SIZE,
            /* cache */ LSC_LDCC_L1C_L3C);
#endif
#if KEY_ZERO_POINTS == QUANTIZE_2D
    cooperative_prefetch_2d_maybe_rem(
            /* ptr */ K_zp,
            /* r */ k,
            /* c */ num_key_groups,
            /* rmax */ ugemm_kq_wg_tile_m,
            /* cmax */ D_MAX / KEY_GROUP_SIZE,
            /* ld */ ldkq,
            /* sg_id */ sg_ij,
            /* n_sg */ sg_per_wg,
            /* sg_size */ SUBGROUP_SIZE,
            /* cache */ LSC_LDCC_L1C_L3C);
#endif
#endif

    /* Initialize S column sums in SLM to -inf */
    const uint n_col_sg = DIV_UP(ugemm_kq_wg_tile_n, SUBGROUP_SIZE * sg_per_wg);
    const float neg_inf = -INFINITY;

#pragma unroll
    for (int q = 0; q < n_col_sg; q++)
        intel_sub_group_block_write(
                (local uint *)&S_max_slm[(q + sg_ij * n_col_sg)
                        * SUBGROUP_SIZE],
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

    const int tiles_per_page = DIV_UP(PAGE_SIZE, ugemm_kq_wg_tile_m);
    const int num_multiplies = num_blocks * tiles_per_page;
    const int partial_num_rows
            = PAGE_SIZE - (PAGE_SIZE / ugemm_kq_wg_tile_m) * ugemm_kq_wg_tile_m;

    /* Main loop over k blocks */
    int k0 = 0;
    for (int multiply_no = 0; multiply_no < num_multiplies; multiply_no++) {

        const int block_index_no = multiply_no / tiles_per_page;
        const int block_no = block_indices[block_index_no];

        const int key_offset = KEY_OFF(block_no, 0, 0, 0);
        const int value_offset = VAL_OFF(block_no, 0, 0, 0);

        const bool first = (multiply_no == 0);
        const bool last = (multiply_no == num_multiplies - 1);

        uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
        uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;

#if WITH_ATTN_MASK
        /* Load mask. No remainder handling needed assuming k block size is a power of 2. */
        mask_tile_type mask_tile;
#if BROADCAST_MASK_Q
#if BLOCK_MSK
        tile_load_block(&mask_tile, msk, 0, k0 + sg_i0_kq, 0);
#else
        if (mask_aligned) {
            tile_load_block(&mask_tile, msk, 0, k0 + sg_i0_kq, 0);
        } else {
            tile_load_full(&mask_tile, msk, 0, k0 + sg_i0_kq, 0);
        }
#endif
#else
        tile_load_t(&mask_tile, msk, q, k, sg_j0_kq + wg_j0, k0 + sg_i0_kq);
#endif
#endif

#if REMAINDER_K
        /* Prepare k mask: NaN in bounds, -inf out of bounds */
        kmask_tile_type_float k_mask;
#pragma unroll
        for (int ii = 0; ii < ugemm_kq_sg_tile_m / SUBGROUP_SIZE; ii++) {
            k_mask.x[0][ii]
                    = (sg_i0_kq + ii * SUBGROUP_SIZE + get_sub_group_local_id()
                              < PAGE_SIZE)
                    ? nan(0u)
                    : -INFINITY;
        }
#endif

        /* Calculate S = (K^T) * Q */
        s_tile_type S_tile = ugemm_kq(K + key_offset, ldk, Q_slm, D_MAX,
                PAGE_SIZE, ugemm_kq_wg_tile_n, d, 0, 0, 0, sg_i_kq, sg_j_kq,
                (local char *)ugemm_slm
#if KEY_SCALES == QUANTIZE_2D
                ,
                K_scales
#endif
#if KEY_ZERO_POINTS
                ,
                K_zp
#endif
#if (KEY_SCALES == QUANTIZE_2D) || KEY_ZERO_POINTS
                ,
                ldkq
#endif
        );

#if KEY_SCALES == QUANTIZE_COMMON
#define k_scale_op(x) ((x)*k_scale)
        tile_elementwise(S_tile, k_scale_op);
#endif

        /* Apply attention mask */
#if WITH_ATTN_MASK
#define unscale(x) ((x)*iscale)
        mask_tile_type_float mask_tile_float;
        tile_copy_reblock(mask_tile, &mask_tile_float);
#if WITH_ATTN_SCALE
        tile_elementwise(mask_tile_float, unscale);
#endif
#if BROADCAST_MASK_Q
        tile_hbroadcast_add(&S_tile, mask_tile_float);
#else
        tile_binary(S_tile, mask_tile_float, binary_add);
#endif
#endif

        /* Apply k mask */
#if REMAINDER_K
        tile_hbroadcast_min(&S_tile, k_mask);
#endif

#if WITH_CAUSAL_MASK
#define greater_than(offset_k, offset_q) (offset_k > offset_q)
        /* Apply causal mask */
        tile_predicated_assignment_t(S_tile, k0 + sg_i0_kq, wg_j0 + sg_j0_kq,
                greater_than, -INFINITY, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
                ugemm_kq_c_type_nblock1);
#endif

        /* Before softmax, we will need to scale columns by maximum values to avoid overflow. */

        /* Compute our maxima and reduce across SLM */
        tile_vreduce_max(S_tile, &S_max_tile);
        tile_atomic_max_full(
                S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        int k_chunk = min(k - k0, ugemm_kq_wg_tile_m);
#ifdef PREFETCH_V
        /* Prefetch V tile. */
        cooperative_prefetch_2d_maybe_rem(
                /* ptr */ V + value_offset,
                /* r */ d,
                /* c */ k - k0,
                /* rmax */ PREFETCH_D_MAX,
                /* cmax */ ugemm_kq_wg_tile_m,
                /* ld */ ldv,
                /* sg_id */ sg_ij,
                /* n_sg */ sg_per_wg,
                /* sg_size */ SUBGROUP_SIZE,
                /* cache */ LSC_LDCC_L1C_L3C);

#if VAL_SCALES == QUANTIZE_2D
        /* Prefetch V scales. */
        cooperative_prefetch_2d_maybe_rem(
                /* ptr */ V_scales,
                /* r */ num_val_groups,
                /* c */ k - k0,
                /* rmax */ PREFETCH_D_MAX / VAL_GROUP_SIZE,
                /* cmax */ k_chunk,
                /* ld */ ldvq,
                /* sg_id */ sg_ij,
                /* n_sg */ sg_per_wg,
                /* sg_size */ SUBGROUP_SIZE,
                /* cache */ LSC_LDCC_L1C_L3C);
#endif
#if VAL_ZERO_POINTS == QUANTIZE_2D
        /* Prefetch V zero points. */
        cooperative_prefetch_2d_maybe_rem(
                /* ptr */ V_zp,
                /* r */ num_val_groups,
                /* c */ k - k0,
                /* rmax */ PREFETCH_D_MAX / VAL_GROUP_SIZE,
                /* cmax */ k_chunk,
                /* ld */ ldvq,
                /* sg_id */ sg_ij,
                /* n_sg */ sg_per_wg,
                /* sg_size */ SUBGROUP_SIZE,
                /* cache */ LSC_LDCC_L1C_L3C);
#endif
#endif
#ifndef ALT_MAX
        /* Read back WG-wide maxima */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        tile_load_full(&S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);
#endif

        tile_vbroadcast_sub(&S_tile, S_max_tile);

/* Scale + exponentiate */
#define scaled_exp(x) native_vexp2(x *scale)
        tile_elementwise(S_tile, scaled_exp);

#ifdef ALT_MAX
        /* Read back WG-wide maxima and adjust S to match */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        s_sum_tile_type S_max_tile1;
        tile_copy(S_max_tile, S_max_tile1);
        tile_load_full(&S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);

#define binary_exp_neg(x, y) native_vexp2(scale *((x) - (y)))
        tile_binary(S_max_tile1, S_max_tile, binary_exp_neg);
        tile_vbroadcast_mul(&S_tile, S_max_tile1);
#endif

        /* Accumulate sums. S tile is transposed for easy summation. */
        s_sum_tile_type S_sum_tile1;
        tile_fill(S_sum_tile1, 0.0f);
        tile_vreduce_add(S_tile, &S_sum_tile1);

        /* Convert to half or bf16, VNNI format */
        s_tile_type_packed S_tile_packed;
        tile_copy_to_vec2(S_tile, S_tile_packed, VEC_TYPE2);

        /* Store to SLM, in packed format */
        tile_store_t_sys_src2(S_tile_packed, (local uint *)S_slm,
                ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2, sg_i0_kq / 2,
                sg_j0_kq);
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        /* Rescale existing accumulator and sums to match new maxima */
        if (!first) {
#define binary_exp_sub(x, y) native_vexp2(scale *((x) - (y)))
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
        tile_binary(S_sum_tile, S_sum_tile1, binary_add);

        /* Save maxima */
        tile_copy(S_max_tile, S_max_tile_old);

        /* Last iteration: store column sums in SLM */
        if (last) {
            tile_store_full(S_sum_tile, S_sum_slm, ugemm_kq_wg_tile_n, sg_j0_kq,
                    sg_i_kq);
        }

#ifdef PREFETCH_K
        /* Prefetch next K tile. */
        if (!last) {
#if TRANSPOSE_K
            const uint stride_k = ldk;
#else
            const uint stride_k = 1;
#endif

            cooperative_prefetch_2d_k(
                    /* ptr */ K + (k0 + ugemm_kq_wg_tile_m) * stride_k,
                    /* r */ k - k0 - ugemm_kq_wg_tile_m,
                    /* c */ d,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ D_MAX,
                    /* ld*/ ldk,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache*/ LSC_LDCC_L1C_L3C);
#if KEY_SCALES == QUANTIZE_2D
            cooperative_prefetch_2d_maybe_rem(
                    /* ptr */ K_scales + (k0 + ugemm_kq_wg_tile_m),
                    /* r */ k - k0 - ugemm_kq_wg_tile_m,
                    /* c */ num_key_groups,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ D_MAX / KEY_GROUP_SIZE,
                    /* ld */ ldkq,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache */ LSC_LDCC_L1C_L3C);
#endif
#if KEY_ZERO_POINTS == QUANTIZE_2D
            cooperative_prefetch_2d_maybe_rem(
                    /* ptr */ K_zp + (k0 + ugemm_kq_wg_tile_m),
                    /* r */ k - k0 - ugemm_kq_wg_tile_m,
                    /* c */ num_key_groups,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ D_MAX / KEY_GROUP_SIZE,
                    /* ld */ ldkq,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache */ LSC_LDCC_L1C_L3C);
#endif
        }
#endif

#if WITH_ATTN_MASK && defined(PREFETCH_MASK)
        /* Prefetch next mask tile. */
        if (!last) {
#if BROADCAST_MASK_Q
            cooperative_prefetch_2d_maybe_rem(
                    /* ptr */ msk + k0 + ugemm_kq_wg_tile_m,
                    /* r */ k - k0,
                    /* c */ 1,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ 1,
                    /* ld */ 0,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache */ LSC_LDCC_L1C_L3C);
#else
            cooperative_prefetch_2d_maybe_rem(
                    /* ptr */ msk + k0 + ugemm_kq_sg_tile_m + (wg_j0)*ldmsk,
                    /* r */ k - k0 - ugemm_kq_wg_tile_m,
                    /* c */ q - wg_j0,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ (ugemm_kq_wg_tile_n * PREFETCH_D_MAX) / D_MAX,
                    /* ld */ ldmsk,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache */ LSC_LDCC_L1UC_L3C);
#endif
        }
#endif

        /* Wait for S stores */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        /* Last iteration: signal column sums are ready */
        if (last && need_sum_barrier)
            intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        /* Accumulate A += V * S */
        a_tile_type A_tile1 = ugemm_vs(V + value_offset, ldv, S_slm,
                ugemm_kq_wg_tile_m, d, ugemm_kq_wg_tile_n, k_chunk, 0, 0, 0,
                sg_i_vs, sg_j_vs, (local char *)ugemm_slm
#if VAL_SCALES == QUANTIZE_2D
                ,
                V_scales
#endif
#if VAL_ZERO_POINTS
                ,
                V_zp
#endif
#if (VAL_SCALES == QUANTIZE_2D) || VAL_ZERO_POINTS
                ,
                ldvq
#endif
        );

        V = V_backup + value_offset;
#if VAL_SCALES == QUANTIZE_2D
        V_scales += ldvq * ugemm_kq_wg_tile_m;
#endif
#if VAL_ZERO_POINTS == QUANTIZE_2D
        V_zp += ldvq * ugemm_kq_wg_tile_m / VAL_ZP_ELEMENTS_PER_BYTE;
#endif
        tile_binary(A_tile, A_tile1, binary_add);

        if ((multiply_no + 1) % tiles_per_page == 0 && partial_num_rows > 0) {
            k0 += partial_num_rows;
        } else {
            k0 += ugemm_kq_wg_tile_m;
        }
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

#if VAL_SCALES == QUANTIZE_COMMON
#define v_scale_op(x) ((x)*v_scale)
    tile_elementwise(A_tile, v_scale_op);
#endif

    /* Rescale by 1 / (column sums) */
    tile_elementwise(A_scale_tile, native_vrecip);
    tile_hbroadcast_mul(&A_tile, A_scale_tile);

    /* Convert to half precision and store */
    a_tile_type_dst A_tile_dst;
    tile_copy_reblock(A_tile, &A_tile_dst);

    uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
    uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n + wg_j0;

#ifdef BLOCK_2D_A
    tile_store_block2d(A_tile_dst, A, d, q, lda, sg_i0_vs, sg_j0_vs);
#elif defined(BLOCK_A)
    tile_store_block_rem_q(A_tile_dst, A, q, lda, sg_i0_vs, sg_j0_vs);
#else
    tile_store(A_tile_dst, A, d, q, lda, sg_i0_vs, sg_j0_vs);
#endif
}
