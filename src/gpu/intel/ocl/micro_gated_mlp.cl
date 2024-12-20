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
#include "gpu/intel/ocl/tile_ops.h"

/* Microkernel headers -- generated at runtime */
#include "gemm_gateup.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

#define sg_per_wg (ugemm_wgu_sg_per_wg_m * ugemm_wgu_sg_per_wg_n)
#define wgu_tile_sg_n DIV_UP(ugemm_wgu_wg_tile_n, sg_per_wg)
#define wgu_tile_sg_m DIV_UP(ugemm_wgu_wg_tile_m, sg_per_wg)

typedef ugemm_wgu_c_type s_tile_type;

DECLARE_2D_TILE(wgu_tile_type, uint, SUBGROUP_SIZE, ugemm_wgu_wg_tile_m/2, 1, 1, wgu_tile_sg_n)

#ifdef BLOCK_SRC
    DECLARE_2D_TILE_BLOCK_OPS(
            wgu_tile_type, uint, SUBGROUP_SIZE, ugemm_wgu_wg_tile_m/2, 1, 1, wgu_tile_sg_n)
#elif WGU_ALIGN < 4
    DECLARE_2D_TILE_LOAD_PACKED_HALF(
            wgu_tile_type, SUBGROUP_SIZE, ugemm_wgu_wg_tile_m/2, 1, 1, wgu_tile_sg_n)
#endif

#if REMAINDER_WGU
    #define tile_load_block_rem_wgu tile_load_block
    #define tile_store_block_rem_wgu tile_store_block
#else
    #define tile_load_block_rem_wgu(t, ptr, n, ld, off_r, off_c) \
        tile_load_block(t, ptr, ld, off_r, off_c)
    #define tile_store_block_rem_wgu(t, ptr, n, ld, off_r, off_c) \
        tile_store_block(t, ptr, ld, off_r, off_c)
#endif

#define binary_add(x, y) ((x) + (y))
#define binary_mul(x, y) ((x) * (y))
#define unary_swish(x) ((x) / (1.f + exp(-1.f * (x)))) //TODO: match ACC type

// TMP tile type for half output
DECLARE_2D_TILE(s_tile_type_half, half, SUBGROUP_SIZE, ugemm_wgu_sg_tile_m, 1, 1,
        ugemm_wgu_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_half, half, SUBGROUP_SIZE,
        ugemm_wgu_sg_tile_m, 1, 1, ugemm_wgu_sg_tile_n)
DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE, ugemm_wgu_c_type_block0,
        ugemm_wgu_c_type_block1, ugemm_wgu_c_type_nblock0,
        ugemm_wgu_c_type_nblock1, s_tile_type_half, SUBGROUP_SIZE,
        ugemm_wgu_sg_tile_m, 1, 1, ugemm_wgu_sg_tile_n)


// attempt @ basicb mlp
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void micro_gated_mlp(const __global half *src,
        const __global half *W_gate, const __global half *W_up,
        const __global half *W_down, __global half *dst, long MB,
        long IC, long OC, __global half *tmp_reduce_mem) {
    int sgperwg = sg_per_wg;
    int wgutilesgn = wgu_tile_sg_n;

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);

    // OC -> group - 0
    // MB -> group - 2
    uint wg_j0 = get_group_id(0) * ugemm_wgu_wg_tile_m;
    uint wg_i0 = get_group_id(2) * ugemm_wgu_wg_tile_n;

    /* Leading dimension for matrices */
    uint lds  = SRC_S0; //TODO: add transposable inputs?
    uint ldgu = W_GATE_S0; //TODO: second mm
    //uint ldd = W_DOWN_S0; //TODO: second mm
    uint ldd = W_GATE_S0; //TMP: first mm
    uint lda  = DST_S0;

    /* Subgroup IDs for each GEMM */
    uint sg_i_wgu = sg_ij % ugemm_wgu_sg_per_wg_m; // 0123;0123
    uint sg_j_wgu = sg_ij / ugemm_wgu_sg_per_wg_m; // 0000;11111


// #define WGU_slm_size (B_MAX * ugemm_wgu_wg_tile_n * sizeof(half)) //TODO: which one???
#define WGU_slm_size (ugemm_wgu_wg_tile_m * ugemm_wgu_wg_tile_n * sizeof(half))
#define ugemm_slm_size ugemm_wgu_slm_size

    local char slm[WGU_slm_size + ugemm_wgu_slm_size];

    local half *wg_slm = (local half *)&slm[0];
    local uint *ugemm_slm = (local uint *)&slm[WGU_slm_size];

    /* Load WG, WU tiles, destined for SLM */
    wgu_tile_type src_tile;
    uint wgu0_copy = wgu_tile_sg_n * sg_ij; // [0:2:16) columns

    s_tile_type S_WG_tile,  S_WU_tile;
    tile_fill(S_WG_tile, 0.0f);
    tile_fill(S_WU_tile, 0.0f);

    for (int k0 = 0; k0 < IC; k0 += ugemm_wgu_wg_tile_m) {
#ifdef BLOCK_SRC
        tile_load_block(
                &src_tile, (global uint *)src, MB, lds >> 1, k0/2, wg_i0 + wgu0_copy);

//#elif SRC_ALIGN >= 4 // TODO:
        //tile_load(&src_tile, (global uint *)src, (OC + 1) >> 1, IC, lds >> 1, 0,
                //wgu0_copy);
//#else
        //tile_load_packed_half(&src_tile, src, OC, IC, lds, 0, wgu0_copy);
#endif

        ///* Store src tile to SLM */
        tile_store_t_sys_src1(
                src_tile, (local uint *)&wg_slm[0], ugemm_wgu_wg_tile_n/2, wgu0_copy, 0);

        /* Wait for src data to reach SLM */
        barrier(CLK_LOCAL_MEM_FENCE);

        s_tile_type FC_G_tile
            = ugemm_wgu(W_gate + k0*ldgu, ldgu,
                        wg_slm, ugemm_wgu_wg_tile_m,
                        OC, ugemm_wgu_wg_tile_n, ugemm_wgu_wg_tile_m,
                        wg_j0, 0, 0,
                        sg_i_wgu, sg_j_wgu, (const local char*)ugemm_slm);

        tile_binary(S_WG_tile, FC_G_tile, binary_add);
        s_tile_type FC_U_tile
            = ugemm_wgu(W_up + k0*ldgu, ldgu,
                        wg_slm, ugemm_wgu_wg_tile_m,
                        OC, ugemm_wgu_wg_tile_n, ugemm_wgu_wg_tile_m, //WTF? m,k,n? m,n,k? which one?, swap tile m,n?
                        wg_j0, 0, 0,  // dependent on layout.{N,T} // is assumption of non-slm true w/k?
                        sg_i_wgu, sg_j_wgu, (const local char*)ugemm_slm);
        tile_binary(S_WU_tile, FC_U_tile, binary_add);
    }

    tile_elementwise(S_WG_tile, unary_swish);
    tile_binary(S_WU_tile, S_WG_tile, binary_mul);

    s_tile_type_half S_tile_half;
    tile_copy_reblock(S_WU_tile, &S_tile_half);

    uint sg_i0_wgu = sg_i_wgu * ugemm_wgu_sg_tile_n;
    uint sg_j0_wgu = sg_j_wgu * ugemm_wgu_sg_tile_m;


    tile_store_block_rem_wgu(S_tile_half, dst, OC, lda, wg_i0 + sg_j0_wgu, wg_j0 + sg_i0_wgu);
    //tile_store_block_rem_wgu(S_tile_half, tmp, OC, lda, wg_i0 + sg_j0_wgu, wg_j0 + sg_i0_wgu); //store to tmp for further MM or reduction

    // WTAF O__o
    // no results unless slm print accessed anywhere in kernel
    if(get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        printf("%5.2f ", ugemm_slm[0]);
    }
}
