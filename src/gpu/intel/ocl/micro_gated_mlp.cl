/*******************************************************************************
* Copyright 2025 Intel Corporation
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

/* The quantization parameter may be unique for each token/element */
#define QUANTIZE_2D 2

/* The quantization parameter shares the same value across the work-group */
#define QUANTIZE_COMMON 3

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

#define sg_per_wg (ugemm_wgu_sg_per_wg_m * ugemm_wgu_sg_per_wg_n)
#define wgu_tile_sg_n DIV_UP(ugemm_wgu_wg_tile_n, sg_per_wg)

#define wgu_tile_sg_m DIV_UP(ugemm_wgu_wg_tile_m, sg_per_wg)

typedef ugemm_wgu_c_type s_tile_type;

#ifdef SRC_DT_F16
#define VEC_TYPE2 half2
#elif defined(SRC_DT_BF16)
#define VEC_TYPE2 ushort2
#else
#error "Data type not supported for VEC_TYPE2"
#endif

//#define DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc)
DECLARE_2D_TILE(wgu_tile_type, uint, SUBGROUP_SIZE, ugemm_wgu_wg_tile_m / 2, 1,
        1, wgu_tile_sg_n)

#ifdef BLOCK_SRC
DECLARE_2D_TILE_BLOCK_OPS(wgu_tile_type, uint, SUBGROUP_SIZE,
        ugemm_wgu_wg_tile_m / 2, 1, 1, wgu_tile_sg_n)
#elif SRC_ALIGN < 4 //TODO: only define if src_align<4
DECLARE_2D_TILE_LOAD_PACKED_VEC(wgu_tile_type, SRC_DATA_T, VEC_TYPE2,
        SUBGROUP_SIZE, ugemm_wgu_wg_tile_m / 2, 1, 1, wgu_tile_sg_n)
#endif

#if PREFETCH_REMAINDER
#define cooperative_prefetch_2d_maybe_rem cooperative_prefetch_2d_rem
#else
#define cooperative_prefetch_2d_maybe_rem( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d(ptr, rmax, cmax, ld, sg_id, n_sg, sg_size, caching)
#endif

#define cooperative_prefetch_2d_k( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d_maybe_rem( \
            ptr, c, r, cmax, rmax, ld, sg_id, n_sg, sg_size, caching)
//#define cooperative_prefetch_2d_maybe_rem( \
           //ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
        //cooperative_prefetch_2d(ptr, rmax, cmax, ld, sg_id, n_sg, sg_size, caching)

#if REMAINDER_SRC
#define tile_load_block_rem_src tile_load_block
#define tile_store_block_rem_wgu tile_store_block
#else
#define tile_load_block_rem_src(t, ptr, n, ld, off_r, off_c) \
    tile_load_block(t, ptr, ld, off_r, off_c)
#define tile_store_block_rem_wgu(t, ptr, n, ld, off_r, off_c) \
    tile_store_block(t, ptr, ld, off_r, off_c)
#endif

#define binary_add(x, y) ((x) + (y))
#define binary_mul(x, y) ((x) * (y))

#ifdef ACTIVATION_SWISH

#define unary_activation(x) \
    ((x) / (1.f + exp(-1.f * (x)))) //TODO: match ACC type

#elif defined ACTIVATION_GELU_ERF

#define sqrt_2_over_2 0.707106769084930419921875f
#define unary_activation(x) (0.5f * (x) * (1.f + erf((x)*sqrt_2_over_2)))

#elif defined ACTIVATION_GELU_TANH

#define sqrt_2_over_pi 0.79788458347320556640625f
#define fitting_const 0.044715f
#define unary_activation(x) \
    (0.5f * (x) \
            * (1.f \
                    + tanh(sqrt_2_over_pi * (x) \
                            * (1 + fitting_const * (x) * (x)))))

#else
#error "Unknown activation function defined"
#endif

// TMP tile type for half output
DECLARE_2D_TILE(s_tile_type_half, half, SUBGROUP_SIZE, ugemm_wgu_sg_tile_m, 1,
        1, ugemm_wgu_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_half, half, SUBGROUP_SIZE,
        ugemm_wgu_sg_tile_m, 1, 1, ugemm_wgu_sg_tile_n)
DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE,
        ugemm_wgu_c_type_block0, ugemm_wgu_c_type_block1,
        ugemm_wgu_c_type_nblock0, ugemm_wgu_c_type_nblock1, s_tile_type_half,
        SUBGROUP_SIZE, ugemm_wgu_sg_tile_m, 1, 1, ugemm_wgu_sg_tile_n,
        CONVERT_DATA_T)

// attempt @ basicb mlp
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
micro_gated_mlp(const __global SRC_DATA_T *src,
        const __global WTS_GATE_DATA_T *W_gate,
        const __global WTS_UP_DATA_T *W_up,
        const __global WTS_DOWN_DATA_T *W_down, __global half *dst, long MB,
        long IC, long OC, __global half *tmp_reduce_mem,
        const __global WTS_GATE_ATTR_SCALES_DATA_T *wts_gate_scales,
        const __global WTS_GATE_ATTR_ZP_DATA_T *wts_gate_zp,
        const __global WTS_UP_ATTR_SCALES_DATA_T *wts_up_scales,
        const __global WTS_UP_ATTR_ZP_DATA_T *wts_up_zp,
        const __global WTS_DOWN_ATTR_SCALES_DATA_T *wts_down_scales,
        const __global WTS_DOWN_ATTR_ZP_DATA_T *wts_down_zp) {
    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);

    // OC -> group - 0
    // MB -> group - 2
    uint wg_j0 = get_group_id(0) * ugemm_wgu_wg_tile_m;
    uint wg_i0 = get_group_id(2) * ugemm_wgu_wg_tile_n;

    /* Leading dimension for matrices */
    uint lds = SRC_S0; //TODO: add transposable inputs?
    uint ldgu = W_GATE_S0; //TODO: second mm
    //uint ldd = W_DOWN_S0; //TODO: second mm
    uint ldd = W_GATE_S0; //TMP: first mm
    uint lda = DST_S0;

#if WTS_GATE_SCALES || WTS_GATE_ZERO_POINTS
    //uint ldguq = div_up(IC, WTS_GATE_GROUP_SIZE);
    uint ldguq = OC;
#endif
#if WTS_DOWN_SCALES || WTS_DOWN_ZERO_POINTS
    uint lddq = div_up(OC, WTS_DOWN_GROUP_SIZE);
#endif

#if WTS_GATE_SCALES == QUANTIZE_COMMON
    float wg_scale = convert_float(*wts_gate_scales);
    float wu_scale = convert_float(*wts_up_scales);
#endif

    /* Subgroup IDs for each GEMM */
    uint sg_i_wgu = sg_ij % ugemm_wgu_sg_per_wg_m; // 0123;0123
    uint sg_j_wgu = sg_ij / ugemm_wgu_sg_per_wg_m; // 0000;11111

#define WGU_slm_size (ugemm_wgu_wg_tile_m * ugemm_wgu_wg_tile_n * sizeof(half))
#define ugemm_slm_size ugemm_wgu_slm_size

    local char slm[WGU_slm_size + 2 * ugemm_wgu_slm_size];

    local SRC_DATA_T *wg_slm = (local SRC_DATA_T *)&slm[0];
    local uint *ugemm_g_slm = (local uint *)&slm[WGU_slm_size];
    local uint *ugemm_u_slm
            = (local uint *)&slm[WGU_slm_size + ugemm_wgu_slm_size];

    /* Load WG, WU tiles, destined for SLM */
    wgu_tile_type src_tile;
    uint wgu0_copy = wgu_tile_sg_n * sg_ij; // [0:2:16) columns

    s_tile_type S_WG_tile, S_WU_tile;
    tile_fill(S_WG_tile, 0.0f);
    tile_fill(S_WU_tile, 0.0f);

    //int k0 = get_group_id(1) * ugemm_wgu_wg_tile_m / sg_per_wg;
    //{
    for (int k0 = 0; k0 < IC; k0 += ugemm_wgu_wg_tile_m) {
        bool last = (k0 + ugemm_wgu_wg_tile_m >= IC);

        // LOAD MISSES LAST COLUMN..... TODO
#ifdef BLOCK_SRC
        tile_load_block_rem_src(&src_tile, (global uint *)src, MB, lds >> 1,
                k0 / 2, wg_i0 + wgu0_copy);
#elif SRC_ALIGN >= 4
        //TODO:verify correct w/lds+1 and realistic branch
        tile_load(&src_tile, (global uint *)src, (lds + 1) >> 1, IC, lds >> 1,
                k0 / 2, wg_i0 + wgu0_copy);
#else
        tile_load_packed_vec2(
                &src_tile, src, IC, MB, lds, k0, wg_i0 + wgu0_copy);
#endif

        int target_gid_mb = 0;
        int target_gid_oc = 0;
#if 0
        barrier(CLK_LOCAL_MEM_FENCE);
        //if(get_group_id(2) == 0) {
        if(get_group_id(0) == target_gid_oc && get_group_id(2) == target_gid_mb) {
            for(int s=0; s<16; ++s) {
                if(sg_ij == s && get_sub_group_local_id() == 0) {
                    printf("\n\n"); }
                for(int t=0; t<SUBGROUP_SIZE ; ++t) {
                if (sg_ij == s && get_sub_group_local_id() == t) {
                    printf("k0%d gid02:%zu,%zu sgid%d: ldk = %d @sg_i,j%d,%d\n",k0, get_group_id(0), get_group_id(2), s, (int) ldgu, sg_i_wgu, sg_j_wgu);
                    //printf("k0%d gid02:%d,%d sgid%d: ldk = %d offs%d\n",k0, get_group_id(0), get_group_id(2), s, (int) ldgu, offs);
                    //for (int i = 0; i < 32; i++) {
                    for (int i = 0; i < 16; i++) {
                    //for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 2; j++) {
                            printf(" %f %f", as_half2(src_tile.x[i][j]).s0, as_half2(src_tile.x[i][j]).s1);
                        }
                        printf("\n");
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        ///* Store src tile to SLM */
        tile_store_t_sys_src1(src_tile, (local uint *)&wg_slm[0],
                ugemm_wgu_wg_tile_m / 2, wgu0_copy, 0);
        /* Wait for src data to reach SLM */
        barrier(CLK_LOCAL_MEM_FENCE);
#if 0
    if(get_group_id(0) == target_gid_oc && get_group_id(2) == target_gid_mb &&
       get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
        for(int r=0;r<ugemm_wgu_wg_tile_m; ++r) {
        for(int c=0;c<ugemm_wgu_wg_tile_n; ++c) {
            //printf("[%d,%d]:%5.2f ", r, c, convert_float(src_slm[r*ugemm_wgu_wg_tile_n + c]));
            //printf("%5.2f ", convert_float(wg_slm[r*ugemm_wgu_wg_tile_n + c])); //--row major
            printf("%5.2f ", convert_float(wg_slm[c*ugemm_wgu_wg_tile_m + r])); // --col major
        }
        printf("r%d\n", r);
        //printf("\n");
        }
    }
#endif

        //TODO: mmres no good, revert example to have in order or eye data
        s_tile_type FC_G_tile = ugemm_wgu(
                W_gate + k0 * ldgu / WTS_GATE_ELEMENTS_PER_BYTE, ldgu,
                //    ugemm_wgu(W_gate + k0*ldgu, ldgu,
                wg_slm, ugemm_wgu_wg_tile_m,
                //&S_WG_tile,
                OC, ugemm_wgu_wg_tile_n, ugemm_wgu_wg_tile_m, wg_j0, 0, 0,
                sg_i_wgu, sg_j_wgu, (const local char *)ugemm_g_slm
#if WTS_GATE_SCALES == QUANTIZE_2D
                ,
                wts_gate_scales + (k0 / WTS_GATE_GROUP_SIZE) * ldguq
#endif
#if WTS_GATE_ZERO_POINTS
                ,
                wts_gate_zp
                        + (k0 / WTS_GATE_GROUP_SIZE) * ldguq
                                / WTS_GATE_ZP_ELEMENTS_PER_BYTE
#endif
#if (WTS_GATE_SCALES == QUANTIZE_2D) || WTS_GATE_ZERO_POINTS
                ,
                ldguq //TODO: lda of groups
#endif
        );

#if WTS_GATE_SCALES == QUANTIZE_COMMON
#define wg_scale_op(x) ((x)*wg_scale)
        tile_elementwise(FC_G_tile, wg_scale_op);
#endif

        tile_binary(S_WG_tile, FC_G_tile, binary_add);
        barrier(CLK_LOCAL_MEM_FENCE);

#if 0
            barrier(CLK_LOCAL_MEM_FENCE);
            //if(get_group_id(2) == 0) {
            if(get_group_id(0) == target_gid_oc && get_group_id(2) == target_gid_mb) {
                for(int s=0; s<16; ++s) {
                    if(sg_ij == s && get_sub_group_local_id() == 0) {
                        printf("\n\n");
                    }
                    for(int t=0; t<16; ++t) {
                    if (sg_ij == s && get_sub_group_local_id() == t) {
                        printf("mmres k0%d gid02:%zu,%zu sgid%d: ldk = %d @sg_i,j%d,%d\n", k0, get_group_id(0), get_group_id(2), s, (int) ldgu, sg_i_wgu, sg_j_wgu);
                        //printf("k0%d gid02:%d,%d sgid%d: ldk = %d offs%d\n",k0, get_group_id(0), get_group_id(2), s, (int) ldgu, offs);
                        printf(" %v8hlf %v8hlf \n", FC_G_tile.x[0], FC_G_tile.x[1]);
                        // printf(" %v8hlf %v8hlf %v8hlf %v8hlf %v8hlf %v8hlf %v8hlf %v8hlf \\n", FC_G_tile.x[0], FC_G_tile.x[1], FC_G_tile.x[2], FC_G_tile.x[3], FC_G_tile.x[4], FC_G_tile.x[5], FC_G_tile.x[6], FC_G_tile.x[7]);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
#endif

        s_tile_type FC_U_tile = ugemm_wgu(
                W_up + k0 * ldgu / WTS_UP_ELEMENTS_PER_BYTE, ldgu, wg_slm,
                ugemm_wgu_wg_tile_m, OC, ugemm_wgu_wg_tile_n,
                ugemm_wgu_wg_tile_m, //WTF? m,k,n? m,n,k? which one?, swap tile m,n?
                wg_j0, 0,
                0, // dependent on layout.{N,T} // is assumption of non-slm true w/k?
                sg_i_wgu, sg_j_wgu, (const local char *)ugemm_u_slm
#if WTS_UP_SCALES == QUANTIZE_2D
                ,
                wts_up_scales + (k0 / WTS_UP_GROUP_SIZE) * ldguq
#endif
#if WTS_UP_ZERO_POINTS
                ,
                wts_up_zp
                        + (k0 / WTS_UP_GROUP_SIZE) * ldguq
                                / WTS_UP_ZP_ELEMENTS_PER_BYTE
#endif
#if (WTS_UP_GATE_SCALES == QUANTIZE_2D) || WTS_UP_ZERO_POINTS
                ,
                ldguq
#endif
        );

#if WTS_UP_SCALES == QUANTIZE_COMMON
#define wu_scale_op(x) ((x)*wu_scale)
        tile_elementwise(FC_U_tile, wu_scale_op);
#endif
        tile_binary(S_WU_tile, FC_U_tile, binary_add);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    tile_elementwise(S_WG_tile, unary_activation); // for gate + swish
    tile_binary(S_WU_tile, S_WG_tile, binary_mul); // for gate + swish

    s_tile_type_half S_tile_half;
    tile_copy_reblock(S_WU_tile, &S_tile_half); // for gate + swish
    //tile_copy_reblock(S_WG_tile, &S_tile_half); //for mm only

    uint sg_i0_wgu = sg_i_wgu * ugemm_wgu_sg_tile_n;
    uint sg_j0_wgu = sg_j_wgu * ugemm_wgu_sg_tile_m;

    //if(get_global_id(0) == 0) {dst[0] = wg_slm[0]; }

    //TODO: block2D?
#ifdef BLOCK_DST
    tile_store_block_rem_wgu(
            S_tile_half, dst, OC, lda, wg_i0 + sg_j0_wgu, wg_j0 + sg_i0_wgu);
#else
    tile_store(S_tile_half, dst, MB, OC, lda, wg_i0 + sg_j0_wgu,
            wg_j0 + sg_i0_wgu);
#endif

    //    size_t k_offset = get_group_id(1) / sg_per_wg * OC * MB;
    //#ifdef BLOCK_DST
    //    tile_store_block_rem_wgu(S_tile_half, tmp_reduce_mem + k_offset, OC, lda, wg_i0 + sg_j0_wgu, wg_j0 + sg_i0_wgu);
    //#else
    //    tile_store(S_tile_half, tmp_reduce_mem + k_offset, MB, OC, lda, wg_i0 + sg_j0_wgu, wg_j0 + sg_i0_wgu);
    //#endif
}
