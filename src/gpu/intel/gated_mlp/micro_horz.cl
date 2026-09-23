/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/include/tile_ops.h"
#include "gpu/intel/include/types.h"

#include "gemm_gateup.h"

#define QUANTIZE_2D 2

#define QUANTIZE_COMMON 3

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

#define sg_per_wg (ugemm_wgu_sg_per_wg_m * ugemm_wgu_sg_per_wg_n)
#define wgu_tile_sg_n DIV_UP(ugemm_wgu_wg_tile_n, sg_per_wg)

typedef ugemm_wgu_c_type s_tile_type;

#ifdef SRC_DT_F16
#define VEC_TYPE1 half
#define VEC_TYPE2 half2
#elif defined(SRC_DT_BF16)
#define VEC_TYPE1 ushort
#define VEC_TYPE2 ushort2
#else
#error "Data type not supported for VEC_TYPE2"
#endif

DECLARE_2D_TILE(wgu_tile_type, uint, SUBGROUP_SIZE, ugemm_wgu_wg_tile_m / 2, 1,
        1, wgu_tile_sg_n)

#ifdef BLOCK_SRC
DECLARE_2D_TILE_BLOCK_OPS(wgu_tile_type, uint, SUBGROUP_SIZE,
        ugemm_wgu_wg_tile_m / 2, 1, 1, wgu_tile_sg_n)
#elif SRC_ALIGN < 4
DECLARE_2D_TILE_LOAD_PACKED_VEC(wgu_tile_type, SRC_DATA_T, VEC_TYPE2,
        as_native_layout, 2, SUBGROUP_SIZE, ugemm_wgu_wg_tile_m / 2, 1, 1,
        wgu_tile_sg_n)
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

#if REMAINDER_SRC
#define tile_load_block_rem_src tile_load_block
#else
#define tile_load_block_rem_src(t, ptr, n, ld, off_r, off_c) \
    tile_load_block(t, ptr, ld, off_r, off_c)
#endif

#define binary_add(x, y) ((x) + (y))
#define binary_mul(x, y) ((x) * (y))

#ifdef ACTIVATION_SWISH

#define unary_activation(x) ((x) / (1.f + exp(-1.f * (x))))

#elif defined ACTIVATION_GELU_ERF

#define sqrt_2_over_2 0.707106769084930419921875f
#define unary_activation(x) (0.5f * (x) * (1.f + erf((x) * sqrt_2_over_2)))

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

DECLARE_2D_TILE(s_tile_type_dst, VEC_TYPE1, SUBGROUP_SIZE, ugemm_wgu_sg_tile_m,
        1, 1, ugemm_wgu_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_dst, VEC_TYPE1, SUBGROUP_SIZE,
        ugemm_wgu_sg_tile_m, 1, 1, ugemm_wgu_sg_tile_n)
DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE,
        ugemm_wgu_c_type_block0, ugemm_wgu_c_type_block1,
        ugemm_wgu_c_type_nblock0, ugemm_wgu_c_type_nblock1, s_tile_type_dst,
        SUBGROUP_SIZE, ugemm_wgu_sg_tile_m, 1, 1, ugemm_wgu_sg_tile_n,
        CONVERT_DATA_T)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
micro_gated_mlp_horz(const __global SRC_DATA_T *src,
        const __global WTS_GATE_DATA_T *W_gate,
        const __global WTS_UP_DATA_T *W_up,
        const __global WTS_DOWN_DATA_T *W_down, __global DST_DATA_T *dst,
        long MB, long IC, long OC, __global INTER_DATA_T *tmp_reduce_mem,
        const __global WTS_GATE_ATTR_SCALES_DATA_T *wts_gate_scales,
        const __global WTS_GATE_ATTR_ZP_DATA_T *wts_gate_zp,
        const __global WTS_UP_ATTR_SCALES_DATA_T *wts_up_scales,
        const __global WTS_UP_ATTR_ZP_DATA_T *wts_up_zp,
        const __global WTS_DOWN_ATTR_SCALES_DATA_T *wts_down_scales,
        const __global WTS_DOWN_ATTR_ZP_DATA_T *wts_down_zp) {

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);

    uint wg_j0 = get_group_id(0) * ugemm_wgu_wg_tile_m; // OC
    uint wg_i0 = get_group_id(2) * ugemm_wgu_wg_tile_n; // MB

    uint lds = SRC_S0;
    uint ldg = W_GATE_S1;
    uint ldu = W_UP_S1;
    uint ldi = INTER_S0;

#if WTS_GATE_SCALES || WTS_GATE_ZERO_POINTS
    uint ldgq = OC;
#endif
#if WTS_UP_SCALES || WTS_UP_ZERO_POINTS
    uint lduq = OC;
#endif

#if WTS_GATE_SCALES == QUANTIZE_COMMON
    float wg_scale = convert_float(*wts_gate_scales);
#endif
#if WTS_UP_SCALES == QUANTIZE_COMMON
    float wu_scale = convert_float(*wts_up_scales);
#endif

    uint sg_i_wgu = sg_ij % ugemm_wgu_sg_per_wg_m;
    uint sg_j_wgu = sg_ij / ugemm_wgu_sg_per_wg_m;

#define WGU_slm_size \
    (ugemm_wgu_wg_tile_m * ugemm_wgu_wg_tile_n * sizeof(SRC_DATA_T))

    local char slm[WGU_slm_size + ugemm_wgu_slm_size];

    local SRC_DATA_T *wg_slm = (local SRC_DATA_T *)&slm[0];
    local char *ugemm_gu_slm = &slm[WGU_slm_size];

    wgu_tile_type src_tile;
    uint wgu0_copy = wgu_tile_sg_n * sg_ij;

#ifndef UGEMM_UP_ONLY
    s_tile_type S_WG_tile;
    tile_fill(S_WG_tile, 0.0f);
#endif
    s_tile_type S_WU_tile;
    tile_fill(S_WU_tile, 0.0f);

    for (int k0 = 0; k0 < IC; k0 += ugemm_wgu_wg_tile_m) {

#ifdef BLOCK_SRC
        tile_load_block_rem_src(&src_tile, (global uint *)src, MB, lds >> 1,
                k0 / 2, wg_i0 + wgu0_copy);
#elif SRC_ALIGN >= 4
        tile_load(&src_tile, (global uint *)src, (lds + 1) >> 1, IC, lds >> 1,
                k0 / 2, wg_i0 + wgu0_copy);
#else
        tile_load_packed_vec(
                &src_tile, src, IC, MB, lds, k0, wg_i0 + wgu0_copy);
#endif

        int target_gid_mb = 0;
        int target_gid_oc = 0;

        tile_store_t_sys_src1(src_tile, (local uint *)&wg_slm[0],
                ugemm_wgu_wg_tile_m / 2, wgu0_copy, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

#ifndef UGEMM_UP_ONLY
        s_tile_type FC_G_tile
                = ugemm_wgu(W_gate + k0 / WTS_GATE_ELEMENTS_PER_BYTE, ldg,
                        wg_slm, ugemm_wgu_wg_tile_m, OC, ugemm_wgu_wg_tile_n,
                        ugemm_wgu_wg_tile_m, wg_j0, 0, 0, sg_i_wgu, sg_j_wgu,
                        ugemm_gu_slm
#if WTS_GATE_SCALES == QUANTIZE_2D
                        ,
                        wts_gate_scales + (k0 / WTS_GATE_GROUP_SIZE) * ldgq
#endif
#if WTS_GATE_ZERO_POINTS
                        ,
                        wts_gate_zp
                                + (k0 / WTS_GATE_GROUP_SIZE) * ldgq
                                        / WTS_GATE_ZP_ELEMENTS_PER_BYTE
#endif
#if (WTS_GATE_SCALES == QUANTIZE_2D) || WTS_GATE_ZERO_POINTS
                        ,
                        ldgq
#endif
                );

#if WTS_GATE_SCALES == QUANTIZE_COMMON
#define wg_scale_op(x) ((x) * wg_scale)
        tile_elementwise(FC_G_tile, wg_scale_op);
#endif

        // TODO: S_W[G,U]_tile might end up clobbered at each ukernel call!
        //       The proper solution for now is to acccumulate right to SLM.
        tile_binary(S_WG_tile, FC_G_tile, binary_add);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif // UGEMM_UP_ONLY

        s_tile_type FC_U_tile = ugemm_wgu(W_up + k0 / WTS_UP_ELEMENTS_PER_BYTE,
                ldu, wg_slm, ugemm_wgu_wg_tile_m, OC, ugemm_wgu_wg_tile_n,
                ugemm_wgu_wg_tile_m, wg_j0, 0, 0, sg_i_wgu, sg_j_wgu,
                ugemm_gu_slm
#if WTS_UP_SCALES == QUANTIZE_2D
                ,
                wts_up_scales + (k0 / WTS_UP_GROUP_SIZE) * lduq
#endif
#if WTS_UP_ZERO_POINTS
                ,
                wts_up_zp
                        + (k0 / WTS_UP_GROUP_SIZE) * lduq
                                / WTS_UP_ZP_ELEMENTS_PER_BYTE
#endif
#if (WTS_UP_SCALES == QUANTIZE_2D) || WTS_UP_ZERO_POINTS
                ,
                lduq
#endif
        );

#if WTS_UP_SCALES == QUANTIZE_COMMON
#define wu_scale_op(x) ((x) * wu_scale)
        tile_elementwise(FC_U_tile, wu_scale_op);
#endif
        tile_binary(S_WU_tile, FC_U_tile, binary_add);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#ifndef UGEMM_UP_ONLY
    tile_elementwise(S_WG_tile, unary_activation);
    tile_binary(S_WU_tile, S_WG_tile, binary_mul);
#endif

    s_tile_type_dst S_tile_dst;
    tile_copy_reblock(S_WU_tile, &S_tile_dst);

    uint sg_i0_wgu = sg_i_wgu * ugemm_wgu_sg_tile_n;
    uint sg_j0_wgu = sg_j_wgu * ugemm_wgu_sg_tile_m;

    size_t k_offset = get_group_id(1) / sg_per_wg * OC * MB;
    tile_store_t(S_tile_dst, tmp_reduce_mem + k_offset, MB, OC, ldi,
            wg_i0 + sg_j0_wgu, wg_j0 + sg_i0_wgu);
}
