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

#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types.h"

// Grouped GEMM OCL reference kernel
//
// Per group g: C[g] = A[g] * B[g] (+ bias / scales / zero-points for 2Dx3D)
//
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// partner_offsets buffer is dst_offsets
//
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
// partner_offsets buffer is wei_offsets
//
// Both patterns are unified via runtime strides + per-group bases:
//   idx = base + i * stride_i + j * stride_j
//   base = group_start * group_stride
//
// get_global_id(0): group index
// get_global_id(1): output row (m)
// get_global_id(2): output column (n)
//
// Supported:
//  f32/f16/bf16
// For 2Dx3D only:
//  u8/s8 src, u8/s8/s4/u4/f8/f4 wei (incl. WOQ)
//  row/col-wise, K-grouped or KxN-grouped (wei) scales & zero-points
//  bias [group_count, N] and post-ops
__kernel void ref_grouped_gemm_matmul(__global const SRC_DATA_T *src,
        __global const int *src_offsets, __global const WEI_DATA_T *wei,
        __global const int *partner_offsets, __global DST_DATA_T *dst,
        const int group_count, const int is_2dby2d, const long m_fixed,
        const long k_fixed, const long N, const long src_stride_m,
        const long src_stride_k, const long wei_stride_k,
        const long wei_stride_n, const long dst_stride_m,
        const long dst_stride_n, const long src_group_stride,
        const long wei_group_stride, const long dst_group_stride,
        const long n_k_groups
#if WITH_BIAS
        ,
        __global const BIA_DATA_T *bias // Bias [group_count, N]
#endif
#if WITH_SRC_SCALES
        ,
        __global const SRC_SCALES_DATA_T *src_scales,
        const long src_scale_ngroups_k
#endif
#if WITH_WEI_SCALES
        ,
        __global const WEI_SCALES_DATA_T *wei_scales,
        const long wei_scale_ngroups_k, const long wei_scale_group_n
#endif
#if WITH_SRC_ZPOINTS
        ,
        __global const SRC_ZP_DATA_T *src_zero_points,
        const long src_zp_ngroups_k
#endif
#if WITH_WEI_ZPOINTS
        ,
        __global const WEI_ZP_DATA_T *wei_zero_points,
        const long wei_zp_ngroups_k,
        const long wei_zp_group_n
#endif
                POST_OP_ARGS) {
    const off_t group_id = get_global_id(0);
    const long m = get_global_id(1);
    const long n = get_global_id(2);

    if (group_id >= group_count) return;
    if (n >= N) return;

    const int src_start = (group_id == 0) ? 0 : src_offsets[group_id - 1];
    const int src_end = src_offsets[group_id];
    const int var_extent
            = src_end - src_start; // M_g or K_g depending on the pattern

    long M_g, K_g;
    off_t src_group_start, wei_group_start, dst_group_start;
    if (is_2dby2d) {
        // partner_offsets == wei_offsets
        if (partner_offsets[group_id] != src_end
                || (group_id > 0 && partner_offsets[group_id - 1] != src_start))
            return;
        M_g = m_fixed;
        K_g = var_extent;
        src_group_start = src_start; // along total_K
        wei_group_start = src_start; // along total_K
        dst_group_start = group_id; // dense [G, M, N]
    } else {
        // partner_offsets == dst_offsets
        const int dst_start
                = (group_id == 0) ? 0 : partner_offsets[group_id - 1];
        const int dst_end = partner_offsets[group_id];
        if (dst_end - dst_start != var_extent) return;
        M_g = var_extent;
        K_g = k_fixed;
        src_group_start = src_start; // along total_M
        wei_group_start = group_id; // dense [G, K, N]
        dst_group_start = dst_start; // along total_M
    }

    // Note, that K_g == 0 must still write zeros
    if (M_g == 0 || m >= M_g) return;

    const long src_base = src_group_start * src_group_stride;
    const long wei_base = wei_group_start * wei_group_stride;
    const long dst_base = dst_group_start * dst_group_stride;

    // src attribute row (global token) = group start + local row
    const long src_attr_row = src_group_start + m;
    const long k_group_size = K_g / n_k_groups;

    FLT_ACC_DATA_T acc = 0;
    for (long i_group = 0; i_group < n_k_groups; i_group++) {
        ACC_DATA_T acc_g = (ACC_DATA_T)0;
        for (long k = 0; k < k_group_size; k++) {
            const long k_abs = k + i_group * k_group_size;
            const long src_idx
                    = src_base + m * src_stride_m + k_abs * src_stride_k;
            const long wei_idx
                    = wei_base + k_abs * wei_stride_k + n * wei_stride_n;
            int src_zp = 0;
#if WITH_SRC_ZPOINTS
            src_zp = SRC_ZP_TO_REF(src_zero_points,
                    src_attr_row * src_zp_ngroups_k
                            + i_group * src_zp_ngroups_k / n_k_groups);
#endif
            int wei_zp = 0;
#if WITH_WEI_ZPOINTS
            wei_zp = WEI_ZP_TO_REF(wei_zero_points,
                    (group_id * wei_zp_ngroups_k
                            + i_group * wei_zp_ngroups_k / n_k_groups)
                                    * (N / wei_zp_group_n)
                            + n / wei_zp_group_n);
#endif
#if SRC_DT_F4_E2M1
            ACC_DATA_T s
                    = TO_ACC(SRC_TO_REF(GET_HALF_BYTE(src, src_idx)) - src_zp);
#else
            ACC_DATA_T s = TO_ACC(SRC_TO_REF(src[src_idx]) - src_zp);
#endif
#if WEI_DT_U3
            ACC_DATA_T w = TO_ACC(WEI_TO_REF(GET_U3(wei, wei_idx)) - wei_zp);
#elif WEI_DT_S4 || WEI_DT_U4 || WEI_DT_F4_E2M1
            ACC_DATA_T w
                    = TO_ACC(WEI_TO_REF(GET_HALF_BYTE(wei, wei_idx)) - wei_zp);
#else
            ACC_DATA_T w = TO_ACC(WEI_TO_REF(wei[wei_idx]) - wei_zp);
#endif
            acc_g += s * w;
        }
        FLT_ACC_DATA_T src_scale = 1.f;
        FLT_ACC_DATA_T wei_scale = 1.f;
#if WITH_SRC_SCALES
        src_scale = SRC_SCALES_TO_REF(
                src_scales[src_attr_row * src_scale_ngroups_k
                        + i_group * src_scale_ngroups_k / n_k_groups]);
#endif
#if WITH_WEI_SCALES
        wei_scale = WEI_SCALES_TO_REF(
                wei_scales[(group_id * wei_scale_ngroups_k
                                   + i_group * wei_scale_ngroups_k / n_k_groups)
                                * (N / wei_scale_group_n)
                        + n / wei_scale_group_n]);
#endif
        acc += ACC_TO_REF(acc_g) * src_scale * wei_scale;
    }

#if WITH_BIAS
    acc += BIA_TO_REF(bias[group_id * N + n]);
#endif

    // Post-ops apply to the 2Dx3D pattern only
    POST_OP_DATA_T po_acc = acc;
    POST_OP_DATA_T sum_src = 0;
    APPLY_POST_OPS_SERIAL(
            po_acc, sum_src, group_id, dst_group_start + m, n, 0, 0, 0);
    acc = po_acc;

    const long dst_idx = dst_base + m * dst_stride_m + n * dst_stride_n;
    dst[dst_idx] = REF_TO_DST(acc);
}
