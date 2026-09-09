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

#include "gpu/intel/include/conversion.h"
#include "gpu/intel/include/tile_ops.h"
#include "gpu/intel/include/types_interop.h"
#include "gpu/intel/include/utils.h"

#include "gemm_grouped.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define DST_TILE_DATA_T NATIVE_LAYOUT_TYPE(DST_DATA_T)
#define BIA_TILE_DATA_T NATIVE_LAYOUT_TYPE(BIA_DATA_T)

#ifdef UGEMM_RESULT_DT_S32
DECLARE_2D_TILE(c_tile_type_float, float, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1)
#else
#define c_tile_type_float ugemm_grouped_c_type
#endif

#if WITH_BIAS
#define bias_br ugemm_grouped_sg_tile_m
#define bias_bc 1
#define bias_nbr 1
#define bias_nbc 1

DECLARE_2D_TILE(bias_tile_type, float, SUBGROUP_SIZE, bias_br, bias_bc,
        bias_nbr, bias_nbc)
DECLARE_2D_TILE_VREDUCE(c_tile_type_float, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1,
        bias_tile_type, SUBGROUP_SIZE, bias_br, bias_bc, bias_nbr, bias_nbc)

#define BIA_TILE_TO_REF(v) into_float(AS_NATIVE_LAYOUT_TYPE(BIA_DATA_T, v))

#ifndef BIA_DT_F32
DECLARE_2D_TILE(bias_in_tile_type, BIA_TILE_DATA_T, SUBGROUP_SIZE, bias_br,
        bias_bc, bias_nbr, bias_nbc)
#endif

void load_bias(
        bias_tile_type *tile, const global BIA_DATA_T *ptr, int n, int sg_i0) {
#if BIA_DT_F32
    tile_load(tile, ptr, n, 1, 0, sg_i0, 0);
#else
    bias_in_tile_type bias_in_tile;
    tile_load(&bias_in_tile, (const global BIA_TILE_DATA_T *)ptr, n, 1, 0,
            sg_i0, 0);
    tile_convert(bias_in_tile, (*tile), BIA_TILE_TO_REF);
#endif
}
#endif

typedef NATIVE_LAYOUT_TYPE(
        BINARY_SCALE_GROUPED_DATA_T) binary_scale_grouped_tile_data_t;
#define BINARY_SCALE_GROUPED_TO_FLOAT(value) \
    into_float(AS_NATIVE_LAYOUT_TYPE(BINARY_SCALE_GROUPED_DATA_T, value))

typedef NATIVE_LAYOUT_TYPE(
        BINARY_SCALE_DENSE_DATA_T) binary_scale_dense_tile_data_t;
#define BINARY_SCALE_DENSE_TO_FLOAT(value) \
    into_float(AS_NATIVE_LAYOUT_TYPE(BINARY_SCALE_DENSE_DATA_T, value))

#if WITH_POST_OP
#define ELTWISE_VECTOR_API
#include "gpu/intel/include/eltwise.h"

#if WITH_BINARY_GROUPED_SCALE
// One column-block of the [M,N] grouped scale, applied a block at a time.
DECLARE_2D_TILE(binary_group_chunk_type, float, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, 1)
#if !BINARY_SCALE_GROUPED_DT_F32
DECLARE_2D_TILE(binary_group_chunk_in_type, binary_scale_grouped_tile_data_t,
        SUBGROUP_SIZE, ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, 1)
#endif
#endif

#if WITH_BINARY_DENSE_SCALE
#define binary_dense_scale_br ugemm_grouped_sg_tile_m
#define binary_dense_scale_bc 1
#define binary_dense_scale_nbr 1
#define binary_dense_scale_nbc 1
DECLARE_2D_TILE(binary_dense_tile_type, float, SUBGROUP_SIZE,
        binary_dense_scale_br, binary_dense_scale_bc, binary_dense_scale_nbr,
        binary_dense_scale_nbc)
#if !BINARY_SCALE_DENSE_DT_F32
// Intermediate tile for loading non-float binary dense scale before converting
DECLARE_2D_TILE(binary_dense_in_tile_type, binary_scale_dense_tile_data_t,
        SUBGROUP_SIZE, binary_dense_scale_br, binary_dense_scale_bc,
        binary_dense_scale_nbr, binary_dense_scale_nbc)
#endif
DECLARE_2D_TILE_HREDUCE(c_tile_type_float, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1,
        binary_dense_tile_type, SUBGROUP_SIZE, binary_dense_scale_br,
        binary_dense_scale_bc, binary_dense_scale_nbr, binary_dense_scale_nbc)
#endif

#include "grouped_post_ops.h"
#endif

#if WITH_SPARSE_GROUPS
#define offsets_tile_br SUBGROUP_SIZE
#define offsets_tile_bc 1
#define offsets_tile_nbr \
    MAX(1, \
            NUM_GROUPS / (ugemm_grouped_sg_per_wg_m * SUBGROUP_SIZE) \
                    / ugemm_grouped_sg_per_wg_n)
#define offsets_tile_nbc 1
DECLARE_2D_TILE(offsets_tile_type, int, SUBGROUP_SIZE, offsets_tile_br,
        offsets_tile_bc, offsets_tile_nbr, offsets_tile_nbc)

#define slm_src_offsets_size sizeof(off_t) * 2
#define slm_batch_size sizeof(off_t)
#define slm_sg_last_size \
    sizeof(int) * ugemm_grouped_sg_per_wg_m *ugemm_grouped_sg_per_wg_n
#define slm_sparse_total_size \
    slm_src_offsets_size + slm_batch_size + slm_sg_last_size

void find_sparse_batch(off_t *batch, int2 *src_range,
        const global int *src_offsets, off_t flat_token, local char *slm) {
    local off_t *slm_src_offset = (local off_t *)slm;
    local off_t *slm_batch = slm_src_offset + 2;
    local int *slm_sg_last = (local int *)(slm_batch + 1);

    offsets_tile_type offsets_tile;
    off_t sg_ij0 = sub_group_broadcast(get_local_linear_id(), 0);
    int sg_idx = get_sub_group_id();
    off_t sg_batch0 = sg_ij0 * offsets_tile_nbr;

    tile_load(&offsets_tile, src_offsets, NUM_GROUPS, 1, 0, sg_batch0, 0);

    // Share each subgroup's last tile value for cross-subgroup carry
    if (get_sub_group_local_id() == SUBGROUP_SIZE - 1)
        slm_sg_last[sg_idx] = offsets_tile.x[offsets_tile_nbr - 1][0];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    int sg_carry = (sg_idx > 0) ? slm_sg_last[sg_idx - 1] : 0;

    // Find the tile containing the current batch, use shuffle_up to
    // query the next src_offset.
    off_t carry_b = sg_carry;
#pragma unroll
    for (int b = 0, idx = sg_batch0 + get_sub_group_local_id();
            b < offsets_tile_nbr; b++, idx += SUBGROUP_SIZE) {
        off_t curr = offsets_tile.x[b][0];
        off_t prev = intel_sub_group_shuffle_up(carry_b, curr, 1);

        // Only one work-item should satisfy this condition
        if (curr > flat_token && prev <= flat_token && idx < NUM_GROUPS) {
            slm_src_offset[0] = prev;
            slm_src_offset[1] = curr;
            *slm_batch = idx;
        }
        carry_b = sub_group_broadcast(curr, SUBGROUP_SIZE - 1);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    *batch = *slm_batch;
    *src_range = (int2)(slm_src_offset[0], slm_src_offset[1]);
#if WITH_SLM
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
#endif
}
#else
#define slm_sparse_total_size 0
#endif

#define CONVERT_TILE_DATA_T(v) as_native_layout(CONVERT_DATA_T(v))

#ifndef DST_DT_F32
DECLARE_2D_TILE(c_tile_type_dst, DST_TILE_DATA_T, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1)
#endif

/* Optional quantization parameters */
#define SRC_SCALE_ARGS \
    OPTIONAL(AND(WITH_SRC_SCALES, SRC_SCALES_GROUPED), src_attr_scales)
#define SRC_ZP_ARGS OPTIONAL(WITH_SRC_ZP, src_attr_zp)
#define SRC_LD_ARGS OPTIONAL(OR(WITH_SRC_ZP, SRC_SCALES_GROUPED), ldsrcq)
#define WEI_SCALE_ARGS \
    OPTIONAL(AND(WITH_WEI_SCALES, WEI_SCALES_GROUPED), wei_attr_scales)
#define WEI_ZP_ARGS OPTIONAL(WITH_WEI_ZP, wei_attr_zp)
#define WEI_LD_ARGS OPTIONAL(OR(WITH_WEI_ZP, WEI_SCALES_GROUPED), ldweiq)
#define K_PARALLEL_LOCAL_ARGS OPTIONAL(K_PARALLEL_LOCAL, sg_k)

void store_results(c_tile_type_float *tile, global DST_DATA_T *ptr, int n,
        int m, int lddst, int sg_i0, int sg_j0) {
#if DST_DT_F32
    tile_store(*tile, ptr, n, m, lddst, sg_i0, sg_j0);
    //tile_store_t_block2d(c_tile, dst, n, m, lddst, sg_j0, sg_i0);
#else
    c_tile_type_dst tile_dst;
    tile_convert((*tile), tile_dst, CONVERT_TILE_DATA_T);
    tile_store(
            tile_dst, (global DST_TILE_DATA_T *)ptr, n, m, lddst, sg_i0, sg_j0);
    //tile_store_block2d(c_tile_dst, dst, n, m, lddst, sg_j0, sg_i0);
#endif
}

#if WITH_SRC_SCALES && !SRC_SCALES_GROUPED
#define src_attr_scales_br ugemm_grouped_sg_tile_m
#define src_attr_scales_bc 1
#define src_attr_scales_nbr \
    MAX(1, \
            (ugemm_grouped_wg_tile_n + ugemm_grouped_sg_tile_n - 1) \
                    / ugemm_grouped_sg_tile_m)
#define src_attr_scales_nbc 1
DECLARE_2D_TILE(src_attr_scales_tile_type, float, SUBGROUP_SIZE,
        src_attr_scales_br, src_attr_scales_bc, src_attr_scales_nbr,
        src_attr_scales_nbc)
DECLARE_2D_TILE_HREDUCE(c_tile_type_float, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1,
        src_attr_scales_tile_type, SUBGROUP_SIZE, src_attr_scales_br,
        src_attr_scales_bc, src_attr_scales_nbr, src_attr_scales_nbc)

#ifndef SRC_SCALES_DT_F32
typedef NATIVE_LAYOUT_TYPE(SRC_SCALES_DATA_T) src_attr_scales_in_tile_data_t;
DECLARE_2D_TILE(src_attr_scales_in_tile_type, src_attr_scales_in_tile_data_t,
        SUBGROUP_SIZE, src_attr_scales_br, src_attr_scales_bc,
        src_attr_scales_nbr, src_attr_scales_nbc)
#endif

void load_src_attr_scales(src_attr_scales_tile_type *tile,
        const global SRC_SCALES_DATA_T *ptr, int m, int ldsrcq, int sg_j0) {
#if SRC_SCALES_DT_F32
    tile_load(tile, ptr, m, 1, ldsrcq, sg_j0, 0);
#else
#define CONVERT_TILE_SRC_SCALES(v) \
    into_float(AS_NATIVE_LAYOUT_TYPE(SRC_SCALES_DATA_T, v))
    src_attr_scales_in_tile_type src_attr_scales_in_tile;
    tile_load(&src_attr_scales_in_tile,
            (const global src_attr_scales_in_tile_data_t *)ptr, m, 1, ldsrcq,
            sg_j0, 0);
    tile_convert(src_attr_scales_in_tile, (*tile), CONVERT_TILE_SRC_SCALES);
#endif
}
#endif

#if WITH_WEI_SCALES && !WEI_SCALES_GROUPED
#define wei_attr_scales_br ugemm_grouped_sg_tile_m
#define wei_attr_scales_bc 1
#define wei_attr_scales_nbr 1
#define wei_attr_scales_nbc 1
DECLARE_2D_TILE(wei_attr_scales_tile_type, float, SUBGROUP_SIZE,
        wei_attr_scales_br, wei_attr_scales_bc, wei_attr_scales_nbr,
        wei_attr_scales_nbc)
DECLARE_2D_TILE_VREDUCE(c_tile_type_float, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1,
        wei_attr_scales_tile_type, SUBGROUP_SIZE, wei_attr_scales_br,
        wei_attr_scales_bc, wei_attr_scales_nbr, wei_attr_scales_nbc)

#ifndef WEI_SCALES_DT_F32
typedef NATIVE_LAYOUT_TYPE(WEI_SCALES_DATA_T) wei_attr_scales_in_tile_data_t;
DECLARE_2D_TILE(wei_attr_scales_in_tile_type, wei_attr_scales_in_tile_data_t,
        SUBGROUP_SIZE, wei_attr_scales_br, wei_attr_scales_bc,
        wei_attr_scales_nbr, wei_attr_scales_nbc)
#endif

void load_wei_attr_scales(wei_attr_scales_tile_type *tile,
        const global WEI_SCALES_DATA_T *ptr, int n, int ldweiq, int sg_i0) {
#if WEI_SCALES_DT_F32
    tile_load(tile, ptr, n, 1, ldweiq, sg_i0, 0);
#else
#define CONVERT_TILE_WEI_SCALES(v) \
    into_float(AS_NATIVE_LAYOUT_TYPE(WEI_SCALES_DATA_T, v))
    wei_attr_scales_in_tile_type wei_attr_scales_in_tile;
    tile_load(&wei_attr_scales_in_tile,
            (const global wei_attr_scales_in_tile_data_t *)ptr, n, 1, ldweiq,
            sg_i0, 0);
    tile_convert(wei_attr_scales_in_tile, (*tile), CONVERT_TILE_WEI_SCALES);
#endif
}
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__attribute__((reqd_work_group_size(ugemm_grouped_sg_per_wg_m * SUBGROUP_SIZE,
        ugemm_grouped_sg_per_wg_n, ugemm_grouped_sg_per_wg_k))) kernel void
grouped_micro_gemm_m_axis(const global SRC_DATA_T *src, long ldsrc,
        const global WEI_DATA_T *wei, long4 wei_strides, global DST_DATA_T *dst,
        long lddst, const global int *src_offsets,
        const global int *dst_offsets,
        const global SRC_SCALES_DATA_T *src_attr_scales,
        const global SRC_ZP_DATA_T *src_attr_zp, const long ldsrcq,
        const global WEI_SCALES_DATA_T *wei_attr_scales,
        const global WEI_ZP_DATA_T *wei_attr_zp, const long ldweiq,
        const long n, const long k, const global BIA_DATA_T *bias,
        const global binary_scale_grouped_tile_data_t *binary_grouped_scale,
        const global binary_scale_dense_tile_data_t *binary_dense_scale,
        const global float *binary_nvfp4_scale) {
#if WITH_SLM
    local char slm[MAX(ugemm_grouped_slm_size, slm_sparse_total_size)];
#else
#if WITH_SPARSE_GROUPS
    local char slm[slm_sparse_total_size];
#else
    local char *slm = NULL;
#endif
#endif

    off_t sg_i = sub_group_broadcast(get_local_id(0) / SUBGROUP_SIZE, 0);
    off_t sg_j = sub_group_broadcast(get_local_id(1), 0);
#if K_PARALLEL_LOCAL
    off_t sg_k = sub_group_broadcast(get_local_id(2), 0);
#endif

    off_t wg_i0 = get_group_id(0) * ugemm_grouped_wg_tile_m;
    off_t wg_j0;
    off_t batch;
    off_t m;
    off_t src_offset;

#if WITH_SPARSE_GROUPS
    off_t flat_token = get_group_id(2);
    int2 src_range;
    find_sparse_batch(&batch, &src_range, src_offsets, flat_token, slm);
    m = (src_range.y - src_range.x);
    src_offset = src_range.x;
    wg_j0 = flat_token - src_range.x;

    // Early exit for wg that can be processed by the previous ugemm tile
    if (m > 1 && (wg_j0 % ugemm_grouped_wg_tile_n) != 0) { return; }
#else
    batch = sub_group_broadcast(get_group_id(2), 0);
    int2 src_range
            = *(global int2 *)(src_offsets + (batch > 0 ? batch - 1 : batch));

    m = batch > 0 ? (src_range.y - src_range.x) : src_range.x;
    src_offset = batch > 0 ? src_range.x : 0;
    wg_j0 = get_group_id(1) * ugemm_grouped_wg_tile_n;

    if (wg_j0 >= m) return; /* early exit if outside batch */
#endif

    off_t sg_i0 = wg_i0 + sg_i * ugemm_grouped_sg_tile_m;
    off_t sg_j0 = wg_j0 + sg_j * ugemm_grouped_sg_tile_n;

    src += src_offset * ldsrc / SRC_ELEMS_PER_BYTE;
    wei += batch * WEI_STRIDE0;
    dst += src_offset * lddst;

    off_t ldwei = wei_strides[2] == 1 ? wei_strides[1] : wei_strides[2];

#if WITH_SRC_SCALES
    src_attr_scales += src_offset * ldsrcq;
#endif
#if WITH_SRC_ZP
    src_attr_zp += src_offset * ldsrcq / SRC_ZP_ELEMS_PER_BYTE;
#endif
#if WITH_WEI_SCALES
    wei_attr_scales += batch * (n / WEI_N_GROUP_SIZE) * (k / WEI_K_GROUP_SIZE);
#endif
#if WITH_WEI_ZP
    wei_attr_zp += batch * (n / WEI_N_GROUP_SIZE) * (k / WEI_K_GROUP_SIZE)
            / WEI_ZP_ELEMS_PER_BYTE;
#endif

    ugemm_grouped_c_type c_tile_result = ugemm_grouped(wei, ldwei, src, ldsrc,
            n, m, k, wg_i0, wg_j0, 0, sg_i, sg_j K_PARALLEL_LOCAL_ARGS,
            slm WEI_SCALE_ARGS WEI_ZP_ARGS WEI_LD_ARGS SRC_SCALE_ARGS
                    SRC_ZP_ARGS SRC_LD_ARGS);
#ifdef UGEMM_RESULT_DT_S32
    c_tile_type_float c_tile;
    tile_convert(c_tile_result, c_tile, convert_float);
#else
#define c_tile c_tile_result
#endif

#if K_PARALLEL_LOCAL
    if (sg_k > 0) return;
#endif

#if WITH_SRC_SCALES && !SRC_SCALES_GROUPED
    src_attr_scales_tile_type src_attr_scales_tile;
    load_src_attr_scales(
            &src_attr_scales_tile, src_attr_scales, m, ldsrcq, sg_j0);
    tile_hbroadcast_mul(&c_tile, src_attr_scales_tile);
#endif

#if WITH_WEI_SCALES && !WEI_SCALES_GROUPED
    wei_attr_scales_tile_type wei_attr_scales_tile;
    load_wei_attr_scales(
            &wei_attr_scales_tile, wei_attr_scales, n, ldweiq, sg_i0);
    tile_vbroadcast_mul(&c_tile, wei_attr_scales_tile);
#endif

#if WITH_BIAS
    bias += batch * n;
    bias_tile_type bias_tile;
    load_bias(&bias_tile, bias, n, sg_i0);
    tile_vbroadcast_add(&c_tile, bias_tile);
#endif

#if WITH_POST_OP
    apply_post_ops_chain(&c_tile, n, m, lddst, sg_i0, sg_j0, src_offset, batch,
            binary_grouped_scale, binary_dense_scale, binary_nvfp4_scale);
#endif

    store_results(&c_tile, dst, n, m, lddst, sg_i0, sg_j0);
}
