/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include "gpu/intel/ocl/ocl_post_ops.h"
#include "gpu/intel/ocl/ocl_types.h"
#include "gpu/intel/ocl/ocl_utils.h"
#include "gpu/intel/ocl/reduction/ocl_reduction.h"

#ifdef OCL_DEBUG
#define DUMP(str, ...) \
    do { \
        const size_t gid[3] \
                = {get_global_id(0), get_global_id(1), get_global_id(2)}; \
        const size_t lid[3] \
                = {get_local_id(0), get_local_id(1), get_local_id(2)}; \
        const size_t wgid[3] \
                = {get_group_id(0), get_group_id(1), get_group_id(2)}; \
        const size_t lin_g = get_global_linear_id(); \
        const size_t lin_l = get_local_linear_id(); \
        const uint sglid = get_sub_group_local_id(); \
        DEBUG_PRINT( \
                "gid=(%zu,%zu,%zu) lid=(%zu,%zu,%zu) " \
                "linear=(%zug/%zul/%usg): " str, \
                gid[0], gid[1], gid[2], lid[0], lid[1], lid[2], lin_g, lin_l, \
                sglid, ##__VA_ARGS__) \
    } while (0)
#else
#define DUMP(...)
#endif

// Define how to read data
#define BLOCK_READ_DATA_T(data_ptr) \
    AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)data_ptr))
#define READ_DATA(val) WITH_BLOCK_READ ? BLOCK_READ_DATA_T(&val) : val

// Zero-padding defines
#if NUM_SRC_ZPAD > 2 || NUM_DST_ZPAD > 2
#error "At most 2 zero pad patterns permitted"
#endif

bool is_dst_zero_padded(const dim_t dst_off) {
    bool ret = false;
#if NUM_DST_ZPAD >= 1
    {
        const dim_t outer_idx = (dst_off / DST_Z0_STRIDE1) % DST_Z0_SIZE1;
        const dim_t inner_idx = (dst_off / DST_Z0_STRIDE0) % DST_Z0_SIZE0;
        ret |= (outer_idx * DST_Z0_SIZE0 + inner_idx >= DST_Z0_SIZE);
    }
#endif
#if NUM_DST_ZPAD >= 2
    {
        const dim_t outer_idx = (dst_off / DST_Z1_STRIDE1) % DST_Z1_SIZE1;
        const dim_t inner_idx = (dst_off / DST_Z1_STRIDE0) % DST_Z1_SIZE0;
        ret |= (outer_idx * DST_Z1_SIZE0 + inner_idx >= DST_Z1_SIZE);
    }
#endif
    return ret;
}

// For dst zero-padding on reduced dimensions:
// increase strides to skip over zeros
// XXX: Relies on zero padding being sorted by inner stride
// i.e. DST_Z0_STRIDE0 < DST_Z1_STRIDE0
dim_t dst_off_w_zero_padding(dim_t outer, dim_t inner) {
    dim_t outer_stride = INNER_DIM_SIZE;

#if NUM_DST_ZPAD >= 1 && DST_Z0_IS_REDUCED
#if DST_Z0_SIZE1 != 1 || DST_Z0_SIZE != 1
#error "Reduced zpad #1 doesn't match expected pattern!"
#endif
    // Increase to account for DST_Z0
    outer_stride *= DST_Z0_SIZE0;

    // In cases with split-reductions (i.e. aBx16b for 8x1024x32:8x1x32)
    // the zero-padding is inserted in the middle (potentially) of the inner
    // block, so we need to wrap inner indexing around it
    const dim_t inside0 = inner % DST_Z0_STRIDE0;
    const dim_t idx0 = inner / DST_Z0_STRIDE0;
    inner = inside0 + idx0 * DST_Z0_SIZE0 * DST_Z0_STRIDE0;
#endif
#if NUM_DST_ZPAD >= 2 && DST_Z1_IS_REDUCED
#if DST_Z1_SIZE1 != 1 || DST_Z1_SIZE != 1
#error "Reduced zpad #2 doesn't match expected pattern!"
#endif
    // Increase to account for DST_Z1
    outer_stride *= DST_Z1_SIZE0;
    const dim_t inside1 = inner % DST_Z1_STRIDE0;
    const dim_t idx1 = inner / DST_Z1_STRIDE0;
    inner = inside1 + idx1 * DST_Z1_SIZE0 * DST_Z1_STRIDE0;
#endif
    return outer * outer_stride + inner;
}

#define _SRC_OFF(outer, reduction, inner) \
    ((outer)*REDUCTION_SIZE * INNER_DIM_SIZE + (reduction)*INNER_DIM_SIZE \
            + (inner))

#define _DST_OFF(outer, inner) dst_off_w_zero_padding(outer, inner)

#if NUM_DST_ZPAD == 0
#define PADDED_NELEMS OUTER_SIZE *INNER_DIM_SIZE
#elif NUM_DST_ZPAD == 1
#define PADDED_NELEMS OUTER_SIZE *INNER_DIM_SIZE *DST_Z0_SIZE0 *DST_Z0_SIZE1
#elif NUM_DST_ZPAD == 2
#define PADDED_NELEMS \
    OUTER_SIZE *INNER_DIM_SIZE *DST_Z0_SIZE0 *DST_Z0_SIZE1 *DST_Z1_SIZE0 \
            *DST_Z1_SIZE1
#endif

#if WITH_POST_OP
void reverse_indexing(dim_t dst_off, int *res) {
    // Reconstruct dimension indices from dst_off
    res[0] = (DST_S0 == 0) ? 0
                           : dst_off / DST_S0 % div_up(DST_D0, DST_B0) * DST_B0
                    + dst_off / DST_SB0 % DST_B0;
    res[1] = (DST_S1 == 0) ? 0
                           : dst_off / DST_S1 % div_up(DST_D1, DST_B1) * DST_B1
                    + dst_off / DST_SB1 % DST_B1;
    res[2] = (DST_S2 == 0) ? 0
                           : dst_off / DST_S2 % div_up(DST_D2, DST_B2) * DST_B2
                    + dst_off / DST_SB2 % DST_B2;
    res[3] = (DST_S3 == 0) ? 0
                           : dst_off / DST_S3 % div_up(DST_D3, DST_B3) * DST_B3
                    + dst_off / DST_SB3 % DST_B3;
    res[4] = (DST_S4 == 0) ? 0
                           : dst_off / DST_S4 % div_up(DST_D4, DST_B4) * DST_B4
                    + dst_off / DST_SB4 % DST_B4;
    res[5] = (DST_S5 == 0) ? 0
                           : dst_off / DST_S5 % div_up(DST_D5, DST_B5) * DST_B5
                    + dst_off / DST_SB5 % DST_B5;
}
#endif

__attribute__((overloadable)) void write_dst(
        __global DST_DATA_T *dst, DST_DATA_T val) {
    *dst = val;
}

__attribute__((overloadable)) DST_DATA_T load(__global DST_DATA_T *dst) {
    return *dst;
}

void write_padded_zeros(__global DST_DATA_T *dst) {
#if DST_Z0_IS_REDUCED && DST_Z1_IS_REDUCED
    for (int i = 0; i < DST_Z0_SIZE0; i++) {
        for (int j = 0; j < DST_Z1_SIZE0; j++) {
            if (i == 0 && j == 0) continue;
            write_dst(dst + i * DST_Z0_STRIDE0 + j * DST_Z1_STRIDE0,
                    TO_DST(0.0f));
        }
    }
#elif DST_Z0_IS_REDUCED
    for (int i = 1; i < DST_Z0_SIZE0; i++) {
        write_dst(dst + i * DST_Z0_STRIDE0, TO_DST(0.0f));
    }
#elif DST_Z1_IS_REDUCED
    for (int j = 1; j < DST_Z1_SIZE0; j++) {
        write_dst(dst + j * DST_Z1_STRIDE0, TO_DST(0.0f));
    }
#endif
}

#if INNER_DIM_SIZE < SUBGROUP_SIZE
#if INNER_DIM_SIZE == 0
#define SLM_PER_SG 1
#else
#define SLM_PER_SG INNER_DIM_SIZE
#endif // INNER_DIM_SIZE == 0
#else
#define SLM_PER_SG SUBGROUP_SIZE
#endif

// Specifying wg size since larger work groups reduce performance.
// TODO: Look into why this is the case
__attribute__((reqd_work_group_size(LWS_SIZE, 1, 1))) // attr:no-format
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) // attr:no-format
__kernel void
combined_reduce(
        __global SRC_DATA_T *src, __global DST_DATA_T *dst POST_OP_ARGS) {
    // Compute constants deriving from defined constants
    const int sg_per_inner_dim = div_up(INNER_DIM_SIZE, SUBGROUP_SIZE);
    const int red_per_sg
            = min(REDUCTION_SIZE, max(1, SUBGROUP_SIZE / INNER_DIM_SIZE));
    const int wg_reductions = LWS_SIZE / SUBGROUP_SIZE;
    const int other_reductions = red_per_sg * wg_reductions;
    const int num_horiz_reductions = REDUCTION_SIZE / other_reductions;
    const int tail_reductions = REDUCTION_SIZE % other_reductions;

    // Direct indices from gws
    const int sgid = get_sub_group_id();
    ASSUME(sgid < wg_reductions);
    ASSUME(sgid >= 0);
    const int tgid = get_global_id(0) / LWS_SIZE;
    const int inner_idx_start = (tgid % sg_per_inner_dim) * SUBGROUP_SIZE;
    const int outer_idx_start = tgid / sg_per_inner_dim * OUTER_TILE_SIZE;

    // Handle inner vector packing into subgroups
    const int sglid = get_sub_group_local_id();
    ASSUME(sglid < SUBGROUP_SIZE);
    ASSUME(sglid >= 0);
    const int inner_idx = (inner_idx_start + sglid) % INNER_DIM_SIZE;
    const int red_off_sg = (inner_idx_start + sglid) / INNER_DIM_SIZE;
    const int red_off_tg = red_off_sg + sgid * red_per_sg;

    const int active_channels = min(
            SUBGROUP_SIZE, red_per_sg * (INNER_DIM_SIZE - inner_idx_start));
    ASSUME(active_channels == SUBGROUP_SIZE || !WITH_BLOCK_READ);

    const int loop_stride = _SRC_OFF(0, other_reductions, 0);
    __local DEF_ACC_DATA_T slm_acc[SLM_PER_SG * wg_reductions];
    unroll_for(int oid = 0; oid < OUTER_TILE_SIZE; oid++) {
        const int outer_idx = outer_idx_start + oid;

        // ---- Work item (loop) reductions ----

        DEF_ACC_DATA_T acc;
        init_acc(REDUCTION_ALG, &acc);
        // Each thread reduces in a loop
        if (sglid < active_channels) {
            // red_off_tg - red_off_sg to get the starting point for the subgroup
            int src_off = _SRC_OFF(
                    outer_idx, red_off_tg - red_off_sg, inner_idx_start);
            if (!WITH_BLOCK_READ) src_off += sglid;
            for (int iters = num_horiz_reductions; iters > 0; --iters) {
                const DATA_T src_val = READ_DATA(src[src_off]);
                acc = reduce(
                        REDUCTION_ALG, acc, TO_DEF_ACC_DATA_T(src_val), POWER);
                DUMP("(iter +%d) src[%d] = %f\n", iters, src_off,
                        CONVERT_FLOAT_T(src_val));
                src_off += loop_stride;
            }
            if (red_off_tg < tail_reductions) {
                const DATA_T src_val = READ_DATA(src[src_off]);
                acc = reduce(
                        REDUCTION_ALG, acc, TO_DEF_ACC_DATA_T(src_val), POWER);
                DUMP("(tail) src[%d] = %f\n", src_off,
                        CONVERT_FLOAT_T(src_val));
            }
        }

        // ---- Subgroup (intra-thread) reductions ----
        // Reduces data-carrying channels from active_channels to num_dst_writes

        DEF_ACC_DATA_T init;
        init_acc(SECONDARY_REDUCTION_ALG, &init);
        unroll_for(int shift = INNER_DIM_SIZE; shift < active_channels;
                   shift *= 2) {
            DEF_ACC_DATA_T next
                    = intel_sub_group_shuffle_down(acc, init, shift);
            acc = reduce(SECONDARY_REDUCTION_ALG, acc, next, POWER);
            DUMP("(sg) acc from sglid %d: %f\n", sglid + shift,
                    convert_float(next));
        }

        // ---- Work group (inter-thread/SLM) reductions ----
        // Reduces data-carrying threads to 1 per thread group

        if (wg_reductions > 1) {
            const int local_idx = sgid * SLM_PER_SG + sglid;
            if (red_off_sg == 0 && inner_idx < INNER_DIM_SIZE) {
                slm_acc[local_idx] = acc;
            }
            init_acc(SECONDARY_REDUCTION_ALG, &acc);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (red_off_tg == 0) {
                unroll_for(int i = 0; i < wg_reductions; i++) {
                    const int idx = i * SLM_PER_SG + sglid;
                    acc = reduce(
                            SECONDARY_REDUCTION_ALG, acc, slm_acc[idx], POWER);
                    DUMP("(wg) acc from wg %d/%d: %f\n", i, idx,
                            convert_float(slm_acc[idx]));
                }
            }
        }

        const dim_t dst_off = _DST_OFF(outer_idx, inner_idx);

        // ---- Finalize results and clean up ----

        if (red_off_tg == 0) {
            float res = IS_FINAL ? finalize(REDUCTION_ALG, acc, DIV, POWER, EPS)
                                 : acc;
#if WITH_POST_OP
            float dst_val;
#if WITH_SUM
            dst_val = DST_TO_REF(load(dst + dst_off));
#endif // WITH_SUM
            int idxs[6];
            reverse_indexing(dst_off, idxs);

            // Only use post-ops on non-zero-padded elements
            if (idxs[0] < DST_D0 && idxs[1] < DST_D1 && idxs[2] < DST_D2
                    && idxs[3] < DST_D3 && idxs[4] < DST_D4
                    && idxs[5] < DST_D5) {
                APPLY_POST_OPS_SERIAL(res, float, dst_val, float, idxs[0], 1,
                        idxs[1], 1, idxs[2], 1, idxs[3], 1, idxs[4], 1, idxs[5],
                        1);
            }
#endif
            if (is_dst_zero_padded(dst_off)) res = 0.0f;
            write_dst(dst + dst_off, IS_FINAL ? TO_DST(res) : res);
            DUMP("Wrote dst[%ld] = %f\n", dst_off, res);
            write_padded_zeros(dst + dst_off);
            DUMP("dst[%ld] <- %f\n", dst_off, TO_DST(res));
        }
    }
}
