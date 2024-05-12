/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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
#include "gpu/intel/ocl/reduction/ocl_reduction.h"

// Define how to read data
#define BLOCK_READ_DATA_T(data_ptr) \
    AS_VECT_DATA_T(VECT_BLOCK_READ((const __global BLOCK_DATA_T *)data_ptr))
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

// Specifying wg size since larger work groups reduce performance.
// TODO: Look into why this is the case
__attribute__((reqd_work_group_size(LWS_SIZE, 1, 1))) // attr:no-format
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) // attr:no-format
__kernel void
combined_reduce(
        __global SRC_DATA_T *src, __global DST_DATA_T *dst POST_OP_ARGS) {
    // Compute constants deriving from defined constants
    const int sg_per_inner_dim
            = div_up(div_up(INNER_DIM_SIZE, VECT_DT_N), SUBGROUP_SIZE);
    const int inner_dims_per_sg
            = min(REDUCTION_SIZE, max(1, SUBGROUP_SIZE / INNER_DIM_SIZE));
    const int num_horiz_reductions = REDUCTION_SIZE / inner_dims_per_sg
            / (REDUCE_VECTOR ? VECT_DT_N : 1);
    const int tail_reductions = REDUCTION_SIZE % inner_dims_per_sg;

    // Direct indices from gws
    const int sgid = get_global_id(0) / SUBGROUP_SIZE;
    const int inner_idx_start
            = (sgid % sg_per_inner_dim) * SUBGROUP_SIZE * VECT_DT_N;

    // Handle inner vector packing into subgroups
    const int sglid = get_sub_group_local_id();
    const int inner_idx = inner_idx_start + (sglid % INNER_DIM_SIZE);
    const int red_off = sglid / INNER_DIM_SIZE;

    // Case happens when inner_dim_size is not a multiple/factor of subgroup size
    if (inner_idx >= INNER_DIM_SIZE
            || sglid >= INNER_DIM_SIZE * inner_dims_per_sg)
        return;

    const int loop_stride = _SRC_OFF(
            0, inner_dims_per_sg * (REDUCE_VECTOR ? VECT_DT_N : 1), 0);

    unroll_for(int oid = 0; oid < OUTER_TILE_SIZE; oid++) {
        const int outer_idx = sgid / sg_per_inner_dim * OUTER_TILE_SIZE + oid;
        DEF_ACC_DATA_T acc[VECT_DT_N];
        unroll_for(int v = 0; v < VECT_DT_N; v++) {
            init_acc(REDUCTION_ALG, &acc[v]);
        }

        int src_off = _SRC_OFF(outer_idx, WITH_BLOCK_READ ? 0 : red_off,
                WITH_BLOCK_READ ? inner_idx_start : inner_idx);
        __attribute__((opencl_unroll_hint(UNROLL_FACTOR))) // attr:no-format
        for (int off = 0; off < num_horiz_reductions;
                off++, src_off += loop_stride) {
            // Load
            const VECT_DATA_T src_val = READ_DATA(src[src_off]);
            const DATA_T *next_val = (DATA_T *)&src_val;

            // Accumulate
            unroll_for(int v = 0; v < VECT_DT_N; v++) {
                acc[v] = reduce(REDUCTION_ALG, acc[v],
                        TO_DEF_ACC_DATA_T(next_val[v]), POWER);
            }
        }
        if (red_off < tail_reductions) {
            // Load
            const VECT_DATA_T src_val = READ_DATA(src[src_off]);
            const DATA_T *next_val = (DATA_T *)&src_val;

            // Accumulate
            unroll_for(int v = 0; v < VECT_DT_N; v++) {
                acc[v] = reduce(REDUCTION_ALG, acc[v],
                        TO_DEF_ACC_DATA_T(next_val[v]), POWER);
            }
        }

        // Potentially accumulate within the subgroup too
        // TODO: Change to tree-based reduce to help large inner_dims_per_sg cases
        DEF_ACC_DATA_T acc_sg[VECT_DT_N];
        for (int v = 0; v < VECT_DT_N; v++) {
            init_acc(SECONDARY_REDUCTION_ALG, &acc_sg[v]);
        }
        unroll_for(int i = 0; i < inner_dims_per_sg; i++) {
            unroll_for(int v = 0; v < VECT_DT_N; v++) {
                DEF_ACC_DATA_T next = intel_sub_group_shuffle_down(acc[v],
                        SPECIAL(DEF_ACC_DATA_T, zero), i * INNER_DIM_SIZE);
                acc_sg[v] = reduce(
                        SECONDARY_REDUCTION_ALG, acc_sg[v], next, POWER);
            }
        }

        if (sglid < INNER_DIM_SIZE) {
            const int final_vec_size = REDUCE_VECTOR ? 1 : VECT_DT_N;
#if REDUCE_VECTOR
            DEF_ACC_DATA_T final_acc[1];
            init_acc(SECONDARY_REDUCTION_ALG, final_acc);
            unroll_for(int v = 0; v < VECT_DT_N; v++) {
                final_acc[0] = reduce(SECONDARY_REDUCTION_ALG, acc_sg[v],
                        final_acc[0], POWER);
            }
#else
            // Just rename the variable to match the REDUCE_VECTOR case
            DEF_ACC_DATA_T final_acc[VECT_DT_N];
            for (int v = 0; v < VECT_DT_N; v++) {
                final_acc[v] = acc_sg[v];
            }
#endif // REDUCE_VECTOR

            // For each result:
            // 1. (if IS_FINAL) finalize the result
            // 2. (if IS_FINAL) apply post-ops
            // 3. write to dst
            for (int v = 0; v < final_vec_size; v++) {
                const dim_t dst_off
                        = _DST_OFF(outer_idx, inner_idx + v * SUBGROUP_SIZE);
                // finalize the result
#if IS_FINAL
                float res = finalize(REDUCTION_ALG, convert_float(final_acc[v]),
                        DIV, POWER, EPS);

                // Apply post-ops
#if WITH_POST_OP
                float dst_val;
#if WITH_SUM
                dst_val = DST_TO_REF(dst[dst_off]);
#endif // WITH_SUM

                // Reconstruct MB/C/D/H/W indices from dst_off
                const int mb = (DST_S0 == 0)
                        ? 0
                        : dst_off / DST_S0 % div_up(DST_D0, DST_B0) * DST_B0
                                + dst_off / DST_SB0 % DST_B0;
                const int c = (DST_S1 == 0)
                        ? 0
                        : dst_off / DST_S1 % div_up(DST_D1, DST_B1) * DST_B1
                                + dst_off / DST_SB1 % DST_B1;
                const int d = (DST_S2 == 0)
                        ? 0
                        : dst_off / DST_S2 % div_up(DST_D2, DST_B2) * DST_B2
                                + dst_off / DST_SB2 % DST_B2;
                const int h = (DST_S3 == 0)
                        ? 0
                        : dst_off / DST_S3 % div_up(DST_D3, DST_B3) * DST_B3
                                + dst_off / DST_SB3 % DST_B3;
                const int w = (DST_S4 == 0)
                        ? 0
                        : dst_off / DST_S4 % div_up(DST_D4, DST_B4) * DST_B4
                                + dst_off / DST_SB4 % DST_B4;

                // Only use post-ops on non-zero-padded elements
                if (mb < DST_D0 && c < DST_D1 && d < DST_D2 && h < DST_D3
                        && w < DST_D4) {
                    APPLY_POST_OPS_SERIAL(res, float, dst_val, float, mb, 1, c,
                            1, d, 1, h, 1, w, 1, 0, 1);
                }
#endif // WITH_POST_OP
#else
                float res = final_acc[v];
#endif // IS_FINAL

                // Write to dst
                if (is_dst_zero_padded(dst_off)) res = 0.0f;
                dst[dst_off] = IS_FINAL ? TO_DST(res) : res;

                // Reduced + zero-padded dims need extra zeros written
#if DST_Z0_IS_REDUCED && DST_Z1_IS_REDUCED
                for (int i = 0; i < DST_Z0_SIZE0; i++) {
                    for (int j = 0; j < DST_Z1_SIZE0; j++) {
                        if (i == 0 && j == 0) continue;
                        dst[dst_off + i * DST_Z0_STRIDE0 + j * DST_Z1_STRIDE0]
                                = TO_DST(0.0f);
                    }
                }
#elif DST_Z0_IS_REDUCED
                for (int i = 1; i < DST_Z0_SIZE0; i++) {
                    dst[dst_off + i * DST_Z0_STRIDE0] = TO_DST(0.0f);
                }
#elif DST_Z1_IS_REDUCED
                for (int j = 1; j < DST_Z1_SIZE0; j++) {
                    dst[dst_off + j * DST_Z1_STRIDE0] = TO_DST(0.0f);
                }
#endif
            }
        }
    }
}
