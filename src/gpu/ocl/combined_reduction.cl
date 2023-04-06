/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

// Initialize for different algorithms
#if defined(IS_MAX)
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_MIN)
#elif defined(IS_MIN)
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_MAX)
#elif defined(IS_MUL)
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_ONE)
#else
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_ZERO)
#endif

// Integer data types have different max/min functions
#if defined(SRC_DT_S8) || defined(SRC_DT_U8) || defined(SRC_DT_S32)
#define MAX_FUNC max
#define MIN_FUNC min
#else
#define MAX_FUNC fmax
#define MIN_FUNC fmin
#endif

// Define accumulation functions
#if defined(IS_MAX)
#define ACCUMULATE_INITIAL(x, y) MAX_FUNC(x, y)
#elif defined(IS_MIN)
#define ACCUMULATE_INITIAL(x, y) MIN_FUNC(x, y)
#elif defined(IS_MEAN) || defined(IS_SUM)
#define ACCUMULATE_INITIAL(x, y) (x + y)
#elif defined(IS_MUL)
#define ACCUMULATE_INITIAL(x, y) (x * y)
#else
#define ACCUMULATE_INITIAL(x, y) (x + pow(fabs(y), POWER))
#endif

// Define secondary accumulation functions
#if defined(IS_MAX) || defined(IS_MIN) || defined(IS_MEAN) || defined(IS_MUL)
#define ACCUMULATE_FURTHER ACCUMULATE_INITIAL
#else
#define ACCUMULATE_FURTHER(x, y) (x + y)
#endif

// Define which accumulate function we use
#if IS_FIRST
#define ACCUMULATE ACCUMULATE_INITIAL
#else
#define ACCUMULATE ACCUMULATE_FURTHER
#endif

// Finalize reduction at the end of the kernel
#if defined(IS_MEAN)
#define FINALIZE(x) (x / DIV)
#elif defined(IS_LP_MAX)
#define FINALIZE(x) rootn(fmax(x, EPS), POWER)
#elif defined(IS_LP_SUM)
#define FINALIZE(x) rootn(x + EPS, POWER)
#elif defined(IS_P_MAX)
#define FINALIZE(x) fmax(x, EPS)
#elif defined(IS_P_SUM)
#define FINALIZE(x) (x + EPS)
#else
#define FINALIZE(x) (x)
#endif

#define BLOCK_READ_DATA_T(data_ptr) \
    AS_VECT_DATA_T(VECT_BLOCK_READ((const __global BLOCK_DATA_T *)data_ptr))

// Define how to read data
#define BLOCK_READ_DATA_T(data_ptr) \
    AS_VECT_DATA_T(VECT_BLOCK_READ((const __global BLOCK_DATA_T *)data_ptr))
#define READ_DATA(val) WITH_BLOCK_READ ? BLOCK_READ_DATA_T(&val) : val

#define _SRC_OFF(outer, reduction, inner) \
    (outer) * REDUCTION_SIZE *INNER_DIM_SIZE + (reduction)*INNER_DIM_SIZE \
            + (inner)

#define _DST_OFF(outer, inner) (outer) * INNER_DIM_SIZE + (inner)

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
    const int outer_idx = sgid / sg_per_inner_dim;

    // Handle inner vector packing into subgroups
    const int sglid = get_sub_group_local_id();
    const int inner_idx = inner_idx_start + (sglid % INNER_DIM_SIZE);
    const int red_off = sglid / INNER_DIM_SIZE;

    // Case happens when inner_dim_size is not a multiple/factor of subgroup size
    if (inner_idx >= INNER_DIM_SIZE
            || sglid >= INNER_DIM_SIZE * inner_dims_per_sg)
        return;

    const int dst_off = _DST_OFF(outer_idx, inner_idx);
    VECT_DEF_ACC_DATA_T acc = INIT_ACC;

    const int loop_stride = _SRC_OFF(
            0, inner_dims_per_sg * (REDUCE_VECTOR ? VECT_DT_N : 1), 0);
    int src_off = _SRC_OFF(outer_idx, WITH_BLOCK_READ ? 0 : red_off,
            WITH_BLOCK_READ ? inner_idx_start : inner_idx);
    __attribute__((opencl_unroll_hint(UNROLL_FACTOR))) // attr:no-format
    for (int off = 0; off < num_horiz_reductions;
            off++, src_off += loop_stride) {
        // Load
        const VECT_DATA_T src_val = READ_DATA(src[src_off]);

        // Accumulate
        acc = ACCUMULATE(acc, AS_VECT_DEF_ACC_DATA_T(src_val));
    }
    if (red_off < tail_reductions) {
        // Load
        const VECT_DATA_T src_val = READ_DATA(src[src_off]);

        // Accumulate
        acc = ACCUMULATE(acc, AS_VECT_DEF_ACC_DATA_T(src_val));
    }

    // Potentially accumulate within the subgroup too
    // TODO: Change to tree-based reduce to help large inner_dims_per_sg cases
    unroll_for(int i = 1; i < inner_dims_per_sg; i++) {
        const VECT_DEF_ACC_DATA_T other
                = intel_sub_group_shuffle_down(acc, INIT_ACC, INNER_DIM_SIZE);
        if (get_sub_group_local_id() < INNER_DIM_SIZE) {
            acc = ACCUMULATE_FURTHER(acc, other);
        } else {
            acc = other; // For further passing down
        }
    }

    // If the vector of results should be reduced as well, do it now.
    // After this, the vector size will either be 1 or VECT_DT_N (stored in final_vect_size)
    if (get_sub_group_local_id() < INNER_DIM_SIZE) {
#if REDUCE_VECTOR
        const int final_vect_size = 1;
        DEF_ACC_DATA_T final_acc[1] = {INIT_ACC};
        for (int i = 0; i < VECT_DT_N; i++) {
            final_acc[0] = ACCUMULATE_FURTHER(acc[i], final_acc[0]);
        }
#else
        const int final_vect_size = VECT_DT_N;
        const DEF_ACC_DATA_T *final_acc = &acc;
#endif // REDUCE_VECTOR

        // For each result:
        // 1. (if IS_FINAL) finalize the result
        // 2. (if IS_FINAL) apply post-ops
        // 3. write to dst
        for (int i = 0; i < final_vect_size; i++) {
            const int dst_offi = dst_off + _DST_OFF(0, i * SUBGROUP_SIZE);
            // finalize the result
#if IS_FINAL
            float res = FINALIZE(convert_float(final_acc[i]));

            // Apply post-ops
#if WITH_POST_OP
            float dst_val;
#if WITH_SUM
            dst_val = DST_TO_REF(dst[dst_offi]);
#endif // WITH_SUM

            // Reconstruct MB/C/D/H/W indices from dst_offi
            const int mb = (DST_S0 == 0)
                    ? 0
                    : dst_offi / DST_S0 % div_up(DST_D0, DST_B0) * DST_B0
                            + dst_offi / DST_SB0 % DST_B0;
            const int c = (DST_S1 == 0)
                    ? 0
                    : dst_offi / DST_S1 % div_up(DST_D1, DST_B1) * DST_B1
                            + dst_offi / DST_SB1 % DST_B1;
            const int d = (DST_S2 == 0)
                    ? 0
                    : dst_offi / DST_S2 % div_up(DST_D2, DST_B2) * DST_B2
                            + dst_offi / DST_SB2 % DST_B2;
            const int h = (DST_S3 == 0)
                    ? 0
                    : dst_offi / DST_S3 % div_up(DST_D3, DST_B3) * DST_B3
                            + dst_offi / DST_SB3 % DST_B3;
            const int w = (DST_S4 == 0)
                    ? 0
                    : dst_offi / DST_S4 % div_up(DST_D4, DST_B4) * DST_B4
                            + dst_offi / DST_SB4 % DST_B4;

            // Only use post-ops on non-zero-padded elements
            if (mb < DST_D0 && c < DST_D1 && d < DST_D2 && h < DST_D3
                    && w < DST_D4) {
                APPLY_POST_OPS_SERIAL(res, float, dst_val, float, mb, 1, c, 1,
                        d, 1, h, 1, w, 1, 0, 1);
            }
#endif // WITH_POST_OP
#else
            const float res = final_acc[i];
#endif // IS_FINAL

            // Write to dst
            dst[dst_offi] = IS_FINAL ? TO_DST(res) : res;
        }
    }
}
