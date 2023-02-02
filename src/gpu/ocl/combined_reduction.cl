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
    AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)data_ptr))

#define _SRC_OFF(outer, reduction, inner) \
    (outer) * REDUCTION_SIZE *INNER_DIM_SIZE + (reduction)*INNER_DIM_SIZE \
            + (inner)

#define _DST_OFF(outer, inner) (outer) * INNER_DIM_SIZE + (inner)

// Specifying wg size since larger work groups reduce performance.
// TODO: Look into why this is the case
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, 1, 1))) // attr:no-format
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) // attr:no-format
__kernel void
combined_reduce(__global SRC_DATA_T *src, __global DST_DATA_T *dst) {
    // Compute constants deriving from defined constants
    const int sg_per_inner_dim = div_up(INNER_DIM_SIZE, SUBGROUP_SIZE);
    const int inner_dims_per_sg
            = min(REDUCTION_SIZE, max(1, SUBGROUP_SIZE / INNER_DIM_SIZE));
    const int num_horiz_reductions = div_up(REDUCTION_SIZE, inner_dims_per_sg);

    // Direct indices from gws
    const int sgid = get_global_id(0) / SUBGROUP_SIZE;
    const int inner_idx_start = (sgid % sg_per_inner_dim) * SUBGROUP_SIZE;
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
    DEF_ACC_DATA_T acc = INIT_ACC;

    int src_off = _SRC_OFF(
            outer_idx, red_off, WITH_BLOCK_READ ? inner_idx_start : inner_idx);
    int off = 0;
    for (; off < num_horiz_reductions - 1;
            off++, src_off += _SRC_OFF(0, inner_dims_per_sg, 0)) {
        // Load
        const DATA_T src_val = WITH_BLOCK_READ
                ? BLOCK_READ_DATA_T(&src[src_off])
                : src[src_off];

        // Accumulate
        acc = ACCUMULATE(acc, TO_DEF_ACC_DATA_T(src_val));
    }

    // Check final iteration -- some work items skip this one
    if (off * inner_dims_per_sg + red_off < REDUCTION_SIZE) {
        // Load
        const DATA_T src_val = WITH_BLOCK_READ
                ? BLOCK_READ_DATA_T(&src[src_off])
                : src[src_off];
        // Accumulate
        acc = ACCUMULATE(acc, TO_DEF_ACC_DATA_T(src_val));
    }

    // Potentially accumulate within the subgroup too
    // TODO: Change to tree-based reduce to help large inner_dims_per_sg cases
    unroll_for(int i = 1; i < inner_dims_per_sg; i++) {
        const DEF_ACC_DATA_T other
                = intel_sub_group_shuffle_down(acc, INIT_ACC, INNER_DIM_SIZE);
        if (get_sub_group_local_id() < INNER_DIM_SIZE) {
            acc = ACCUMULATE_FURTHER(acc, other);
        } else {
            acc = other; // For further passing down
        }
    }

    if (get_sub_group_local_id() < INNER_DIM_SIZE) {
#if IS_FINAL

        float res = convert_float(acc);
        res = FINALIZE(res);

        dst[dst_off] = TO_DST(res);
#else
        dst[dst_off] = acc;
#endif
    }
}
