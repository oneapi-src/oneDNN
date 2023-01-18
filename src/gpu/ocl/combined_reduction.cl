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

#define _SRC_OFF(outer, reduction, inner) \
    (outer) * REDUCTION_START_SIZE *INNER_DIM_SIZE \
            + (reduction)*INNER_DIM_SIZE + (inner)

#define _DST_OFF(outer, reduction_chunk, inner) \
    (outer) * INNER_DIM_SIZE *REDUCTION_END_SIZE \
            + (reduction_chunk)*INNER_DIM_SIZE + inner

KERNEL_ATTR
__kernel void combined_reduce(
        __global SRC_DATA_T *src, __global DST_DATA_T *dst) {
    // Direct indices from gws
    const int outer_idx = (get_global_id(0) / OUTER_DIM_STRIDE);
    const int reduction_chunk
            = (get_global_id(0) / PADDED_INNER_DIM_SIZE) % REDUCTION_CHUNK_SIZE;
    const int inner = (get_global_id(0) % PADDED_INNER_DIM_SIZE);

    // Handle padded GWS
    if (outer_idx >= OUTER_DIM_SIZE) return;

    // Only inner changes within a subgroup - unpack to inner_idx and reduction index
    // Break out components that change within each subgroup
    const int inner_idx = inner % INNER_DIM_SIZE;
    const int red_off = inner / INNER_DIM_SIZE;
    const int reduction_idx = reduction_chunk * REDUCTION_SIZE + red_off;

    // outer idx needs no sg adjusting
    const int reduction_idx_start = reduction_idx - red_off;
    const int inner_idx_start = inner - get_sub_group_local_id();

    // Deal with padded inner dims
    if (inner >= INNER_DIMS_PER_WI * INNER_DIM_SIZE) return;

    const int dst_off
            = _DST_OFF(outer_idx, reduction_chunk + red_off, inner_idx);
    DEF_ACC_DATA_T acc = INIT_ACC;

    int off = 0;
    for (; off < REDUCTIONS_PER_WI - 1; off++) {
        // Load
#if WITH_BLOCK_READ
        const int src_off = _SRC_OFF(outer_idx,
                off * INNER_DIMS_PER_WI + reduction_idx_start, inner_idx_start);
        const SRC_DATA_T src_val = AS_DATA_T(
                BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off]));
#else
        const int src_off = _SRC_OFF(
                outer_idx, off * INNER_DIMS_PER_WI + reduction_idx, inner_idx);
        const SRC_DATA_T src_val = src[src_off];
#endif
        // Accumulate
        acc = ACCUMULATE(acc, TO_DEF_ACC_DATA_T(src_val));
    }
    // Check final iteration -- some work items skip this one
    if (off * INNER_DIMS_PER_WI + reduction_idx < REDUCTION_START_SIZE) {
        // Load
#if WITH_BLOCK_READ
        const int src_off = _SRC_OFF(outer_idx,
                off * INNER_DIMS_PER_WI + reduction_idx_start, inner_idx_start);
        const SRC_DATA_T src_val = AS_DATA_T(
                BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off]));
#else
        const int src_off = _SRC_OFF(
                outer_idx, off * INNER_DIMS_PER_WI + reduction_idx, inner_idx);
        const SRC_DATA_T src_val = src[src_off];
#endif
        // Accumulate
        acc = ACCUMULATE(acc, TO_DEF_ACC_DATA_T(src_val));
    }

    // Potentially accumulate within the subgroup too
    // TODO: Change to tree-based reduce to help large INNER_DIMS_PER_WI cases
    unroll_for(int i = 1; i < INNER_DIMS_PER_WI; i++) {
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
