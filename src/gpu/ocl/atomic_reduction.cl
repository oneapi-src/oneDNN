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
#define ATOMIC_ACCUMULATE(atomic_p, data) atomic_max_global(atomic_p, data)
#elif defined(IS_MIN)
#define ACCUMULATE_INITIAL(x, y) MIN_FUNC(x, y)
#define ATOMIC_ACCUMULATE(atomic_p, data) atomic_min_global(atomic_p, data)
#elif defined(IS_MEAN) || defined(IS_SUM)
#define ACCUMULATE_INITIAL(x, y) (x + y)
#define ATOMIC_ACCUMULATE(atomic_p, data) atomic_add_global(atomic_p, data)
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

// Finalize only applies in the final stage
#if !IS_FINAL
#undef FINALIZE
#define FINALIZE(x) (x)
#endif

#if ATOMIC_REDUCTION_SIZE > 1
#define ATOMIC(x) CONCAT2(atomic_, x)
#else
#define ATOMIC(x) x
#endif

// Define how to read data
#define BLOCK_READ_DATA_T(data_ptr) \
    AS_VECT_DATA_T(VECT_BLOCK_READ((const __global BLOCK_DATA_T *)data_ptr))

#if VECT_DT_N == 1
#define GET_ELEM(x, idx) (x)
#else
#define GET_ELEM(x, idx) x[idx]
#endif

#define WG_PER_INNER (INNER_DIM_SIZE / VECT_DT_N / SUBGROUP_SIZE)

#define SRC_ATOMIC_STRIDE INNER_DIM_SIZE
#define SRC_LOCAL_STRIDE (SRC_ATOMIC_STRIDE * ATOMIC_REDUCTION_SIZE)
#define SRC_SIMD_STRIDE (SRC_LOCAL_STRIDE * LOCAL_SIZE)
#define SRC_INNER_STRIDE (SUBGROUP_SIZE * VECT_DT_N)
#define SRC_OUTER_STRIDE (REDUCTION_SIZE * INNER_DIM_SIZE)

#define DST_INNER_STRIDE (SUBGROUP_SIZE * VECT_DT_N)
#define DST_OUTER_STRIDE INNER_DIM_SIZE

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) // attr:no-format
__kernel void
atomic_reduce(__global SRC_DATA_T *src, __global ATOMIC(DST_DATA_T) * dst) {
    // GWS dim 0 is just subgroup local ID
    const int sglid = get_global_id(0);

    // GWS dim 1 is combined wg_inner and wg_outer indices
    const int inner_idx = get_global_id(1) % WG_PER_INNER;
    const int outer_idx = get_global_id(1) / WG_PER_INNER;

    // GWS dim 2 is combined atomic/local reduction indices
    const int local_idx = get_global_id(2) % LOCAL_SIZE;
    const int atomic_idx
            = (get_global_id(2) / LOCAL_SIZE) % ATOMIC_REDUCTION_SIZE;

    const int src_off_start = atomic_idx * SRC_ATOMIC_STRIDE
            + local_idx * SRC_LOCAL_STRIDE + inner_idx * SRC_INNER_STRIDE
            + outer_idx * SRC_OUTER_STRIDE;
    VECT_DEF_ACC_DATA_T acc = INIT_ACC;
    int i = 0;
    for (; i < REDUCTION_SIZE / ATOMIC_REDUCTION_SIZE / LOCAL_SIZE; i++) {
        const int src_off = src_off_start + i * SRC_SIMD_STRIDE;
        const VECT_DATA_T src_val = BLOCK_READ_DATA_T(&src[src_off]);
        acc = ACCUMULATE(acc, AS_VECT_DEF_ACC_DATA_T(src_val));
    }

    // Check the final iteration, some SIMD reductions may still be needed
    const int src_off = src_off_start + i * SRC_SIMD_STRIDE;
    if (src_off < (outer_idx + 1) * SRC_OUTER_STRIDE) {
        const VECT_DATA_T src_val = BLOCK_READ_DATA_T(&src[src_off]);
        acc = ACCUMULATE(acc, AS_VECT_DEF_ACC_DATA_T(src_val));
    }

#if LOCAL_SIZE > 1
    // Store results to SLM
    __local DEF_ACC_DATA_T local_acc_buf[LOCAL_SIZE][SUBGROUP_SIZE * VECT_DT_N];
    unroll_for(int i = 0; i < VECT_DT_N; i++) {
        local_acc_buf[local_idx][sglid + i * SUBGROUP_SIZE] = GET_ELEM(acc, i);
    }

    // Wait for all subgroups to finish
    barrier(CLK_LOCAL_MEM_FENCE);

    // In the first subgroup of each work group:
    if (local_idx == 0) {
        // Perform the SLM reduction
        VECT_DEF_ACC_DATA_T local_acc = acc;

        for (int slm_off = 1; slm_off < LOCAL_SIZE; slm_off++) {
            for (int v = 0; v < VECT_DT_N; v++) {
                DEF_ACC_DATA_T slm_data
                        = local_acc_buf[slm_off][sglid + v * SUBGROUP_SIZE];
                GET_ELEM(local_acc, v)
                        = ACCUMULATE_FURTHER(GET_ELEM(local_acc, v), slm_data);
            }
        }
#else
    {
        VECT_DEF_ACC_DATA_T local_acc = acc;
#endif

        // Finalize data, then (atomically) accumulate into to dst
        const int dst_off = outer_idx * DST_OUTER_STRIDE
                + inner_idx * DST_INNER_STRIDE + sglid;
        for (int v = 0; v < VECT_DT_N; v++) {
#if ATOMIC_REDUCTION_SIZE > 1
            DST_DATA_T dst_data = TO_DST(convert_float(GET_ELEM(local_acc, v)));
            DST_DATA_T old_val = ATOMIC_ACCUMULATE(
                    &dst[dst_off + v * SUBGROUP_SIZE], dst_data);
#else
            DST_DATA_T dst_data
                    = TO_DST(FINALIZE(convert_float(GET_ELEM(local_acc, v))));
            dst[dst_off + v * SUBGROUP_SIZE] = dst_data;
#endif
        }
    }
}
