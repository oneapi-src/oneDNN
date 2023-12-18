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

#include "gpu/ocl/dispatch.h"
#include "gpu/ocl/ocl_types.h"
#include "gpu/ocl/types_interop.h"

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

#define REDUCTION_WI_COUNT (ATOMIC_REDUCTION_SIZE * LOCAL_SIZE)
#define SRC_SIMD_STRIDE (INNER_DIM_SIZE * REDUCTION_WI_COUNT)

KERNEL_ATTR
__kernel void atomic_reduce(__global SRC_DATA_T *src,
        __global ATOMIC(DST_DATA_T) * dst, off_t num_reductions,
        dispatch_gws_rt_params_t gws_params) {
    const uint local_idx = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();
    const uint subgroup_size = get_max_sub_group_size();

    // src inner dim is split into the subgroup, vectorization, and inner_group blocks
    // subgroup and inner_group blocks are dispatched, and the vectorization block
    // is handled by a single work item. Since the vectorization size is compiled in,
    // update offsets using it: the inner_group stride is subgroup_size * vec_size
    off_t SRC_OFF = GWS_GET_OFF(SRC, gws_params) - sglid;
    off_t src_ig_idx = SRC_OFF / subgroup_size;
    SRC_OFF += src_ig_idx * (VECT_DT_N - 1) * subgroup_size;
    src += SRC_OFF;

    // Do the same for dst
    off_t DST_OFF = GWS_GET_OFF(DST, gws_params);
    off_t dst_ig_idx = DST_OFF / subgroup_size;
    DST_OFF += dst_ig_idx * (VECT_DT_N - 1) * subgroup_size;
    dst += DST_OFF;

    off_t red_idx = GWS_GET_OFF(REDUCE, gws_params);

    VECT_DEF_ACC_DATA_T acc = INIT_ACC;
    int i = 0;
    unroll_16_for(; i < num_reductions / REDUCTION_WI_COUNT; i++) {
        const int src_off = i * SRC_SIMD_STRIDE;
        const VECT_DATA_T src_val = BLOCK_READ_DATA_T(&src[src_off]);
        acc = ACCUMULATE(acc, AS_VECT_DEF_ACC_DATA_T(src_val));
    }

    // Check the final iteration, some SIMD reductions may still be needed
    if (i * REDUCTION_WI_COUNT + red_idx < num_reductions) {
        const int src_off = i * SRC_SIMD_STRIDE;
        const VECT_DATA_T src_val = BLOCK_READ_DATA_T(&src[src_off]);
        acc = ACCUMULATE(acc, AS_VECT_DEF_ACC_DATA_T(src_val));
    }

#if LOCAL_SIZE > 1
    // Store results to SLM
    __local DEF_ACC_DATA_T
            local_acc_buf[LOCAL_SIZE][GWS_SGS_DEFAULT * VECT_DT_N];
    unroll_for(int i = 0; i < VECT_DT_N; i++) {
        local_acc_buf[local_idx][sglid + i * subgroup_size] = GET_ELEM(acc, i);
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
                        = local_acc_buf[slm_off][sglid + v * subgroup_size];
                GET_ELEM(local_acc, v)
                        = ACCUMULATE_FURTHER(GET_ELEM(local_acc, v), slm_data);
            }
        }
#else
    {
        VECT_DEF_ACC_DATA_T local_acc = acc;
#endif

        // Finalize data, then (atomically) accumulate into to dst
        for (int v = 0; v < VECT_DT_N; v++) {
#if ATOMIC_REDUCTION_SIZE > 1
            DST_DATA_T dst_data = TO_DST(GET_ELEM(local_acc, v));
            DST_DATA_T old_val
                    = ATOMIC_ACCUMULATE(&dst[v * subgroup_size], dst_data);
#else
            DST_DATA_T dst_data
                    = TO_DST(FINALIZE(convert_float(GET_ELEM(local_acc, v))));
            dst[v * subgroup_size] = dst_data;
#endif
        }
    }
}
