/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_RNN_CELL_COMPUTE_H
#define GPU_INTEL_OCL_RNN_CELL_COMPUTE_H

#include "gpu/intel/ocl/ocl_conversion.h"
#include "gpu/intel/ocl/rnn/rnn_common.h"

#if CELL_COMP_ENABLED
#define DHC_TG get_local_size(0)
#define DHC_LOCAL (CELL_DHC_THR * DHC_TG)

#define BATCH_TG get_local_size(1)
#define BATCH_LOCAL (CELL_BATCH_THR * BATCH_TG)

#define M_THR_BLOCK CELL_BATCH_THR
#define N_THR_BLOCK CELL_DHC_THR
#define N_OUTER_BLOCK n_gates

#define K_TG_BLOCK SUBGROUP_SIZE
#define K_THR_BLOCK 1
const int gemm_k_block = SUBGROUP_SIZE;
typedef int64_t cell_offset_t;
typedef int64_t grid_offset_t;
typedef dim_t cell_dim_t;

typedef struct {
    cell_dim_t m;
    cell_dim_t k;
    cell_dim_t n_inner;
    cell_dim_t n_outer;
} gemm_dims_t;

typedef struct {
    bool m;
    bool k;
    bool n;
} gemm_overflow_t;

// Handles strides required in computation. A different structure is used
// so that there is a consistent type used for data transfer. Because of this,
// we can easily change the offset type used to improve performance when
// overflow is not a concern. The only dimensions iterated over iter, mb, and
// channel. The channel dimension is required to be dense, so is dropped from
// these structures.
typedef struct {
    cell_offset_t mb;
    cell_offset_t slc;
    cell_offset_t sic;
} cell_strides_t;

typedef struct {
    grid_offset_t iter;
    cell_strides_t cell;
} grid_strides_t;

typedef struct {
    __global const WEI_LAYER_DATA_T *ptr;
    cell_strides_t strides;
} const_wei_layer_cell_t;

typedef struct {
    __global const WEI_ITER_DATA_T *ptr;
    cell_strides_t strides;
} const_wei_iter_cell_t;

typedef struct {
    __global AUX_DATA_T *ptr;
    cell_strides_t strides;
} aux_cell_t;

typedef struct {
    __global const AUX_DATA_T *ptr;
    cell_strides_t strides;
} const_aux_cell_t;

typedef struct {
    __global WS_STATE_DATA_T *ptr;
    cell_strides_t strides;
} ws_state_cell_t;

typedef struct {
    __global const WS_STATE_DATA_T *ptr;
    cell_strides_t strides;
} const_ws_state_cell_t;

typedef struct {
    cell_dim_t mb;
    cell_dim_t dhc;
    cell_dim_t slc;
    cell_dim_t sic;
} cell_dims_t;

// Cell function data used in non-GEMM calculation
typedef struct {
    union {
        struct {
            float alpha;
            __global BIAS_DATA_T *bias;
            __global float *tm_scales;
        } rnn;

        struct {
            __global AUX_DATA_T *c_states;
            __global const AUX_DATA_T *c_states_iter;
            __global BIAS_DATA_T *bias;
            __global float *tm_scales;
            float tm_cscale;
        } lstm;
    };
} cell_ctx_t;

typedef struct {
    cell_dim_t mb;
    cell_dim_t dhc;
} cell_loops_t;

#define NEED_SCRATCH_GATES \
    (!(CELL_COMPUTE_GEMM_LAYER && CELL_COMPUTE_GEMM_ITER))

inline void __attribute__((overloadable))
load(float *s, const __global float *data, bool is_valid) {
    *s = is_valid ? data[get_sub_group_local_id()] : 0;
}

inline void __attribute__((overloadable))
load(float *s, const __global half *data, bool is_valid) {
    *s = is_valid ? into_float(data[get_sub_group_local_id()]) : 0;
}

// Bfloat 16
inline void __attribute__((overloadable))
load(float *s, const __global ushort *data, bool is_valid) {
    *s = is_valid ? into_float(as_bf16(data[get_sub_group_local_id()])) : 0;
}

inline float __attribute__((overloadable)) sg_get(float s, int offset) {
    return intel_sub_group_shuffle(s, offset);
}

inline bool __attribute__((overloadable))
sg_get(bool s[gemm_k_block / SUBGROUP_SIZE], int offset) {
    return intel_sub_group_shuffle(
            s[offset / SUBGROUP_SIZE], offset % SUBGROUP_SIZE);
}
#endif
#endif
