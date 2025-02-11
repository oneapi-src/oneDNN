/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "gpu/intel/ocl/ocl_eltwise.h"
#include "gpu/intel/ocl/ocl_io.h"
#include "gpu/intel/ocl/ocl_post_ops.h"

__attribute__((intel_reqd_sub_group_size(SIMD))) __kernel void gen9_eltwise_fwd(
        __global DATA_T *src, __global DATA_T *dst, dim_t nelems, float alpha,
        float beta) {
    const dim_t grsize = get_local_size(0);
    const dim_t grid = get_group_id(0);
    const dim_t sgid = get_sub_group_id();
    const dim_t lid = get_sub_group_local_id();

    const dim_t gid = get_global_id(0);

    dim_t offset
            = (grid * grsize + sgid * get_max_sub_group_size()) * VECT_DT_N;

    const int nel_per_read = SIMD * VECT_DT_N;

    if (!NELEMS_OVERFLOW || offset + nel_per_read < nelems) {
        POST_OP_DATA_T val[VECT_DT_N];
        block_load(val, src + offset, VECT_DT_N);
        for (int i = 0; i < VECT_DT_N; ++i) {
            val[i] = fwd_eltwise(val[i], alpha, beta, 1.0f);
        }
        block_write(dst + offset, val, VECT_DT_N);
    } else {
        dim_t pos = offset + lid;
        for (int i = 0; i < VECT_DT_N && pos < nelems; ++i) {
            POST_OP_DATA_T val = load(val, src + pos);
            val = fwd_eltwise(val, alpha, beta, 1.0f);
            write(dst + pos, val);
            pos += SIMD;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SIMD))) __kernel void gen9_eltwise_bwd(
        __global DATA_T *src, __global DATA_T *diff_src,
        __global DATA_T *diff_dst, dim_t nelems, float alpha, float beta) {
    const dim_t grsize = get_local_size(0);
    const dim_t grid = get_group_id(0);
    const dim_t sgid = get_sub_group_id();
    const dim_t lid = get_sub_group_local_id();

    dim_t offset = (grid * grsize + sgid * SIMD) * VECT_DT_N;
    const int nel_per_read = SIMD * VECT_DT_N;

    // READ
    if (!NELEMS_OVERFLOW || offset + nel_per_read < nelems) {
        POST_OP_DATA_T val_dd[VECT_DT_N];
        POST_OP_DATA_T val_src[VECT_DT_N];
        block_load(val_dd, diff_dst + offset, VECT_DT_N);
        block_load(val_src, src + offset, VECT_DT_N);
        for (int i = 0; i < VECT_DT_N; ++i) {
            val_dd[i] = bwd_eltwise(val_dd[i], val_src[i], alpha, beta);
        }
        block_write(diff_src + offset, val_dd, VECT_DT_N);
    } else {
        // read data in the same access pattern block_reads would
        dim_t pos = offset + lid;
        for (int i = 0; i < VECT_DT_N && pos < nelems; ++i) {
            POST_OP_DATA_T val_dd = load(val_dd, diff_dst + pos);
            POST_OP_DATA_T val_src = load(val_src, src + pos);
            val_dd = bwd_eltwise(val_dd, val_src, alpha, beta);
            write(diff_src + pos, val_dd);
            pos += SIMD;
        }
    }
}
