/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "gpu/ocl/ocl_types.h"

#define LOAD_DATA_8x16(ptr) \
    CONVERT_FLOAT8_T( \
            AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)(ptr))))

#define STORE_DATA_8x16(ptr, val) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)ptr, \
            AS_BLOCK_DATA8_T(CONVERT_DATA8_T(val)))

#define VECT_SIZE 8
#define NUM_BUF (SOFTMAX_AXIS_SIZE / SUB_GROUP_SIZE / VECT_SIZE)

#if IS_FWD

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_softmax_fwd(__global DATA_T *src, __global DATA_T *dst) {
    const int data_off = (get_global_id(0) / GROUP_SIZE) * SOFTMAX_AXIS_SIZE;

    float8 d[NUM_BUF];

    float max_ = -FLT_MAX;
    float denom_ = 0.f;

    src += data_off;

    for (int k = 0; k < NUM_BUF; ++k) {
        d[k] = LOAD_DATA_8x16(&src[k * VECT_SIZE * SUB_GROUP_SIZE]);
        for (int i = 0; i < VECT_SIZE; ++i) {
            max_ = max(d[k][i], max_);
        }
    }

    max_ = sub_group_reduce_max(max_);

    for (int k = 0; k < NUM_BUF; ++k) {
#if LOGSOFTMAX
        for (int i = 0; i < VECT_SIZE; ++i)
            denom_ += exp(d[k][i] - max_);
#else
        d[k] = exp(d[k] - max_);
        for (int i = 0; i < VECT_SIZE; ++i)
            denom_ += d[k][i];
#endif
    }

    denom_ = sub_group_reduce_add(denom_);

#if LOGSOFTMAX
    denom_ = log(denom_);
#else
    denom_ = 1.0 / denom_;
#endif

    dst += data_off;
    for (int k = 0; k < NUM_BUF; ++k) {
#if LOGSOFTMAX
        d[k] = d[k] - max_ - denom_;
#else
        d[k] = d[k] * denom_;
#endif
        STORE_DATA_8x16(&dst[k * VECT_SIZE * SUB_GROUP_SIZE], d[k]);
    }
}

#endif
#if IS_BWD
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_softmax_bwd(__global DATA_T *dst, __global DATA_T *diff_src,
        __global DATA_T *diff_dst) {
#if IS_NHWC || IS_16C
    const int groups = get_global_id(0) / GROUP_SIZE;
    const int batch = groups / IC_PADDED;
    const int ic = groups % IC_PADDED;
    const int sub_grp_id = get_sub_group_local_id();
    const int local_id = get_local_id(0);
    const int slice = local_id / SUB_GROUP_SIZE;
    const int ic_buff = IC * VECT_SIZE;

#if IS_16C
    const int ic_blk = ic / IC;
    const int ic_in_blk = ic % IC;
    int data_off = BATCH * IC * SOFTMAX_AXIS_SIZE * ic_blk
            + batch * IC * SOFTMAX_AXIS_SIZE + ic_buff * sub_grp_id + ic_in_blk;
#else
    int data_off = batch * IC * SOFTMAX_AXIS_SIZE + ic_buff * sub_grp_id + ic;
#endif

    float sbr = 0.f;
    float diff_d[VECT_SIZE];
    float dst_[VECT_SIZE];

    diff_dst += data_off;
    dst += data_off;

    for (int i = 0, idx = IC * slice * SOFTMAX_BUF; i < VECT_SIZE;
            ++i, idx += IC) {
        diff_d[i] = CONVERT_FLOAT_T(diff_dst[idx]);
        dst_[i] = CONVERT_FLOAT_T(dst[idx]);
#if LOGSOFTMAX
        sbr += diff_d[i];
#else
        sbr += dst_[i] * diff_d[i];
#endif
    }
#if GROUP_SIZE == SUB_GROUP_SIZE
    sbr = sub_group_reduce_add(sbr);
#else
    sbr = work_group_reduce_add(sbr);
#endif
    diff_src += data_off;

    for (int i = 0, idx = IC * slice * SOFTMAX_BUF; i < VECT_SIZE;
            ++i, idx += IC) {
#if LOGSOFTMAX
        diff_d[i] = diff_d[i] - exp(dst_[i]) * sbr;
#else
        diff_d[i] = (diff_d[i] - sbr) * dst_[i];
#endif
        diff_src[idx] = TO_DATA_T(diff_d[i]);
    }

#else
    const int data_off = (get_global_id(0) / GROUP_SIZE) * SOFTMAX_AXIS_SIZE;

    float sbr = 0.f;

    float8 diff_d[NUM_BUF];
    float8 dst_[NUM_BUF];

    diff_dst += data_off;
    dst += data_off;
    for (int k = 0; k < NUM_BUF; ++k) {
        diff_d[k] = LOAD_DATA_8x16(&diff_dst[k * VECT_SIZE * SUB_GROUP_SIZE]);
        dst_[k] = LOAD_DATA_8x16(&dst[k * VECT_SIZE * SUB_GROUP_SIZE]);
        for (int i = 0; i < VECT_SIZE; ++i) {
#if LOGSOFTMAX
            sbr += diff_d[k][i];
#else
            sbr += dst_[k][i] * diff_d[k][i];
#endif
        }
    }

    sbr = sub_group_reduce_add(sbr);

    diff_src += data_off;

    for (int k = 0; k < NUM_BUF; ++k) {
#if LOGSOFTMAX
        diff_d[k] = diff_d[k] - exp(dst_[k]) * sbr;
#else
        diff_d[k] = (diff_d[k] - sbr) * dst_[k];
#endif
        STORE_DATA_8x16(&diff_src[k * VECT_SIZE * SUB_GROUP_SIZE], diff_d[k]);
    }
#endif
}
#endif
