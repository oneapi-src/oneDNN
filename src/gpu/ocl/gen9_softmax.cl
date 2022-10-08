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

#define LOAD_FLOAT8(prefix, ptr) \
    DATA_TO_FLOAT8(prefix, \
            BLOCK_TO_DATA8(prefix, \
                    READ_BLOCK8(prefix, \
                            (__global BLOCK_T(ALIAS(prefix)) *)(ptr))))

#define STORE_FLOAT8(prefix, ptr, val) \
    WRITE_BLOCK8(prefix, (__global BLOCK_T(ALIAS(prefix)) *)(ptr), \
            DATA_TO_BLOCK8(prefix, FLOAT_TO_DATA8(prefix, val)))

#define VECT_SIZE 8
#define NUM_BUF (SOFTMAX_AXIS_SIZE / SUB_GROUP_SIZE / VECT_SIZE)

#if IS_FWD

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_softmax_fwd(__global SRC_DATA_T *src, __global DST_DATA_T *dst,
        __global float *scale) {
#if IS_NHWC || IS_BLOCKED
    // gws is the combination of mb and axis size
    const int group = get_global_id(0) / GROUP_SIZE;
    const int mb = group / OC_PADDED;
    const int local_oc = group % OC_PADDED;

    const int local_id = get_local_id(0);
    const int axis_chunk = (local_id / SUB_GROUP_SIZE) * SOFTMAX_BUF;

    const int subgroup_id = get_sub_group_local_id();
    const int oc_chunk = OC * VECT_SIZE * subgroup_id;

#if IS_BLOCKED
    const int oc_block_id = local_oc / OC;
    const int oc_in_block = local_oc % OC;

    int data_off = (MB * oc_block_id + mb) * OC * SOFTMAX_AXIS_SIZE + oc_chunk
            + oc_in_block;
#else
    int data_off = mb * OC_PADDED * SOFTMAX_AXIS_SIZE + oc_chunk + local_oc;
#endif

    float d[VECT_SIZE];
    float max_ = -FLT_MAX;
    float denom_ = 0.f;

    src += data_off;
    for (int k = 0, axis_channel_id = OC * axis_chunk; k < VECT_SIZE;
            ++k, axis_channel_id += OC) {
        d[k] = DATA_TO_FLOAT(SRC, src[axis_channel_id]);
        max_ = max(d[k], max_);
    }

#if GROUP_SIZE == SUB_GROUP_SIZE
    max_ = sub_group_reduce_max(max_);
#else
    max_ = work_group_reduce_max(max_);
#endif

    for (int k = 0; k < VECT_SIZE; ++k) {
#if LOGSOFTMAX
        denom_ += exp(d[k] - max_);
#else
        d[k] = exp(d[k] - max_);
        denom_ += d[k];
#endif
    }

#if GROUP_SIZE == SUB_GROUP_SIZE
    denom_ = sub_group_reduce_add(denom_);
#else
    denom_ = work_group_reduce_add(denom_);
#endif

#if LOGSOFTMAX
    denom_ = log(denom_);
#else
    denom_ = 1.0 / denom_;
#endif

    dst += data_off;

    for (int k = 0, axis_channel_id = OC * axis_chunk; k < VECT_SIZE;
            ++k, axis_channel_id += OC) {
#if LOGSOFTMAX
        d[k] = d[k] - max_ - denom_;
#else
        d[k] = d[k] * denom_;
#endif
#if WITH_SCALES
        dst[axis_channel_id] = FLOAT_TO_DATA(DST, d[k] * scale[0]);
#else
        dst[axis_channel_id] = FLOAT_TO_DATA(DST, d[k]);
#endif
    }

#else
    const int data_off = (get_global_id(0) / GROUP_SIZE) * SOFTMAX_AXIS_SIZE;

    float8 d[NUM_BUF];
    float max_ = -FLT_MAX;
    float denom_ = 0.f;

    src += data_off;

    for (int k = 0; k < NUM_BUF; ++k) {
        d[k] = LOAD_FLOAT8(SRC, &src[k * VECT_SIZE * SUB_GROUP_SIZE]);
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
#if WITH_SCALES
        STORE_FLOAT8(
                DST, &dst[k * VECT_SIZE * SUB_GROUP_SIZE], scale[0] * d[k]);
#else
        STORE_FLOAT8(DST, &dst[k * VECT_SIZE * SUB_GROUP_SIZE], d[k]);
#endif
    }
#endif
}

#endif
#if IS_BWD
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_softmax_bwd(__global DST_DATA_T *dst, __global SRC_DATA_T *diff_src,
        __global DST_DATA_T *diff_dst) {
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
        diff_d[i] = DATA_TO_FLOAT(DST, diff_dst[idx]);
        dst_[i] = DATA_TO_FLOAT(DST, dst[idx]);
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
        diff_src[idx] = FLOAT_TO_DATA(SRC, diff_d[i]);
    }

#else
    const int data_off = (get_global_id(0) / GROUP_SIZE) * SOFTMAX_AXIS_SIZE;

    float sbr = 0.f;

    float8 diff_d[NUM_BUF];
    float8 dst_[NUM_BUF];

    diff_dst += data_off;
    dst += data_off;
    for (int k = 0; k < NUM_BUF; ++k) {
        diff_d[k] = LOAD_FLOAT8(DST, &diff_dst[k * VECT_SIZE * SUB_GROUP_SIZE]);
        dst_[k] = LOAD_FLOAT8(DST, &dst[k * VECT_SIZE * SUB_GROUP_SIZE]);
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
        STORE_FLOAT8(SRC, &diff_src[k * VECT_SIZE * SUB_GROUP_SIZE], diff_d[k]);
    }
#endif
}
#endif
