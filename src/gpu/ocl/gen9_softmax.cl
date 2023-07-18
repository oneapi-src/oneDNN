/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#define LOAD_DOUBLE8(prefix, ptr) \
    DATA_TO_DOUBLE8(prefix, \
            BLOCK_TO_DATA8(prefix, \
                    READ_BLOCK8(prefix, \
                            (__global BLOCK_T(ALIAS(prefix)) *)(ptr))))

#define STORE_DOUBLE8(prefix, ptr, val) \
    WRITE_BLOCK8(prefix, (__global BLOCK_T(ALIAS(prefix)) *)(ptr), \
            DATA_TO_BLOCK8(prefix, DOUBLE_TO_DATA8(prefix, val)))

#if DST_DT_F64
#define UP_CASE_DATA DOUBLE
#define COMMON_DATA_T double
#define COMMON_DATA_MAX DBL_MAX
#define COMMON_DATA_ZERO 0.0
#else
#define UP_CASE_DATA FLOAT
#define COMMON_DATA_T float
#define COMMON_DATA_MAX FLT_MAX
#define COMMON_DATA_ZERO 0.0f
#endif

#define COMMON_DATA8_T CONCAT2(COMMON_DATA_T, 8)

#define COMMON_DATA_TO_X(x, y) CONCAT2(DATA_TO_, UP_CASE_DATA)(x, y)
#define COMMON_X_TO_DATA(x, y) CONCAT2(UP_CASE_DATA, _TO_DATA)(x, y)

#define COMMON_LOAD_DATA8(x, y) CONCAT3(LOAD_, UP_CASE_DATA, 8)(x, y)
#define COMMON_STORE_DATA8(x, y, z) CONCAT3(STORE_, UP_CASE_DATA, 8)(x, y, z)

#define VECT_SIZE 8
#define SV (GROUP_SIZE * VECT_SIZE)
#define HAS_TAIL (SOFTMAX_AXIS_SIZE % SV != 0)
#define NUM_BUF ((SOFTMAX_AXIS_SIZE + SV - 1) / SV)
#define SMALL_BUFFER (REPEAT_SUBGRP_BUF_SIZE < THREAD_BUF_SIZE)

#if IS_FWD

int find_axis_offset(int index, int local_id, int subgroup) {
    const int group_axis_block = GROUP_SIZE * THREAD_BUF_SIZE * CHANNELS;
    int offset = CHANNELS * THREAD_BUF_SIZE * local_id;
    if (index > 0) { offset += group_axis_block; }
    if (index > 0 && subgroup == (SUBGROUPS_REPEATED - 1) && SMALL_BUFFER) {
        offset = group_axis_block
                + (CHANNELS * REPEAT_SUBGRP_BUF_SIZE * local_id);
    }
    return offset;
}

int get_local_off(int subgroup, int subgroup_local_id) {
    if (subgroup == (SUBGROUPS_REPEATED - 1)) {
        if (SMALL_BUFFER) {
            return ((GROUP_SIZE + SUB_GROUP_SIZE * subgroup) * THREAD_BUF_SIZE
                    + subgroup_local_id * REPEAT_SUBGRP_BUF_SIZE);
        } else { // THREAD_BUF_SIZE == REPEAT_SUBGRP_BUF_SIZE i.e. 8 == 8
            return ((GROUP_SIZE + SUB_GROUP_SIZE * subgroup + subgroup_local_id)
                    * THREAD_BUF_SIZE);
        }
    } else {
        return 1; // subgroup thread within normal range
    }
}

int get_buffer_size(int index, int subgroup, int local_off) {
    if (index > 0 && subgroup == (SUBGROUPS_REPEATED - 1)) {
        if (SMALL_BUFFER) {
            int tail = SOFTMAX_AXIS_SIZE
                    - ((GROUP_SIZE + subgroup) * THREAD_BUF_SIZE);
            return ((local_off + REPEAT_SUBGRP_BUF_SIZE) <= SOFTMAX_AXIS_SIZE)
                    ? REPEAT_SUBGRP_BUF_SIZE
                    : tail % REPEAT_SUBGRP_BUF_SIZE;
        } else {
            return ((local_off + THREAD_BUF_SIZE) <= SOFTMAX_AXIS_SIZE)
                    ? THREAD_BUF_SIZE
                    : SOFTMAX_AXIS_SIZE % THREAD_BUF_SIZE;
        }
    } else {
        return THREAD_BUF_SIZE;
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_softmax_fwd(__global SRC_DATA_T *src, __global DST_DATA_T *dst,
        __global float *src_scale, __global float *dst_scale) {
    float scale = 1.0f;
#if WITH_SRC_SCALES
    scale *= src_scale[0];
#endif
#if WITH_DST_SCALES
    scale /= dst_scale[0];
#endif
#if IS_NHWC || IS_BLOCKED
    // gws is the combination of mb and axis size
    const int group = get_global_id(0) / GROUP_SIZE;
    const int mb = group / CHANNELS_PADDED;
    const int channel_id = group % CHANNELS_PADDED;

    const int local_id = get_local_id(0);
    const int subgroup_local_id = get_sub_group_local_id();
    const int subgroup_id = get_sub_group_id();

    int buf_chunk = (local_id / SUB_GROUP_SIZE) * SOFTMAX_BUF;
    COMMON_DATA_T max_ = -COMMON_DATA_MAX;
    COMMON_DATA_T denom_ = COMMON_DATA_ZERO;

#if SUBGROUPS_REPEATED // only for NHWC kernel
    // total reads should be num_buf x THREAD_BUF_SIZE

    COMMON_DATA_T d[2][THREAD_BUF_SIZE];

    int num_buf = (subgroup_id < SUBGROUPS_REPEATED) ? 2 : 1;

    int local_off = get_local_off(subgroup_id, subgroup_local_id);

    for (int i = 0; i < num_buf; i++) {
        if (i > 0 && local_off >= SOFTMAX_AXIS_SIZE) break;

        __global SRC_DATA_T *src_copy = src;
        int axis_offset = find_axis_offset(i, subgroup_local_id, subgroup_id);
        int data_off = mb * CHANNELS_PADDED * SOFTMAX_AXIS_SIZE + axis_offset
                + channel_id;
        int buf_reads = get_buffer_size(i, subgroup_id, local_off);

        src_copy += data_off;
        for (int k = 0, axis_channel_id = CHANNELS * buf_chunk; k < buf_reads;
                ++k, axis_channel_id += CHANNELS) {
            d[i][k] = COMMON_DATA_TO_X(SRC, src_copy[axis_channel_id]);
            max_ = max(d[i][k], max_);
        }
    }

#if GROUP_SIZE == SUB_GROUP_SIZE
    max_ = sub_group_reduce_max(max_);
#else
    max_ = work_group_reduce_max(max_);
#endif

    for (int i = 0; i < num_buf; i++) {
        if (i > 0 && local_off >= SOFTMAX_AXIS_SIZE) break;

        int buf_reads = get_buffer_size(i, subgroup_id, local_off);
        for (int k = 0; k < buf_reads; ++k) {
#if LOGSOFTMAX
            denom_ += exp(d[i][k] - max_);
#else
            d[i][k] = exp(d[i][k] - max_);
            denom_ += d[i][k];
#endif
        }
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

    for (int i = 0; i < num_buf; i++) {
        if (i > 0 && local_off >= SOFTMAX_AXIS_SIZE) break;

        __global DST_DATA_T *dst_copy = dst;
        int axis_offset = find_axis_offset(i, subgroup_local_id, subgroup_id);
        int data_off = mb * CHANNELS_PADDED * SOFTMAX_AXIS_SIZE + axis_offset
                + channel_id;
        int buf_reads = get_buffer_size(i, subgroup_id, local_off);
        dst_copy += data_off;
        for (int k = 0, axis_channel_id = CHANNELS * buf_chunk; k < buf_reads;
                ++k, axis_channel_id += CHANNELS) {
#if LOGSOFTMAX
            d[i][k] = d[i][k] - max_ - denom_;
#else
            d[i][k] = d[i][k] * denom_;
#endif
            dst_copy[axis_channel_id] = COMMON_X_TO_DATA(DST, d[i][k] * scale);
        }
    }

#else
    // NHWC kernel for lws size < max_lws or multiples of max_lws
    // Blocked layout kernel for 128-byte reads and writes
    const int channel_offset = CHANNELS * THREAD_BUF_SIZE * subgroup_local_id;

#if IS_BLOCKED
    const int channel_block = channel_id / CHANNELS;
    const int channel_in_block = channel_id % CHANNELS;

    int data_off = mb * CHANNELS * SOFTMAX_AXIS_SIZE + channel_offset
            + channel_in_block;
    const int buf_reads = THREAD_BUF_SIZE;
#else
    int data_off = mb * CHANNELS_PADDED * SOFTMAX_AXIS_SIZE + channel_offset
            + channel_id;

    const int local_off = local_id * THREAD_BUF_SIZE;
    int buf_reads;
    if (local_off >= SOFTMAX_AXIS_SIZE) {
        buf_reads = 0;
    } else {
        buf_reads = ((local_off + THREAD_BUF_SIZE) <= SOFTMAX_AXIS_SIZE)
                ? THREAD_BUF_SIZE
                : (SOFTMAX_AXIS_SIZE % THREAD_BUF_SIZE);
    }
#endif

    COMMON_DATA_T d[THREAD_BUF_SIZE];
    src += data_off;
    for (int k = 0, axis_channel_id = CHANNELS * buf_chunk; k < buf_reads;
            ++k, axis_channel_id += CHANNELS) {
        d[k] = COMMON_DATA_TO_X(SRC, src[axis_channel_id]);
        max_ = max(d[k], max_);
    }

#if GROUP_SIZE == SUB_GROUP_SIZE
    max_ = sub_group_reduce_max(max_);
#else
    max_ = work_group_reduce_max(max_);
#endif
    for (int k = 0; k < buf_reads; ++k) {
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

    for (int k = 0, axis_channel_id = CHANNELS * buf_chunk; k < buf_reads;
            ++k, axis_channel_id += CHANNELS) {
#if LOGSOFTMAX
        d[k] = d[k] - max_ - denom_;
#else
        d[k] = d[k] * denom_;
#endif
        dst[axis_channel_id] = COMMON_X_TO_DATA(DST, d[k] * scale);
    }
#endif

#else // NCHW kernel starts here
    const int data_off = (get_global_id(0) / GROUP_SIZE) * SOFTMAX_AXIS_SIZE;

    COMMON_DATA8_T d[NUM_BUF];
    COMMON_DATA_T max_ = -COMMON_DATA_MAX;
    COMMON_DATA_T denom_ = COMMON_DATA_ZERO;

    int last_buf = HAS_TAIL ? (NUM_BUF - 1) : NUM_BUF;

    src += data_off;
    int sid = get_sub_group_id();

#if IS_READ_ALIGNED
    for (int k = 0; k < last_buf; ++k) {
#if GROUP_SIZE == SUB_GROUP_SIZE
        int idx = k * SUB_GROUP_SIZE;
#else
        int idx = HAS_TAIL ? k * SUB_GROUP_SIZE : sid * SUB_GROUP_SIZE;
#endif
        d[k] = COMMON_LOAD_DATA8(SRC, &src[idx * VECT_SIZE]);
        for (int i = 0; i < VECT_SIZE; ++i) {
            max_ = max(d[k][i], max_);
        }
    }
#if HAS_TAIL
    {
        int k = last_buf;
        for (int i = 0; i < VECT_SIZE; ++i) {
            int off = k * VECT_SIZE * SUB_GROUP_SIZE + i * SUB_GROUP_SIZE
                    + get_sub_group_local_id();
            d[k][i] = (off < SOFTMAX_AXIS_SIZE ? COMMON_DATA_TO_X(SRC, src[off])
                                               : -COMMON_DATA_MAX);
            max_ = max(d[k][i], max_);
        }
    }
#endif
#else // subgroup block read requires 4-byte alignment
    for (int k = 0; k < NUM_BUF; ++k) {
        for (int i = 0; i < VECT_SIZE; ++i) {
            int off = k * VECT_SIZE * SUB_GROUP_SIZE + i * SUB_GROUP_SIZE
                    + get_sub_group_local_id();
            d[k][i] = (off < SOFTMAX_AXIS_SIZE ? DATA_TO_FLOAT(SRC, src[off])
                                               : -FLT_MAX);
            max_ = max(d[k][i], max_);
        }
    }
#endif

#if GROUP_SIZE == SUB_GROUP_SIZE
    max_ = sub_group_reduce_max(max_);
#else
    max_ = work_group_reduce_max(max_);
#endif

    for (int k = 0; k < last_buf; ++k) {
#if LOGSOFTMAX
        for (int i = 0; i < VECT_SIZE; ++i)
            denom_ += exp(d[k][i] - max_);
#else
        d[k] = exp(d[k] - max_);
        for (int i = 0; i < VECT_SIZE; ++i)
            denom_ += d[k][i];
#endif
    }

#if HAS_TAIL
    {
        int k = last_buf;
#if LOGSOFTMAX
        for (int i = 0; i < VECT_SIZE; ++i) {
            int off = k * VECT_SIZE * SUB_GROUP_SIZE + i * SUB_GROUP_SIZE
                    + get_sub_group_local_id();
            if (off < SOFTMAX_AXIS_SIZE) denom_ += exp(d[k][i] - max_);
        }
#else
        d[k] = exp(d[k] - max_);
        for (int i = 0; i < VECT_SIZE; ++i) {
            int off = k * VECT_SIZE * SUB_GROUP_SIZE + i * SUB_GROUP_SIZE
                    + get_sub_group_local_id();
            if (off < SOFTMAX_AXIS_SIZE) denom_ += d[k][i];
        }
#endif
    }
#endif

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
#if IS_WRITE_ALIGNED
    for (int k = 0; k < last_buf; ++k) {
#if GROUP_SIZE == SUB_GROUP_SIZE
        int idx = k * SUB_GROUP_SIZE;
#else
        int idx = HAS_TAIL ? k * SUB_GROUP_SIZE : sid * SUB_GROUP_SIZE;
#endif
#if LOGSOFTMAX
        d[k] = d[k] - max_ - denom_;
#else
        d[k] = d[k] * denom_;
#endif

        COMMON_STORE_DATA8(DST, &dst[idx * VECT_SIZE], scale * d[k]);
    }
#if HAS_TAIL // subgroup block write requires 16-byte alignment
    {
        int k = last_buf;
#if LOGSOFTMAX
        d[k] = d[k] - max_ - denom_;
#else
        d[k] = d[k] * denom_;
#endif
        for (int i = 0; i < VECT_SIZE; i++) {
            int off = k * VECT_SIZE * SUB_GROUP_SIZE + i * SUB_GROUP_SIZE
                    + get_sub_group_local_id();
            if (off < SOFTMAX_AXIS_SIZE)
                dst[off] = COMMON_X_TO_DATA(DST, scale * d[k][i]);
        }
    }
#endif
#else // for test-cases not aligned by 16 bytes needed for block write
    for (int k = 0; k < NUM_BUF; k++) {
#if LOGSOFTMAX
        d[k] = d[k] - max_ - denom_;
#else
        d[k] = d[k] * denom_;
#endif
        for (int i = 0; i < VECT_SIZE; i++) {
            int off = k * VECT_SIZE * SUB_GROUP_SIZE + i * SUB_GROUP_SIZE
                    + get_sub_group_local_id();
            if (off < SOFTMAX_AXIS_SIZE)
                dst[off] = COMMON_X_TO_DATA(DST, scale * d[k][i]);
        }
    }
#endif
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

    COMMON_DATA_T sbr = COMMON_DATA_ZERO;
    COMMON_DATA_T diff_d[VECT_SIZE];
    COMMON_DATA_T dst_[VECT_SIZE];

    diff_dst += data_off;
    dst += data_off;

    for (int i = 0, idx = IC * slice * SOFTMAX_BUF; i < VECT_SIZE;
            ++i, idx += IC) {
        diff_d[i] = COMMON_DATA_TO_X(DST, diff_dst[idx]);
        dst_[i] = COMMON_DATA_TO_X(DST, dst[idx]);
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
        diff_src[idx] = COMMON_X_TO_DATA(SRC, diff_d[i]);
    }

#else
    const int data_off = (get_global_id(0) / GROUP_SIZE) * SOFTMAX_AXIS_SIZE;

    COMMON_DATA_T sbr = COMMON_DATA_ZERO;
    COMMON_DATA8_T diff_d[NUM_BUF];
    COMMON_DATA8_T dst_[NUM_BUF];

    diff_dst += data_off;
    dst += data_off;
    for (int k = 0; k < NUM_BUF; ++k) {
        diff_d[k] = COMMON_LOAD_DATA8(
                DST, &diff_dst[k * VECT_SIZE * SUB_GROUP_SIZE]);
        dst_[k] = COMMON_LOAD_DATA8(DST, &dst[k * VECT_SIZE * SUB_GROUP_SIZE]);

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
        COMMON_STORE_DATA8(
                SRC, &diff_src[k * VECT_SIZE * SUB_GROUP_SIZE], diff_d[k]);
    }
#endif
}
#endif
