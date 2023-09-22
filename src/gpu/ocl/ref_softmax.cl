/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)

#define IS_HALF_half 1
#define IS_HALF(dt) CONCAT2(IS_HALF_, dt)

#if IS_FWD
#define DD(i) CONCAt2(DST_D, i)
#elif IS_BWD
#define DD(i) CONCAt2(SRC_D, i)
#else
#error unsupported data parameter
#endif

#define OFF(dim, idx) \
    (dim % CONCAT2(DATA_B, idx)) * CONCAT2(DATA_SB, idx) \
            + (dim / CONCAT2(DATA_B, idx)) * CONCAT2(DATA_S, idx)

#if SOFTMAX_AXIS_IDX == 0
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(softmax_dim, 0) + OFF(dim0, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    softmax_dim >= DD(0) || dim0 >= DD(1) || dim1 >= DD(2) || dim2 >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 1
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(softmax_dim, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || softmax_dim >= DD(1) || dim1 >= DD(2) || dim2 >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 2
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(softmax_dim, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || softmax_dim >= DD(2) || dim2 >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 3
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(softmax_dim, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || softmax_dim >= DD(3) \
            || dim3 >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 4
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) \
            + OFF(softmax_dim, 4) + OFF(dim4, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || dim3 >= DD(3) \
            || softmax_dim >= DD(4) || dim4 >= DD(5)
#elif SOFTMAX_AXIS_IDX == 5
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) + OFF(dim4, 4) \
            + OFF(softmax_dim, 5)
#define NEEDS_PADDING(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    dim0 >= DD(0) || dim1 >= DD(1) || dim2 >= DD(2) || dim3 >= DD(3) \
            || dim4 >= DD(4) || softmax_dim >= DD(5)
#else
#error unsupported softmax dimension
#endif

#if IS_FWD

#if SUB_GROUP_SIZE == 16
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#endif

__kernel void
ref_softmax_fwd_generic(__global SRC_DATA_T *src, __global DATA_T *dst,
        __global float *src_scale, __global float *dst_scale POST_OP_ARGS) {

    const int dim[] = {
            (get_global_id(0) / GROUP_SIZE) % BLOCK_0,
            get_global_id(1) % BLOCK_1,
            get_global_id(2) % BLOCK_2,
            (get_global_id(0) / GROUP_SIZE) / BLOCK_0,
            get_global_id(1) / BLOCK_1,
            get_global_id(2) / BLOCK_2,
    };

    float scale = 1.0f;
#if WITH_SRC_SCALES
    scale *= src_scale[0];
#endif

#if SUB_GROUP_SIZE == 16
    const int local_id = get_local_id(0);

    // SOFTMAX_AXIS is the size of axis around which softmax operation is
    // performed

    // begin and end indices calculated according to thread's id
    const int begin = local_id * (SOFTMAX_AXIS / GROUP_SIZE);
    const int end = (local_id == GROUP_SIZE - 1)
            ? SOFTMAX_AXIS
            : (local_id + 1) * (SOFTMAX_AXIS / GROUP_SIZE);
#if SOFTMAX_AXIS - (GROUP_SIZE - 1) * (SOFTMAX_AXIS / GROUP_SIZE) \
        > SOFTMAX_AXIS / GROUP_SIZE
    const int buf_size
            = SOFTMAX_AXIS - (GROUP_SIZE - 1) * (SOFTMAX_AXIS / GROUP_SIZE);
#else
    const int buf_size = SOFTMAX_AXIS / GROUP_SIZE;
#endif
#else
    const int begin = 0;
    const int end = SOFTMAX_AXIS;
    const int buf_size = SOFTMAX_AXIS;
#endif

#if IS_HALF(SRC_DATA_T) == 1 && IS_HALF(DST_DATA_TYPE) == 1
    typedef half acc_t;
    const acc_t acc_max = HALF_MAX;
    const acc_t acc_zero = 0.h;
#elif DT_F64 || DST_DT_F64
    typedef double acc_t;
    const acc_t acc_max = DBL_MAX;
    const acc_t acc_zero = 0.0;
#else
    typedef float acc_t;
    const acc_t acc_max = FLT_MAX;
    const acc_t acc_zero = 0.f;
#endif

    acc_t d[buf_size];
    acc_t max_ = -acc_max;
    acc_t denom_ = acc_zero;

    // finding max value for each sub_group
    if (!(NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], begin))) {
        for (int i = begin; i < end && i < DD(SOFTMAX_AXIS_IDX); ++i) {
            size_t data_off
                    = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);
            d[i - begin] = SRC_TO_REF(src[data_off]);
            max_ = max(max_, d[i - begin]);
        }
    }
#if SUB_GROUP_SIZE == 16
    // reduce using work_group_reduce if no. of subgroups > 1, for e.g.
    // if group_size is 32, there will be 2 sub-groups (size of each sub-group
    // is 16 which is an optimal value)
#if GROUP_SIZE == SUB_GROUP_SIZE
    max_ = sub_group_reduce_max(max_);
#else
    max_ = work_group_reduce_max(max_);
#endif
#endif

    // updating dst tensor and accumulating denom for last step
    if (!(NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], begin))) {
        for (int i = begin; i < end && i < DD(SOFTMAX_AXIS_IDX); ++i) {
#if LOGSOFTMAX
            denom_ += exp(d[i - begin] - max_);
#else
            d[i - begin] = exp(d[i - begin] - max_);
            denom_ += d[i - begin];
#endif
        }
    }
#if SUB_GROUP_SIZE == 16
#if GROUP_SIZE == SUB_GROUP_SIZE
    denom_ = sub_group_reduce_add(denom_);
#else
    denom_ = work_group_reduce_add(denom_);
#endif
#endif

#if LOGSOFTMAX
    denom_ = log(denom_);
#else
    denom_ = 1.0 / denom_;
#endif

    for (int i = begin; i < end; ++i) {
        size_t data_off = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);

        POST_OP_DATA_T tmp;
        if (NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], i)) {
            tmp = (POST_OP_DATA_T)(acc_zero);
        } else {
            float unscaled;
#if LOGSOFTMAX
            unscaled = d[i - begin] - max_ - denom_;
#else
            unscaled = d[i - begin] * denom_;
#endif
            tmp = (POST_OP_DATA_T)(scale * unscaled);
        }
        // post op service
        POST_OP_DATA_T sum_src;
#if WITH_SUM
        sum_src = (POST_OP_DATA_T)DATA_TO_REF(dst[data_off]);
#endif
#if NDIMS == 3
#if IS_CHANNEL_LAST
        const unsigned po_d2 = (data_off / OC) % SPATIAL_DIM_0;
#else
        const unsigned po_d2 = data_off % SPATIAL_DIM_0;
#endif
        const unsigned po_d3 = 0;
        const unsigned po_d4 = 0;
#elif NDIMS == 4
#if IS_CHANNEL_LAST
        const unsigned po_d2 = data_off / OC / SPATIAL_DIM_1;
        const unsigned po_d3 = (data_off / OC) % SPATIAL_DIM_1;
#else
        const unsigned po_d2 = data_off / SPATIAL_DIM_1;
        const unsigned po_d3 = data_off % SPATIAL_DIM_1;
#endif
        const unsigned po_d4 = 0;
#elif NDIMS == 5
#if IS_CHANNEL_LAST
        const unsigned po_d2 = (data_off / OC) / SPATIAL_DIM_1 / SPATIAL_DIM_0;
        const unsigned po_d3 = (data_off / OC) / SPATIAL_DIM_1;
        const unsigned po_d4 = (data_off / OC) % SPATIAL_DIM_1;
#else
        const unsigned po_d2 = data_off / SPATIAL_DIM_1 / SPATIAL_DIM_0;
        const unsigned po_d3 = data_off / SPATIAL_DIM_1;
        const unsigned po_d4 = data_off % SPATIAL_DIM_1;
#endif
#else
        const unsigned po_d2 = 0;
        const unsigned po_d3 = 0;
        const unsigned po_d4 = 0;
#endif
        const int mb = data_off / SPATIAL_DIMS_SIZE / OC;
#if IS_CHANNEL_LAST
        const int oc = data_off % OC;
#else
        const int oc = data_off / SPATIAL_DIMS_SIZE;
#endif
        APPLY_POST_OPS_SERIAL(tmp, POST_OP_DATA_T, sum_src, POST_OP_DATA_T, mb,
                1, oc, 1, po_d2, 1, po_d3, 1, po_d4, 1, 0, 1);
#if WITH_DST_SCALES
        tmp /= dst_scale[0];
#endif
        dst[data_off] = TO_DST(tmp);
    }
}

#endif

#if IS_BWD

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))

__kernel void
ref_softmax_bwd_generic(__global DST_DATA_T *dst, __global SRC_DATA_T *diff_src,
        __global DST_DATA_T *diff_dst) {

    const int dim[] = {
            (get_global_id(0) / GROUP_SIZE) % BLOCK_0,
            get_global_id(1) % BLOCK_1,
            get_global_id(2) % BLOCK_2,
            (get_global_id(0) / GROUP_SIZE) / BLOCK_0,
            get_global_id(1) / BLOCK_1,
            get_global_id(2) / BLOCK_2,
    };

    const int local_id = get_local_id(0);

    // begin and end indices calculated according to thread's id
    const int begin = local_id * (SOFTMAX_AXIS / GROUP_SIZE);
    const int end = (local_id == GROUP_SIZE - 1)
            ? SOFTMAX_AXIS
            : (local_id + 1) * (SOFTMAX_AXIS / GROUP_SIZE);
#if SOFTMAX_AXIS - (GROUP_SIZE - 1) * (SOFTMAX_AXIS / GROUP_SIZE) \
        > SOFTMAX_AXIS / GROUP_SIZE
    const int buf_size
            = SOFTMAX_AXIS - (GROUP_SIZE - 1) * (SOFTMAX_AXIS / GROUP_SIZE);
#else
    const int buf_size = SOFTMAX_AXIS / GROUP_SIZE;
#endif

#if IS_HALF(SRC_DATA_T) == 1 && IS_HALF(DST_DATA_TYPE) == 1
    typedef half acc_t;
    const acc_t acc_zero = 0.h;
#elif DT_F64 || DST_DT_F64
    typedef double acc_t;
    const acc_t acc_zero = 0.0;
#else
    typedef float acc_t;
    const acc_t acc_zero = 0.f;
#endif

    acc_t diff_d[buf_size];
    acc_t d[buf_size];
    acc_t sbr = acc_zero;

    if (!(NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], begin))) {
        for (int i = begin; i < end && i < DD(SOFTMAX_AXIS_IDX); ++i) {
            size_t idx = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);
            diff_d[i - begin] = DST_TO_REF(diff_dst[idx]);
            d[i - begin] = DST_TO_REF(dst[idx]);
#if LOGSOFTMAX
            sbr += diff_d[i - begin];
#else
            sbr += diff_d[i - begin] * d[i - begin];
#endif
        }
    }

#if GROUP_SIZE == SUB_GROUP_SIZE
    sbr = sub_group_reduce_add(sbr);
#else
    sbr = work_group_reduce_add(sbr);
#endif

    for (int i = begin; i < end; ++i) {
        size_t idx = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);

        if (NEEDS_PADDING(dim[0], dim[1], dim[2], dim[3], dim[4], i)) {
            diff_src[idx] = REF_TO_SRC(acc_zero);
        } else {
#if LOGSOFTMAX
            diff_src[idx]
                    = REF_TO_SRC(diff_d[i - begin] - exp(d[i - begin]) * sbr);
#else
            acc_t inner_data = diff_d[i - begin] - sbr;
            diff_src[idx] = REF_TO_SRC(d[i - begin] * inner_data);
#endif
        }
    }
}
#endif
