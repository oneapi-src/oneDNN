/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/intel/ocl/dispatch.h"
#include "gpu/intel/ocl/ocl_types.h"
#include "gpu/intel/ocl/types_interop.h"

#if ONE_REDUCTION_PER_SUBGROUP == 1
__attribute__((reqd_work_group_size(16, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
#endif
__kernel void
reusable_softmax_fwd_generic(__global SRC_DATA_T *src, __global DST_DATA_T *dst,
        __global float *src_scale, __global float *dst_scale,
        dim_t softmax_axis_size, dim_t softmax_axis_stride,
        dim_t softmax_axis_chunk_size, dispatch_gws_rt_params_t gws_params) {
    dst = GWS_GET_BUFFER_POS(DST, gws_params, dst);

    FLT_ACC_DATA_T max_ = TO_FLT_ACC_DATA_T(DATA_MIN);
    FLT_ACC_DATA_T denom_ = TO_FLT_ACC_DATA_T(DATA_ZERO);

    const size_t begin
            = GWS_GET_OFF_NAMED(SRC, DEFAULT_DISPATCHER_SUFFIX, gws_params);

#if MANY_REDUCTIONS_PER_WORKGROUP == 1
    const size_t end = begin + softmax_axis_stride * softmax_axis_size;
#else
    const size_t axis_begin = GWS_GET_OFF_NAMED(
            ORIGINAL, DEFAULT_DISPATCHER_SUFFIX, gws_params);
    const size_t end = min(axis_begin + softmax_axis_stride * softmax_axis_size,
            begin + softmax_axis_stride * softmax_axis_chunk_size);
#endif

    for (off_t c = begin; c < end; c += softmax_axis_stride) {
        max_ = max(max_, TO_FLT_ACC_DATA_T(src[c]));
    }
    if (USE_WORKGROUP_REDUCTION) { max_ = work_group_reduce_max(max_); }
    if (USE_SUBGROUP_REDUCTION) { max_ = sub_group_reduce_max(max_); }

    for (off_t c = begin; c < end; c += softmax_axis_stride) {
        denom_ += exp(TO_FLT_ACC_DATA_T(src[c]) - max_);
    }
    if (USE_WORKGROUP_REDUCTION) { denom_ = work_group_reduce_add(denom_); }
    if (USE_SUBGROUP_REDUCTION) { denom_ = sub_group_reduce_add(denom_); }

    denom_ = LOGSOFTMAX ? log(denom_) : 1.0f / denom_;

    for (off_t c = begin; c < end; c += softmax_axis_stride) {
        FLT_ACC_DATA_T unscaled = LOGSOFTMAX
                ? TO_FLT_ACC_DATA_T(src[c]) - max_ - denom_
                : exp(TO_FLT_ACC_DATA_T(src[c]) - max_) * denom_;

        float scale = 1.0f;
        if (src_scale) { scale = *src_scale; }
        if (dst_scale) { scale /= *dst_scale; }

        dst[c - begin] = TO_DST(unscaled * scale);
    }
}
