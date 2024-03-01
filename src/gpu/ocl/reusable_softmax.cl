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

#include "gpu/ocl/dispatch.h"
#include "gpu/ocl/ocl_types.h"
#include "gpu/ocl/types_interop.h"

__kernel void reusable_softmax_fwd_generic(__global SRC_DATA_T *src,
        __global DST_DATA_T *dst, __global float *src_scale,
        __global float *dst_scale, dim_t softmax_axis_size,
        dim_t softmax_axis_stride, dispatch_gws_rt_params_t gws_params) {
    src = GWS_GET_BUFFER_POS(SRC, gws_params, src);
    dst = GWS_GET_BUFFER_POS(DST, gws_params, dst);

    FLT_ACC_DATA_T max_ = TO_FLT_ACC_DATA_T(DATA_MIN);
    FLT_ACC_DATA_T denom_ = TO_FLT_ACC_DATA_T(DATA_ZERO);

    unroll_16_for(int c = 0; c < softmax_axis_size; c++) {
        max_ = max(max_, TO_FLT_ACC_DATA_T(src[c * softmax_axis_stride]));
    }

    unroll_16_for(int c = 0; c < softmax_axis_size; c++) {
        denom_ += exp(TO_FLT_ACC_DATA_T(src[c * softmax_axis_stride]) - max_);
    }

    denom_ = LOGSOFTMAX ? log(denom_) : 1.0f / denom_;

    for (int c = 0; c < softmax_axis_size; c++) {
        size_t data_off = c * softmax_axis_stride;
        FLT_ACC_DATA_T unscaled = LOGSOFTMAX
                ? TO_FLT_ACC_DATA_T(src[data_off]) - max_ - denom_
                : exp(TO_FLT_ACC_DATA_T(src[data_off]) - max_) * denom_;

        float scale = 1.0f;
        if (src_scale) { scale = *src_scale; }
        if (dst_scale) { scale /= *dst_scale; }

        dst[data_off] = TO_DST(unscaled * scale);
    }
}
