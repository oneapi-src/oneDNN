/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef USE_VECTORIZED_KERNEL
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
#endif

#ifdef USE_VECTORIZED_KERNEL

#define VECT_SIZE 8
#define DIVUP(a, b) (((a) + (b)-1) / (b))

#define STORE_FLOAT8(prefix, ptr, val) \
    WRITE_BLOCK8(prefix, (__global BLOCK_T(ALIAS(prefix)) *)(ptr), \
            DATA_TO_BLOCK8(prefix, FLOAT_TO_DATA8(prefix, val)))

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

#define COMMON_DATA_TO_X(x, y) CONCAT2(DATA_TO_, UP_CASE_DATA)(x, y)
#define COMMON_STORE_DATA8(x, y, z) CONCAT3(STORE_, UP_CASE_DATA, 8)(x, y, z)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) __kernel void
reusable_softmax_fwd_generic(__global DATA_T *src, __global DST_DATA_T *dst,
        __global float *src_scale, __global float *dst_scale,
        dim_t SOFTMAX_AXIS_SIZE, dim_t softmax_axis_stride,
        dim_t softmax_axis_chunk_size, dispatch_gws_rt_params_t gws_params) {
    float scale = 1.0f;
    const int data_off = (get_global_id(0) / 16) * SOFTMAX_AXIS_SIZE;
    VECT_FLOAT_T dk;
    float max_ = -FLT_MAX;
    float denom_ = 0.0f;
    const bool has_tail = SOFTMAX_AXIS_SIZE % (16 * 8);
    int last_buf = DIVUP(SOFTMAX_AXIS_SIZE, (16 * 8));
    if (has_tail) last_buf--;
    src += data_off;
    for (int k = 0; k < last_buf; ++k) {
        int idx = k * 16;
        dk = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[idx * 8])));
        for (int i = 0; i < 8; ++i) {
            max_ = max(dk[i], max_);
        }
    }
    if (has_tail) {
        int k = last_buf;
        for (int i = 0; i < 8; ++i) {
            int off = k * 8 * 16 + i * 16 + get_sub_group_local_id();
            float d = (off < SOFTMAX_AXIS_SIZE ? COMMON_DATA_TO_X(SRC, src[off])
                                               : -COMMON_DATA_MAX);
            max_ = max(d, max_);
        }
    }
    max_ = sub_group_reduce_max(max_);

    for (int k = 0; k < last_buf; ++k) {
        int idx = k * 16;
        dk = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[idx * 8])));
        dk = exp(dk - max_);
        for (int i = 0; i < 8; ++i)
            denom_ += dk[i];
    }
    if (has_tail) {
        int k = last_buf;
        int idx = k * 16;
        dk = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[idx * 8])));
        dk = exp(dk - max_);
        for (int i = 0; i < 8; ++i) {
            int off = k * 8 * 16 + i * 16 + get_sub_group_local_id();
            if (off < SOFTMAX_AXIS_SIZE) denom_ += dk[i];
        }
    }
    denom_ = sub_group_reduce_add(denom_);
    denom_ = LOGSOFTMAX ? log(denom_) : 1.0f / denom_;

    dst += data_off;

    for (int k = 0; k < last_buf; ++k) {
        int idx = k * 16;
        dk = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[idx * 8])));
        dk = LOGSOFTMAX ? dk - max_ - denom_ : exp(dk - max_) * denom_;

        COMMON_STORE_DATA8(DST, &dst[idx * VECT_SIZE], scale * dk);
    }
    if (has_tail) {
        int k = last_buf;
        int idx = k * 16;
        dk = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[idx * 8])));
        dk = LOGSOFTMAX ? dk - max_ - denom_ : exp(dk - max_) * denom_;

        for (int i = 0; i < 8; i++) {
            int off = k * 8 * 16 + i * 16 + get_sub_group_local_id();
            if (off < SOFTMAX_AXIS_SIZE)
                dst[off] = convert_float(scale * dk[i]);
        }
    }
}
#endif
