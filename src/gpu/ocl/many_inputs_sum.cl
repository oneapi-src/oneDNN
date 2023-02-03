/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#define INIT_N_INPUTS(i) \
    if (i < N_INPUTS) inputs[i] = input##i;
#define INIT_MAX_N_INPUTS(i) \
    if (i < MAX_N_INPUTS) inputs[i] = input##i;

float get_value(__global SRC_DATA_T *src, ptrdiff_t offset) {
    if (offset >= N_ELEMS) return 0;
    return CONVERT_FLOAT_T(src[offset]);
}
#define many_inputs_sum_impl(inputs, output, scales, num_inputs, local_val) \
    const uint group_id = get_group_id(0); \
    const uint group_size = get_local_size(0); \
    const uint gid = get_global_id(0); \
    ptrdiff_t offset = gid / num_inputs; \
    const int tensor_idx = gid % num_inputs; \
    const int local_id = get_local_id(0); \
    local_val[local_id] \
            = get_value(inputs[tensor_idx], offset) * scales[tensor_idx]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    if (tensor_idx == 0 && offset < N_ELEMS) { \
        float final_val = 0; \
        for (int i = 0; i < num_inputs; i++) { \
            final_val += local_val[local_id + i]; \
        } \
        output[offset] = TO_DST(final_val + CONVERT_FLOAT_T(output[offset])); \
    }

__kernel void many_inputs_sum(__global SRC_DATA_T *input0,
        __global SRC_DATA_T *input1, __global SRC_DATA_T *input2,
        __global SRC_DATA_T *input3, __global SRC_DATA_T *input4,
        __global SRC_DATA_T *input5, __global SRC_DATA_T *input6,
        __global SRC_DATA_T *input7, __global SRC_DATA_T *input8,
        __global SRC_DATA_T *input9, __global SRC_DATA_T *input10,
        __global SRC_DATA_T *input11, __global SRC_DATA_T *input12,
        __global SRC_DATA_T *input13, __global SRC_DATA_T *input14,
        __global SRC_DATA_T *input15, __global DST_DATA_T *output,
        __global float *scales) {

    __local float local_val[256];
    __global SRC_DATA_T *inputs[16];

    INIT_N_INPUTS(0);
    INIT_N_INPUTS(1);
    INIT_N_INPUTS(2);
    INIT_N_INPUTS(3);
    INIT_N_INPUTS(4);
    INIT_N_INPUTS(5);
    INIT_N_INPUTS(6);
    INIT_N_INPUTS(7);
    INIT_N_INPUTS(8);
    INIT_N_INPUTS(9);
    INIT_N_INPUTS(10);
    INIT_N_INPUTS(11);
    INIT_N_INPUTS(12);
    INIT_N_INPUTS(13);
    INIT_N_INPUTS(14);
    INIT_N_INPUTS(15);
    many_inputs_sum_impl(inputs, output, scales, N_INPUTS, local_val);
}

__kernel void many_inputs_sum_batched(__global SRC_DATA_T *input0,
        __global SRC_DATA_T *input1, __global SRC_DATA_T *input2,
        __global SRC_DATA_T *input3, __global SRC_DATA_T *input4,
        __global SRC_DATA_T *input5, __global SRC_DATA_T *input6,
        __global SRC_DATA_T *input7, __global SRC_DATA_T *input8,
        __global SRC_DATA_T *input9, __global SRC_DATA_T *input10,
        __global SRC_DATA_T *input11, __global SRC_DATA_T *input12,
        __global SRC_DATA_T *input13, __global SRC_DATA_T *input14,
        __global SRC_DATA_T *input15, __global DST_DATA_T *output,
        __global float *scales) {

    __local float local_val[256];
    __global SRC_DATA_T *inputs[16];

    INIT_MAX_N_INPUTS(0);
    INIT_MAX_N_INPUTS(1);
    INIT_MAX_N_INPUTS(2);
    INIT_MAX_N_INPUTS(3);
    INIT_MAX_N_INPUTS(4);
    INIT_MAX_N_INPUTS(5);
    INIT_MAX_N_INPUTS(6);
    INIT_MAX_N_INPUTS(7);
    INIT_MAX_N_INPUTS(8);
    INIT_MAX_N_INPUTS(9);
    INIT_MAX_N_INPUTS(10);
    INIT_MAX_N_INPUTS(11);
    INIT_MAX_N_INPUTS(12);
    INIT_MAX_N_INPUTS(13);
    INIT_MAX_N_INPUTS(14);
    INIT_MAX_N_INPUTS(15);
    many_inputs_sum_impl(inputs, output, scales, MAX_N_INPUTS, local_val);
}
