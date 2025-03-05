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

#include "gpu/intel/ocl/ocl_post_ops.h"
#include "gpu/intel/ocl/ocl_types.h"

__kernel void ref_gated_mlp(const __global SRC_DATA_T *src,
        const __global W_GATE_DATA_T *W_gate, const __global W_UP_DATA_T *W_up,
        const __global W_DOWN_DATA_T *W_down, __global DST_DATA_T *dst, long MB,
        long IC, long OC) {

    // TODO: name indices reasonably...
    long r = get_global_id(0);

    float fused_g_l[SIZE_OC];

    // Multiply (row of input V)*W_gate
    for (long c = 0; c < SIZE_OC; c++) {
        if(r < MB) {
            float acc_gate = 0;
            float acc_linear = 0;
            for(long h=0; h<IC; ++h) {
                long v_off = r * IC + h;
                long W_g_off = h * OC + c;

                float s = src[v_off];

                acc_gate   += s * convert_float(W_gate[W_g_off]);
                acc_linear += s * convert_float(W_up[W_g_off]);
            }
            float swish_acc = acc_gate / (1.f + exp(-1.f * acc_gate)); // swish
            fused_g_l[c] = swish_acc * acc_linear; // elemwise mul
        }
        else {
            fused_g_l[c] = 0;
        }
    }


    // matmul W_down
    for (long c = 0; c < IC; c++) {
        // printf("%f ", fused_subrow[get_local_id(0)]);
        float acc = 0;
        for(long h=0; h < OC; ++h) {

            long W_down_off = h * IC + c;
            if(r < MB) {
                acc += fused_g_l[h] * convert_float(W_down[W_down_off]);
            }
        }
        long output_off = r * IC + c;
        dst[output_off] = acc;
    }
}
