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

#include "gpu/ocl/gen9_bnorm.h"

// Zeroing and finalization kernels are required for atomics-based
// reduction (FUSED_ATOMICS_REDUCTION definition).

NAMED_KERNEL_ATTR(AUX)
__kernel void gen9_fused_reduce_init(
#if IS_FWD
        __global float *mean, __global float *variance
#else
        __global float *diff_scale, __global float *diff_shift
#endif
) {
    const int c = GWS_GET_IC_AUX();
#if IS_FWD
    mean[c] = 0.0f;
    variance[c] = 0.0f;
#else
    diff_scale[c] = 0.0f;
#if DIFF_SHIFT == 1
    diff_shift[c] = 0.0f;
#else
    diff_shift[IC + IC * REDUCE_STAT_NBLOCKS + c] = 0.0f;
#endif
#endif
    return;
}

#if IS_FWD
NAMED_KERNEL_ATTR(AUX)
__kernel void gen9_fused_reduce_final(
#if USE_STATS_ONE_PASS
        __global float *mean, __global float *variance
#else
        __global float *data_reduce
#endif
) {
    const int c = GWS_GET_IC_AUX();
#if USE_STATS_ONE_PASS
    mean[c] = mean[c] / (MB * ID * IH * IW);
    float tmp_var = max(
            0.0f, (variance[c] / (MB * ID * IH * IW)) - (mean[c] * mean[c]));
    variance[c] = tmp_var;
#else
    data_reduce[c] = data_reduce[c] / (MB * ID * IH * IW);
#endif
    return;
}
#else
NAMED_KERNEL_ATTR(AUX)
__kernel void gen9_fused_reduce_final(
        __global float *diff_scale, __global float *variance, float eps) {
    const int c = GWS_GET_IC_AUX();
    diff_scale[c] *= 1.0f / sqrt(variance[c] + eps);
    return;
}
#endif // IS_FWD
