/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_OCL_OCL_SCALES_H
#define GPU_OCL_OCL_SCALES_H

inline void block_read_scales(float4 *data, int idx, int sg_local_id,
        __global float *runtime_scales) {
    if (OC > idx + (SUB_GROUP_SIZE * 4)) {
        *data = as_float4(intel_sub_group_block_read4(
                (__global uint *)&runtime_scales[idx]));
    } else {
        float local_dat[4] = {};
        for (int i = 0; i < 4; ++i)
            if (idx + ((i + 1) * SUB_GROUP_SIZE) <= OC) {
                local_dat[i] = as_float(intel_sub_group_block_read(
                        (__global uint *)&runtime_scales[idx
                                + (SUB_GROUP_SIZE * i)]));
            } else if (idx + (i * SUB_GROUP_SIZE) + sg_local_id < OC) {
                local_dat[i] = runtime_scales[idx + (SUB_GROUP_SIZE * i)
                        + sg_local_id];
            }
        (*data).s0 = local_dat[0];
        (*data).s1 = local_dat[1];
        (*data).s2 = local_dat[2];
        (*data).s3 = local_dat[3];
    }
}

#endif
