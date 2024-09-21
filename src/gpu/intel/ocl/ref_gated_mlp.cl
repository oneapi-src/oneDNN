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

//STF: TODO
__kernel void ref_gated_mlp(const __global SRC_DATA_T *src,
        const __global W_GATE_DATA_T *W_gate, const __global W_UP_DATA_T *W_up,
        const __global W_DOWN_DATA_T *W_down, __global DST_DATA_T *dst, long MB,
        long IC, long OC) {

    /*
*/
}
