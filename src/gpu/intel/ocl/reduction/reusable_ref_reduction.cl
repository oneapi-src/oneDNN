/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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
#include "gpu/intel/ocl/reduction/ocl_reduction.h"
#include "gpu/intel/ocl/types_interop.h"

KERNEL_ATTR
__kernel void reusable_ref_reduce(__global DATA_T *src,
        __global DST_DATA_T *dst, off_t reduction_stride, off_t reduction_size,
        off_t div, float power, float eps,
        dispatch_gws_rt_params_t gws_params) {
    ASSUME(reduction_size > 0);
    ASSUME(reduction_stride > 0);
    ASSUME(div > 0);

    src = GWS_GET_BUFFER_POS(SRC, gws_params, src);
    dst = GWS_GET_BUFFER_POS(DST, gws_params, dst);

    DEF_ACC_DATA_T acc;
    init_acc(REDUCTION_ALG, &acc);
    for (off_t i = 0; i < reduction_size; i++) {
        const off_t src_off = i * reduction_stride;
        const DEF_ACC_DATA_T src_val = TO_DEF_ACC_DATA_T(src[src_off]);
        acc = reduce(REDUCTION_ALG, acc, src_val, power);
    }

    // Finalize data, then accumulate into to dst
    DST_DATA_T dst_data = TO_DST(
            finalize(REDUCTION_ALG, convert_float(acc), div, power, eps));
    *dst = dst_data;
}
