/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include "ocl/ocl_types.h"

#undef DST_OFF

#define SRC0_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC0, x0, x1, x2, x3, x4, x5)
#define SRC1_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC1, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)

__kernel void ref_binary(
        __global DATA_T *src0, __global DATA_T *src1, __global DATA_T *dst) {

    // since gws = no. of total elems in A, id will be the logical offset
    int id = get_global_id(0);
    int dims0[6] = {0};
    // dimensions passed as separate values due to opencl constraints
    // creating an array for writing shorter and manageable code
    // done for dims_src0 and bcast_dims
    int dims_src0[] = {DIM0, DIM1, DIM2, DIM3, DIM4, DIM5};

    // calculating dims from logical offset for src0
    for (int d = NDIMS - 1; d >= 0; --d) {
        dims0[d] = id % dims_src0[d];
        id /= dims_src0[d];
    }

    int src0_off = SRC0_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    int src1_off = 0;
    int dst_off = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);

    // IS_TENSOR_OP is true in case of no broadcasting for src1
#if IS_TENSOR_OP
    src1_off = SRC1_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
#else
    int bcast_dims[] = {BCAST_DIM0, BCAST_DIM1, BCAST_DIM2, BCAST_DIM3,
            BCAST_DIM4, BCAST_DIM5};

    for (int d = 0; d < NDIMS; ++d) {
        dims0[d] *= (!bcast_dims[d]);
    }

    src1_off = SRC1_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
#endif

    // using tmp vars to handle float calculations for bf16 datatypes
    // DATA_TO_REF is a macro defined in ocl_types.h according to datatype
    POST_OP_DATA_T tmp_src0 = DATA_TO_REF(src0[src0_off]);
    POST_OP_DATA_T tmp_src1 = DATA_TO_REF(src1[src1_off]);

#if IS_ADD
    dst[dst_off] = CONVERT_DATA_T(tmp_src0 + tmp_src1);
#elif IS_MUL
    dst[dst_off] = CONVERT_DATA_T(tmp_src0 * tmp_src1);
#endif
}
