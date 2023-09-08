/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "gpu/ocl/binary_types.h"
#include "gpu/ocl/dispatch.h"

#if IS_TENSOR_OP && IS_DENSE && IS_SAME_MD && !WITH_BINARY_POST_OP
KERNEL_ATTR
__kernel void ref_binary(__global DATA_T *src0, __global DATA_T *src1,
        __global DST_DATA_T *dst POST_OP_ARGS, __global float *src0_scale,
        __global float *src1_scale) {
    int off = GWS_GET_IDX();

    float tmp_src0 = SRC0_TO_FLOAT(src0[off]);
    float tmp_src1 = SRC1_TO_FLOAT(src1[off]);
    float d = 0;

#if WITH_SRC0_SCALE
    tmp_src0 = tmp_src0 * (*src0_scale);
#endif
#if WITH_SRC1_SCALE
    tmp_src1 = tmp_src1 * (*src1_scale);
#endif

    d = get_eltwise_op(tmp_src0, tmp_src1);

    float dst_data;
#if WITH_SUM
    dst_data = CONVERT_FLOAT_T(dst[off]);
#endif

    APPLY_POST_OPS_SERIAL(
            d, float, dst_data, float, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    dst[off] = TO_DST(d);
}
#else
KERNEL_ATTR
__kernel void ref_binary(__global SRC0_DATA_T *src0, __global SRC1_DATA_T *src1,
        __global DST_DATA_T *dst POST_OP_ARGS, __global float *src0_scale,
        __global float *src1_scale) {

    // since gws = no. of total elems in A, id will be the logical offset
    int dims0[6] = {0};
    dims0[0] = GWS_GET_D0();
    dims0[1] = GWS_GET_D1();
    dims0[2] = GWS_GET_D2();
    dims0[3] = GWS_GET_D3();
    dims0[4] = GWS_GET_D4();
    dims0[5] = GWS_GET_D5();
    int d1_block = GWS_GET_D1_BLOCK();
    int dims0_po[6]
            = {dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]};
    int d1_init = GWS_GET_D1();

    int dst_off = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
#if TENSOR_OP
    int src0_off = SRC0_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    int src1_off = SRC1_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
#else
    int src0_off
            = SRC0_OFF(dims0[0] * !SRC0_BCAST_DIM0, dims0[1] * !SRC0_BCAST_DIM1,
                    dims0[2] * !SRC0_BCAST_DIM2, dims0[3] * !SRC0_BCAST_DIM3,
                    dims0[4] * !SRC0_BCAST_DIM4, dims0[5] * !SRC0_BCAST_DIM5);
    int src1_off
            = SRC1_OFF(dims0[0] * !SRC1_BCAST_DIM0, dims0[1] * !SRC1_BCAST_DIM1,
                    dims0[2] * !SRC1_BCAST_DIM2, dims0[3] * !SRC1_BCAST_DIM3,
                    dims0[4] * !SRC1_BCAST_DIM4, dims0[5] * !SRC1_BCAST_DIM5);
#endif

    // SRC1_D1 = IC for SRC1, using the dispatch ap
    int block_size = d1_block;

    if (dims0[0] >= DST_D0) {
        for (int ic = 0; ic < block_size; ++ic) {
            dst[dst_off] = TO_DST(0.0f);
            dst_off++;
        }

        return;
    }

    if (d1_init + block_size <= DST_D1) {
        for (int ic = 0; ic < block_size; ++ic) {
            float tmp_src0 = SRC0_TO_FLOAT(src0[src0_off]);
            float tmp_src1 = SRC1_TO_FLOAT(src1[src1_off]);
            float d = 0;

#if WITH_SRC0_SCALE
            tmp_src0 = tmp_src0 * (*src0_scale);
#endif
#if WITH_SRC1_SCALE
            tmp_src1 = tmp_src1 * (*src1_scale);
#endif
            d = get_eltwise_op(tmp_src0, tmp_src1);

            float dst_data;
#if WITH_SUM
            dst_data = CONVERT_FLOAT_T(dst[dst_off]);
#endif
            APPLY_POST_OPS_SERIAL(d, float, dst_data, float, dims0_po[0], 1,
                    dims0_po[1], 1, dims0_po[2], 1, dims0_po[3], 1, dims0_po[4],
                    1, dims0_po[5], 1);

            dst[dst_off] = TO_DST(d);

#if USE_UNROLL_16B || SRC0_UNROLL_16B
            src0_off++;
            dst_off++;
            ++dims0_po[1];
            if (USE_UNROLL_16B && (SRC1_D1 > 1)) {
                src1_off++;
            } else if (SRC0_UNROLL_16B && (SRC1_D1 > 1)) {
                src1_off += SRC1_S1_0; // Equilvalent stride in plain format
            }
#endif
        }
    } else {
        for (int ic = 0; ic < DST_D1 - d1_init; ic++) {
            float tmp_src0 = SRC0_TO_FLOAT(src0[src0_off]);
            float tmp_src1 = SRC1_TO_FLOAT(src1[src1_off]);
            float d = 0;

#if WITH_SRC0_SCALE
            tmp_src0 = tmp_src0 * (*src0_scale);
#endif
#if WITH_SRC1_SCALE
            tmp_src1 = tmp_src1 * (*src1_scale);
#endif
            d = get_eltwise_op(tmp_src0, tmp_src1);

            float dst_data;
#if WITH_SUM
            dst_data = CONVERT_FLOAT_T(dst[dst_off]);
#endif
            APPLY_POST_OPS_SERIAL(d, float, dst_data, float, dims0_po[0], 1,
                    dims0_po[1], 1, dims0_po[2], 1, dims0_po[3], 1, dims0_po[4],
                    1, dims0_po[5], 1);

            dst[dst_off] = TO_DST(d);

#if USE_UNROLL_16B || SRC0_UNROLL_16B
            src0_off++;
            dst_off++;
            ++dims0_po[1];
            if (USE_UNROLL_16B && (SRC1_D1 > 1)) {
                src1_off++;
            } else if (SRC0_UNROLL_16B && (SRC1_D1 > 1)) {
                src1_off += SRC1_S1_0; // Equilvalent stride in plain format
            }
#endif
        }
#if DST_D1 != DST_PD1
        for (int ic = 0; ic < min(DST_PD1 - DST_D1, block_size); ic++) {
            dst[dst_off] = TO_DST(0.0f);
            dst_off++;
        }
#endif
    }
}
#endif
