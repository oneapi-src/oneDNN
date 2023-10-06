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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#define offset6D(d0, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5) \
    ((d0) * (s0) + (d1) * (s1) + (d2) * (s2) + (d3) * (s3) + (d4) * (s4) \
            + (d5) * (s5))

__kernel void ref_matmul(__global SRC_DATA_T *A, __global WEI_DATA_T *B,
        __global DST_DATA_T *C, __global BIA_DATA_T *bia, __global int *a0,
        __global int *b0, __global int *c0, __global float *src_scales,
        __global float *wei_scales, long wei_scale_stride,
        __global float *dst_scales, long K, long N, long M, long D0, long D1,
        long D2, long bia_stride_d3, long bia_stride_d2, long bia_stride_d1,
        long bia_stride_d0, long bia_stride_m, long bia_stride_n,
        long a_stride_d3, long a_stride_d2, long a_stride_d1, long a_stride_d0,
        long a_stride_m, long a_stride_k, long b_stride_d3, long b_stride_d2,
        long b_stride_d1, long b_stride_d0, long b_stride_k, long b_stride_n,
        long c_stride_d3, long c_stride_d2, long c_stride_d1, long c_stride_d0,
        long c_stride_m, long c_stride_n POST_OP_ARGS) {

    int n = get_global_id(1);
    int mb = get_global_id(2);

#if WITH_SRC_ZPOINTS
    int src_zp = a0[0];
#else
    int src_zp = 0;
#endif
#if WITH_WEI_ZPOINTS
    int wei_zp = b0[0];
#else
    int wei_zp = 0;
#endif
#if WITH_DST_ZPOINTS
    int dst_zp = c0[0];
#else
    int dst_zp = 0;
#endif
    // NOTE: non-standard notation
    // In matmul all dimensions above 2 are considered as batch
    // In below code that deals with broadcast of batch dimensions,
    // d0 means lowest batch dimension and d3 the highest

    // decompose mb into batch dimensions (d0..d3)
    long d3 = mb / D0 / D1 / D2;
    long d2 = (mb / D0 / D1) % D2;
    long d1 = (mb / D0) % D1;
    long d0 = mb % D0;

    for (long m = 0; m < M; ++m) {
        ACC_DATA_T acc = 0;
        for (long k = 0; k < K; ++k) {
#if RUNTIME_DIMS
            long src_off
                    = offset6D(m, k, d0, d1, d2, d3, a_stride_m, a_stride_k,
                            a_stride_d0, a_stride_d1, a_stride_d2, a_stride_d3);
            long wei_off
                    = offset6D(k, n, d0, d1, d2, d3, b_stride_k, b_stride_n,
                            b_stride_d0, b_stride_d1, b_stride_d2, b_stride_d3);
#else
#if NDIMS == 5
            long src_off = SRC_OFF(d2 % SRC_D0, d1 % SRC_D1, d0 % SRC_D2, m, k);
            long wei_off
                    = WEI_OFF(0, d2 % WEI_D0, d1 % WEI_D1, d0 % WEI_D2, k, n);
#elif NDIMS == 4
            long src_off = SRC_OFF(d1 % SRC_D0, d0 % SRC_D1, 0, m, k);
            long wei_off = WEI_OFF(0, d1 % WEI_D0, d0 % WEI_D1, 0, k, n);
#elif NDIMS == 3
            long src_off = SRC_OFF(d0 % SRC_D0, m, 0, 0, k);
            long wei_off = WEI_OFF(0, d0 % WEI_D0, k, 0, 0, n);
#else
            long src_off = SRC_OFF(m, k, 0, 0, 0);
            long wei_off = WEI_OFF(0, k, n, 0, 0, 0);
#endif
#endif
            acc += TO_ACC(SRC_TO_REF(A[src_off]) - src_zp)
                    * TO_ACC(WEI_TO_REF(B[wei_off]) - wei_zp);
        }

#if RUNTIME_DIMS
        long dst_off = offset6D(m, n, d0, d1, d2, d3, c_stride_m, c_stride_n,
                c_stride_d0, c_stride_d1, c_stride_d2, c_stride_d3);
#else
#if NDIMS == 5
        long dst_off = DST_OFF(d2 % DST_D0, d1 % DST_D1, d0 % DST_D2, m, n);
#elif NDIMS == 4
        long dst_off = DST_OFF(d1 % DST_D0, d0 % DST_D1, 0, m, n);
#elif NDIMS == 3
        long dst_off = DST_OFF(d0 % DST_D0, m, 0, 0, n);
#else
        long dst_off = DST_OFF(m, n, 0, 0, 0);
#endif
#endif

#if WITH_BIAS || NON_DEFAULT_ATTRS
        POST_OP_DATA_T temp = (POST_OP_DATA_T)acc;
#if WITH_SRC_SCALES
        temp *= src_scales[0];
#endif
#if WITH_WEI_SCALES
        temp *= wei_scales[wei_scale_stride * n];
#endif
#if WITH_BIAS
        long bia_off = offset6D(m, n, d0, d1, d2, d3, bia_stride_m,
                bia_stride_n, bia_stride_d0, bia_stride_d1, bia_stride_d2,
                bia_stride_d3);
        temp += BIA_TO_REF(bia[bia_off]);
#endif // WITH_BIAS

        float dst_data;
#if WITH_SUM
        dst_data = convert_float(DATA_TO_REF(C[dst_off]));
#endif // WITH_SUM

        float po_acc = convert_float(temp);

        if (DST_NDIMS == 2)
            APPLY_POST_OPS_SERIAL(po_acc, float, dst_data, float, m, 1, n, 1, 0,
                    1, 0, 1, 0, 1, 0, 1);
        if (DST_NDIMS == 3)
            APPLY_POST_OPS_SERIAL(po_acc, float, dst_data, float, d0, 1, m, 1,
                    n, 1, 0, 1, 0, 1, 0, 1);
        if (DST_NDIMS == 4)
            APPLY_POST_OPS_SERIAL(po_acc, float, dst_data, float, d1, 1, d0, 1,
                    m, 1, n, 1, 0, 1, 0, 1);
        if (DST_NDIMS == 5)
            APPLY_POST_OPS_SERIAL(po_acc, float, dst_data, float, d2, 1, d1, 1,
                    d0, 1, m, 1, n, 1, 0, 1);
        if (DST_NDIMS == 6)
            APPLY_POST_OPS_SERIAL(po_acc, float, dst_data, float, d3, 1, d2, 1,
                    d1, 1, d0, 1, m, 1, n, 1);

#if WITH_DST_SCALES
        po_acc /= dst_scales[0];
#endif
        po_acc += dst_zp;
        C[dst_off] = TO_DST(po_acc);
#else // WITH_BIAS || NON_DEFAULT_ATTRS
        C[dst_off] = TO_DST(acc);
#endif // WITH_BIAS || NON_DEFAULT_ATTRS
    }
}
