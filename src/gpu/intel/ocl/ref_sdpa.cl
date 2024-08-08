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
#include "gpu/intel/ocl/sdpa_utils.h"

__kernel void ref_sdpa(const __global QRY_DATA_T *Q,
        const __global KEY_DATA_T *K, const __global VAL_DATA_T *V,
        __global DST_DATA_T *dst, const __global SCALE_DATA_T *scale_ptr,
        const __global MSK_DATA_T *mask, long nv, long nd) {

    long q = get_global_id(0);
    long b0 = get_global_id(1);
    long b1 = get_global_id(2);

    float s[SIZE_K];

    // Multiply (row of Q)*K
    for (long k = 0; k < SIZE_K; k++) {
        float acc = 0;
        for (long h = 0; h < nd; h++) {
            long qry_off = QRY_OFF(b1 % QRY_D0, b0 % QRY_D1, q, h);
            long key_off = KEY_OFF(b1 % KEY_D0, b0 % KEY_D1, h, k);

            acc += convert_float(Q[qry_off]) * convert_float(K[key_off]);
        }
        s[k] = acc;
    }

    // Scale + shift + softmax
    float s_sum = 0;

#if WITH_ATTN_SCALE
    SCALE_DATA_T scale = *scale_ptr;
#if INVERT_SCALE
    scale = 1.f / scale;
#endif
    for (long k = 0; k < SIZE_K; k++) {
        s[k] *= scale;
#endif

#if WITH_ATTN_MASK
        long msk_off = MSK_OFF(b1 % MSK_D0, b0 % MSK_D1, q, k);
        s[k] += convert_float(mask[msk_off]);
#endif
        s[k] = exp(s[k]);
        s_sum += s[k];
    }

    for (long k = 0; k < SIZE_K; k++)
        s[k] /= s_sum;

    // Multiply (row of S)*V
    for (long v = 0; v < nv; v++) {
        float acc = 0;
        for (long k = 0; k < SIZE_K; k++) {
            long val_off = VAL_OFF(b1 % VAL_D0, b0 % VAL_D1, k, v);
            acc += convert_float(V[val_off]) * s[k];
        }

        long dst_off = DST_OFF(b1 % DST_D0, b0 % DST_D1, 0, q, v);
        dst[dst_off] = TO_DST(acc);
    }
}
