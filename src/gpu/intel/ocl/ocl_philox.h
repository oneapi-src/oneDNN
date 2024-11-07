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

#define DT_UNDEF 1
#include "gpu/intel/ocl/ocl_types.h"

uint philox_4x32(long idx, uint seed) {
#define PHILOX_4UINT_ROUND(mul, ctr, key) \
    as_uint4(convert_ulong2(ctr.s31) * mul) ^ (uint4)(ctr.s20 ^ key, 0, 0).s3120

    uint4 ctr = 0;
    const ulong2 ctr_mul = (ulong2)(0xD2511F53uL, 0xCD9E8D57uL);
    const ulong key_add = as_ulong((uint2)(0x9E3779B9u, 0xBB67AE85u));
    const uint16 key0 = (uint16)(seed)
            + as_uint16((ulong8)(key_add))
                    * (uint16)(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const uint4 key1
            = (uint4)(seed) + as_uint4((ulong2)(key_add)) * (uint4)(8, 8, 9, 9);
    ctr = (uint4)(idx & ~3L) + (uint4)(3, 2, 1, 0);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key0.s01);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key0.s23);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key0.s45);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key0.s67);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key0.s89);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key0.sAB);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key0.sCD);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key0.sEF);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key1.s01);
    ctr = PHILOX_4UINT_ROUND(ctr_mul, ctr, key1.s23);
    return ctr[~idx & 3L];
}

ushort philox_8x16(long idx, uint seed) {
    return as_ushort2(philox_4x32(idx >> 1, seed))[idx & 1];
}

uchar philox_16x8(long idx, uint seed) {
    return as_uchar4(philox_4x32(idx >> 2, seed))[idx & 3];
}

#if WITH_SROUND

#if DST_DT_DIGITS > 24
#error "Invalid dst digits"
#endif

float stochastic_round_fwd(float s, long idx, uint seed) {
    if (isnan(s) || isinf(s)) return s;
    uint truncation_mask = 0xffffffff << (24 - DST_DT_DIGITS);
    uint bias_val = sizeof(DST_DATA_T) == 2 ? philox_16x8(idx, seed)
                                            : philox_8x16(idx, seed);
    uint rnd_bias = (uint)(bias_val & ~truncation_mask);
    float r = as_float((as_uint(s) + rnd_bias) & truncation_mask);
    r = fmin(fmax((float)DST_DATA_FLOW, r), (float)DST_DATA_FMAX);
    if (fabs(r) > 0 && fabs(r) < DST_DATA_FMIN) r = 0;
    return r;
}
#endif
