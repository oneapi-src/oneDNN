/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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
#include "gpu/intel/ocl/ocl_philox.cl"

#define DT_UNDEF 1

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
