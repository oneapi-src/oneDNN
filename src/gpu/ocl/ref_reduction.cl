/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gpu/ocl/ocl_types.h"

#if defined(IS_MAX)
#define INIT_ACC -INFINITY
#elif defined(IS_MIN)
#define INIT_ACC INFINITY
#elif defined(IS_MUL)
#define INIT_ACC 1.0f
#else
#define INIT_ACC 0.0f
#endif

#if defined(IS_MAX)
#define ACCUMULATE(x, y) fmax(x, y)
#elif defined(IS_MIN)
#define ACCUMULATE(x, y) fmin(x, y)
#elif defined(IS_MEAN) || defined(IS_SUM)
#define ACCUMULATE(x, y) (x + y)
#elif defined(IS_MUL)
#define ACCUMULATE(x, y) (x * y)
#else
#define ACCUMULATE(x, y) (x + pow(fabs(y), POWER))
#endif

#if defined(IS_MEAN)
#define FINALIZE(x) (x / DIV)
#elif defined(IS_LP_MAX)
#define FINALIZE(x) rootn(fmax(x, EPS), POWER)
#elif defined(IS_LP_SUM)
#define FINALIZE(x) rootn(x + EPS, POWER)
#elif defined(IS_P_MAX)
#define FINALIZE(x) fmax(x, EPS)
#elif defined(IS_P_SUM)
#define FINALIZE(x) (x + EPS)
#else
#define FINALIZE(x) (x)
#endif

#if !defined(SRC_OFF) && !defined(DST_OFF) && NDIMS == 1
#define SRC_OFF(n, c, d, h, w) (n)
#define DST_OFF(n, c, d, h, w) (n)
#endif

KERNEL_ATTR
__kernel void ref_reduce(__global SRC_DATA_T *src, __global DST_DATA_T *dst) {
    int n = GWS_GET_IN();
    int c = GWS_GET_IC();
    int d = GWS_GET_ID();
    int h = GWS_GET_IH();
    int w = GWS_GET_IW();

    float acc = INIT_ACC;
    for_(int n_off = 0; n_off < REDUCTION_IN; n_off++)
    for_(int c_off = 0; c_off < REDUCTION_IC; c_off++)
    for_(int d_off = 0; d_off < REDUCTION_ID; d_off++)
    for_(int h_off = 0; h_off < REDUCTION_IH; h_off++)
    for_(int w_off = 0; w_off < REDUCTION_IW; w_off++)
    {
        acc = ACCUMULATE(acc,
                CONVERT_FLOAT_T(src[SRC_OFF(n + n_off, c + c_off, d + d_off,
                        h + h_off, w + w_off)]));
    }

    acc = FINALIZE(acc);
    dst[DST_OFF(n, c, d, h, w)] = TO_DST(acc);
}
