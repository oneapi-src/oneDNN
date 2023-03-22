/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#if defined(IS_MAX)
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_MIN)
#elif defined(IS_MIN)
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_MAX)
#elif defined(IS_MUL)
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_ONE)
#else
#define INIT_ACC TO_DEF_ACC_DATA_T(DATA_ZERO)
#endif

#if defined(IS_MAX)
#if defined(SRC_DT_S8) || defined(SRC_DT_U8)
#define ACCUMULATE(x, y) max(x, y)
#else
#define ACCUMULATE(x, y) fmax(x, y)
#endif
#elif defined(IS_MIN)
#if defined(SRC_DT_S8) || defined(SRC_DT_U8)
#define ACCUMULATE(x, y) min(x, y)
#else
#define ACCUMULATE(x, y) fmin(x, y)
#endif
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

#define _SRC_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC, x0, x1, x2, x3, x4, x5)
#define _DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)

__kernel void ref_reduce(
        __global SRC_DATA_T *src, __global DST_DATA_T *dst POST_OP_ARGS) {
    int d0 = GWS_GET_D0();
    int d1 = GWS_GET_D1();
    int d2 = GWS_GET_D2();
    int d3 = GWS_GET_D3();
    int d4 = GWS_GET_D4();
    int d5 = GWS_GET_D5();

    // If the index combination is supposed to be zero-padded, write a zero and quit
    const int dst_off = _DST_OFF(d0, d1, d2, d3, d4, d5);
    if (d0 >= DST_D0 || d1 >= DST_D1 || d2 >= DST_D2 || d3 >= DST_D3
            || d4 >= DST_D4 || d5 >= DST_D5) {
        dst[dst_off] = DATA_ZERO;
        return;
    }

    DEF_ACC_DATA_T acc = INIT_ACC;
    for_(int d0_off = 0; d0_off < REDUCTION_D0; d0_off++)
    for_(int d1_off = 0; d1_off < REDUCTION_D1; d1_off++)
    for_(int d2_off = 0; d2_off < REDUCTION_D2; d2_off++)
    for_(int d3_off = 0; d3_off < REDUCTION_D3; d3_off++)
    for_(int d4_off = 0; d4_off < REDUCTION_D4; d4_off++)
    for_(int d5_off = 0; d5_off < REDUCTION_D5; d5_off++)
    {
        const int src_off = _SRC_OFF(d0 + d0_off, d1 + d1_off, d2 + d2_off,
                d3 + d3_off, d4 + d4_off, d5 + d5_off);
        acc = ACCUMULATE(acc, TO_DEF_ACC_DATA_T(src[src_off]));
    }

    float res = convert_float(acc);
    res = FINALIZE(res);

    float dst_val;
#if WITH_SUM
    dst_val = DST_TO_REF(dst[dst_off]);
#endif

    APPLY_POST_OPS_SERIAL(res, float, dst_val, float, d0, 1, d1, 1, d2, 1, d3,
            1, d4, 1, d5, 1);

    dst[dst_off] = TO_DST(res);
}
