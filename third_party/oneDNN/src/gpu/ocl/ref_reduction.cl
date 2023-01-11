/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#if NDIMS == 6
#define _SRC_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC, x0, x1, x2, x3, x4, x5)
#define _DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#elif NDIMS == 1
#define _SRC_OFF(x0, x1, x2, x3, x4, x5) (x0)
#define _DST_OFF(x0, x1, x2, x3, x4, x5) (x0)
#else
#define _SRC_OFF(x0, x1, ignore, x3, x4, x5) SRC_OFF(x0, x1, x3, x4, x5)
#define _DST_OFF(x0, x1, ignore, x3, x4, x5) DST_OFF(x0, x1, x3, x4, x5)
#endif

// Remove unnecessary loop as performance degradation was observed
// for some cases with multiple additional loops
#if NDIMS == 6
#define ITERATE_OVER_REDUCTION_D2 \
    for_(int d2_off = 0; d2_off < REDUCTION_D2; d2_off++)
#define D2_OFF d2_off
#else
#define ITERATE_OVER_REDUCTION_D2
#define D2_OFF 0
#endif

#define _DST_OFF_MODULO_DIM(x0, x1, x2, x3, x4, x5) \
    ({ \
        int ret_val; \
        if (NDIMS == 1) \
            ret_val = _DST_OFF(x0 % DST_D0, 0, 0, 0, 0, 0); \
        else if (NDIMS == 2) \
            ret_val = _DST_OFF(x0 % DST_D0, x1 % DST_D1, 0, 0, 0, 0); \
        else if (NDIMS == 3) \
            ret_val = _DST_OFF( \
                    x0 % DST_D0, x1 % DST_D1, 0, 0, 0, x5 % DST_D2); \
        else if (NDIMS == 4) \
            ret_val = _DST_OFF( \
                    x0 % DST_D0, x1 % DST_D1, 0, 0, x4 % DST_D2, x5 % DST_D3); \
        else if (NDIMS == 5) \
            ret_val = _DST_OFF(x0 % DST_D0, x1 % DST_D1, 0, x3 % DST_D2, \
                    x4 % DST_D3, x5 % DST_D4); \
        else \
            ret_val = _DST_OFF(x0 % DST_D0, x1 % DST_D1, x2 % DST_D2, \
                    x3 % DST_D3, x4 % DST_D4, x5 % DST_D5); \
        ret_val; \
    })

__kernel void ref_reduce(
        __global SRC_DATA_T *src, __global DST_DATA_T *dst POST_OP_ARGS) {
    int d0 = GWS_GET_D0();
    int d1 = GWS_GET_D1();
    int d2 = GWS_GET_D2();
    int d3 = GWS_GET_D3();
    int d4 = GWS_GET_D4();
    int d5 = GWS_GET_D5();

    DEF_ACC_DATA_T acc = INIT_ACC;
    for_(int d0_off = 0; d0_off < REDUCTION_D0; d0_off++)
    for_(int d1_off = 0; d1_off < REDUCTION_D1; d1_off++)
    ITERATE_OVER_REDUCTION_D2
    for_(int d3_off = 0; d3_off < REDUCTION_D3; d3_off++)
    for_(int d4_off = 0; d4_off < REDUCTION_D4; d4_off++)
    for_(int d5_off = 0; d5_off < REDUCTION_D5; d5_off++)
    {
        const int src_off = _SRC_OFF(d0 + d0_off, d1 + d1_off, d2 + D2_OFF,
                d3 + d3_off, d4 + d4_off, d5 + d5_off);
        acc = ACCUMULATE(acc, TO_DEF_ACC_DATA_T(src[src_off]));
    }

    float res = convert_float(acc);
    res = FINALIZE(res);

    const int dst_off = _DST_OFF_MODULO_DIM(d0, d1, d2, d3, d4, d5);
    const int dst_off_pd = _DST_OFF(d0, d1, d2, d3, d4, d5);

    float dst_val;
#if WITH_SUM
    dst_val = DST_TO_REF(dst[dst_off]);
#endif
#if NDIMS == 4
#if REDUCTION_D1 != 1
    d1 = 0;
    d2 = d4;
    d3 = d5;
#elif REDUCTION_D4 != 1
    d2 = 0;
    d3 = d5;
#elif REDUCTION_D5 != 1
    d2 = d4;
    d3 = 0;
#endif
    APPLY_POST_OPS_SERIAL(
            res, float, dst_val, float, d0, 1, d1, 1, d2, 1, d3, 1, 0, 1, 0, 1);
#elif NDIMS == 5
#if REDUCTION_D1 != 1
    d1 = 0;
#elif REDUCTION_D5 != 1
    d5 = 0;
#endif
    APPLY_POST_OPS_SERIAL(res, float, dst_val, float, d0, 1, d1, 1, d3, 1, d4,
            1, d5, 1, 0, 1);
#else
    APPLY_POST_OPS_SERIAL(res, float, dst_val, float, d0, 1, d1, 1, d2, 1, d3,
            1, d4, 1, d5, 1);
#endif

    if (dst_off_pd != dst_off) res = 0.f;
    dst[dst_off_pd] = TO_DST(res);
}
