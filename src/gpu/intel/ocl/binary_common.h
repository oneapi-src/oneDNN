/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_BINARY_COMMON_H
#define GPU_INTEL_OCL_BINARY_COMMON_H

#include "gpu/intel/ocl/dispatch.h"
#include "gpu/intel/ocl/ocl_post_ops.h"
#include "gpu/intel/ocl/ocl_types.h"
#include "gpu/intel/ocl/ocl_utils.h"

#undef DST_OFF
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#define SRC0_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC0, x0, x1, x2, x3, x4, x5)
#define SRC1_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC1, x0, x1, x2, x3, x4, x5)

#define DEF_binary_op(dt, special_dt) \
    dt __attribute__((overloadable)) binary_op(int alg, dt src0, dt src1) { \
        switch (alg) { \
            case (BINARY_ADD): return src0 + src1; \
            case (BINARY_MUL): return src0 * src1; \
            case (BINARY_MAX): return max(src0, src1); \
            case (BINARY_MIN): return min(src0, src1); \
            case (BINARY_DIV): return src0 / src1; \
            case (BINARY_SUB): return src0 - src1; \
            case (BINARY_GE): \
                return (src0 >= src1) ? SPECIAL(special_dt, one) \
                                      : SPECIAL(special_dt, zero); \
            case (BINARY_GT): \
                return (src0 > src1) ? SPECIAL(special_dt, one) \
                                     : SPECIAL(special_dt, zero); \
            case (BINARY_LE): \
                return (src0 <= src1) ? SPECIAL(special_dt, one) \
                                      : SPECIAL(special_dt, zero); \
            case (BINARY_LT): \
                return (src0 < src1) ? SPECIAL(special_dt, one) \
                                     : SPECIAL(special_dt, zero); \
            case (BINARY_EQ): \
                return (src0 == src1) ? SPECIAL(special_dt, one) \
                                      : SPECIAL(special_dt, zero); \
            case (BINARY_NE): \
                return (src0 != src1) ? SPECIAL(special_dt, one) \
                                      : SPECIAL(special_dt, zero); \
        } \
        DEBUG_PRINT("Invalid binary op: %d\n", alg); \
        return SPECIAL(special_dt, max); \
    }

DEF_binary_op(float, float);
DEF_binary_op(float2, float);
DEF_binary_op(float4, float);
DEF_binary_op(float8, float);
#undef DEF_binary_op
#endif
