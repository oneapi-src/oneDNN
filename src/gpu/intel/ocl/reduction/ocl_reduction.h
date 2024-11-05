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

#ifndef GPU_INTEL_OCL_REDUCTION_OCL_REDUCTION_H
#define GPU_INTEL_OCL_REDUCTION_OCL_REDUCTION_H

#include "gpu/intel/ocl/ocl_utils.h"

// ********* Integer vs floating point functions *********** //

// floating point types use fmax/fmin/fabs, but integer
// types use max/min/abs. Create aliases so we can use
// max/min/abs for all types.
#define DEF_fp_minmax_abs(dt) \
    dt __attribute__((overloadable)) min(dt lhs, dt rhs) { \
        return fmin(lhs, rhs); \
    } \
    dt __attribute__((overloadable)) max(dt lhs, dt rhs) { \
        return fmax(lhs, rhs); \
    } \
    dt __attribute__((overloadable)) abs(dt val) { return fabs(val); }

DEF_fp_minmax_abs(float);
IF_HALF_SUPPORTED(DEF_fp_minmax_abs(half));
IF_DOUBLE_SUPPORTED(DEF_fp_minmax_abs(double));

#undef DEF_fp_minmax_abs

// *********** Generic reduction functions ************* //

#define DEF_reduce(dt) \
    dt __attribute__((overloadable)) \
            reduce(int alg, dt lhs, dt rhs, float power) { \
        switch (alg) { \
            case (REDUCTION_MAX): return max(lhs, rhs); \
            case (REDUCTION_MIN): return min(lhs, rhs); \
            case (REDUCTION_MUL): return lhs * rhs; \
            case (REDUCTION_LP_NORM_MAX): \
            case (REDUCTION_LP_NORM_SUM): \
            case (REDUCTION_LP_NORM_POWER_P_MAX): \
            case (REDUCTION_LP_NORM_POWER_P_SUM): \
            case (REDUCTION_LP_NORM_POWER_P): \
                return (lhs + (dt)pow(convert_float(abs(rhs)), power)); \
            case (REDUCTION_SUM): \
            case (REDUCTION_FINAL_MAX): \
            case (REDUCTION_FINAL_SUM): \
            case (REDUCTION_PTH_ROOT_MAX): \
            case (REDUCTION_PTH_ROOT_SUM): \
            case (REDUCTION_MEAN): return lhs + rhs; \
            default: return lhs; \
        } \
    }

DEF_reduce(float);
DEF_reduce(int);
IF_DOUBLE_SUPPORTED(DEF_reduce(double));
IF_HALF_SUPPORTED(DEF_reduce(half));

#undef DEF_reduce

#if !DETERMINISTIC && ATOMICS_SUPPORTED
// Atomic reduction does not support mul... Must be checked on the caller side
#define DEF_atomic_reduce(dt) \
    dt __attribute__((overloadable)) atomic_reduce( \
            int alg, __global ATOMIC(dt) * dst, dt rhs, float power) { \
        switch (alg) { \
            case (REDUCTION_MAX): return atomic_max_global(dst, rhs); \
            case (REDUCTION_MIN): return atomic_min_global(dst, rhs); \
            case (REDUCTION_LP_NORM_MAX): \
            case (REDUCTION_LP_NORM_SUM): \
            case (REDUCTION_LP_NORM_POWER_P_MAX): \
            case (REDUCTION_LP_NORM_POWER_P_SUM): \
            case (REDUCTION_LP_NORM_POWER_P): \
                return atomic_add_global( \
                        dst, (dt)pow(convert_float(abs(rhs)), power)); \
            case (REDUCTION_SUM): \
            case (REDUCTION_FINAL_MAX): \
            case (REDUCTION_FINAL_SUM): \
            case (REDUCTION_PTH_ROOT_MAX): \
            case (REDUCTION_PTH_ROOT_SUM): \
            case (REDUCTION_MEAN): return atomic_add_global(dst, rhs); \
        } \
        printf("Atomic reduction on unsupported alg: %d\n", alg); \
        return SPECIAL(dt, zero); \
    }

#if ATOMIC_FLOAT_SUPPORTED
DEF_atomic_reduce(float);
#endif
DEF_atomic_reduce(int);

#undef DEF_atomic_reduce
#endif

// ************ Initialization functions ************* //

// These functions have to be defined for all vector types, since we
// can't get the address of vector elements to pass here
#define internal_def_init_acc(dt, min_val, max_val, zero_val, one_val) \
    void __attribute__((overloadable)) init_acc(int alg, dt *val) { \
        switch (alg) { \
            case (REDUCTION_MAX): *val = min_val; return; \
            case (REDUCTION_MIN): *val = max_val; return; \
            case (REDUCTION_MUL): *val = one_val; return; \
            default: *val = zero_val; return; \
        } \
    }

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)

#define DEF_init_acc(dt, min_val, max_val, zero_val, one_val) \
    internal_def_init_acc(dt, min_val, max_val, zero_val, one_val); \
    internal_def_init_acc( \
            CONCAT2(dt, 2), min_val, max_val, zero_val, one_val); \
    internal_def_init_acc( \
            CONCAT2(dt, 4), min_val, max_val, zero_val, one_val); \
    internal_def_init_acc(CONCAT2(dt, 8), min_val, max_val, zero_val, one_val)

DEF_init_acc(float, -FLT_MAX, FLT_MAX, 0.0f, 1.0f);
DEF_init_acc(int, INT_MIN, INT_MAX, 0, 1);

IF_DOUBLE_SUPPORTED(DEF_init_acc(double, -DBL_MAX, DBL_MAX, 0.0, 1.0));
IF_HALF_SUPPORTED(DEF_init_acc(half, -HALF_MAX, HALF_MAX, 0.0h, 1.0h));

#undef DEF_init_acc
#undef internal_def_init_acc

// ********** Finalization functions ********** //

float finalize(int alg, float val, float div, float power, float eps) {
    switch (alg) {
        case (REDUCTION_MEAN): return val / div;
        case (REDUCTION_LP_NORM_MAX):
        case (REDUCTION_PTH_ROOT_MAX): return rootn(fmax(val, eps), power);
        case (REDUCTION_LP_NORM_SUM):
        case (REDUCTION_PTH_ROOT_SUM): return rootn(val + eps, power);
        case (REDUCTION_LP_NORM_POWER_P_MAX):
        case (REDUCTION_FINAL_MAX): return fmax(val, eps);
        case (REDUCTION_LP_NORM_POWER_P_SUM):
        case (REDUCTION_FINAL_SUM): return val + eps;
        default: return val;
    }
}

#endif
