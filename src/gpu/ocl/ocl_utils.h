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

#ifndef GPU_OCL_OCL_UTILS_H
#define GPU_OCL_OCL_UTILS_H

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)
#define CONCAT3(a, b, c) CONCAT2(CONCAT2(a, b), c)

#ifdef cl_khr_fp64
#define IF_DOUBLE_SUPPORTED(x) x
#else
#define IF_DOUBLE_SUPPORTED(x)
#endif

#ifdef cl_khr_fp16
#define IF_HALF_SUPPORTED(x) x
#else
#define IF_HALF_SUPPORTED(x)
#endif

#ifdef OCL_DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__);
#else
#define DEBUG_PRINT(...)
#endif

// Defines (for example) float_zero, float_one, float_min, and float_max
// Can be used in a data-type agnostic way with the SPECIAL macro below
#define DEF_special_vals(dt, zero_val, one_val, min_val, max_val) \
    __constant dt CONCAT2(dt, _zero) = zero_val; \
    __constant dt CONCAT2(dt, _one) = one_val; \
    __constant dt CONCAT2(dt, _min) = min_val; \
    __constant dt CONCAT2(dt, _max) = max_val;

DEF_special_vals(float, 0.0f, 1.0f, -FLT_MAX, FLT_MAX);
DEF_special_vals(int, 0, 1, INT_MIN, INT_MAX);
IF_DOUBLE_SUPPORTED(DEF_special_vals(double, 0.0, 1.0, -DBL_MAX, DBL_MAX));
IF_HALF_SUPPORTED(DEF_special_vals(half, 0.0h, 1.0h, -HALF_MAX, HALF_MAX));

#define SPECIAL(dt, val) CONCAT3(dt, _, val)

#endif // GPU_OCL_OCL_UTILS_H