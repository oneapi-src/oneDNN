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

#ifndef GPU_INTEL_OCL_OCL_IO_H
#define GPU_INTEL_OCL_OCL_IO_H

#include "gpu/intel/ocl/ocl_conversion.h"
#include "gpu/intel/ocl/ocl_custom_types.h"
#include "gpu/intel/ocl/ocl_utils.h"

//******* Always-available load/writes *********//

#define DEF_load(dst_dt, src_dt) \
    void __attribute__((overloadable)) \
            load(__private dst_dt *dst, __global src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    }
#define DEF_write(dst_dt, src_dt) \
    void __attribute__((overloadable)) \
            write(__global dst_dt *dst, __private src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    }

// Loads
DEF_load(float, float);
DEF_load(float, char);
DEF_load(float, uchar);
DEF_load(int, char);
DEF_load(int, uchar);
DEF_load(float, bf16);

// Writes
DEF_write(char, float);
DEF_write(uchar, float);
DEF_write(bf16, float);
DEF_write(float, float);

DEF_write(char, int);
DEF_write(uchar, int);
DEF_write(bf16, int);
DEF_write(float, int);

//******* Conditionally-available load/writes *********//

#ifdef cl_khr_fp16
// Loads
DEF_load(half, half);
DEF_load(float, half);

// Writes
DEF_write(half, float);
DEF_write(half, int);
DEF_write(half, half);
DEF_write(char, half);
DEF_write(uchar, half);
DEF_write(float, half);
DEF_write(bf16, half);

#ifdef MATH_UTILS_DECLARE_BF8
// Loads
DEF_load(half, f8_e5m2);
DEF_load(float, f8_e5m2);
DEF_load(half, f8_e4m3);
DEF_load(float, f8_e4m3);

// Writes
DEF_write(f8_e5m2, half);
DEF_write(f8_e5m2, float);
DEF_write(f8_e5m2, int);
DEF_write(f8_e4m3, half);
DEF_write(f8_e4m3, float);
DEF_write(f8_e4m3, int);

#endif // MATH_UTILS_DECLARE_BF8
#endif // cl_khr_fp16

#ifdef cl_khr_fp64
DEF_load(float, double); // Needed for src=f64, dst=f32
DEF_load(double, bf16);
DEF_load(double, float);
DEF_load(double, double);

DEF_write(char, double);
DEF_write(uchar, double);
DEF_write(bf16, double);
DEF_write(float, double);
DEF_write(double, double);
DEF_write(double, float);
#endif

//******* Interactions between extended data types *********//

#if defined(cl_khr_fp16) && defined(cl_khr_fp64)

DEF_load(double, half);
DEF_write(half, double);

#ifdef MATH_UTILS_DECLARE_BF8
DEF_load(double, f8_e5m2);
DEF_load(double, f8_e4m3);
DEF_write(f8_e5m2, double);
DEF_write(f8_e4m3, double);
#endif // MATH_UTILS_DECLARE_BF8
#endif

#endif
