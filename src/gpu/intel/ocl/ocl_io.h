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

#ifndef GPU_INTEL_OCL_OCL_IO_H
#define GPU_INTEL_OCL_OCL_IO_H

#include "gpu/intel/ocl/ocl_conversion.h"
#include "gpu/intel/ocl/ocl_custom_types.h"
#include "gpu/intel/ocl/ocl_utils.h"

//******* Always-available load/writes *********//

#define BLOCK_READ_FUNC_1 intel_sub_group_block_read_uc
#define BLOCK_READ_FUNC_2 intel_sub_group_block_read_us
#define BLOCK_READ_FUNC_4 intel_sub_group_block_read
#define BLOCK_READ_FUNC_8 intel_sub_group_block_read_ul

#define BLOCK_DT_1 uchar
#define BLOCK_DT_2 ushort
#define BLOCK_DT_4 uint
#define BLOCK_DT_8 ulong

#define SIZE_undef_data 1
#define SIZE_char 1
#define SIZE_uchar 1
#define SIZE_f8_e5m2 1
#define SIZE_f8_e4m3 1
#define SIZE_e8m0 1
#define SIZE_bf16 2
#define SIZE_half 2
#define SIZE_int 4
#define SIZE_float 4
#define SIZE_double 8

#define SIZE(dt) CONCAT2(SIZE_, dt)
#define BLOCK_DT(dt) CONCAT2(BLOCK_DT_, SIZE(dt))
#define BLOCK_READ_FUNC(dt) CONCAT2(BLOCK_READ_FUNC_, SIZE(dt))

#define DEF_load(dst_dt, src_dt) \
    void __attribute__((overloadable)) \
            load(__private dst_dt *dst, __global const src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    dst_dt __attribute__((overloadable, warn_unused_result)) \
            load(dst_dt dst, __global const src_dt *val) { \
        return CONCAT2(into_, dst_dt)(*val); \
    } \
    __attribute__((overloadable)) void block_load( \
            __private dst_dt *dst, __global src_dt *src) { \
        __global BLOCK_DT(src_dt) *data = (__global BLOCK_DT(src_dt) *)(src); \
        BLOCK_DT(src_dt) block_val = BLOCK_READ_FUNC(src_dt)(data); \
        src_dt src_val = CONCAT2(as_, src_dt)(block_val); \
        *dst = CONCAT2(into_, dst_dt)(src_val); \
    }

#define DEF_write(dst_dt, src_dt) \
    void __attribute__((overloadable)) \
            write(__global dst_dt *dst, __private src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    void __attribute__((overloadable)) \
            write(__global dst_dt *dst, __private src_dt val) { \
        *dst = CONCAT2(into_, dst_dt)(val); \
    }

// Loads
DEF_load(float, int);
DEF_load(float, float);
DEF_load(float, char);
DEF_load(float, uchar);
DEF_load(int, char);
DEF_load(int, uchar);
DEF_load(int, int);
DEF_load(float, bf16);

// Included for compile time compatibility
DEF_load(int, undef_data);
DEF_load(float, undef_data);

// Writes
DEF_write(char, float);
DEF_write(uchar, float);
DEF_write(bf16, float);
DEF_write(float, float);

DEF_write(char, int);
DEF_write(uchar, int);
DEF_write(bf16, int);
DEF_write(int, int);
DEF_write(float, int);
DEF_write(int, float);

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

// Writes
DEF_write(f8_e5m2, half);
DEF_write(f8_e5m2, float);
DEF_write(f8_e5m2, int);
#endif // MATH_UTILS_DECLARE_BF8

#ifdef MATH_UTILS_DECLARE_HF8
// Loads
DEF_load(half, f8_e4m3);
DEF_load(float, f8_e4m3);

// Writes
DEF_write(f8_e4m3, half);
DEF_write(f8_e4m3, float);
DEF_write(f8_e4m3, int);

#endif // MATH_UTILS_DECLARE_HF8

#ifdef MATH_UTILS_DECLARE_E8M0
// Loads
DEF_load(float, e8m0);

#endif

#endif // cl_khr_fp16

#ifdef cl_khr_fp64
// Included for compile time compatibility
DEF_load(double, undef_data);

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
DEF_write(f8_e5m2, double);
#endif // MATH_UTILS_DECLARE_BF8

#ifdef MATH_UTILS_DECLARE_HF8
DEF_load(double, f8_e4m3);
DEF_write(f8_e4m3, double);
#endif // MATH_UTILS_DECLARE_HF8
#endif

#endif
