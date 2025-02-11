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

#define BLOCK_WRITE_FUNC_1 intel_sub_group_block_write_uc
#define BLOCK_WRITE_FUNC_2 intel_sub_group_block_write_us
#define BLOCK_WRITE_FUNC_4 intel_sub_group_block_write
#define BLOCK_WRITE_FUNC_8 intel_sub_group_block_write_ul

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
#define BLOCK_WRITE_FUNC(dt) CONCAT2(BLOCK_WRITE_FUNC_, SIZE(dt))

#define BLOCK_DT_N(dt, n) CONCAT2(BLOCK_DT(dt), n)
#define BLOCK_READ_FUNC_N(dt, n) CONCAT2(BLOCK_READ_FUNC(dt), n)
#define BLOCK_WRITE_FUNC_N(dt, n) CONCAT2(BLOCK_WRITE_FUNC(dt), n)

#define DECLARE_AS_BLOCK(t) \
    BLOCK_DT(t) __attribute__((overloadable)) as_block_data(t a) { \
        return CONCAT2(as_, BLOCK_DT(t))(a); \
    }

#define DECLARE_AS_STRUCT_BLOCK(t) \
    BLOCK_DT(t) __attribute__((overloadable)) as_block_data(t a) { \
        return CONCAT2(as_, BLOCK_DT(t))(a.data); \
    }

DECLARE_AS_BLOCK(char)
DECLARE_AS_BLOCK(uchar)
DECLARE_AS_STRUCT_BLOCK(f8_e5m2)
DECLARE_AS_STRUCT_BLOCK(f8_e4m3)
DECLARE_AS_STRUCT_BLOCK(e8m0)
DECLARE_AS_STRUCT_BLOCK(bf16)
DECLARE_AS_BLOCK(half)
DECLARE_AS_BLOCK(int)
DECLARE_AS_BLOCK(float)
DECLARE_AS_BLOCK(double)

#undef DECLARE_AS_BLOCK
#undef DECLARE_AS_STRUCT_BLOCK

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
            __private dst_dt *dst, __global const src_dt *src, int n) { \
        __attribute__((opencl_unroll_hint)) while (n >= 8) { \
            __global BLOCK_DT(src_dt) *data \
                    = (__global BLOCK_DT(src_dt) *)(src); \
            BLOCK_DT_N(src_dt, 8) \
            block_val = BLOCK_READ_FUNC_N(src_dt, 8)(data); \
            for (int i = 0; i < 8; i++) { \
                src_dt src_val = CONCAT2(as_, src_dt)(block_val[i]); \
                dst[i] = CONCAT2(into_, dst_dt)(src_val); \
            } \
            dst += 8; \
            src += 8 * get_max_sub_group_size(); \
            n -= 8; \
        } \
        if (n >= 4) { \
            __global BLOCK_DT(src_dt) *data \
                    = (__global BLOCK_DT(src_dt) *)(src); \
            BLOCK_DT_N(src_dt, 4) \
            block_val = BLOCK_READ_FUNC_N(src_dt, 4)(data); \
            for (int i = 0; i < 4; i++) { \
                src_dt src_val = CONCAT2(as_, src_dt)(block_val[i]); \
                dst[i] = CONCAT2(into_, dst_dt)(src_val); \
            } \
            dst += 4; \
            src += 4 * get_max_sub_group_size(); \
            n -= 4; \
        } \
        if (n >= 2) { \
            __global BLOCK_DT(src_dt) *data \
                    = (__global BLOCK_DT(src_dt) *)(src); \
            BLOCK_DT_N(src_dt, 2) \
            block_val = BLOCK_READ_FUNC_N(src_dt, 2)(data); \
            for (int i = 0; i < 2; i++) { \
                src_dt src_val = CONCAT2(as_, src_dt)(block_val[i]); \
                dst[i] = CONCAT2(into_, dst_dt)(src_val); \
            } \
            dst += 2; \
            src += 2 * get_max_sub_group_size(); \
            n -= 2; \
        } \
        if (n >= 1) { \
            __global BLOCK_DT(src_dt) *data \
                    = (__global BLOCK_DT(src_dt) *)(src); \
            BLOCK_DT(src_dt) \
            block_val = BLOCK_READ_FUNC(src_dt)(data); \
            src_dt src_val = CONCAT2(as_, src_dt)(block_val); \
            *dst = CONCAT2(into_, dst_dt)(src_val); \
        } \
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
    } \
    __attribute__((overloadable)) void block_write( \
            __global dst_dt *dst, __private const src_dt *src, int n) { \
        __attribute__((opencl_unroll_hint)) while (n >= 8) { \
            BLOCK_DT_N(dst_dt, 8) block_val; \
            for (int i = 0; i < 8; i++) { \
                dst_dt val = CONCAT2(into_, dst_dt)(src[i]); \
                block_val[i] = as_block_data(val); \
            } \
            __global BLOCK_DT(dst_dt) *data \
                    = (__global BLOCK_DT(dst_dt) *)(dst); \
            BLOCK_WRITE_FUNC_N(dst_dt, 8)(data, block_val); \
            dst += 8 * get_max_sub_group_size(); \
            src += 8; \
            n -= 8; \
        } \
        if (n >= 4) { \
            BLOCK_DT_N(dst_dt, 4) block_val; \
            for (int i = 0; i < 4; i++) { \
                dst_dt val = CONCAT2(into_, dst_dt)(src[i]); \
                block_val[i] = as_block_data(val); \
            } \
            __global BLOCK_DT(dst_dt) *data \
                    = (__global BLOCK_DT(dst_dt) *)(dst); \
            BLOCK_WRITE_FUNC_N(dst_dt, 4)(data, block_val); \
            dst += 4 * get_max_sub_group_size(); \
            src += 4; \
            n -= 4; \
        } \
        if (n >= 2) { \
            BLOCK_DT_N(dst_dt, 2) block_val; \
            for (int i = 0; i < 2; i++) { \
                dst_dt val = CONCAT2(into_, dst_dt)(src[i]); \
                block_val[i] = as_block_data(val); \
            } \
            __global BLOCK_DT(dst_dt) *data \
                    = (__global BLOCK_DT(dst_dt) *)(dst); \
            BLOCK_WRITE_FUNC_N(dst_dt, 2)(data, block_val); \
            dst += 2 * get_max_sub_group_size(); \
            src += 2; \
            n -= 2; \
        } \
        if (n >= 1) { \
            BLOCK_DT(dst_dt) block_val; \
            dst_dt val = CONCAT2(into_, dst_dt)(*src); \
            block_val = as_block_data(val); \
            __global BLOCK_DT(dst_dt) *data \
                    = (__global BLOCK_DT(dst_dt) *)(dst); \
            BLOCK_WRITE_FUNC(dst_dt)(data, block_val); \
        } \
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
