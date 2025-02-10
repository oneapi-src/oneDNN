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

#ifndef GPU_INTEL_OCL_OCL_CONVERSION_H
#define GPU_INTEL_OCL_OCL_CONVERSION_H

#include "gpu/intel/ocl/ocl_custom_types.h"
#include "gpu/intel/ocl/ocl_math_utils.h"
#include "gpu/intel/ocl/ocl_utils.h"

// Idea: We want to have a uniform method of conversion between types within
// onednn, but convert_* functions cannot be overloaded due to an igc bug
// currently. Furthermore, future versions of opencl may define new convert_*
// functions that clash with these names. Instead, use into_<type> within oneDNN
// for conversion of various types.

// Identity conversions
#define def_identity_into(dt) \
    dt __attribute__((overloadable)) CONCAT2(into_, dt)(dt val) { return val; }

def_identity_into(float);
def_identity_into(char);
def_identity_into(uchar);
def_identity_into(int);
def_identity_into(bf16);
IF_DOUBLE_SUPPORTED(def_identity_into(double));
IF_HALF_SUPPORTED(def_identity_into(half));

#ifdef MATH_UTILS_DECLARE_BF8
def_identity_into(f8_e5m2);
#endif // MATH_UTILS_DECLARE_BF8

#ifdef MATH_UTILS_DECLARE_HF8
def_identity_into(f8_e4m3);
#endif // MATH_UTILS_DECLARE_HF8

#undef def_identity_into

// Standard-standard conversions, utilizing convert_* builtins
#define def_std_into(out_type, in_type) \
    out_type __attribute__((overloadable)) \
            CONCAT2(into_, out_type)(in_type val) { \
        return CONCAT2(convert_, out_type)(val); \
    }
#define def_std_into_sat(out_type, in_type) \
    out_type __attribute__((overloadable)) \
            CONCAT2(into_, out_type)(in_type val) { \
        return CONCAT3(convert_, out_type, _sat_rte)(val); \
    }

def_std_into_sat(char, uchar);
def_std_into_sat(char, float);
def_std_into_sat(char, int);
IF_DOUBLE_SUPPORTED(def_std_into_sat(char, double));
IF_HALF_SUPPORTED(def_std_into_sat(char, half));

def_std_into_sat(uchar, char);
def_std_into_sat(uchar, float);
def_std_into_sat(uchar, int);
IF_DOUBLE_SUPPORTED(def_std_into_sat(uchar, double));
IF_HALF_SUPPORTED(def_std_into_sat(uchar, half));

def_std_into_sat(int, float);

IF_HALF_SUPPORTED(def_std_into(half, char));
IF_HALF_SUPPORTED(def_std_into(half, uchar));
IF_HALF_SUPPORTED(def_std_into(half, float));
IF_HALF_SUPPORTED(def_std_into(half, int));
IF_HALF_SUPPORTED(IF_DOUBLE_SUPPORTED(def_std_into(half, double)));

def_std_into(float, char);
def_std_into(float, uchar);
def_std_into(float, int);
IF_DOUBLE_SUPPORTED(def_std_into(float, double));
IF_HALF_SUPPORTED(def_std_into(float, half));

IF_DOUBLE_SUPPORTED(def_std_into(double, char));
IF_DOUBLE_SUPPORTED(def_std_into(double, uchar));
IF_DOUBLE_SUPPORTED(def_std_into(double, float));
IF_DOUBLE_SUPPORTED(def_std_into(double, int));
IF_DOUBLE_SUPPORTED(IF_HALF_SUPPORTED(def_std_into(double, half)));

#define def_undef_into(out_type) \
    out_type __attribute__((overloadable)) \
            CONCAT2(into_, out_type)(undef_data val) { \
        DEBUG_PRINT("Error: unexpected conversion from undefined data"); \
        return 0xbadbad; \
    }

def_undef_into(float);
def_undef_into(int);
IF_DOUBLE_SUPPORTED(def_undef_into(double));

#undef def_std_into
#undef def_std_into_sat

// Conversions to/from custom types

// XXX: The following macro requires the following function definitions:
// - mid_type into_<mid_type>(in_type)
// - out_type into_<out_type>(mid_type)
#define def_two_step_conversion(out_type, in_type, mid_type) \
    out_type __attribute__((overloadable)) \
            CONCAT2(into_, out_type)(in_type val) { \
        return CONCAT2(into_, out_type)(CONCAT2(into_, mid_type)(val)); \
    }

#ifdef cl_future_bf16_cvt
// Builtin conversions
// clang-format off
// f32 -> bf16 conversion builtins (rte rounding mode)
short __builtin_IB_ftobf_1(float a) __attribute__((const));
short2 __builtin_IB_ftobf_2(float2 a) __attribute__((const));
short4 __builtin_IB_ftobf_4(float4 a) __attribute__((const));
short8 __builtin_IB_ftobf_8(float8 a) __attribute__((const));
short16 __builtin_IB_ftobf_16(float16 a) __attribute__((const));

// bf16 -> f32 conversion builtins (precise conversion)
float __builtin_IB_bftof_1(short a) __attribute__((const));
float2 __builtin_IB_bftof_2(short2 a) __attribute__((const));
float4 __builtin_IB_bftof_4(short4 a) __attribute__((const));
float8 __builtin_IB_bftof_8(short8 a) __attribute__((const));
float16 __builtin_IB_bftof_16(short16 a) __attribute__((const));

bf16 __attribute__((overloadable)) into_bf16(float x) {
    return as_bf16(__builtin_IB_ftobf_1(x));
}
float __attribute__((overloadable)) into_float(bf16 x) {
    return __builtin_IB_bftof_1(x.data);
}
#else
// Emulated conversions
bf16 __attribute__((overloadable)) into_bf16(float f) {
    uint i = as_uint(f);
    i += 0x00007FFF + ((i & 0x10000) >> 16);
    ushort2 r = as_ushort2(i);
    return as_bf16(r[1]);
}
float __attribute__((overloadable)) into_float(bf16 b) {
    ushort2 r = {0, b.data};
    float f = as_float(r);
    return f;
}
#endif
def_two_step_conversion(bf16, int, float);
IF_HALF_SUPPORTED(def_two_step_conversion(bf16, half, float));

// BF8 conversion emulations
#ifdef MATH_UTILS_DECLARE_BF8

f8_e5m2 __attribute__((overloadable)) into_f8_e5m2(half f) {
    return as_f8_e5m2(cvt_hf_to_f8_e5m2(f));
}

half __attribute__((overloadable)) into_half(f8_e5m2 b) {
    uchar2 iraw = {0, b.data};
    return as_half(iraw);
}

def_two_step_conversion(f8_e5m2, float, half);
def_two_step_conversion(f8_e5m2, int, half);
def_two_step_conversion(float, f8_e5m2, half);

IF_DOUBLE_SUPPORTED(def_two_step_conversion(f8_e5m2, double, float));
IF_DOUBLE_SUPPORTED(def_two_step_conversion(double, f8_e5m2, float));
#endif

#ifdef MATH_UTILS_DECLARE_HF8
f8_e4m3 __attribute__((overloadable)) into_f8_e4m3(half f) {
    return as_f8_e4m3(cvt_hf_to_f8_e4m3(f));
}

half __attribute__((overloadable)) into_half(f8_e4m3 b) {
    return cvt_f8_e4m3_to_hf(b.data);
}

def_two_step_conversion(f8_e4m3, float, half);
def_two_step_conversion(f8_e4m3, int, half);
def_two_step_conversion(float, f8_e4m3, half);

IF_DOUBLE_SUPPORTED(def_two_step_conversion(f8_e4m3, double, float));
IF_DOUBLE_SUPPORTED(def_two_step_conversion(double, f8_e4m3, float));
#endif // MATH_UTILS_DECLARE_HF8

#ifdef MATH_UTILS_DECLARE_F4_E2M1
f4_e2m1 __attribute__((overloadable)) into_f4_e2m1(float f) {
    return as_f4_e2m1(cvt_f32_to_f4_e2m1(f));
}

half __attribute__((overloadable)) into_float(f4_e2m1 b) {
    return cvt_f4_e2m1_to_f32(b.data);
}

def_two_step_conversion(f4_e2m1, half, float);
def_two_step_conversion(f4_e2m1, int, float);
def_two_step_conversion(half, f4_e2m1, float);

IF_DOUBLE_SUPPORTED(def_two_step_conversion(f4_e2m1, double, float));
IF_DOUBLE_SUPPORTED(def_two_step_conversion(double, f4_e2m1, float));
#endif // MATH_UTILS_DECLARE_F4_E2M1

#ifdef MATH_UTILS_DECLARE_F4_E3M0
f4_e3m0 __attribute__((overloadable)) into_f4_e3m0(float f) {
    return as_f4_e3m0(cvt_f32_to_f4_e3m0(f));
}

half __attribute__((overloadable)) into_float(f4_e3m0 b) {
    return cvt_f4_e3m0_to_f32(b.data);
}

def_two_step_conversion(f4_e3m0, half, float);
def_two_step_conversion(f4_e3m0, int, float);
def_two_step_conversion(half, f4_e3m0, float);

IF_DOUBLE_SUPPORTED(def_two_step_conversion(f4_e3m0, double, float));
IF_DOUBLE_SUPPORTED(def_two_step_conversion(double, f4_e3m0, float));
#endif // MATH_UTILS_DECLARE_F4_E3M0

#ifdef MATH_UTILS_DECLARE_E8M0
// Copy-paste from `cvt_e8m0_to_f32`.
float __attribute__((overloadable)) into_float(e8m0 b) {
    if (b.data == (char)0xff) return as_float(0xffc00000);
    uint bits = b.data << 23;
    return as_float(bits);
}
#endif

IF_DOUBLE_SUPPORTED(def_two_step_conversion(bf16, double, float));
IF_DOUBLE_SUPPORTED(def_two_step_conversion(double, bf16, float));

#undef def_two_step_conversion

#endif
