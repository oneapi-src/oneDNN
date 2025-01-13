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
    // we just need to apply rounding
    ushort fraw = as_ushort(f);
    ushort naninf_mask = 0x7c00;

    bool is_special = (fraw & naninf_mask) == naninf_mask;
    bool is_nan = is_special && (fraw & 0x03ff); // one of the lsb is non zero

    // we always return R ind for Nan input as there is no good
    // conversion of payload
    if (is_nan) { return as_f8_e5m2((fraw >> 8) | 0x02); }

    // if infinity, we just return it as is
    if (is_special) {
        uchar raw_bits = fraw >> 8;
        return as_f8_e5m2(raw_bits);
    }

    // otherwise we just round and return
    ushort rounding_nudge = 0x007f + ((fraw & 0x0100) >> 8);
    fraw = fraw + rounding_nudge;
    uchar raw_bits = fraw >> 8;
    return as_f8_e5m2(raw_bits);
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
    // Here the idea is to add a large constant to the float16_t to force the
    // proper rounding to f8_e4m3 accuracy.
    uchar raw_bits = 0;
    ushort fraw = as_ushort(f);

    // first we extract the sign and make the input positive
    uint s8 = (fraw & 0x8000) >> 8;
    fraw = fraw & 0x7fff;

    // we filter out overlow, nan
    if (fraw >= 0x5f40) {
        raw_bits = s8 | 0x7f;
        return as_f8_e4m3(raw_bits);
    }
    // we filter out underflow when f <= 2^-10
    if (fraw <= 0x1400) {
        raw_bits = s8;
        return as_f8_e4m3(raw_bits);
    }

    // compute the rounding shifter by taking its exponent + 0x1p7
    // Lucky us, it does not overflow as fraw <= 448.
    ushort a = 0x7c00, b = 0x1c00;
    ushort shifter = (fraw & a) + b;
    // e8 = e16 - e16_bias + e8_bias = e16 - 15 + 7
    // e8 will be denorm if e8 <= 0 or e16 + 7 < 16
    const int exp_threshold = 0x4000; // raw bits of exponent = 16
    ushort is_denorm = shifter < exp_threshold;
    if (is_denorm) shifter = exp_threshold;

    ushort rounded
            = as_ushort((as_half(fraw) + as_half(shifter)) - as_half(shifter));

    int e8 = ((rounded & 0x7c00) >> 10) - 8;
    uchar m8 = (rounded & 0x03ff) >> 7;

    // we need to make the implicit f32 mantissa bit explicit for
    // denorm f8_e4m3
    if (is_denorm) {
        m8 = (m8 | 0x08) >> (-e8 + 1);
        e8 = 0;
    }

    raw_bits = s8 | (e8 << 3) | m8;
    return as_f8_e4m3(raw_bits);
}

half __attribute__((overloadable)) into_half(f8_e4m3 b) {
    uchar raw_bits_ = b.data;
    ushort s8 = (raw_bits_ & 0x80) >> 7;
    ushort e8 = (raw_bits_ & 0x78) >> 3;
    ushort m8 = (raw_bits_ & 0x7);
    ushort s16 = s8;
    ushort e16 = e8 + 8; /* 15 - 7 = e16_bias - e8_bias */
    ushort m16 = m8;

    // Need to convert f8_e4m3 denormal into f16 normal.
    if (e8 == 0 && m8 != 0) {
        ushort count = 2;
        count = m8 > 0x1 ? 1 : count;
        count = m8 > 0x3 ? 0 : count;
        e16 -= count;
        m16 = (m16 << (count + 1)) & 0x7;
    } else if (e8 == 0 && m8 == 0) {
        e16 = 0;
    } else if (e8 == 0xf && m8 == 0x7) {
        e16 = 0x1f;
        m16 = 0x4; // Real Indefinite (a qNaN)
    }
    s16 <<= 15;
    e16 <<= 10;
    m16 <<= 7;

    ushort u16 = s16 | e16 | m16;
    return as_half(u16);
}

def_two_step_conversion(f8_e4m3, float, half);
def_two_step_conversion(f8_e4m3, int, half);
def_two_step_conversion(float, f8_e4m3, half);

IF_DOUBLE_SUPPORTED(def_two_step_conversion(f8_e4m3, double, float));
IF_DOUBLE_SUPPORTED(def_two_step_conversion(double, f8_e4m3, float));
#endif // MATH_UTILS_DECLARE_HF8

IF_DOUBLE_SUPPORTED(def_two_step_conversion(bf16, double, float));
IF_DOUBLE_SUPPORTED(def_two_step_conversion(double, bf16, float));

#undef def_two_step_conversion

#endif
