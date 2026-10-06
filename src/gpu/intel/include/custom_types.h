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

#ifndef GPU_INTEL_INCLUDE_CUSTOM_TYPES_H
#define GPU_INTEL_INCLUDE_CUSTOM_TYPES_H

#include "gpu/intel/include/utils.h"

// Fixed to 64 bit per the OpenCL specification to align with same type in C++
// source code
typedef long dim_t;

// Signed offset used to support the (rarely) used negative strides
#ifdef USE_INT32_OFFSET
typedef int off_t;
#else
typedef long off_t;
#endif

typedef struct {
    ushort data;
} bf16;

bf16 as_bf16(ushort data) {
    bf16 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    uchar data;
} f8_e5m2;

f8_e5m2 as_f8_e5m2(uchar data) {
    f8_e5m2 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    uchar data;
} f8_e4m3;

f8_e4m3 as_f8_e4m3(uchar data) {
    f8_e4m3 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    unsigned char data;
} e8m0;

e8m0 as_e8m0(unsigned char data) {
    e8m0 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    uchar data;
} f4_e2m1;

f4_e2m1 as_f4_e2m1(unsigned char data) {
    f4_e2m1 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    char data;
} s4;

s4 as_s4(uchar data) {
    s4 res;
    res.data = (char)data;
    return res;
}

/*****************************/

typedef struct {
    uchar data;
} u4;

u4 as_u4(uchar data) {
    u4 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    uchar data;
} u2;

u2 as_u2(uchar data) {
    u2 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    uchar data;
} u3;

u3 as_u3(uchar data) {
    u3 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    char invalid_data;
} undef_data;

undef_data as_undef_data(char data) {
    undef_data ret = {0xba};
    return ret;
}

#define NATIVE_TYPE_CASE(t) t : (0, ((t *)0)->data)

// Returns the native scalar type used to store the punned `type`.
#define NATIVE_LAYOUT_TYPE(type) \
    __typeof__(_Generic(*(type *)0, \
            NATIVE_TYPE_CASE(bf16), \
            NATIVE_TYPE_CASE(f8_e5m2), \
            NATIVE_TYPE_CASE(f8_e4m3), \
            NATIVE_TYPE_CASE(e8m0), \
            NATIVE_TYPE_CASE(f4_e2m1), \
            NATIVE_TYPE_CASE(s4), \
            NATIVE_TYPE_CASE(u4), \
            default: (0, *(type *)0)))

// Macro to generate as_<type> function call for a given dt
#define AS_NATIVE_LAYOUT_TYPE(dt, v) CONCAT2(as_, dt)(v)

#endif
