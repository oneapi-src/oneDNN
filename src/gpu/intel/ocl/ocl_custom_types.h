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

#ifndef GPU_INTEL_OCL_OCL_CUSTOM_TYPES_H
#define GPU_INTEL_OCL_OCL_CUSTOM_TYPES_H

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
    short data;
} bf16;

bf16 as_bf16(short data) {
    bf16 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    char data;
} f8_e5m2;

f8_e5m2 as_f8_e5m2(char data) {
    f8_e5m2 res;
    res.data = data;
    return res;
}

/*****************************/

typedef struct {
    char data;
} f8_e4m3;

f8_e4m3 as_f8_e4m3(char data) {
    f8_e4m3 res;
    res.data = data;
    return res;
}

#endif
