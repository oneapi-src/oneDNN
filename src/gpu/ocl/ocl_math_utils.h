/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_OCL_OCL_MATH_UTILS_H
#define GPU_OCL_OCL_MATH_UTILS_H

ushort convert_f32_to_bf16(float f) {
    uint i = as_uint(f);
    i += 0x00007FFF + ((i & 0x10000) >> 16);
    ushort2 r = as_ushort2(i);
    return r[1];
}

float convert_bf16_to_f32(ushort b) {
    ushort2 r = {0, b};
    float f = as_float(r);
    return f;
}

ushort2 convert_f32_to_bf16_vec2(float2 f) {
    ushort2 r;
    for (int i = 0; i < 2; i++) {
        r[i] = convert_f32_to_bf16(f[i]);
    }
    return r;
}

ushort4 convert_f32_to_bf16_vec4(float4 f) {
    ushort4 r;
    for (int i = 0; i < 4; i++) {
        r[i] = convert_f32_to_bf16(f[i]);
    }
    return r;
}

ushort8 convert_f32_to_bf16_vec8(float8 f) {
    ushort8 r;
    for (int i = 0; i < 8; i++) {
        r[i] = convert_f32_to_bf16(f[i]);
    }
    return r;
}

float2 convert_bf16_to_f32_vec2(ushort2 b) {
    float2 f;
    for (int i = 0; i < 2; i++) {
        f[i] = convert_bf16_to_f32(b[i]);
    }
    return f;
}

float4 convert_bf16_to_f32_vec4(ushort4 b) {
    float4 f;
    for (int i = 0; i < 4; i++) {
        f[i] = convert_bf16_to_f32(b[i]);
    }
    return f;
}

float8 convert_bf16_to_f32_vec8(ushort8 b) {
    float8 f;
    for (int i = 0; i < 8; i++) {
        f[i] = convert_bf16_to_f32(b[i]);
    }
    return f;
}

#endif
