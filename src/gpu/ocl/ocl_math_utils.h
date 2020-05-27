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

float convert_bf16_to_f32(ushort b) {
    ushort2 r = {0, b};
    float f = as_float(r);
    return f;
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

#ifdef cl_intel_subgroups_char
void __attribute__((overloadable))
intel_sub_group_block_write_uc16(__global uchar *p, uchar16 data);

uchar16 __attribute__((overloadable))
intel_sub_group_block_read_uc16(const __global uchar *p);
#endif

inline int __imad(char4 a, char4 b, int c) __attribute__((overloadable)) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}
inline int __imad(uchar4 a, uchar4 b, int c) __attribute__((overloadable)) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}
inline int __imad(char4 a, uchar4 b, int c) __attribute__((overloadable)) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}
inline int __imad(uchar4 a, char4 b, int c) __attribute__((overloadable)) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

#define IMAD(_O, _I, _W) __imad(_O, _I, _W)

#ifdef cl_intel_subgroup_local_block_io

inline uint8 sub_group_block_read_local8(const __local uint *p)
        __attribute__((overloadable)) {
    uint8 __builtin_IB_simd_block_read_8_local(const __local uint *p)
            __attribute__((const));
    return __builtin_IB_simd_block_read_8_local(p);
}

inline uint sub_group_block_read_local(const __local uint *p)
        __attribute__((overloadable)) {
    uint __builtin_IB_simd_block_read_1_local(const __local uint *p)
            __attribute__((const));
    return __builtin_IB_simd_block_read_1_local(p);
}

#define READ_LOCAL_8(_P) sub_group_block_read_local8(_P)
#define READ_LOCAL_1(_P) sub_group_block_read_local(_P)
#define WRITE_LOCAL_8(_P, _V) intel_sub_group_block_write8(_P, _V)
#define WRITE_LOCAL_4(_P, _V) intel_sub_group_block_write4(_P, _V)
#define WRITE_LOCAL_2(_P, _V) intel_sub_group_block_write2(_P, _V)
#define WRITE_LOCAL_1(_P, _V) intel_sub_group_block_write(_P, _V)
#define WRITE_LOCAL_SHORT_1(_P, _V) intel_sub_group_block_write_us(_P, _V)

#else

#define READ_LOCAL_8(_P) subgroup_block_read_uint8(_P)
#define READ_LOCAL_1(_P) subgroup_block_read_uint(_P)
#define WRITE_LOCAL_8(_P, _V) subgroup_block_write_uint8(_P, _V)
#define WRITE_LOCAL_4(_P, _V) subgroup_block_write_uint4(_P, _V)
#define WRITE_LOCAL_2(_P, _V) subgroup_block_write_uint2(_P, _V)
#define WRITE_LOCAL_1(_P, _V) subgroup_block_write_uint(_P, _V)
#define WRITE_LOCAL_SHORT_1(_P, _V) subgroup_block_write_ushort(_P, _V)

#endif

uint subgroup_block_read_uint(const __local uint *p) {
    uint ret;
    uint idx = get_sub_group_local_id();
    ret = p[idx];
    return ret;
}

uint8 subgroup_block_read_uint8(const __local uint *p) {
    uint8 ret;
    uint idx = get_sub_group_local_id();
    ret.s0 = p[idx];
    idx += get_max_sub_group_size();
    ret.s1 = p[idx];
    idx += get_max_sub_group_size();
    ret.s2 = p[idx];
    idx += get_max_sub_group_size();
    ret.s3 = p[idx];
    idx += get_max_sub_group_size();
    ret.s4 = p[idx];
    idx += get_max_sub_group_size();
    ret.s5 = p[idx];
    idx += get_max_sub_group_size();
    ret.s6 = p[idx];
    idx += get_max_sub_group_size();
    ret.s7 = p[idx];
    return ret;
}

void subgroup_block_write_ushort(__local ushort *p, ushort v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v;
}

void subgroup_block_write_uint(__local uint *p, uint v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v;
}

void subgroup_block_write_uint2(__local uint *p, uint2 v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v.s0;
    p += get_max_sub_group_size();
    p[idx] = v.s1;
}

void subgroup_block_write_uint4(__local uint *p, uint4 v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v.s0;
    p += get_max_sub_group_size();
    p[idx] = v.s1;
    p += get_max_sub_group_size();
    p[idx] = v.s2;
    p += get_max_sub_group_size();
    p[idx] = v.s3;
}

void subgroup_block_write_uint8(__local uint *p, uint8 v) {
    uint idx = get_sub_group_local_id();
    p[idx] = v.s0;
    p += get_max_sub_group_size();
    p[idx] = v.s1;
    p += get_max_sub_group_size();
    p[idx] = v.s2;
    p += get_max_sub_group_size();
    p[idx] = v.s3;
    p += get_max_sub_group_size();
    p[idx] = v.s4;
    p += get_max_sub_group_size();
    p[idx] = v.s5;
    p += get_max_sub_group_size();
    p[idx] = v.s6;
    p += get_max_sub_group_size();
    p[idx] = v.s7;
}

inline int mmad_4(uchar4 input, char4 weight, int acc)
        __attribute__((overloadable)) {
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int mmad_4(char4 input, char4 weight, int acc)
        __attribute__((overloadable)) {
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int mmad_4(char4 input, uchar4 weight, int acc)
        __attribute__((overloadable)) {
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int mmad_4(uchar4 input, uchar4 weight, int acc)
        __attribute__((overloadable)) {
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int idot4(int src, int wei, int acc) __attribute__((overloadable)) {
    return IMAD(as_char4(src), as_char4(wei), acc);
}

inline int idot4(uint src, int wei, int acc) __attribute__((overloadable)) {
    return IMAD(as_uchar4(src), as_char4(wei), acc);
}

#define DECLARE_MMAD(_suffix, _src_type, _acc_type, _int_type, _owb, _icb) \
    inline int mmad_part##_suffix(_int_type a, int8 b, int acc) \
            __attribute__((overloadable)) { \
        for (uint i = 0; i < _icb; ++i) \
            acc = idot4(a[i], b[i], acc); \
        return acc; \
    } \
    inline _acc_type mmad##_suffix(_src_type a, int8 b, _acc_type acc) \
            __attribute__((overloadable)) { \
        _acc_type ret; \
        for (uint i = 0; i < _owb; ++i) { \
            _int_type c; \
            for (int j = 0; j < _icb; ++j) \
                c[j] = sub_group_broadcast(a[i], j); \
            ret[i] = mmad_part##_suffix(c, b, acc[i]); \
        } \
        return ret; \
    }

inline int mmad8(int8 A_scalars, int8 B_vectors, int acc)
        __attribute__((overloadable)) {
    for (uint i = 0; i < 8; ++i)
        acc = idot4(A_scalars[i], B_vectors[i], acc);
    return acc;
}

inline int mmad8(uint8 A_scalars, int8 B_vectors, int acc)
        __attribute__((overloadable)) {
    for (uint i = 0; i < 8; ++i)
        acc = idot4(A_scalars[i], B_vectors[i], acc);
    return acc;
}

inline int4 mmad8x4(uint4 A_vectors, int8 B_vectors, int4 acc)
        __attribute__((overloadable)) {
    int4 ret;
    for (uint i = 0; i < 4; ++i) {
        uint8 A_scalars;
        for (uint j = 0; j < 8; ++j)
            A_scalars[j] = sub_group_broadcast(A_vectors[i], j);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int4 mmad8x4(int4 A_vectors, int8 B_vectors, int4 acc)
        __attribute__((overloadable)) {
    int4 ret;
    for (uint i = 0; i < 4; ++i) {
        int8 A_scalars;
        for (uint j = 0; j < 8; ++j)
            A_scalars[j] = sub_group_broadcast(A_vectors[i], j);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int8 mmad8x8(uint8 A_vectors, int8 B_vectors, int8 acc)
        __attribute__((overloadable)) {
    int8 ret;
    for (uint i = 0; i < 8; ++i) {
        uint8 A_scalars;
        for (uint j = 0; j < 8; ++j)
            A_scalars[j] = sub_group_broadcast(A_vectors[i], j);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int8 mmad8x8(int8 A_vectors, int8 B_vectors, int8 acc)
        __attribute__((overloadable)) {
    int8 ret;
    for (uint i = 0; i < 8; ++i) {
        int8 A_scalars;
        for (uint j = 0; j < 8; ++j)
            A_scalars[j] = sub_group_broadcast(A_vectors[i], j);
        ret[i] = mmad8(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

#endif
