/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#if DATA_TYPE_SIZE == 4
#define DATA_T uint
#define DATA2_T uint2
#define DATA4_T uint4
#define DATA8_T uint8
#define BLOCK_READ intel_sub_group_block_read
#define BLOCK_WRITE intel_sub_group_block_write
#define BLOCK_READ2 intel_sub_group_block_read2
#define BLOCK_WRITE2 intel_sub_group_block_write2
#define BLOCK_READ4 intel_sub_group_block_read4
#define BLOCK_WRITE4 intel_sub_group_block_write4
#define BLOCK_READ8 intel_sub_group_block_read8
#define BLOCK_WRITE8 intel_sub_group_block_write8
#elif DATA_TYPE_SIZE == 2
#define DATA_T ushort
#define DATA2_T ushort2
#define DATA4_T ushort4
#define DATA8_T ushort8
#define BLOCK_READ intel_sub_group_block_read_us
#define BLOCK_WRITE intel_sub_group_block_write_us
#define BLOCK_READ2 intel_sub_group_block_read_us2
#define BLOCK_WRITE2 intel_sub_group_block_write_us2
#define BLOCK_READ4 intel_sub_group_block_read_us4
#define BLOCK_WRITE4 intel_sub_group_block_write_us4
#define BLOCK_READ8 intel_sub_group_block_read_us8
#define BLOCK_WRITE8 intel_sub_group_block_write_us8
#elif DATA_TYPE_SIZE == 1
#define DATA_T uchar
#define DATA2_T uchar2
#define DATA4_T uchar4
#define DATA8_T uchar8
#define DATA16_T uchar16
#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ2 intel_sub_group_block_read_uc2
#define BLOCK_WRITE2 intel_sub_group_block_write_uc2
#define BLOCK_READ4 intel_sub_group_block_read_uc4
#define BLOCK_WRITE4 intel_sub_group_block_write_uc4
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
#define BLOCK_READ16 intel_sub_group_block_read_uc16
#define BLOCK_WRITE16 intel_sub_group_block_write_uc16
#endif

#include "gpu/ocl/concat_common.h"

#define SRC_PTR(n) __global const DATA_T *src##n
#define SRC_PTRS REDUCE(N_INPUTS, JOIN_COMMA, SRC_PTR)
#define JOIN_ELSE(x, y) y else x
#define CHECK_AND_GET(n) \
    if (get_global_id(2) >= OFFSET##n) \
        src = src##n + get_global_id(1) * SRC##n##_EXT_OFFSET + x \
                - OFFSET##n * INNER_OFFSET;
#define SET_SRC REDUCE(N_INPUTS, JOIN_ELSE, CHECK_AND_GET)

#if SIMD != 1
__attribute__((intel_reqd_sub_group_size(SIMD)))
#endif
__kernel void
simple_concat(__global DATA_T *dst, long dst_offset0, SRC_PTRS) {
    const size_t x = (get_global_id(0) / SIMD) * BLOCK
            + get_global_id(2) * INNER_OFFSET;
    __global const DATA_T *src;

    SET_SRC;

#if SIMD == 1
    dst += dst_offset0 + get_global_id(1) * DST_EXT_OFFSET + x;

    for (int i = 0; i < BLOCK; ++i)
        dst[i] = src[i];
#else
    union {
#if DATA_TYPE_SIZE == 1
        DATA16_T v16[2];
#endif
        DATA8_T v8[4];
        DATA4_T v4[8];
        DATA2_T v2[16];
        DATA_T v1[32];
    } buf;

    const int thr_elems = BLOCK / SIMD;
    const int lane = get_sub_group_local_id();

#if (BLOCK * DATA_TYPE_SIZE % 4 != 0)
    for (int i = 0; i < thr_elems; ++i)
        buf.v1[i] = src[i * SIMD + lane];
#else
    const uint v8_off = thr_elems & ~0xf;
    const uint v4_off = thr_elems & ~0x7;
    const uint v2_off = thr_elems & ~0x3;
    const uint v1_off = thr_elems & ~0x1;
#if DATA_TYPE_SIZE == 1
    for (int i = 0; i < thr_elems / 16; ++i)
        buf.v16[i] = BLOCK_READ16(&src[16 * i * SIMD]);
    if (thr_elems & 8) buf.v8[v8_off / 8] = BLOCK_READ8(&src[v8_off * SIMD]);
#else
    for (int i = 0; i < thr_elems / 8; ++i)
        buf.v8[i] = BLOCK_READ8(&src[8 * i * SIMD]);
#endif
    if (thr_elems & 4) buf.v4[v4_off / 4] = BLOCK_READ4(&src[v4_off * SIMD]);
    if (thr_elems & 2) buf.v2[v2_off / 2] = BLOCK_READ2(&src[v2_off * SIMD]);
    if (thr_elems & 1) buf.v1[v1_off / 1] = BLOCK_READ(&src[v1_off * SIMD]);
#endif
    if (lane < BLOCK % SIMD) buf.v1[thr_elems] = src[thr_elems * SIMD + lane];

    dst += dst_offset0 + get_global_id(1) * DST_EXT_OFFSET + x;

#if (BLOCK * DATA_TYPE_SIZE % 16 != 0)
    for (int i = 0; i < thr_elems; ++i)
        dst[i * SIMD + lane] = buf.v1[i];
#else
#if DATA_TYPE_SIZE == 1
    for (int i = 0; i < thr_elems / 16; ++i)
        BLOCK_WRITE16(&dst[16 * i * SIMD], buf.v16[i]);
    if (thr_elems & 8) BLOCK_WRITE8(&dst[v8_off * SIMD], buf.v8[v8_off / 8]);
#else
    for (int i = 0; i < thr_elems / 8; ++i)
        BLOCK_WRITE8(&dst[8 * i * SIMD], buf.v8[i]);
#endif
    if (thr_elems & 4) BLOCK_WRITE4(&dst[v4_off * SIMD], buf.v4[v4_off / 4]);
    if (thr_elems & 2) BLOCK_WRITE2(&dst[v2_off * SIMD], buf.v2[v2_off / 2]);
    if (thr_elems & 1) BLOCK_WRITE(&dst[v1_off * SIMD], buf.v1[v1_off / 1]);
#endif
    if (lane < BLOCK % SIMD) dst[thr_elems * SIMD + lane] = buf.v1[thr_elems];
#endif
}
