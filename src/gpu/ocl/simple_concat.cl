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
#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ2 intel_sub_group_block_read_uc2
#define BLOCK_WRITE2 intel_sub_group_block_write_uc2
#define BLOCK_READ4 intel_sub_group_block_read_uc4
#define BLOCK_WRITE4 intel_sub_group_block_write_uc4
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
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

#if BLOCK != 1
__attribute__((intel_reqd_sub_group_size(SIMD)))
#endif
__kernel void
simple_concat(__global DATA_T *dst, long dst_offset0, SRC_PTRS) {
    const size_t x = (get_global_id(0) / SIMD) * BLOCK
            + get_global_id(2) * INNER_OFFSET;
    __global const DATA_T *src;

    SET_SRC;

#if BLOCK == 1
    dst += dst_offset0 + get_global_id(1) * DST_EXT_OFFSET + x;
    dst[0] = src[0];
#else
    DATA8_T A[4];
    DATA_T B;
    DATA2_T C;
    DATA4_T D;

    const int repeats = BLOCK / SIMD;
    const int vec8s = repeats / 8;
    const int d_offset = 8 * vec8s * SIMD;
    const int c_offset = d_offset + (repeats & 4) * SIMD;
    const int b_offset = c_offset + (repeats & 2) * SIMD;

    for (int i = 0; i < vec8s; ++i)
        A[i] = BLOCK_READ8(&src[8 * i * SIMD]);
    if (repeats & 4) D = BLOCK_READ4(&src[d_offset]);
    if (repeats & 2) C = BLOCK_READ2(&src[c_offset]);
    if (repeats & 1) B = BLOCK_READ(&src[b_offset]);

    dst += dst_offset0 + get_global_id(1) * DST_EXT_OFFSET + x;

    for (int i = 0; i < vec8s; ++i)
        BLOCK_WRITE8(&dst[8 * i * SIMD], A[i]);
    if (repeats & 4) BLOCK_WRITE4(&dst[d_offset], D);
    if (repeats & 2) BLOCK_WRITE2(&dst[c_offset], C);
    if (repeats & 1) BLOCK_WRITE(&dst[b_offset], B);
#endif
}
