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
    DATA8_T A0, A1, A2, A3;
    DATA_T B;
    DATA2_T C;
    DATA4_T D;
    const size_t x = (get_global_id(0) / SIMD) * BLOCK
            + get_global_id(2) * INNER_OFFSET;
    __global const DATA_T *src;

    SET_SRC;

#if BLOCK == 1
    B = src[0];
#elif BLOCK == SIMD
    B = BLOCK_READ(src);
#elif BLOCK == 2 * SIMD
    C = BLOCK_READ2(src);
#elif BLOCK == 3 * SIMD
    C = BLOCK_READ2(src);
    B = BLOCK_READ(&src[2 * SIMD]);
#elif BLOCK == 4 * SIMD
    D = BLOCK_READ4(src);
#elif BLOCK == 5 * SIMD
    D = BLOCK_READ4(src);
    B = BLOCK_READ(&src[4 * SIMD]);
#elif BLOCK == 6 * SIMD
    D = BLOCK_READ4(src);
    C = BLOCK_READ2(&src[4 * SIMD]);
#elif BLOCK == 7 * SIMD
    B = BLOCK_READ(src);
    C = BLOCK_READ2(&src[SIMD]);
    D = BLOCK_READ4(&src[3 * SIMD]);
#elif BLOCK >= 8 * SIMD
    A0 = BLOCK_READ8(src);
#elif BLOCK >= 16 * SIMD
    A1 = BLOCK_READ8(&src[8 * SIMD]);
#elif BLOCK >= 24 * SIMD
    A2 = BLOCK_READ8(&src[16 * SIMD]);
#elif BLOCK >= 32 * SIMD
    A3 = BLOCK_READ8(&src[24 * SIMD]);
#endif
    dst += dst_offset0 + get_global_id(1) * DST_EXT_OFFSET + x;
#if BLOCK == 1
    dst[0] = B;
#elif BLOCK == SIMD
    BLOCK_WRITE(dst, B);
#elif BLOCK == 2 * SIMD
    BLOCK_WRITE2(dst, C);
#elif BLOCK == 3 * SIMD
    BLOCK_WRITE2(dst, C);
    BLOCK_WRITE(&dst[2 * SIMD], B);
#elif BLOCK == 4 * SIMD
    BLOCK_WRITE4(dst, D);
#elif BLOCK == 5 * SIMD
    BLOCK_WRITE4(dst, D);
    BLOCK_WRITE(&dst[4 * SIMD], B);
#elif BLOCK == 6 * SIMD
    BLOCK_WRITE4(dst, D);
    BLOCK_WRITE2(&dst[4 * SIMD], C);
#elif BLOCK == 7 * SIMD
    BLOCK_WRITE(dst, B);
    BLOCK_WRITE2(&dst[SIMD], C);
    BLOCK_WRITE4(&dst[3 * SIMD], D);
#elif BLOCK >= 8 * SIMD
    BLOCK_WRITE8(dst, A0);
#elif BLOCK >= 16 * SIMD
    BLOCK_WRITE8(&dst[8 * SIMD], A1);
#elif BLOCK >= 24 * SIMD
    BLOCK_WRITE8(&dst[16 * SIMD], A2);
#elif BLOCK >= 32 * SIMD
    BLOCK_WRITE8(&dst[24 * SIMD], A3);
#endif
}
