/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#define BLOCK_READ intel_sub_group_block_read
#define BLOCK_WRITE intel_sub_group_block_write
#define BLOCK_READ2 intel_sub_group_block_read2
#define BLOCK_WRITE2 intel_sub_group_block_write2
#define BLOCK_READ4 intel_sub_group_block_read4
#define BLOCK_WRITE4 intel_sub_group_block_write4
#define BLOCK_READ8 intel_sub_group_block_read8
#define BLOCK_WRITE8 intel_sub_group_block_write8

#define CHECK_AND_GET(N, M) \
    if (get_group_id(2) >= OFFSET##N \
            && (M == N_INPUTS || get_group_id(2) < OFFSET##M)) { \
        src = src##N + get_group_id(1) * SRC##N##_EXT_OFFSET + x \
                - OFFSET##N * INNER_OFFSET; \
    }

#if BLOCK != 1
__attribute__((intel_reqd_sub_group_size(16)))
#endif
__kernel void
simple_concat(__global uint *dst, __global const uint *src0
#if N_INPUTS > 1
        ,
        __global const uint *src1
#endif
#if N_INPUTS > 2
        ,
        __global const uint *src2
#endif
#if N_INPUTS > 3
        ,
        __global const uint *src3
#endif
#if N_INPUTS > 4
        ,
        __global const uint *src4
#endif
#if N_INPUTS > 5
        ,
        __global const uint *src5
#endif
#if N_INPUTS > 6
        ,
        __global const uint *src6
#endif
#if N_INPUTS > 7
        ,
        __global const uint *src7
#endif
#if N_INPUTS > 8
        ,
        __global const uint *src8
#endif
#if N_INPUTS > 9
        ,
        __global const uint *src9
#endif
#if N_INPUTS > 10
        ,
        __global const uint *src10
#endif
#if N_INPUTS > 11
        ,
        __global const uint *src11
#endif
#if N_INPUTS > 12
        ,
        __global const uint *src12
#endif
#if N_INPUTS > 13
        ,
        __global const uint *src13
#endif
#if N_INPUTS > 14
        ,
        __global const uint *src14
#endif
#if N_INPUTS > 15
        ,
        __global const uint *src15
#endif
) {
    uint8 A0, A1, A2, A3;
    uint B;
    uint2 C;
    uint4 D;
    const size_t x = get_group_id(0) * BLOCK + get_sub_group_id() * BLOCK
            + get_group_id(2) * INNER_OFFSET;
    __global const uint *src;

    CHECK_AND_GET(0, 1)
#if N_INPUTS > 1
    CHECK_AND_GET(1, 2)
#endif
#if N_INPUTS > 2
    CHECK_AND_GET(2, 3)
#endif
#if N_INPUTS > 3
    CHECK_AND_GET(3, 4)
#endif
#if N_INPUTS > 4
    CHECK_AND_GET(4, 5)
#endif
#if N_INPUTS > 5
    CHECK_AND_GET(5, 6)
#endif
#if N_INPUTS > 6
    CHECK_AND_GET(6, 7)
#endif
#if N_INPUTS > 7
    CHECK_AND_GET(7, 8)
#endif
#if N_INPUTS > 8
    CHECK_AND_GET(8, 9)
#endif
#if N_INPUTS > 9
    CHECK_AND_GET(9, 10)
#endif
#if N_INPUTS > 10
    CHECK_AND_GET(10, 11)
#endif
#if N_INPUTS > 11
    CHECK_AND_GET(11, 12)
#endif
#if N_INPUTS > 12
    CHECK_AND_GET(12, 13)
#endif
#if N_INPUTS > 13
    CHECK_AND_GET(13, 14)
#endif
#if N_INPUTS > 14
    CHECK_AND_GET(14, 15)
#endif
#if N_INPUTS > 15
    CHECK_AND_GET(15, 16)
#endif

#if BLOCK == 1
    B = src[0];
#elif BLOCK == 16
    B = BLOCK_READ(src);
#elif BLOCK == 32
    C = BLOCK_READ2(src);
#elif BLOCK == 48
    C = BLOCK_READ2(src);
    B = BLOCK_READ(&src[32]);
#elif BLOCK == 64
    D = BLOCK_READ4(src);
#elif BLOCK == 80
    D = BLOCK_READ4(src);
    B = BLOCK_READ(&src[64]);
#elif BLOCK == 96
    D = BLOCK_READ4(src);
    C = BLOCK_READ2(&src[64]);
#elif BLOCK == 112
    B = BLOCK_READ(src);
    C = BLOCK_READ2(&src[16]);
    D = BLOCK_READ4(&src[48]);
#elif BLOCK >= 128
    A0 = BLOCK_READ8(src);
#elif BLOCK >= 256
    A1 = BLOCK_READ8(&src[128]);
#elif BLOCK >= 384
    A2 = BLOCK_READ8(&src[256]);
#elif BLOCK >= 512
    A3 = BLOCK_READ8(&src[384]);
#endif
    dst += get_group_id(1) * DST_EXT_OFFSET + x;
#if BLOCK == 1
    dst[0] = B;
#elif BLOCK == 16
    BLOCK_WRITE(dst, B);
#elif BLOCK == 32
    BLOCK_WRITE2(dst, C);
#elif BLOCK == 48
    BLOCK_WRITE2(dst, C);
    BLOCK_WRITE(&dst[32], B);
#elif BLOCK == 64
    BLOCK_WRITE4(dst, D);
#elif BLOCK == 80
    BLOCK_WRITE4(dst, D);
    BLOCK_WRITE(&dst[64], B);
#elif BLOCK == 96
    BLOCK_WRITE4(dst, D);
    BLOCK_WRITE2(&dst[64], C);
#elif BLOCK == 112
    BLOCK_WRITE(dst, B);
    BLOCK_WRITE2(&dst[16], C);
    BLOCK_WRITE4(&dst[48], D);
#elif BLOCK >= 128
    BLOCK_WRITE8(dst, A0);
#elif BLOCK >= 256
    BLOCK_WRITE8(&dst[128], A1);
#elif BLOCK >= 384
    BLOCK_WRITE8(&dst[256], A2);
#elif BLOCK >= 512
    BLOCK_WRITE8(&dst[384], A3);
#endif
}
