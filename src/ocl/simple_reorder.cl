/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define IN_OFF(x0, x1, x2, x3, x4, x5)                             \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0          \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 \
            + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)

#if IN_OIHW8O16I2O
#    undef IN_OFF
#    if WITH_GROUP
#        define IN_OFF(x0, x1, x2, x3, x4, x5)                               \
            (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 + (x1 % 2) \
                    + ((x1 % 16) / 2) * 32 + ((x1) / SRC_B1) * SRC_S1        \
                    + 2 * (x2 % 16) + ((x2) / SRC_B2) * SRC_S2               \
                    + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3   \
                    + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4   \
                    + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)
#    else
#        define IN_OFF(x0, x1, x2, x3, x4, x5)                             \
            ((x0 % 2) + ((x0 % 16) / 2) * 32 + ((x0) / SRC_B0) * SRC_S0    \
                    + 2 * (x1 % 16) + ((x1) / SRC_B1) * SRC_S1             \
                    + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
                    + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
                    + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 \
                    + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)
#    endif
#endif

#if IN_OIHW8I16O2I
#    undef IN_OFF
#    if WITH_GROUP
#        define IN_OFF(x0, x1, x2, x3, x4, x5)                               \
            (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 + (x2 % 2) \
                    + ((x2 % 16) / 2) * 32 + ((x2) / SRC_B2) * SRC_S2        \
                    + 2 * (x1 % 16) + ((x1) / SRC_B1) * SRC_S1               \
                    + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3   \
                    + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4   \
                    + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)
#    else
#        define IN_OFF(x0, x1, x2, x3, x4, x5)                             \
            ((x1 % 2) + ((x1 % 16) / 2) * 32 + ((x1) / SRC_B1) * SRC_S1    \
                    + 2 * (x0 % 16) + ((x0) / SRC_B0) * SRC_S0             \
                    + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
                    + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
                    + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 \
                    + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)
#    endif
#endif

#if IN_OIHW4O8I8O4I
#    undef IN_OFF
#    if WITH_GROUP
#        define IN_OFF(x0, x1, x2, x3, x4, x5)                             \
            (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0          \
                    + 4 * (p[x1 % 32] % 8) + ((p[x1 % 32]) / 8) * 256      \
                    + ((x1) / SRC_B1) * SRC_S1 + (x2 % 4)                  \
                    + ((x2 % 32) / 4) * 32 + ((x2) / SRC_B2) * SRC_S2      \
                    + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
                    + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 \
                    + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)
#    else
#        define IN_OFF(x0, x1, x2, x3, x4, x5)                               \
            (4 * (x0 % 8) + ((x0 % 32) / 8) * 256 + ((x0) / SRC_B0) * SRC_S0 \
                    + (x1 % 4) + ((x1 % 32) / 4) * 32                        \
                    + ((x1) / SRC_B1) * SRC_S1 + ((x2) % SRC_B2) * SRC_SB2   \
                    + ((x2) / SRC_B2) * SRC_S2 + ((x3) % SRC_B3) * SRC_SB3   \
                    + ((x3) / SRC_B3) * SRC_S3 + ((x4) % SRC_B4) * SRC_SB4   \
                    + ((x4) / SRC_B4) * SRC_S4 + ((x5) % SRC_B5) * SRC_SB5   \
                    + ((x5) / SRC_B5) * SRC_S5)
#    endif
#endif
#if IN_OIHW2O8I8O2I
#    undef IN_OFF
#    if WITH_GROUP
#        define IN_OFF(x0, x1, x2, x3, x4, x5)                             \
            (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0          \
                    + 2 * (x1 % 8) + ((x1 % 16) / 8) * 128                 \
                    + ((x1) / SRC_B1) * SRC_S1 + (x2 % 2)                  \
                    + ((x2 % 16) / 2) * 16 + ((x2) / SRC_B2) * SRC_S2      \
                    + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
                    + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 \
                    + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)
#    else
#        define IN_OFF(x0, x1, x2, x3, x4, x5)                               \
            (2 * (x0 % 8) + ((x0 % 16) / 8) * 128 + ((x0) / SRC_B0) * SRC_S0 \
                    + (x1 % 2) + ((x1 % 16) / 2) * 16                        \
                    + ((x1) / SRC_B1) * SRC_S1 + ((x2) % SRC_B2) * SRC_SB2   \
                    + ((x2) / SRC_B2) * SRC_S2 + ((x3) % SRC_B3) * SRC_SB3   \
                    + ((x3) / SRC_B3) * SRC_S3 + ((x4) % SRC_B4) * SRC_SB4   \
                    + ((x4) / SRC_B4) * SRC_S4 + ((x5) % SRC_B5) * SRC_SB5   \
                    + ((x5) / SRC_B5) * SRC_S5)
#    endif
#endif

#define OUT_OFF(x0, x1, x2, x3, x4, x5)                            \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0          \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4 \
            + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)

#if OUT_OIHW8O16I2O
#    undef OUT_OFF
#    if WITH_GROUP
#        define OUT_OFF(x0, x1, x2, x3, x4, x5)                              \
            (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 + (x1 % 2) \
                    + ((x1 % 16) / 2) * 32 + ((x1) / DST_B1) * DST_S1        \
                    + 2 * (x2 % 16) + ((x2) / DST_B2) * DST_S2               \
                    + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3   \
                    + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4   \
                    + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#    else
#        define OUT_OFF(x0, x1, x2, x3, x4, x5)                            \
            ((x0 % 2) + ((x0 % 16) / 2) * 32 + ((x0) / DST_B0) * DST_S0    \
                    + 2 * (x1 % 16) + ((x1) / DST_B1) * DST_S1             \
                    + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
                    + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
                    + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4 \
                    + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#    endif
#endif

#if OUT_OIHW8I16O2I
#    undef OUT_OFF
#    if WITH_GROUP
#        define OUT_OFF(x0, x1, x2, x3, x4, x5)                              \
            (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 + (x2 % 2) \
                    + ((x2 % 16) / 2) * 32 + ((x2) / DST_B2) * DST_S2        \
                    + 2 * (x1 % 16) + ((x1) / DST_B1) * DST_S1               \
                    + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3   \
                    + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4   \
                    + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#    else
#        define OUT_OFF(x0, x1, x2, x3, x4, x5)                            \
            ((x1 % 2) + ((x1 % 16) / 2) * 32 + ((x1) / DST_B1) * DST_S1    \
                    + 2 * (x0 % 16) + ((x0) / DST_B0) * DST_S0             \
                    + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
                    + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
                    + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4 \
                    + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#    endif
#endif

#if OUT_OIHW4O8I8O4I
#    undef OUT_OFF
#    if WITH_GROUP
#        define OUT_OFF(x0, x1, x2, x3, x4, x5)                            \
            (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0          \
                    + 4 * (p[x1 % 32] % 8) + ((p[x1 % 32]) / 8) * 256      \
                    + ((x1) / DST_B1) * DST_S1 + (x2 % 4)                  \
                    + ((x2 % 32) / 4) * 32 + ((x2) / DST_B2) * DST_S2      \
                    + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
                    + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4 \
                    + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#    else
#        define OUT_OFF(x0, x1, x2, x3, x4, x5)                              \
            (4 * (x0 % 8) + ((x0 % 32) / 8) * 256 + ((x0) / DST_B0) * DST_S0 \
                    + (x1 % 4) + ((x1 % 32) / 4) * 32                        \
                    + ((x1) / DST_B1) * DST_S1 + ((x2) % DST_B2) * DST_SB2   \
                    + ((x2) / DST_B2) * DST_S2 + ((x3) % DST_B3) * DST_SB3   \
                    + ((x3) / DST_B3) * DST_S3 + ((x4) % DST_B4) * DST_SB4   \
                    + ((x4) / DST_B4) * DST_S4 + ((x5) % DST_B5) * DST_SB5   \
                    + ((x5) / DST_B5) * DST_S5)
#    endif
#endif
#if OUT_OIHW2O8I8O2I
#    undef OUT_OFF
#    if WITH_GROUP
#        define OUT_OFF(x0, x1, x2, x3, x4, x5)                            \
            (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0          \
                    + 2 * (x1 % 8) + ((x1 % 16) / 8) * 128                 \
                    + ((x1) / DST_B1) * DST_S1 + (x2 % 2)                  \
                    + ((x2 % 16) / 2) * 16 + ((x2) / DST_B2) * DST_S2      \
                    + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
                    + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4 \
                    + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#    else
#        define OUT_OFF(x0, x1, x2, x3, x4, x5)                              \
            (2 * (x0 % 8) + ((x0 % 32) / 8) * 128 + ((x0) / DST_B0) * DST_S0 \
                    + (x1 % 2) + ((x1 % 32) / 2) * 16                        \
                    + ((x1) / DST_B1) * DST_S1 + ((x2) % DST_B2) * DST_SB2   \
                    + ((x2) / DST_B2) * DST_S2 + ((x3) % DST_B3) * DST_SB3   \
                    + ((x3) / DST_B3) * DST_S3 + ((x4) % DST_B4) * DST_SB4   \
                    + ((x4) / DST_B4) * DST_S4 + ((x5) % DST_B5) * DST_SB5   \
                    + ((x5) / DST_B5) * DST_S5)
#    endif
#endif

#if WITH_GROUP
#    define IN_OFF_G(gr, x0, x1, x2, x3, x4) IN_OFF(gr, x0, x1, x2, x3, x4)
#    define OUT_OFF_G(gr, x0, x1, x2, x3, x4) OUT_OFF(gr, x0, x1, x2, x3, x4)
#else
#    define IN_OFF_G(gr, x0, x1, x2, x3, x4) IN_OFF(x0, x1, x2, x3, x4, 0)
#    define OUT_OFF_G(gr, x0, x1, x2, x3, x4) OUT_OFF(x0, x1, x2, x3, x4, 0)
#endif

#if IN_TYPE_S8
#    define DT_IN char
#    define DT_IN8 char8
#    define AS_DATA_T_IN as_char
#    define AS_DATA8_T_IN as_char8
#endif
#if IN_TYPE_U8
#    define DT_IN uchar
#    define DT_IN8 uchar8
#    define AS_DATA_T_IN as_uchar
#    define AS_DATA8_T_IN as_uchar8
#endif
#if IN_TYPE_F16
#    define DT_IN half
#    define DT_IN8 half8
#    define AS_UINT_T_IN as_ushort
#    define AS_UINT8_T_IN as_ushort8
#    define AS_DATA_T_IN as_half
#    define AS_DATA8_T_IN as_half8
#    define BLOCK_READ8_IN intel_sub_group_block_read_us8
#    define BLOCK_WRITE8_IN intel_sub_group_block_write_us8
#    define BLOCK_READ_IN intel_sub_group_block_read_us
#    define BLOCK_WRITE_IN intel_sub_group_block_write_us
#    define GLOBAL_UINT_PT_IN const __global ushort *
#endif
#if IN_TYPE_S32
#    define DT_IN int
#    define DT_IN8 int8
#    define AS_UINT_T_IN as_uint
#    define AS_UINT8_T_IN as_uint8
#    define AS_DATA_T_IN as_int
#    define AS_DATA8_T_IN as_int8
#    define BLOCK_READ8_IN intel_sub_group_block_read8
#    define BLOCK_WRITE8_IN intel_sub_group_block_write8
#    define BLOCK_READ_IN intel_sub_group_block_read
#    define BLOCK_WRITE_IN intel_sub_group_block_write
#    define GLOBAL_UINT_PT_IN const __global uint *
#endif
#if IN_TYPE_F32
#    define DT_IN float
#    define DT_IN8 float8
#    define AS_UINT_T_IN as_uint
#    define AS_UINT8_T_IN as_uint8
#    define AS_DATA_T_IN as_float
#    define AS_DATA8_T_IN as_float8
#    define BLOCK_READ8_IN intel_sub_group_block_read8
#    define BLOCK_WRITE8_IN intel_sub_group_block_write8
#    define BLOCK_READ_IN intel_sub_group_block_read
#    define BLOCK_WRITE_IN intel_sub_group_block_write
#    define GLOBAL_UINT_PT_IN const __global uint *
#endif
#if OUT_TYPE_S8
#    define DT_OUT char
#    define DT_OUT8 char8
#    define AS_DATA_T_OUT as_char
#    define AS_DATA8_T_OUT as_char8
#endif
#if OUT_TYPE_U8
#    define DT_OUT uchar
#    define DT_OUT8 uchar8
#    define AS_DATA_T_OUT as_uchar
#    define AS_DATA8_T_OUT as_uchar8
#endif
#if OUT_TYPE_F16
#    define DT_OUT half
#    define DT_OUT8 half8
#    define AS_UINT_T_OUT as_ushort
#    define AS_UINT8_T_OUT as_ushort8
#    define AS_DATA_T_OUT as_half
#    define AS_DATA8_T_OUT as_half8
#    define BLOCK_READ8_OUT intel_sub_group_block_read_us8
#    define BLOCK_WRITE8_OUT intel_sub_group_block_write_us8
#    define BLOCK_READ_OUT intel_sub_group_block_read_us
#    define BLOCK_WRITE_OUT intel_sub_group_block_write_us
#    define GLOBAL_UINT_PT_OUT __global ushort *
#endif
#if OUT_TYPE_F32
#    define DT_OUT float
#    define DT_OUT8 float8
#    define AS_UINT_T_OUT as_uint
#    define AS_UINT8_T_OUT as_uint8
#    define AS_DATA_T_OUT as_float
#    define AS_DATA8_T_OUT as_float8
#    define BLOCK_READ8_OUT intel_sub_group_block_read8
#    define BLOCK_WRITE8_OUT intel_sub_group_block_write8
#    define BLOCK_READ_OUT intel_sub_group_block_read
#    define BLOCK_WRITE_OUT intel_sub_group_block_write
#    define GLOBAL_UINT_PT_OUT __global uint *
#endif
#if OUT_TYPE_S32
#    define DT_OUT int
#    define DT_OUT8 int8
#    define AS_UINT_T_OUT as_uint
#    define AS_UINT8_T_OUT as_uint8
#    define AS_DATA_T_OUT as_int
#    define AS_DATA8_T_OUT as_int8
#    define BLOCK_READ8_OUT intel_sub_group_block_read8
#    define BLOCK_WRITE8_OUT intel_sub_group_block_write8
#    define BLOCK_READ_OUT intel_sub_group_block_read
#    define BLOCK_WRITE_OUT intel_sub_group_block_write
#    define GLOBAL_UINT_PT_OUT __global uint *
#endif

#if IN_TYPE_F16 || OUT_TYPE_F16
#    pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if SUB_GROUP_SIZE != 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#endif
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
__kernel void any2any_kernel(__global DT_IN *input, __global DT_OUT *output,
        float alpha, float beta) {

    input += SRC_OFFSET_PAD;
    output += DST_OFFSET_PAD;

#if REF_REORDER

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    d0 = get_global_id(0) / BLOCK_0;
#    if NDIMS > 1
    d1 = get_global_id(0) % BLOCK_0;
#    endif
#    if NDIMS > 2
    d2 = get_global_id(1) / BLOCK_1;
#    endif
#    if NDIMS > 3
    d3 = get_global_id(1) % BLOCK_1;
#    endif
#    if NDIMS > 4
    d4 = get_global_id(2) / BLOCK_2;
#    endif
#    if NDIMS > 5
    d5 = get_global_id(2) % BLOCK_2;
#    endif

    int p[32] = { 0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27, 4,
        12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31 };

#    if PAD_FILL_ZERO
    if (d0 >= SRC_DIM0
#        if NDIMS > 1
        || d1 >= SRC_DIM1
#        endif
#        if NDIMS > 2
        || d2 >= SRC_DIM2
#        endif
#        if NDIMS > 3
        || d3 >= SRC_DIM3
#        endif
#        if NDIMS > 4
        || d4 >= SRC_DIM4
#        endif
#        if NDIMS > 5
        || d5 >= SRC_DIM5
#        endif
    ) {
        output[OUT_OFF(d0,d1,d2,d3,d4,d5)] = 0;
        return;
    }
#    endif
    const int in_off = IN_OFF(d0,d1,d2,d3,d4,d5);
    const int out_off = OUT_OFF(d0,d1,d2,d3,d4,d5);
#    if ALPHA_BETA
    const float a = convert_float(input[in_off]);
    const float b = convert_float(output[out_off]);
    const float tmp = (beta == 0) ? alpha * a : alpha * a + beta * b;
#        if OUT_TYPE_F16
        output[out_off] = convert_half(tmp);
#        else
        output[out_off] = tmp;
#        endif
#    else
#        if IN_TYPE_F16 && !OUT_TYPE_F16
    output[out_off] = convert_float(input[in_off]);
#        elif !IN_TYPE_F16 && OUT_TYPE_F16
    output[out_off] = convert_half(input[in_off]);
#        else
    output[out_off] = input[in_off];
#        endif
#   endif
    return;
#else

#    if IN_NCHW16N16C == 1 || OUT_NCHW16N16C == 1
    const int d0 = get_group_id(0) * 16;
    const int d1 = get_group_id(1) * 16;
    const int d2 = get_group_id(2) / BLOCK_2;
    const int d3 = get_group_id(2) % BLOCK_2;
    const int local_id = get_local_id(1);

#        if IN_NCHW16N16C == 1
    input += IN_OFF(d0, d1, d2, d3, 0, 0);
    output += OUT_OFF(d0, d1 + local_id, d2, d3, 0, 0);

    DT_IN8 blockI0
            = AS_DATA8_T_IN(BLOCK_READ8_IN((GLOBAL_UINT_PT_IN)&input[0]));
    DT_IN8 blockI1
            = AS_DATA8_T_IN(BLOCK_READ8_IN((GLOBAL_UINT_PT_IN)&input[8 * 16]));

    for (int i = 0; i < 8; i++) {
        output[OUT_OFF(i, 0, 0, 0, 0, 0)] = blockI0[i];
        output[OUT_OFF(i + 8, 0, 0, 0, 0, 0)] = blockI1[i];
    }
#        else
    DT_OUT8 blockI0, blockI1;
    input += IN_OFF(d0, d1 + local_id, d2, d3, 0, 0);
    output += OUT_OFF(d0, d1, d2, d3, 0, 0);

    for (int i = 0; i < 8; i++) {
        blockI0[i] = input[IN_OFF(i, 0, 0, 0, 0, 0)];
        blockI1[i] = input[IN_OFF(i + 8, 0, 0, 0, 0, 0)];
    }

    BLOCK_WRITE8_OUT((GLOBAL_UINT_PT_OUT)&output[0], AS_UINT8_T_OUT(blockI0));
    BLOCK_WRITE8_OUT(
            (GLOBAL_UINT_PT_OUT)&output[8 * 16], AS_UINT8_T_OUT(blockI1));
#        endif
#    elif IN_NCHW16C == 1 || OUT_NCHW16C == 1
    const int d0 = get_group_id(0);
    const int d1 = get_group_id(1) * 16;
    const int d2 = get_group_id(2) / BLOCK_2;
    const int d3 = get_group_id(2) % BLOCK_2;
    const int local_id = get_local_id(1);

#        if IN_NCHW16C == 1
    input += IN_OFF(d0, d1, d2, d3, 0, 0);
    output += OUT_OFF(d0, d1 + local_id, d2, d3, 0, 0);

    DT_IN blockI0 = AS_DATA_T_IN(BLOCK_READ_IN((GLOBAL_UINT_PT_IN)&input[0]));

    output[0] = blockI0;
#        else
    DT_OUT blockI0;
    input += IN_OFF(d0, d1 + local_id, d2, d3, 0, 0);
    output += OUT_OFF(d0, d1, d2, d3, 0, 0);

    blockI0 = input[0];

    BLOCK_WRITE_OUT((GLOBAL_UINT_PT_OUT)&output[0], AS_UINT_T_OUT(blockI0));
#        endif
#    elif IN_IOHW16I16O == 1 || OUT_IOHW16I16O == 1 || IN_OIHW16O16I == 1 \
            || OUT_OIHW16O16I == 1
    const int g = get_global_id(0) / BLOCK_0;
    const int d0 = ((get_global_id(0) % BLOCK_0) / 16) * 16;
    const int d1 = get_group_id(1) * 16;
    const int d2 = get_group_id(2) / BLOCK_2;
    const int d3 = get_group_id(2) % BLOCK_2;
    const int local_id = get_local_id(0);

#        if IN_IOHW16I16O == 1 || IN_OIHW16O16I == 1
    input += IN_OFF_G(g, d0, d1, d2, d3, 0);
#        endif
#        if IN_REF_FORMAT == 1
#            if OUT_IOHW16I16O == 1
    input += IN_OFF_G(g, d0 + local_id, d1, d2, d3, 0);
#            else
    input += IN_OFF_G(g, d0, d1 + local_id, d2, d3, 0);
#            endif
#        endif

#        if OUT_IOHW16I16O == 1 || OUT_OIHW16O16I == 1
    output += OUT_OFF_G(g, d0, d1, d2, d3, 0);
#        endif
#        if OUT_REF_FORMAT == 1
#            if IN_IOHW16I16O == 1
    output += OUT_OFF_G(g, d0 + local_id, d1, d2, d3, 0);
#            else
    output += OUT_OFF_G(g, d0, d1 + local_id, d2, d3, 0);
#            endif
#        endif

    float8 blockI0, blockI1;

#        if IN_IOHW16I16O == 1 || IN_OIHW16O16I == 1
    blockI0 = as_float8(
            intel_sub_group_block_read8((const __global uint *)&input[0]));
    blockI1 = as_float8(
            intel_sub_group_block_read8((const __global uint *)&input[8 * 16]));
#        else
    for (int i = 0; i < 8; i++) {
#            if OUT_IOHW16I16O == 1
        blockI0[i] = input[IN_OFF_G(0, 0, i, 0, 0, 0)];
        blockI1[i] = input[IN_OFF_G(0, 0, i + 8, 0, 0, 0)];
#            else
        blockI0[i] = input[IN_OFF_G(0, i, 0, 0, 0, 0)];
        blockI1[i] = input[IN_OFF_G(0, i + 8, 0, 0, 0, 0)];
#            endif
    }
#        endif

#        if (IN_IOHW16I16O == 1 || IN_OIHW16O16I == 1) \
                && (OUT_IOHW16I16O == 1 || OUT_OIHW16O16I == 1)
    for (int i = 0; i < 8; i++) {
#            if IN_IOHW16I16O == 1
        output[OUT_OFF_G(0, local_id, i, 0, 0, 0)] = blockI0[i];
        output[OUT_OFF_G(0, local_id, i + 8, 0, 0, 0)] = blockI1[i];
#            else
        output[OUT_OFF_G(0, i, local_id, 0, 0, 0)] = blockI0[i];
        output[OUT_OFF_G(0, i + 8, local_id, 0, 0, 0)] = blockI1[i];
#            endif
    }
#        else
#            if OUT_IOHW16I16O == 1 || OUT_OIHW16O16I == 1
    intel_sub_group_block_write8(
            (__global uint *)&output[0], as_uint8(blockI0));
    intel_sub_group_block_write8(
            (__global uint *)&output[8 * 16], as_uint8(blockI1));
#            else
    for (int i = 0; i < 8; i++) {
#                if IN_IOHW16I16O == 1
        output[OUT_OFF_G(0, 0, i, 0, 0, 0)] = blockI0[i];
        output[OUT_OFF_G(0, 0, i + 8, 0, 0, 0)] = blockI1[i];
#                else
        output[OUT_OFF_G(0, i, 0, 0, 0, 0)] = blockI0[i];
        output[OUT_OFF_G(0, i + 8, 0, 0, 0, 0)] = blockI1[i];
#                endif
    }
#            endif
#        endif
#    endif
#endif
}
