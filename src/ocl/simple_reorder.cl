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

#include "ocl/ocl_math_utils.h"

#if IN_TYPE_F16 || OUT_TYPE_F16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define IN_OFF(x0, x1, x2, x3, x4, x5)                             \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0          \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 \
            + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)

#if IN_OIHW8O16I2O
#undef IN_OFF
#if WITH_GROUP
#define IN_OFF(x0, x1, x2, x3, x4, x5)                                        \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 + (x1 % 2)          \
            + ((x1 % 16) / 2) * 32 + ((x1) / SRC_B1) * SRC_S1 + 2 * (x2 % 16) \
            + ((x2) / SRC_B2) * SRC_S2 + ((x3) % SRC_B3) * SRC_SB3            \
            + ((x3) / SRC_B3) * SRC_S3 + ((x4) % SRC_B4) * SRC_SB4            \
            + ((x4) / SRC_B4) * SRC_S4 + ((x5) % SRC_B5) * SRC_SB5            \
            + ((x5) / SRC_B5) * SRC_S5)
#else
#define IN_OFF(x0, x1, x2, x3, x4, x5)                             \
    ((x0 % 2) + ((x0 % 16) / 2) * 32 + ((x0) / SRC_B0) * SRC_S0    \
            + 2 * (x1 % 16) + ((x1) / SRC_B1) * SRC_S1             \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 \
            + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)
#endif
#endif

#if IN_OIHW8I16O2I
#undef IN_OFF
#if WITH_GROUP
#define IN_OFF(x0, x1, x2, x3, x4, x5)                                        \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 + (x2 % 2)          \
            + ((x2 % 16) / 2) * 32 + ((x2) / SRC_B2) * SRC_S2 + 2 * (x1 % 16) \
            + ((x1) / SRC_B1) * SRC_S1 + ((x3) % SRC_B3) * SRC_SB3            \
            + ((x3) / SRC_B3) * SRC_S3 + ((x4) % SRC_B4) * SRC_SB4            \
            + ((x4) / SRC_B4) * SRC_S4 + ((x5) % SRC_B5) * SRC_SB5            \
            + ((x5) / SRC_B5) * SRC_S5)
#else
#define IN_OFF(x0, x1, x2, x3, x4, x5)                             \
    ((x1 % 2) + ((x1 % 16) / 2) * 32 + ((x1) / SRC_B1) * SRC_S1    \
            + 2 * (x0 % 16) + ((x0) / SRC_B0) * SRC_S0             \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 \
            + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)
#endif
#endif

#if IN_OIHW4O8I8O4I
#undef IN_OFF
#if WITH_GROUP
#define IN_OFF(x0, x1, x2, x3, x4, x5)                                   \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0                \
            + 4 * (p[x1 % 32] % 8) + ((p[x1 % 32]) / 8) * 256            \
            + ((x1) / SRC_B1) * SRC_S1 + (x2 % 4) + ((x2 % 32) / 4) * 32 \
            + ((x2) / SRC_B2) * SRC_S2 + ((x3) % SRC_B3) * SRC_SB3       \
            + ((x3) / SRC_B3) * SRC_S3 + ((x4) % SRC_B4) * SRC_SB4       \
            + ((x4) / SRC_B4) * SRC_S4 + ((x5) % SRC_B5) * SRC_SB5       \
            + ((x5) / SRC_B5) * SRC_S5)
#else
#define IN_OFF(x0, x1, x2, x3, x4, x5)                                   \
    (4 * (x0 % 8) + ((x0 % 32) / 8) * 256 + ((x0) / SRC_B0) * SRC_S0     \
            + (x1 % 4) + ((x1 % 32) / 4) * 32 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2       \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3       \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4       \
            + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)
#endif
#endif
#if IN_OIHW2O8I8O2I
#undef IN_OFF
#if WITH_GROUP
#define IN_OFF(x0, x1, x2, x3, x4, x5)                                 \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 + ((x2) % 2) \
            + ((x1) % 8) * 2 + (((x2) % 16) / 2) * 16                  \
            + (((x1) % 16) / 8) * 128 + ((x1) / SRC_B1) * SRC_S1       \
            + ((x2) / SRC_B2) * SRC_S2 + ((x3) % SRC_B3) * SRC_SB3     \
            + ((x3) / SRC_B3) * SRC_S3 + ((x4) % SRC_B4) * SRC_SB4     \
            + ((x4) / SRC_B4) * SRC_S4 + ((x5) % SRC_B5) * SRC_SB5     \
            + ((x5) / SRC_B5) * SRC_S5)
#else
#define IN_OFF(x0, x1, x2, x3, x4, x5)                             \
    (((x1) % 2) + ((x0) % 8) * 2 + (((x1) % 16) / 2) * 16          \
            + (((x0) % 16) / 8) * 128 + ((x0) / SRC_B0) * SRC_S0   \
            + ((x1) / SRC_B1) * SRC_S1 + ((x2) % SRC_B2) * SRC_SB2 \
            + ((x2) / SRC_B2) * SRC_S2 + ((x3) % SRC_B3) * SRC_SB3 \
            + ((x3) / SRC_B3) * SRC_S3 + ((x4) % SRC_B4) * SRC_SB4 \
            + ((x4) / SRC_B4) * SRC_S4 + ((x5) % SRC_B5) * SRC_SB5 \
            + ((x5) / SRC_B5) * SRC_S5)
#endif
#endif

#define OUT_OFF(x0, x1, x2, x3, x4, x5)                            \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0          \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4 \
            + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)

#if OUT_OIHW8O16I2O
#undef OUT_OFF
#if WITH_GROUP
#define OUT_OFF(x0, x1, x2, x3, x4, x5)                                       \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 + (x1 % 2)          \
            + ((x1 % 16) / 2) * 32 + ((x1) / DST_B1) * DST_S1 + 2 * (x2 % 16) \
            + ((x2) / DST_B2) * DST_S2 + ((x3) % DST_B3) * DST_SB3            \
            + ((x3) / DST_B3) * DST_S3 + ((x4) % DST_B4) * DST_SB4            \
            + ((x4) / DST_B4) * DST_S4 + ((x5) % DST_B5) * DST_SB5            \
            + ((x5) / DST_B5) * DST_S5)
#else
#define OUT_OFF(x0, x1, x2, x3, x4, x5)                            \
    ((x0 % 2) + ((x0 % 16) / 2) * 32 + ((x0) / DST_B0) * DST_S0    \
            + 2 * (x1 % 16) + ((x1) / DST_B1) * DST_S1             \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4 \
            + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#endif
#endif

#if OUT_OIHW8I16O2I
#undef OUT_OFF
#if WITH_GROUP
#define OUT_OFF(x0, x1, x2, x3, x4, x5)                                       \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 + (x2 % 2)          \
            + ((x2 % 16) / 2) * 32 + ((x2) / DST_B2) * DST_S2 + 2 * (x1 % 16) \
            + ((x1) / DST_B1) * DST_S1 + ((x3) % DST_B3) * DST_SB3            \
            + ((x3) / DST_B3) * DST_S3 + ((x4) % DST_B4) * DST_SB4            \
            + ((x4) / DST_B4) * DST_S4 + ((x5) % DST_B5) * DST_SB5            \
            + ((x5) / DST_B5) * DST_S5)
#else
#define OUT_OFF(x0, x1, x2, x3, x4, x5)                            \
    ((x1 % 2) + ((x1 % 16) / 2) * 32 + ((x1) / DST_B1) * DST_S1    \
            + 2 * (x0 % 16) + ((x0) / DST_B0) * DST_S0             \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4 \
            + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#endif
#endif

#if OUT_OIHW4O8I8O4I
#undef OUT_OFF
#if WITH_GROUP
#define OUT_OFF(x0, x1, x2, x3, x4, x5)                                  \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0                \
            + 4 * (p[x1 % 32] % 8) + ((p[x1 % 32]) / 8) * 256            \
            + ((x1) / DST_B1) * DST_S1 + (x2 % 4) + ((x2 % 32) / 4) * 32 \
            + ((x2) / DST_B2) * DST_S2 + ((x3) % DST_B3) * DST_SB3       \
            + ((x3) / DST_B3) * DST_S3 + ((x4) % DST_B4) * DST_SB4       \
            + ((x4) / DST_B4) * DST_S4 + ((x5) % DST_B5) * DST_SB5       \
            + ((x5) / DST_B5) * DST_S5)
#else
#define OUT_OFF(x0, x1, x2, x3, x4, x5)                                  \
    (4 * (x0 % 8) + ((x0 % 32) / 8) * 256 + ((x0) / DST_B0) * DST_S0     \
            + (x1 % 4) + ((x1 % 32) / 4) * 32 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2       \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3       \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4       \
            + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#endif
#endif
#if OUT_OIHW2O8I8O2I
#undef OUT_OFF
#if WITH_GROUP
#define OUT_OFF(x0, x1, x2, x3, x4, x5)                                \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 + ((x2) % 2) \
            + ((x1) % 8) * 2 + (((x2) % 16) / 2) * 16                  \
            + (((x1) % 16) / 8) * 128 + ((x1) / DST_B1) * DST_S1       \
            + ((x2) / DST_B2) * DST_S2 + ((x3) % DST_B3) * DST_SB3     \
            + ((x3) / DST_B3) * DST_S3 + ((x4) % DST_B4) * DST_SB4     \
            + ((x4) / DST_B4) * DST_S4 + ((x5) % DST_B5) * DST_SB5     \
            + ((x5) / DST_B5) * DST_S5)
#else
#define OUT_OFF(x0, x1, x2, x3, x4, x5)                                     \
    ((x1 % 2) + (x0 % 8) * 2 + ((x1 % 16) / 2) * 16 + ((x0 % 16) / 8) * 128 \
            + ((x0) / DST_B0) * DST_S0 + ((x1) / DST_B1) * DST_S1           \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2          \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3          \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4          \
            + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)
#endif
#endif

#if WITH_GROUP
#define IN_OFF_G(gr, x0, x1, x2, x3, x4) IN_OFF(gr, x0, x1, x2, x3, x4)
#define OUT_OFF_G(gr, x0, x1, x2, x3, x4) OUT_OFF(gr, x0, x1, x2, x3, x4)
#else
#define IN_OFF_G(gr, x0, x1, x2, x3, x4) IN_OFF(x0, x1, x2, x3, x4, 0)
#define OUT_OFF_G(gr, x0, x1, x2, x3, x4) OUT_OFF(x0, x1, x2, x3, x4, 0)
#endif

#if IN_TYPE_S8
#define DT_IN char
#define DT_IN8 char8
#define SG_READ_IN(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SG_READ8_IN(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define SG_WRITE_IN(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define SG_WRITE8_IN(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // IN_TYPE_S8

#if IN_TYPE_U8
#define DT_IN uchar
#define DT_IN8 uchar8
#define SG_READ_IN(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SG_READ8_IN(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define SG_WRITE_IN(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define SG_WRITE8_IN(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // IN_TYPE_U8

#if IN_TYPE_F16
#define DT_IN half
#define DT_IN8 half8
#define SG_READ_IN(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SG_READ8_IN(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define SG_WRITE_IN(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SG_WRITE8_IN(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // IN_TYPE_F16

#if IN_TYPE_S32
#define DT_IN int
#define DT_IN8 int8
#define SG_READ_IN(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SG_READ8_IN(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#define SG_WRITE_IN(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SG_WRITE8_IN(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // IN_TYPE_S32

#if IN_TYPE_F32
#define DT_IN float
#define DT_IN8 float8
#define SG_READ_IN(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SG_READ8_IN(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#define SG_WRITE_IN(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SG_WRITE8_IN(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // IN_TYPE_F32

#if IN_TYPE_BF16
#define DT_IN ushort
#define DT_IN8 ushort8
#define SG_READ_IN(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SG_READ8_IN(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define SG_WRITE_IN(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SG_WRITE8_IN(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // IN_TYPE_F16

#if OUT_TYPE_S8
#define DT_OUT char
#define DT_OUT8 char8
#define SG_READ_OUT(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SG_READ8_OUT(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define SG_WRITE_OUT(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define SG_WRITE8_OUT(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // OUT_TYPE_S8

#if OUT_TYPE_U8
#define DT_OUT uchar
#define DT_OUT8 uchar8
#define SG_READ_OUT(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SG_READ8_OUT(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define SG_WRITE_OUT(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define SG_WRITE8_OUT(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // IN_TYPE_U8

#if OUT_TYPE_F16
#define DT_OUT half
#define DT_OUT8 half8
#define SG_READ_OUT(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SG_READ8_OUT(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define SG_WRITE_OUT(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SG_WRITE8_OUT(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // OUT_TYPE_F16

#if OUT_TYPE_S32
#define DT_OUT int
#define DT_OUT8 int8
#define SG_READ_OUT(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SG_READ8_OUT(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#define SG_WRITE_OUT(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SG_WRITE8_OUT(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // OUT_TYPE_S32

#if OUT_TYPE_F32
#define DT_OUT float
#define DT_OUT8 float8
#define SG_READ_OUT(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SG_READ8_OUT(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#define SG_WRITE_OUT(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SG_WRITE8_OUT(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // OUT_TYPE_F32

#if OUT_TYPE_BF16
#define DT_OUT ushort
#define DT_OUT8 ushort8
#define SG_READ_OUT(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SG_READ8_OUT(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define SG_WRITE_OUT(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SG_WRITE8_OUT(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // IN_TYPE_F16

#if IN_TYPE_BF16 || OUT_TYPE_BF16
float8 bfloat_to_float8(ushort8 b) {
    float8 r;
    for (int i = 0; i < 8; ++i) {
        r[i] = convert_bf16_to_f32(b[i]);
    }
    return r;
}
ushort8 float_to_bfloat8(float8 b) {
    ushort8 r;
    for (int i = 0; i < 8; ++i) {
        r[i] = convert_f32_to_bf16(b[i]);
    }
    return r;
}
#endif // IN_TYPE_BF16 || OUT_TYPE_BF16

#if IN_TYPE_BF16
#define CONVERT_IN_TO_F32 convert_bf16_to_f32
#define CONVERT_IN_TO_F32_8 bfloat_to_float8
#else
#define CONVERT_IN_TO_F32 convert_float
#define CONVERT_IN_TO_F32_8 convert_float8
#endif

#if OUT_TYPE_BF16
#define CONVERT_OUT_TO_F32 convert_bf16_to_f32
#define CONVERT_OUT_TO_F32_8 bfloat_to_float8
#else
#define CONVERT_OUT_TO_F32 convert_float
#define CONVERT_OUT_TO_F32_8 convert_float8
#endif

#if OUT_TYPE_S8
#define CONVERT_F32_TO_OUT convert_char_sat_rte
#define CONVERT_F32_TO_OUT8 convert_char8_sat_rte
#elif OUT_TYPE_U8
#define CONVERT_F32_TO_OUT convert_uchar_sat_rte
#define CONVERT_F32_TO_OUT8 convert_uchar8_sat_rte
#elif OUT_TYPE_F16
#define CONVERT_F32_TO_OUT convert_half
#define CONVERT_F32_TO_OUT8 convert_half8
#elif OUT_TYPE_F32
#define CONVERT_F32_TO_OUT convert_float
#define CONVERT_F32_TO_OUT8 convert_float8
#elif OUT_TYPE_S32
#define CONVERT_F32_TO_OUT convert_int_sat_rte
#define CONVERT_F32_TO_OUT8 convert_int8_sat_rte
#elif OUT_TYPE_BF16
#define CONVERT_F32_TO_OUT convert_f32_to_bf16
#define CONVERT_F32_TO_OUT8 float_to_bfloat8
#else
#define CONVERT_F32_TO_OUT
#define CONVERT_F32_TO_OUT8
#endif

#if IN_TYPE_BF16 && OUT_TYPE_BF16
#define CONVERT_IN_TO_OUT(x) (x)
#define CONVERT_IN_TO_OUT8(x) (x)
#elif IN_TYPE_BF16 || OUT_TYPE_BF16
#define CONVERT_IN_TO_OUT(x) CONVERT_F32_TO_OUT(CONVERT_IN_TO_F32(x))
#define CONVERT_IN_TO_OUT8(x) CONVERT_F32_TO_OUT8(CONVERT_IN_TO_F32_8(x))
#elif (IN_TYPE_U8 || IN_TYPE_S32) && OUT_TYPE_S8
#define CONVERT_IN_TO_OUT(x) convert_char_sat(x)
#define CONVERT_IN_TO_OUT8(x) convert_char8_sat(x)
#elif (IN_TYPE_S8 || IN_TYPE_S32) && OUT_TYPE_U8
#define CONVERT_IN_TO_OUT(x) convert_uchar_sat(x)
#define CONVERT_IN_TO_OUT8(x) convert_uchar8_sat(x)
#elif (IN_TYPE_S8 || IN_TYPE_U8) && OUT_TYPE_S32
#define CONVERT_IN_TO_OUT(x) convert_int(x)
#define CONVERT_IN_TO_OUT8(x) convert_int8(x)
#else
#define CONVERT_IN_TO_OUT(x) CONVERT_F32_TO_OUT(x)
#define CONVERT_IN_TO_OUT8(x) CONVERT_F32_TO_OUT8(x)
#endif

#if WITH_SUM_A

#define REORDER(_out, _in, _a, _b)               \
    do {                                         \
        const float _x = CONVERT_IN_TO_F32(_in); \
        const float _s = _a * _x;                \
        _out = CONVERT_F32_TO_OUT(_s);           \
    } while (0)
#define REORDER8(_out, _in, _a, _b)                 \
    do {                                            \
        const float8 _x = CONVERT_IN_TO_F32_8(_in); \
        const float8 _s = _a * _x;                  \
        _out = CONVERT_F32_TO_OUT8(_s);             \
    } while (0)

#elif WITH_SUM_AB

#define REORDER(_out, _in, _a, _b)                 \
    do {                                           \
        const float _x = CONVERT_IN_TO_F32(_in);   \
        const float _y = CONVERT_OUT_TO_F32(_out); \
        const float _s = _a * _x + _b * _y;        \
        _out = CONVERT_F32_TO_OUT(_s);             \
    } while (0)
#define REORDER8(_out, _in, _a, _b)                   \
    do {                                              \
        const float8 _x = CONVERT_IN_TO_F32_8(_in);   \
        const float8 _y = CONVERT_OUT_TO_F32_8(_out); \
        const float8 _s = _a * _x + _b * _y;          \
        _out = CONVERT_F32_TO_OUT8(_s);               \
    } while (0)

#else // WITH_SUM_AB == 0

#define REORDER(_out, _in, _a, _b)     \
    do {                               \
        _out = CONVERT_IN_TO_OUT(_in); \
    } while (0)
#define REORDER8(_out, _in, _a, _b)     \
    do {                                \
        _out = CONVERT_IN_TO_OUT8(_in); \
    } while (0)

#endif // WITH_SUM_AB

#if SUB_GROUP_SIZE != 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#endif
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
any2any_kernel(__global DT_IN *input, __global DT_OUT *output, float alpha,
        float beta) {

    input += SRC_OFFSET_PAD;
    output += DST_OFFSET_PAD;

    const int p[32] = { 0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19,
        27, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31 };

#if REF_REORDER

#if NDIMS <= 3
    const int d0 = get_global_id(0);
    const int d1 = get_global_id(1);
    const int d2 = get_global_id(2);
#if PAD_FILL_ZERO
    if (d0 >= SRC_D0 || d1 >= SRC_D1 || d2 >= SRC_D2) {
        output[OUT_OFF(d0, d1, d2, 0, 0, 0)] = 0;
        return;
    }
#endif // PAD_FILL_ZERO
    {
        const int in_off = IN_OFF(d0, d1, d2, 0, 0, 0);
        const int out_off = OUT_OFF(d0, d1, d2, 0, 0, 0);
        REORDER(output[out_off], input[in_off], alpha, beta);
    }
#elif NDIMS <= 5
    const int d0 = get_global_id(0);
    const int d1 = get_global_id(1);
#if PAD_FILL_ZERO
    if (d0 >= SRC_D0 || d1 >= SRC_D1) {
        for (int d2 = 0; d2 < DST_D2; ++d2)
            for (int d3 = 0; d3 < DST_D3; ++d3)
                for (int d4 = 0; d4 < DST_D4; ++d4) {
                    output[OUT_OFF(d0, d1, d2, d3, d4, 0)] = 0;
                }
        return;
    }
#endif // PAD_FILL_ZERO
    for (int d2 = 0; d2 < SRC_D2; ++d2)
        for (int d3 = 0; d3 < SRC_D3; ++d3)
            for (int d4 = 0; d4 < SRC_D4; ++d4) {
                const int in_off = IN_OFF(d0, d1, d2, d3, d4, 0);
                const int out_off = OUT_OFF(d0, d1, d2, d3, d4, 0);
                REORDER(output[out_off], input[in_off], alpha, beta);
            }
#if PAD_FILL_ZERO
    for (int d2 = SRC_D2; d2 < DST_D2; ++d2)
        for (int d3 = SRC_D3; d3 < DST_D3; ++d3)
            for (int d4 = SRC_D4; d4 < DST_D4; ++d4) {
                output[OUT_OFF(d0, d1, d2, d3, d4, 0)] = 0;
            }
#endif // PAD_FILL_ZERO

#elif NDIMS == 6
    const int d0 = get_global_id(0);
    const int d1 = get_global_id(1);
    const int d2 = get_global_id(2);
#if PAD_FILL_ZERO
    if (d0 >= SRC_D0 || d1 >= SRC_D1 || d2 >= SRC_D2) {
        for (int d3 = 0; d3 < DST_D3; ++d3)
            for (int d4 = 0; d4 < DST_D4; ++d4)
                for (int d5 = 0; d5 < DST_D5; ++d5) {
                    output[OUT_OFF(d0, d1, d2, d3, d4, d5)] = 0;
                }
        return;
    }
#endif // PAD_FILL_ZERO
    for (int d3 = 0; d3 < DST_D3; ++d3)
        for (int d4 = 0; d4 < DST_D4; ++d4)
            for (int d5 = 0; d5 < DST_D5; ++d5) {
                const int in_off = IN_OFF(d0, d1, d2, d3, d4, d5);
                const int out_off = OUT_OFF(d0, d1, d2, d3, d4, d5);
                REORDER(output[out_off], input[in_off], alpha, beta);
            }
#if PAD_FILL_ZERO
    for (int d3 = SRC_D3; d3 < DST_D3; ++d3)
        for (int d4 = SRC_D4; d4 < DST_D4; ++d4)
            for (int d5 = SRC_D5; d5 < DST_D5; ++d5) {
                output[OUT_OFF(d0, d1, d2, d3, d4, d5)] = 0;
            }
#endif // PAD_FILL_ZERO
#endif // NDIMS == 6

#else // REF_REORDER == 0

#if IN_16A16B || OUT_16A16B || IN_16B16A || OUT_16B16A
    const int d0 = get_global_id(0) * 16;
    const int d1 = get_group_id(1) * 16;
    const int local_id = get_local_id(1);
    const int sp = get_group_id(2);

    const int d23 = sp / (DST_D4 * DST_D5);
    const int d45 = sp % (DST_D4 * DST_D5);
    const int d2 = d23 / DST_D3;
    const int d3 = d23 % DST_D3;
    const int d4 = d45 / DST_D5;
    const int d5 = d45 % DST_D5;

    input += IN_OFF(d0, d1, d2, d3, d4, d5);
    output += OUT_OFF(d0, d1, d2, d3, d4, d5);

    DT_IN8 in0, in1;
#if IN_16A16B || IN_16B16A
    in0 = SG_READ8_IN(&input[0]);
    in1 = SG_READ8_IN(&input[8 * 16]);
#else
    for (int i = 0; i < 8; i++) {
#if OUT_16B16A
        in0[i] = input[IN_OFF(local_id, i, 0, 0, 0, 0)];
        in1[i] = input[IN_OFF(local_id, i + 8, 0, 0, 0, 0)];
#else
        in0[i] = input[IN_OFF(i, local_id, 0, 0, 0, 0)];
        in1[i] = input[IN_OFF(i + 8, local_id, 0, 0, 0, 0)];
#endif // OUT_16B16A
    }
#endif // IN_16A16B || IN_16B16A

    DT_OUT8 out0, out1;
#if (IN_16A16B || IN_16B16A) && (OUT_16A16B || OUT_16B16A)
#if WITH_SUM_AB
    for (int i = 0; i < 8; i++) {
#if IN_16B16A
        out0[i] = output[OUT_OFF(local_id, i + 0, 0, 0, 0, 0)];
        out1[i] = output[OUT_OFF(local_id, i + 8, 0, 0, 0, 0)];
#else
        out0[i] = output[OUT_OFF(i + 0, local_id, 0, 0, 0, 0)];
        out1[i] = output[OUT_OFF(i + 8, local_id, 0, 0, 0, 0)];
#endif // IN_16B16A
    }
#endif // WITH_SUM_AB
#elif OUT_16A16B || OUT_16B16A
#if WITH_SUM_AB
    out0 = SG_READ8_OUT(&output[0]);
    out1 = SG_READ8_OUT(&output[8 * 16]);
#endif // WITH_SUM_AB
#else
#if WITH_SUM_AB
    for (int i = 0; i < 8; i++) {
#if IN_16B16A
        out0[i] = output[OUT_OFF(local_id, i + 0, 0, 0, 0, 0)];
        out1[i] = output[OUT_OFF(local_id, i + 8, 0, 0, 0, 0)];
#else
        out0[i] = output[OUT_OFF(i + 0, local_id, 0, 0, 0, 0)];
        out1[i] = output[OUT_OFF(i + 8, local_id, 0, 0, 0, 0)];
#endif // IN_16B16A
    }
#endif // WITH_SUM_AB
#endif // (IN_16A16B || IN_16B16A) && (OUT_16A16B || OUT_16B16A)

    REORDER8(out0, in0, alpha, beta);
    REORDER8(out1, in1, alpha, beta);

#if (IN_16A16B || IN_16B16A) && (OUT_16A16B || OUT_16B16A)
    for (int i = 0; i < 8; i++) {
#if IN_16B16A
        output[OUT_OFF(local_id, i + 0, 0, 0, 0, 0)] = out0[i];
        output[OUT_OFF(local_id, i + 8, 0, 0, 0, 0)] = out1[i];
#else
        output[OUT_OFF(i + 0, local_id, 0, 0, 0, 0)] = out0[i];
        output[OUT_OFF(i + 8, local_id, 0, 0, 0, 0)] = out1[i];
#endif // IN_16B16A
    }
#elif OUT_16A16B || OUT_16B16A
    SG_WRITE8_OUT(&output[0], out0);
    SG_WRITE8_OUT(&output[8 * 16], out1);
#else
    for (int i = 0; i < 8; i++) {
#if IN_16B16A
        output[OUT_OFF(local_id, i + 0, 0, 0, 0, 0)] = out0[i];
        output[OUT_OFF(local_id, i + 8, 0, 0, 0, 0)] = out1[i];
#else
        output[OUT_OFF(i + 0, local_id, 0, 0, 0, 0)] = out0[i];
        output[OUT_OFF(i + 8, local_id, 0, 0, 0, 0)] = out1[i];
#endif // IN_16B16A
    }
#endif // (IN_16A16B || IN_16B16A) && (OUT_16A16B || OUT_16B16A)

#elif IN_16B || OUT_16B
    const int d0 = get_global_id(0);
    const int d1 = get_group_id(1) * 16;
    const int local_id = get_local_id(1);
    const int sp = get_group_id(2);

    const int d23 = sp / (DST_D4 * DST_D5);
    const int d45 = sp % (DST_D4 * DST_D5);
    const int d2 = d23 / DST_D3;
    const int d3 = d23 % DST_D3;
    const int d4 = d45 / DST_D5;
    const int d5 = d45 % DST_D5;

    DT_IN in;
#if IN_16B
    input += IN_OFF(d0, d1, d2, d3, d4, d5);
    in = SG_READ_IN(&input[0]);
#else
    input += IN_OFF(d0, d1 + local_id, d2, d3, d4, d5);
    in = input[0];
#endif // IN_16B

    DT_OUT out;
#if OUT_16B
    output += OUT_OFF(d0, d1, d2, d3, d4, d5);
#if WITH_SUM_AB
    out = SG_READ_OUT(&output[0]);
#endif // WITH_SUM_AB
#else
    output += OUT_OFF(d0, d1 + local_id, d2, d3, d4, d5);
#if WITH_SUM_AB
    out = output[0];
#endif // WITH_SUM_AB
#endif // OUT_16B

    REORDER(out, in, alpha, beta);

#if OUT_16B
    SG_WRITE_OUT(&output[0], out);
#else
    output[0] = out;
#endif // OUT_16B

#elif IN_16B16C || OUT_16B16C || IN_16C16B || OUT_16C16B
    const int g = get_global_id(0) / BLOCK_0;
    const int d1 = ((get_global_id(0) % BLOCK_0) / 16) * 16;
    const int d2 = get_group_id(1) * 16;
    const int local_id = get_local_id(0);
    const int sp = get_global_id(2);

    const int d34 = sp / DST_D5;
    const int d3 = d34 / DST_D4;
    const int d4 = d34 % DST_D4;
    const int d5 = sp % DST_D5;

    DT_IN8 in0, in1;
#if IN_16B16C || IN_16C16B
    input += IN_OFF_G(g, d1, d2, d3, d4, d5);
    in0 = SG_READ8_IN(&input[0]);
    in1 = SG_READ8_IN(&input[8 * 16]);
#else
    for (int i = 0; i < 8; i++) {
#if OUT_16C16B
        in0[i] = input[IN_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)];
        in1[i] = input[IN_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)];
#else
        in0[i] = input[IN_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)];
        in1[i] = input[IN_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)];
#endif // OUT_16C16B
    }
#endif // IN_16B16C || IN_16C16B

    DT_OUT8 out0, out1;

#if (IN_16B16C || IN_16C16B) && (OUT_16B16C || OUT_16C16B)
#if WITH_SUM_AB
    for (int i = 0; i < 8; i++) {
#if IN_16C16B
        out0[i] = output[OUT_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)];
        out1[i] = output[OUT_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)];
#else
        out0[i] = output[OUT_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)];
        out1[i] = output[OUT_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)];
#endif // IN_16C16B
    }
#endif // WITH_SUM_AB
#elif OUT_16B16C || OUT_16C16B
    output += OUT_OFF_G(g, d1, d2, d3, d4, d5);
#if WITH_SUM_AB
    out0 = SG_READ8_OUT(&output[0]);
    out1 = SG_READ8_OUT(&output[8 * 16]);
#endif // WITH_SUM_AB
#else
#if WITH_SUM_AB
    for (int i = 0; i < 8; i++) {
#if IN_16C16B
        out0[i] = output[OUT_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)];
        out1[i] = output[OUT_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)];
#else
        out0[i] = output[OUT_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)];
        out1[i] = output[OUT_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)];
#endif // IN_16C16B
    }
#endif // WITH_SUM_AB
#endif // (IN_16B16C || IN_16C16B) && (OUT_16B16C || OUT_16C16B)

    REORDER8(out0, in0, alpha, beta);
    REORDER8(out1, in1, alpha, beta);

#if (IN_16B16C || IN_16C16B) && (OUT_16B16C || OUT_16C16B)
    for (int i = 0; i < 8; i++) {
#if IN_16C16B
        output[OUT_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)] = out0[i];
        output[OUT_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)] = out1[i];
#else
        output[OUT_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)] = out0[i];
        output[OUT_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)] = out1[i];
#endif // IN_16C16B
    }
#elif OUT_16B16C || OUT_16C16B
    SG_WRITE8_OUT(&output[0], out0);
    SG_WRITE8_OUT(&output[8 * 16], out1);
#else
    for (int i = 0; i < 8; i++) {
#if IN_16C16B
        output[OUT_OFF_G(g, d1 + local_id, d2 + i + 0, d3, d4, d5)] = out0[i];
        output[OUT_OFF_G(g, d1 + local_id, d2 + i + 8, d3, d4, d5)] = out1[i];
#else
        output[OUT_OFF_G(g, d1 + i + 0, d2 + local_id, d3, d4, d5)] = out0[i];
        output[OUT_OFF_G(g, d1 + i + 8, d2 + local_id, d3, d4, d5)] = out1[i];
#endif // IN_16C16B
    }
#endif // (IN_16B16C || IN_16C16B) && (OUT_16B16C || OUT_16C16B)
#endif // IN_16B16C || OUT_16B16C || IN_16C16B || OUT_16C16B

#endif // REF_REORDER
}
