/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

// Temporary W/A for bf16 problems in HW and compiler
#undef cl_future_bf16_cvt

#define DT_UNDEF 1
#include "gpu/ocl/ocl_types.h"

#include "gpu/ocl/ocl_math_utils.h"

#if SRC_DT_F16 || DST_DT_F16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#undef SRC_OFF
#undef DST_OFF

#define SRC_OFF(x0, x1, x2, x3, x4, x5) \
    OFF_MD(SRC, (x0), (x1), (x2), (x3), (x4), (x5))
#define DST_OFF(x0, x1, x2, x3, x4, x5) \
    OFF_MD(DST, (x0), (x1), (x2), (x3), (x4), (x5))

#define SRC_OFF_G(gr, x0, x1, x2, x3, x4) \
    OFF_MD(SRC, gr, (x0), (x1), (x2), (x3), (x4))
#define DST_OFF_G(gr, x0, x1, x2, x3, x4) \
    OFF_MD(DST, gr, (x0), (x1), (x2), (x3), (x4))

#if SRC_DT_S8
#define SRC_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // SRC_DT_S8

#if SRC_DT_U8
#define SRC_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // SRC_DT_U8

#if SRC_DT_F16
#define SRC_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // SRC_DT_F16

#if SRC_DT_S32
#define SRC_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // SRC_DT_S32

#if SRC_DT_F32
#define SRC_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // SRC_DT_F32

#if SRC_DT_BF16
#define SRC_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define SRC_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SRC_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // SRC_DT_F16

#if DST_DT_S8
#define DST_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // DST_DT_S8

#if DST_DT_U8
#define DST_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // SRC_DT_U8

#if DST_DT_F16
#define DST_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // DST_DT_F16

#if DST_DT_S32
#define DST_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_S32

#if DST_DT_F32
#define DST_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_F32

#if DST_DT_BF16
#define DST_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // SRC_DT_F16

#if SRC_DT_BF16 && DST_DT_BF16
#define SRC_TO_DST(x) (x)
#define SRC_TO_DST8(x) (x)
#else
#define SRC_TO_DST(x) TO_DST(SRC_TO_REF(x))
#define SRC_TO_DST8(x) TO_DST8(SRC_TO_REF8(x))
#endif

#if WITH_SUM_A
#define REORDER(_dst, _src, _a, _b) \
    do { \
        const float _x = SRC_TO_REF(_src); \
        const float _s = _a * _x; \
        _dst = TO_DST(_s); \
    } while (0)
#define REORDER8(_dst, _src, _a, _b) \
    do { \
        const float8 _x = convert_float8(SRC_TO_REF8(_src)); \
        const float8 _s = _a * _x; \
        _dst = TO_DST8(_s); \
    } while (0)

#elif WITH_SUM_AB
#define REORDER(_dst, _src, _a, _b) \
    do { \
        const float _x = SRC_TO_REF(_src); \
        const float _y = DST_TO_REF(_dst); \
        const float _s = _a * _x + _b * _y; \
        _dst = TO_DST(_s); \
    } while (0)
#define REORDER8(_dst, _src, _a, _b) \
    do { \
        const float8 _x = convert_float8(SRC_TO_REF8(_src)); \
        const float8 _y = convert_float8(DST_TO_REF8(_dst)); \
        const float8 _s = _a * _x + _b * _y; \
        _dst = TO_DST8(_s); \
    } while (0)

#elif SCALE_QUANT
#define REORDER(_out, _src, _a, _b) \
    do { \
        const float _x = SRC_TO_REF(_src); \
        const float _s = _a * _x + _b; \
        _out = TO_DST(_s); \
    } while (0)
#define REORDER8(_out, _src, _a, _b) \
    do { \
        const float8 _x = convert_float8(SRC_TO_REF8(_src)); \
        const float8 _s = _a * _x + _b; \
        _out = TO_DST8(_s); \
    } while (0)

#else // WITH_SUM_AB == 0
#define REORDER(_dst, _src, _a, _b) \
    do { \
        _dst = SRC_TO_DST(_src); \
    } while (0)
#define REORDER8(_dst, _src, _a, _b) \
    do { \
        _dst = SRC_TO_DST8(_src); \
    } while (0)

#endif // WITH_SUM_AB

#if SCALE_QUANT

#define MASK_D(_d) ((SCALE_MASK >> _d) & 1)

#define SCALE_D0 (MASK_D(0) ? SRC_D0 : 1)
#define SCALE_D1 (MASK_D(1) ? SRC_D1 : 1)
#define SCALE_D2 (MASK_D(2) ? SRC_D2 : 1)
#define SCALE_D3 (MASK_D(3) ? SRC_D3 : 1)
#define SCALE_D4 (MASK_D(4) ? SRC_D4 : 1)
#define SCALE_D5 (MASK_D(5) ? SRC_D5 : 1)

#define SCALE_S0 (SCALE_D1 * SCALE_D2 * SCALE_D3 * SCALE_D4 * SCALE_D5)
#define SCALE_S1 (SCALE_D2 * SCALE_D3 * SCALE_D4 * SCALE_D5)
#define SCALE_S2 (SCALE_D3 * SCALE_D4 * SCALE_D5)
#define SCALE_S3 (SCALE_D4 * SCALE_D5)
#define SCALE_S4 (SCALE_D5)
#define SCALE_S5 (1)

#define SCALE_OFF(x0, x1, x2, x3, x4, x5) \
    ((x0)*SCALE_S0 * MASK_D(0) + (x1)*SCALE_S1 * MASK_D(1) \
            + (x2)*SCALE_S2 * MASK_D(2) + (x3)*SCALE_S3 * MASK_D(3) \
            + (x4)*SCALE_S4 * MASK_D(4) + (x5)*SCALE_S5 * MASK_D(5))

#endif // SCALE_QUANT
