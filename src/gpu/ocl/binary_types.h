/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_OCL_BINARY_TYPES_H
#define GPU_OCL_BINARY_TYPES_H
#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#undef DST_OFF
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#define SRC0_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC0, x0, x1, x2, x3, x4, x5)
#define SRC1_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC1, x0, x1, x2, x3, x4, x5)

#if SRC1_DT_BF16
#define SRC1_TO_FLOAT cvt_bf16_to_f32
#else
#define SRC1_TO_FLOAT CONVERT_FLOAT_T
#endif

#if SRC0_DT_BF16
#define SRC0_TO_FLOAT cvt_bf16_to_f32
#else
#define SRC0_TO_FLOAT CONVERT_FLOAT_T
#endif

#if SRC0_DT_S8
#define SRC0_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_char2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_char4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#endif // SRC_DT_S8

#if SRC1_DT_S8
#define SRC1_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_char2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_char4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#endif // SRC_DT_S8

#if SRC0_DT_U8
#define SRC0_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_uchar2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_uchar4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#endif // SRC0_DT_U8

#if SRC1_DT_U8
#define SRC1_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_uchar2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_uchar4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#endif // SRC1_DT_U8

#if SRC0_DT_F16
#define SRC0_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_half2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_half4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#endif // SRC0_DT_F16

#if SRC1_DT_F16
#define SRC1_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_half2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_half4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#endif // SRC1_DT_F16

#if SRC0_DT_S32
#define SRC0_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_int2(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_int4(intel_sub_group_block_read4((const __global uint *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#endif // SRC0_DT_S32

#if SRC1_DT_S32
#define SRC1_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_int2(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_int4(intel_sub_group_block_read4((const __global uint *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#endif // SRC1_DT_S32

#if SRC0_DT_F32
#define SRC0_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_float2(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_float4(intel_sub_group_block_read4((const __global uint *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#endif // SRC0_DT_F32

#if SRC1_DT_F32
#define SRC1_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_float2(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_float4(intel_sub_group_block_read4((const __global uint *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#endif // SRC1_DT_F32

#if SRC0_DT_BF16
#define SRC0_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC0_BLOCK_READ2(src) \
    as_ushort2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC0_BLOCK_READ4(src) \
    as_ushort4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define SRC0_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#endif // SRC0_DT_BF16

#if SRC1_DT_BF16
#define SRC1_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC1_BLOCK_READ2(src) \
    as_ushort2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC1_BLOCK_READ4(src) \
    as_ushort4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define SRC1_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#endif // SRC1_DT_BF16

#if DST_DT_S8
#define DST_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_READ2(src) \
    as_char2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define DST_BLOCK_READ4(src) \
    as_char4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define DST_BLOCK_READ8(src) \
    as_char8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_uc2((__global uchar *)(dst), as_uchar2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write_uc4((__global uchar *)(dst), as_uchar4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // DST_DT_S8

#if DST_DT_U8
#define DST_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_READ2(src) \
    as_uchar2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define DST_BLOCK_READ4(src) \
    as_uchar4(intel_sub_group_block_read_uc4((const __global uchar *)(src)))
#define DST_BLOCK_READ8(src) \
    as_uchar8(intel_sub_group_block_read_uc8((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_uc2((__global uchar *)(dst), as_uchar2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write_uc4((__global uchar *)(dst), as_uchar4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_uc8((__global uchar *)(dst), as_uchar8(val))
#endif // SRC_DT_U8

#if DST_DT_F16
#define DST_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_READ2(src) \
    as_half2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define DST_BLOCK_READ4(src) \
    as_half4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define DST_BLOCK_READ8(src) \
    as_half8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_us2((__global ushort *)(dst), as_ushort2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write_us4((__global ushort *)(dst), as_ushort4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // DST_DT_F16

#if DST_DT_S32
#define DST_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_READ2(src) \
    as_int2(intel_sub_group_block_read2((const __global uint *)(src)))
#define DST_BLOCK_READ4(src) \
    as_int4(intel_sub_group_block_read4((const __global uint *)(src)))
#define DST_BLOCK_READ8(src) \
    as_int8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write2((__global uint *)(dst), as_uint2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write4((__global uint *)(dst), as_uint4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_S32

#if DST_DT_F32
#define DST_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_READ2(src) \
    as_float2(intel_sub_group_block_read2((const __global uint *)(src)))
#define DST_BLOCK_READ4(src) \
    as_float4(intel_sub_group_block_read4((const __global uint *)(src)))
#define DST_BLOCK_READ8(src) \
    as_float8(intel_sub_group_block_read8((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write2((__global uint *)(dst), as_uint2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write4((__global uint *)(dst), as_uint4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write8((__global uint *)(dst), as_uint8(val))
#endif // DST_DT_F32

#if DST_DT_BF16
#define DST_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_READ2(src) \
    as_ushort2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define DST_BLOCK_READ4(src) \
    as_ushort4(intel_sub_group_block_read_us4((const __global ushort *)(src)))
#define DST_BLOCK_READ8(src) \
    as_ushort8(intel_sub_group_block_read_us8((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_us2((__global ushort *)(dst), as_ushort2(val))
#define DST_BLOCK_WRITE4(dst, val) \
    intel_sub_group_block_write_us4((__global ushort *)(dst), as_ushort4(val))
#define DST_BLOCK_WRITE8(dst, val) \
    intel_sub_group_block_write_us8((__global ushort *)(dst), as_ushort8(val))
#endif // SRC_DT_F16

#if NVECT == 1 || IS_PLAIN_LAYOUT
#define ELEM_DATA_T float
#elif NVECT == 2
#define ELEM_DATA_T float2
#elif NVECT == 4
#define ELEM_DATA_T float4
#elif NVECT == 8
#define ELEM_DATA_T float8
#endif

ELEM_DATA_T get_eltwise_op(ELEM_DATA_T src0, ELEM_DATA_T src1) {
    ELEM_DATA_T d = 0;
#if IS_ADD
    d = src0 + src1;
#elif IS_MUL
    d = src0 * src1;
#elif IS_MAX
    d = max(src0, src1);
#elif IS_MIN
    d = min(src0, src1);
#elif IS_DIV
    d = src0 / src1;
#elif IS_SUB
    d = src0 - src1;
#elif IS_GE
    d = (src0 >= src1) ? 1.0f : 0.0f;
#elif IS_GT
    d = (src0 > src1) ? 1.0f : 0.0f;
#elif IS_LE
    d = (src0 <= src1) ? 1.0f : 0.0f;
#elif IS_LT
    d = (src0 < src1) ? 1.0f : 0.0f;
#elif IS_EQ
    d = (src0 == src1) ? 1.0f : 0.0f;
#elif IS_NE
    d = (src0 != src1) ? 1.0f : 0.0f;
#endif
    return d;
}

#define READ_DATA(size, name, source_ptr, dest_ptr, scale) \
    { \
        unsigned offset = 0; \
        unroll_for(unsigned j8 = 0; j8 < size / 8; ++j8) { \
            *((float8 *)(dest_ptr + offset)) = scale \
                    * CONVERT_FLOAT8_T(CONCAT2(name, _BLOCK_READ8)( \
                            (source_ptr + offset * SUB_GROUP_SIZE))); \
            offset += 8; \
        } \
        if ((size % 8) / 4) { \
            *((float4 *)(dest_ptr + offset)) = scale \
                    * CONVERT_FLOAT4_T(CONCAT2(name, _BLOCK_READ4)( \
                            (source_ptr + offset * SUB_GROUP_SIZE))); \
            offset += 4; \
        } \
        if ((size % 4) / 2) { \
            *((float2 *)(dest_ptr + offset)) = scale \
                    * CONVERT_FLOAT2_T(CONCAT2(name, _BLOCK_READ2)( \
                            (source_ptr + offset * SUB_GROUP_SIZE))); \
            offset += 2; \
        } \
        if ((size % 2)) { \
            *((float *)(dest_ptr + offset)) = scale \
                    * CONVERT_FLOAT_T(CONCAT2(name, _BLOCK_READ)( \
                            (source_ptr + offset * SUB_GROUP_SIZE))); \
        } \
    }

#define WRITE_DATA(size, name, source_ptr, dest_ptr) \
    { \
        unsigned offset = 0; \
        unroll_for(unsigned j8 = 0; j8 < size / 8; ++j8) { \
            CONCAT2(name, _BLOCK_WRITE8) \
            ((dest_ptr + offset * SUB_GROUP_SIZE), \
                    TO_DST8(*((float8 *)(source_ptr + offset)))); \
            offset += 8; \
        } \
        if ((size % 8) / 4) { \
            CONCAT2(name, _BLOCK_WRITE4) \
            ((dest_ptr + offset * SUB_GROUP_SIZE), \
                    TO_DST4(*((float4 *)(source_ptr + offset)))); \
            offset += 4; \
        } \
        if ((size % 4) / 2) { \
            CONCAT2(name, _BLOCK_WRITE2) \
            ((dest_ptr + offset * SUB_GROUP_SIZE), \
                    TO_DST2(*((float2 *)(source_ptr + offset)))); \
            offset += 2; \
        } \
        if ((size % 2)) { \
            CONCAT2(name, _BLOCK_WRITE) \
            ((dest_ptr + offset * SUB_GROUP_SIZE), \
                    TO_DST(*((float *)(source_ptr + offset)))); \
        } \
    }

#endif
