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

#include "gpu/ocl/ocl_types.h"
#if WITH_ELTWISE == 1
#include "gpu/ocl/ocl_post_ops.h"
#endif

#undef DST_OFF
#define SRC0_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC0, x0, x1, x2, x3, x4, x5)
#define SRC1_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC1, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)

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

KERNEL_ATTR
__kernel void gen9_binary(__global SRC0_DATA_T *src0,
        __global SRC1_DATA_T *src1, __global DATA_T *dst, float eltwise_alpha,
        float eltwise_beta, float eltwise_scale, float sum_scale,
        float src0_scale, float src1_scale) {

    // since gws = no. of total elems in A, id will be the logical offset
    int dims0[6] = {0};
    dims0[0] = GWS_GET_D0();
    dims0[1] = GWS_GET_D1();
    dims0[2] = GWS_GET_D2();
    dims0[3] = GWS_GET_D3();
    dims0[4] = GWS_GET_D4();
    dims0[5] = GWS_GET_D5();
    int d1_block = GWS_GET_D1_BLOCK();
    int src0_off = SRC0_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    src0 += src0_off;
    int dst_off = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    dst += dst_off;

    int src1_off = SRC1_OFF(dims0[0] * (!BCAST_DIM0), dims0[1] * (!BCAST_DIM1),
            dims0[2] * (!BCAST_DIM2), dims0[3] * (!BCAST_DIM3),
            dims0[4] * (!BCAST_DIM4), dims0[5] * (!BCAST_DIM5));
    src1 += src1_off;
#if NVECT == 1
    float d = 0;
    float tmp_src0 = CONVERT_FLOAT_T(SRC0_BLOCK_READ(&src0[0]));
#elif NVECT == 2
    float2 d = 0;
    float2 tmp_src0 = CONVERT_FLOAT2_T(SRC0_BLOCK_READ2(&src0[0]));
#elif NVECT == 4
    float4 d = 0;
    float4 tmp_src0 = CONVERT_FLOAT4_T(SRC0_BLOCK_READ4(&src0[0]));
#elif NVECT == 8
    float8 d = 0;
    float8 tmp_src0 = CONVERT_FLOAT8_T(SRC0_BLOCK_READ8(&src0[0]));
#endif
    float tmp_src1 = CONVERT_FLOAT_T(SRC1_BLOCK_READ(&src1[0]));

#if SRC0_SCALE != 1
    tmp_src0 = tmp_src0 * src0_scale;
#endif
#if SRC1_SCALE != 1
    tmp_src1 = tmp_src1 * src1_scale;
#endif

#if IS_ADD
    d = tmp_src0 + tmp_src1;
#elif IS_MUL
    d = tmp_src0 * tmp_src1;
#elif IS_MAX
    d = max(tmp_src0, tmp_src1);
#elif IS_MIN
    d = min(tmp_src0, tmp_src1);
#endif

#if WITH_SUM == 1
#if NVECT == 1
    d += sum_scale * CONVERT_FLOAT_T(DST_BLOCK_READ(&src0[0]));
#elif NVECT == 2
    d += sum_scale * CONVERT_FLOAT2_T(DST_BLOCK_READ2(&src0[0]));
#elif NVECT == 4
    d += sum_scale * CONVERT_FLOAT4_T(DST_BLOCK_READ4(&src0[0]));
#elif NVECT == 8
    d += sum_scale * CONVERT_FLOAT8_T(DST_BLOCK_READ8(&src0[0]));
#endif
#endif

#if WITH_ELTWISE == 1
    d = fwd_eltwise(d, eltwise_alpha, eltwise_beta, eltwise_scale);
#endif

#if NVECT == 1
    DST_BLOCK_WRITE(&dst[0], TO_DST(d));
#elif NVECT == 2
    DST_BLOCK_WRITE2(&dst[0], TO_DST2(d));
#elif NVECT == 4
    DST_BLOCK_WRITE4(&dst[0], TO_DST4(d));
#elif NVECT == 8
    DST_BLOCK_WRITE8(&dst[0], TO_DST8(d));
#endif
}
