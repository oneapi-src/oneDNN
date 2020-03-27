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
#include "gpu/ocl/ocl_types.h"
#if WITH_ELTWISE == 1
#include "gpu/ocl/ocl_post_ops.h"
#endif

#undef DST_OFF
#define SRC0_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC0, x0, x1, x2, x3, x4, x5)
#define SRC1_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC1, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)

#if SRC_DT_F16 || DST_DT_F16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if SRC0_DT_S8
#define SRC0_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC0_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#endif // SRC_DT_S8

#if SRC1_DT_S8
#define SRC1_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC1_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#endif // SRC_DT_S8

#if SRC0_DT_U8
#define SRC0_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC0_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#endif // SRC0_DT_U8

#if SRC1_DT_U8
#define SRC1_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define SRC1_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#endif // SRC1_DT_U8

#if SRC0_DT_F16
#define SRC0_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC0_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#endif // SRC0_DT_F16

#if SRC1_DT_F16
#define SRC1_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC1_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#endif // SRC1_DT_F16

#if SRC0_DT_S32
#define SRC0_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC0_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#endif // SRC0_DT_S32

#if SRC1_DT_S32
#define SRC1_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC1_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SRC1_BLOCK_READ2(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC1_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#endif // SRC1_DT_S32

#if SRC0_DT_F32
#define SRC0_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC0_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SRC0_BLOCK_READ2(src) \
    as_float2(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC0_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write2((__global uint *)(dst), as_uint2(val))
#endif // SRC0_DT_F32

#if SRC1_DT_F32
#define SRC1_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define SRC1_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define SRC1_BLOCK_READ2(src) \
    as_float(intel_sub_group_block_read2((const __global uint *)(src)))
#define SRC1_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write2((__global uint *)(dst), as_uint2(val))
#endif // SRC1_DT_F32

#if SRC0_DT_BF16
#define SRC0_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC0_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SRC0_BLOCK_READ2(src) \
    as_ushort(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC0_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_us2((__global ushort *)(dst), as_ushort2(val))
#endif // SRC0_DT_F16

#if SRC1_DT_BF16
#define SRC1_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define SRC1_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define SRC1_BLOCK_READ2(src) \
    as_ushort2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define SRC1_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_us2((__global ushort *)(dst), as_ushort2(val))
#endif // SRC1_DT_F16

#if DST_DT_S8
#define DST_BLOCK_READ(src) \
    as_char(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_READ2(src) \
    as_char2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_uc2((__global uchar *)(dst), as_uchar2(val))
#endif // DST_DT_S8

#if DST_DT_U8
#define DST_BLOCK_READ(src) \
    as_uchar(intel_sub_group_block_read_uc((const __global uchar *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_uc((__global uchar *)(dst), as_uchar(val))
#define DST_BLOCK_READ2(src) \
    as_uchar2(intel_sub_group_block_read_uc2((const __global uchar *)(src)))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_uc2((__global uchar *)(dst), as_uchar2(val))
#endif // SRC_DT_U8

#if DST_DT_F16
#define DST_BLOCK_READ(src) \
    as_half(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_READ2(src) \
    as_half2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_us2((__global ushort *)(dst), as_ushort2(val))
#endif // DST_DT_F16

#if DST_DT_S32
#define DST_BLOCK_READ(src) \
    as_int(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_READ2(src) \
    as_int2(intel_sub_group_block_read2((const __global uint *)(src)))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#endif // DST_DT_S32

#if DST_DT_F32
#define DST_BLOCK_READ(src) \
    as_float(intel_sub_group_block_read((const __global uint *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write((__global uint *)(dst), as_uint(val))
#define DST_BLOCK_READ2(src) \
    as_float2(intel_sub_group_block_read2((const __global uint *)(src)))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write2((__global uint *)(dst), as_uint2(val))
#endif // DST_DT_F32

#if DST_DT_BF16
#define DST_BLOCK_READ(src) \
    as_ushort(intel_sub_group_block_read_us((const __global ushort *)(src)))
#define DST_BLOCK_WRITE(dst, val) \
    intel_sub_group_block_write_us((__global ushort *)(dst), as_ushort(val))
#define DST_BLOCK_READ2(src) \
    as_ushort2(intel_sub_group_block_read_us2((const __global ushort *)(src)))
#define DST_BLOCK_WRITE2(dst, val) \
    intel_sub_group_block_write_us2((__global ushort *)(dst), as_ushort2(val))
#endif // SRC_DT_F16

#if IS_TENSOR_OP && IS_DENSE && IS_SAME_MD
KERNEL_ATTR
__kernel void ref_binary(__global DATA_T *src0, __global DATA_T *src1,
        __global DATA_T *dst, float eltwise_alpha, float eltwise_beta,
        float eltwise_scale, float sum_scale
#if SRC0_SCALE
        ,
        float src0_scale
#endif
#if SRC1_SCALE
        ,
        float src1_scale
#endif
) {

    int off = GWS_GET_IDX();

    POST_OP_DATA_T tmp_src0 = DATA_TO_REF(src0[off]);
    POST_OP_DATA_T tmp_src1 = DATA_TO_REF(src1[off]);
    POST_OP_DATA_T d = 0;

#if SRC0_SCALE
    tmp_src0 = tmp_src0 * src0_scale;
#endif
#if SRC1_SCALE
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
#if SUM_SCALE == 1
    d += DATA_TO_REF(dst[off]);
#else
    d += sum_scale * DATA_TO_REF(dst[off]);
#endif
#endif

#if WITH_ELTWISE == 1
    d = fwd_eltwise(d, eltwise_alpha, eltwise_beta, eltwise_scale);
#endif
    dst[off] = TO_DST(d);
}
#else

#if !SAME_SRC_DT
#if SRC0_S
#define SRC0_DATA_T char
#else
#define SRC0_DATA_T uchar
#endif

#if SRC1_S
#define SRC1_DATA_T char
#else
#define SRC1_DATA_T uchar
#endif

#else
#define SRC1_DATA_T DATA_T
#define SRC0_DATA_T DATA_T
#endif
KERNEL_ATTR
__kernel void ref_binary(__global SRC0_DATA_T *src0, __global SRC1_DATA_T *src1,
        __global DATA_T *dst, float eltwise_alpha, float eltwise_beta,
        float eltwise_scale, float sum_scale
#if SRC0_SCALE
        ,
        float src0_scale
#endif
#if SRC1_SCALE
        ,
        float src1_scale
#endif
) {

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
    int dst_off  = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    dst += dst_off;
#if TENSOR_OP
    src1 += SRC1_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
#else
    int bcast_dims[] = {BCAST_DIM0, BCAST_DIM1, BCAST_DIM2, BCAST_DIM3,
            BCAST_DIM4, BCAST_DIM5};

    src1 += SRC1_OFF(
            dims0[0] * (!bcast_dims[0]), dims0[1] * (!bcast_dims[1]), dims0[2] * (!bcast_dims[2]), dims0[3] * (!bcast_dims[3]), dims0[4] * (!bcast_dims[4]), dims0[5] * (!bcast_dims[5]));

#endif
#if  USE_BLOCK_16B
#if  USE_2D_BLOCK
        float2 d = 0;
        float2 tmp_src0 = as_float2(SRC0_BLOCK_READ2(&src0[0]));
        float tmp_src1 = DATA_TO_REF(SRC1_BLOCK_READ(&src1[0]));
#else
        POST_OP_DATA_T d = 0;
        POST_OP_DATA_T tmp_src0 = DATA_TO_REF(SRC0_BLOCK_READ(&src0[0]));
        POST_OP_DATA_T tmp_src1 = DATA_TO_REF(SRC1_BLOCK_READ(&src1[0]));
#endif
#else
        POST_OP_DATA_T d = 0;
	POST_OP_DATA_T tmp_src0 = DATA_TO_REF(src0[0]);
	POST_OP_DATA_T tmp_src1 = DATA_TO_REF(src1[0]);
#endif
#if SRC0_SCALE
        tmp_src0 = tmp_src0 * src0_scale;
#endif
#if SRC1_SCALE
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

#if USE_BLOCK_16B

#if WITH_SUM == 1
#if SUM_SCALE == 1
        d += DATA_TO_REF(DST_BLOCK_READ(&dst[0]));
#else
        d += sum_scale * DATA_TO_REF(DST_BLOCK_READ(&dst[0]));
#endif
#endif

#else

#if WITH_SUM == 1
#if SUM_SCALE == 1
        d += DATA_TO_REF(dst[0]);
#else
        d += sum_scale * DATA_TO_REF(dst[0]);
#endif
#endif

#endif

#if WITH_ELTWISE == 1
        d = fwd_eltwise(d, eltwise_alpha, eltwise_beta, eltwise_scale);
#endif
#if USE_BLOCK_16B
#if  USE_2D_BLOCK
        DST_BLOCK_WRITE2(&dst[0], TO_DST2(d));
#else
        DST_BLOCK_WRITE(&dst[0], TO_DST(d));
#endif
#else
	dst[0] = TO_DST(d);
#endif
}
#endif
