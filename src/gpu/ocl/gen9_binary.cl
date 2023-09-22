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

#include "gpu/ocl/binary_types.h"

#if IS_PLAIN_LAYOUT
KERNEL_ATTR
__kernel void gen9_binary(__global SRC0_DATA_T *src0,
        __global SRC1_DATA_T *src1, __global DST_DATA_T *dst POST_OP_ARGS,
        __global float *src0_scale, __global float *src1_scale) {

    int dims0[6] = {0};
    int local_id = get_sub_group_local_id();

    unsigned mid_dim = GWS_GET_MIXED_DIM();
    dims0[5] = mid_dim % DST_D5;
    mid_dim /= DST_D5;
    dims0[4] = mid_dim % DST_D4;
    mid_dim /= DST_D4;
    dims0[3] = mid_dim % DST_D3;
    mid_dim /= DST_D3;
    dims0[2] = mid_dim % DST_D2;
    mid_dim /= DST_D2;
    dims0[1] = mid_dim % DST_D1;
    mid_dim /= DST_D1;
    dims0[0] = mid_dim;

#if HAS_TAIL
    if (dims0[0] > (DST_D0 - 1)) { return; }
#endif

    int src0_off = SRC0_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    src0 += src0_off;

    int src1_off = SRC1_OFF(dims0[0] * (!BCAST_DIM0), dims0[1] * (!BCAST_DIM1),
            dims0[2] * (!BCAST_DIM2), dims0[3] * (!BCAST_DIM3),
            dims0[4] * (!BCAST_DIM4), dims0[5] * (!BCAST_DIM5));
    src1 += src1_off;

    int dst_off = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    dst += dst_off;

#if WITH_SRC0_SCALE
#define src0_scale_val src0_scale[0]
#else
#define src0_scale_val 1
#endif
#if WITH_SRC1_SCALE
#define src1_scale_val src1_scale[0]
#else
#define src1_scale_val 1
#endif
    float tmp_src0[NVECT];
    READ_DATA(NVECT, SRC0, (&src0[0]), (&tmp_src0[0]), src0_scale_val);

#if BCAST_AT_INNERMOST_DIM
    float tmp_src1[1];
    tmp_src1[0] = src1_scale_val * CONVERT_FLOAT_T(src1[0]);
#define SRC1_IDX_MASK 0
#else
    float tmp_src1[NVECT];
    READ_DATA(NVECT, SRC1, (&src1[0]), (&tmp_src1[0]), src1_scale_val);
#define SRC1_IDX_MASK 1
#endif

    float tmp[NVECT];
    unroll_for(unsigned idx = 0; idx < NVECT; ++idx) {
        tmp[idx] = get_eltwise_op(tmp_src0[idx], tmp_src1[idx * SRC1_IDX_MASK]);
    }

    float dst_data[NVECT];
#if WITH_SUM
    READ_DATA(NVECT, DST, (&dst[0]), (&dst_data[0]), 1);
#endif

#if HAS_TAIL
    unsigned gws_dim = get_global_id(0);
    int po_dims0[6] = {0};
    po_dims0[5] = gws_dim % DST_D5;
    gws_dim /= DST_D5;
    po_dims0[4] = gws_dim % DST_D4;
    gws_dim /= DST_D4;
    po_dims0[3] = gws_dim % DST_D3;
    gws_dim /= DST_D3;
    po_dims0[2] = gws_dim % DST_D2;
    gws_dim /= DST_D2;
    po_dims0[1] = gws_dim % DST_D1;
    gws_dim /= DST_D1;
    if (gws_dim >= DST_D0) { return; }
    po_dims0[0] = gws_dim;
#else
    int po_dims0[6]
            = {dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]};
    po_dims0[NDIMS - 1] += local_id;
#endif
    unroll_for(unsigned idx = 0; idx < NVECT; ++idx) {
        float d_i = tmp[idx];
        float dst_i = dst_data[idx];
        APPLY_POST_OPS_SERIAL(d_i, float, dst_i, float, po_dims0[0], 1,
                po_dims0[1], 1, po_dims0[2], 1, po_dims0[3], 1, po_dims0[4], 1,
                po_dims0[5], 1);
        tmp[idx] = d_i;
#if !HAS_TAIL
        po_dims0[NDIMS - 1] += SUB_GROUP_SIZE;
#endif
    }

    WRITE_DATA(NVECT, DST, (&tmp[0]), (&dst[0]));
}

#elif PLAIN_TO_ABCD4AXB
KERNEL_ATTR
__kernel void gen9_binary(__global SRC0_DATA_T *src0,
        __global SRC1_DATA_T *src1, __global DST_DATA_T *dst POST_OP_ARGS,
        __global float *src0_scale, __global float *src1_scale) {

    src0 += SRC0_OFFSET0;
    src1 += SRC1_OFFSET0;
    dst += DST_OFFSET0;

    int sglid = get_sub_group_local_id();

    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();
    const int d4 = GWS_GET_D3();
    const int d5 = GWS_GET_D3();

    const int d0_block = GWS_GET_D0_BLOCK();
    const int d1_block = GWS_GET_D1_BLOCK();
    const int d01_block = d0_block * d1_block;

    SRC0_DATA_T tmp_buf0[d01_block] = {0};
    SRC1_DATA_T tmp_buf1[d01_block] = {0};
    DST_DATA_T res_buf[d01_block] = {0};

    const int d0_inner_block = min(d0_block, SRC0_D0);
    const int d1_inner_block = min(d1_block, SRC0_D1);
    for (int d0_inner = 0; d0_inner < d0_inner_block; d0_inner++) {
        for (int d1_inner = 0; d1_inner < d1_inner_block; d1_inner++) {
            if (SRC0_D0 % d0_inner_block != 0 && d0 + d0_inner >= SRC0_D0)
                continue;
            if (SRC0_D1 % d1_inner_block != 0 && d1 + d1_inner >= SRC0_D1)
                continue;
            int src0_off;
            int src1_off;
            if (SRC0_S3_0 == 1) {
                // abcd layout.
                src0_off = SRC0_OFF(d0 + d0_inner, d1 + d1_inner, d2, d3, 0, 0);
                tmp_buf0[d0_inner * d1_block + d1_inner]
                        = SRC0_BLOCK_READ(&src0[src0_off]);
                src1_off = SRC1_OFF((d0 + d0_inner) * (!BCAST_DIM0),
                        (d1 + d1_inner) * (!BCAST_DIM1), d2 * (!BCAST_DIM2),
                        d3 * (!BCAST_DIM3), 0, 0);
            } else {
                // acdb layout.
                src0_off = SRC0_OFF(
                        d0 + d0_inner, d1 + d1_inner, d2, d3 + sglid, 0, 0);
                tmp_buf0[d0_inner * d1_block + d1_inner] = src0[src0_off];
                src1_off = SRC1_OFF((d0 + d0_inner) * (!BCAST_DIM0),
                        (d1 + d1_inner) * (!BCAST_DIM1), d2 * (!BCAST_DIM2),
                        (d3 + sglid) * (!BCAST_DIM3), 0, 0);
            }
#if BCAST_AT_INNERMOST_DIM == 1
            tmp_buf1[d0_inner * d1_block + d1_inner] = src1[src1_off];
#else
            tmp_buf1[d0_inner * d1_block + d1_inner]
                    = SRC1_BLOCK_READ(&src1[src1_off]);
#endif //BCAST_AT_INNERMOST_DIM
        }
    }

    int i = 0;
    for (int d0_i = 0; d0_i < d0_block; d0_i++) {
        for (int d1_i = 0; d1_i < d1_block; d1_i++) {

            float tmp_src0 = CONVERT_FLOAT_T(tmp_buf0[i]);
            float tmp_src1 = CONVERT_FLOAT_T(tmp_buf1[i]);
            float res;
            float dst_data;

#if WITH_SRC0_SCALE
            tmp_src0 = tmp_src0 * src0_scale[0];
#endif
#if WITH_SRC1_SCALE
            tmp_src1 = tmp_src1 * src1_scale[0];
#endif
            res = get_eltwise_op(tmp_src0, tmp_src1);

            APPLY_POST_OPS_SERIAL(res, float, dst_data, float, d0 + d0_i, 1,
                    d1 + d1_i, 1, d2, 1, d3 + sglid, 1, d4, 1, d5, 1);

            res_buf[i] = TO_DST(res);
            ++i;
        }
    }

    DST_DATA_T res_all[d01_block][SUB_GROUP_SIZE];
    for (int i = 0; i < d01_block; i++)
        for (int j = 0; j < SUB_GROUP_SIZE; j++)
            res_all[i][j] = intel_sub_group_shuffle(res_buf[i], j);
    for (int d = 0; d < SUB_GROUP_SIZE; d += 8) {
        DST_DATA8_T res_tmp;
        for (int i = 0; i < 8; i++)
            res_tmp[i] = res_all[sglid][d + i];
        int dst_off = DST_OFF(d0, d1, d2, d3 + d, 0, 0);

        DST_BLOCK_WRITE8(&dst[dst_off], res_tmp);
    }
}
#elif IS_XA16B
KERNEL_ATTR
__kernel void gen9_binary(__global SRC0_DATA_T *src0,
        __global SRC1_DATA_T *src1, __global DST_DATA_T *dst POST_OP_ARGS,
        __global float *src0_scale, __global float *src1_scale) {
    // since gws = no. of total elems in A, id will be the logical offset
    int dims0[6] = {0};
    dims0[0] = GWS_GET_D0();
    dims0[1] = GWS_GET_D1();
    dims0[2] = GWS_GET_D2();
    dims0[3] = GWS_GET_D3();
    dims0[4] = GWS_GET_D4();
    dims0[5] = GWS_GET_D5();

    int src0_off = SRC0_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    int dst_off = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    int src1_off = SRC1_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);

    int sub_grp_id = get_sub_group_local_id();

    for (int channels = 0; channels < SRC0_PD1; channels += GWS_LWS0_DEFAULT) {
        float8 d = 0;
        float8 dst_data;

        __global SRC1_DATA_T *t_src1 = src1 + src1_off;
        __global DST_DATA_T *t_dst = dst + dst_off;
        __global SRC0_DATA_T *t_src0 = src0 + src0_off;

        if ((SRC0_D1 % SUB_GROUP_SIZE != 0)
                && (dims0[1] + sub_grp_id) >= SRC0_D1) {
            d = 0;
        } else {
            float8 tmp_src0 = CONVERT_FLOAT8_T(SRC0_BLOCK_READ8(&t_src0[0]));
            float8 tmp_src1 = CONVERT_FLOAT8_T(SRC1_BLOCK_READ8(&t_src1[0]));
#if WITH_SRC0_SCALE
            tmp_src0 = tmp_src0 * src0_scale[0];
#endif
#if WITH_SRC1_SCALE
            tmp_src1 = tmp_src1 * src1_scale[0];
#endif
            d = get_eltwise_op(tmp_src0, tmp_src1);
#if WITH_SUM
            dst_data = CONVERT_FLOAT8_T(DST_BLOCK_READ8(&t_dst[0]));
#endif

            const int po_mb = dims0[0];
            const int po_oc = dims0[1] + sub_grp_id;
            for (int lcl_mb = 0; lcl_mb < NVECT; ++lcl_mb) {
                if (po_mb + lcl_mb >= SRC0_D0) {
                    d[lcl_mb] = 0;
                } else {
                    float d_i = d[lcl_mb];
                    float dst_i = dst_data[lcl_mb];
                    APPLY_POST_OPS_SERIAL(d_i, float, dst_i, float,
                            po_mb + lcl_mb, 1, po_oc, 1, dims0[2], 1, dims0[3],
                            1, dims0[4], 1, dims0[5], 1);
                    d[lcl_mb] = d_i;
                }
            }
        }

        DST_BLOCK_WRITE8(&t_dst[0], TO_DST8(d));

        src0_off += MB_BLOCK * SUB_GROUP_SIZE * SRC0_PD2 * SRC0_PD3 * SRC0_PD4
                * SRC0_PD5;
        src1_off += MB_BLOCK * SUB_GROUP_SIZE * SRC1_PD2 * SRC1_PD3 * SRC1_PD4
                * SRC1_PD5;
        dst_off += MB_BLOCK * SUB_GROUP_SIZE * DST_PD2 * DST_PD3 * DST_PD4
                * DST_PD5;
    }
}
#else
KERNEL_ATTR
__kernel void gen9_binary(__global SRC0_DATA_T *src0,
        __global SRC1_DATA_T *src1, __global DST_DATA_T *dst POST_OP_ARGS,
        __global float *src0_scale, __global float *src1_scale) {

    // since gws = no. of total elems in A, id will be the logical offset
    int dims0[6] = {0};
    dims0[0] = GWS_GET_D0();
    dims0[1] = GWS_GET_D1();
    dims0[2] = GWS_GET_D2();
    dims0[3] = GWS_GET_D3();
    dims0[4] = GWS_GET_D4();
    dims0[5] = GWS_GET_D5();

#if IS_SRC1_BROADCAST
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
    float dst_data;
    float tmp_src0 = CONVERT_FLOAT_T(SRC0_BLOCK_READ(&src0[0]));
#elif NVECT == 2
    float2 d = 0;
    float2 dst_data;
    float2 tmp_src0 = CONVERT_FLOAT2_T(SRC0_BLOCK_READ2(&src0[0]));
#elif NVECT == 4
    float4 d = 0;
    float4 dst_data;
    float4 tmp_src0 = CONVERT_FLOAT4_T(SRC0_BLOCK_READ4(&src0[0]));
#elif NVECT == 8
    float8 d = 0;
    float8 dst_data;
    float8 tmp_src0 = CONVERT_FLOAT8_T(SRC0_BLOCK_READ8(&src0[0]));
#endif

#if BCAST_DIM1
    float tmp_src1 = CONVERT_FLOAT_T(src1[0]);
#else
#if BCAST_AT_INNERMOST_DIM == 1 || NVECT == 1
    float tmp_src1 = CONVERT_FLOAT_T(SRC1_BLOCK_READ(&src1[0]));
#elif NVECT == 2
    float2 tmp_src1 = CONVERT_FLOAT2_T(SRC1_BLOCK_READ2(&src1[0]));
#elif NVECT == 4
    float4 tmp_src1 = CONVERT_FLOAT4_T(SRC1_BLOCK_READ4(&src1[0]));
#elif NVECT == 8
    float8 tmp_src1 = CONVERT_FLOAT8_T(SRC1_BLOCK_READ8(&src1[0]));
#endif
#endif

#if WITH_SRC0_SCALE
    tmp_src0 = tmp_src0 * src0_scale[0];
#endif
#if WITH_SRC1_SCALE
    tmp_src1 = tmp_src1 * src1_scale[0];
#endif

    d = get_eltwise_op(tmp_src0, tmp_src1);

#if WITH_SUM
#if NVECT == 1
    dst_data = CONVERT_FLOAT_T(DST_BLOCK_READ(&dst[0]));
#elif NVECT == 2
    dst_data = CONVERT_FLOAT2_T(DST_BLOCK_READ2(&dst[0]));
#elif NVECT == 4
    dst_data = CONVERT_FLOAT4_T(DST_BLOCK_READ4(&dst[0]));
#elif NVECT == 8
    dst_data = CONVERT_FLOAT8_T(DST_BLOCK_READ8(&dst[0]));
#endif
#endif

    const int po_mb = dims0[0];
    const int po_oc = dims0[1] + get_sub_group_local_id();
#if NVECT == 1
    APPLY_POST_OPS_SERIAL(d, float, dst_data, float, po_mb, 1, po_oc, 1,
            dims0[2], 1, dims0[3], 1, dims0[4], 1, dims0[5], 1);
#else
    for (int vidx = 0; vidx < NVECT; ++vidx) {
        float d_i = d[vidx];
        float dst_i = dst_data[vidx];
        APPLY_POST_OPS_SERIAL(d_i, float, dst_i, float, po_mb, 1, po_oc, 1,
                dims0[2], 1, dims0[3], 1, dims0[4], 1, dims0[5], 1);
        d[vidx] = d_i;
        ++dims0[NDIMS - 1];
    }
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

#else // mixed_layout with no broadcast in src1
    int local_channel = get_sub_group_local_id();
    int channel_block = get_sub_group_id();
    const int sub_group_size = 16;
    const int CHANNELS = SRC1_PD1;
    int src0_off, src1_off;

#if IS_SRC0_BLOCKED
    src0_off = SRC0_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    src0 += src0_off;

    // Convert Plain to Blocked Reads
    src1_off = dims0[0] * CHANNELS * SRC1_PD2 * SRC1_PD3 * SRC1_PD4
            + (channel_block * sub_group_size + local_channel) * SRC1_PD2
                    * SRC1_PD3 * SRC1_PD4
            + dims0[2] * SRC1_PD3 * SRC1_PD4 + dims0[3] * SRC1_PD4 + dims0[4];
    src1 += src1_off;
#else // IS_SRC1_BLOCKED
    src1_off = SRC1_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    src1 += src1_off;

    // Convert Plain to Blocked Reads
    src0_off = dims0[0] * CHANNELS * SRC0_PD2 * SRC0_PD3 * SRC0_PD4
            + (channel_block * sub_group_size + local_channel) * SRC0_PD2
                    * SRC0_PD3 * SRC0_PD4
            + dims0[2] * SRC0_PD3 * SRC0_PD4 + dims0[3] * SRC0_PD4 + dims0[4];
    src0 += src0_off;
#endif

    int dst_off = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    dst += dst_off;

    for (int idx = 0; idx < sub_group_size; idx++) {
        float d = 0;
        float dst_data;

#if IS_SRC0_BLOCKED
        float tmp_src0
                = CONVERT_FLOAT_T(src0[local_channel + sub_group_size * idx]);
        float tmp_src1 = CONVERT_FLOAT_T(src1[idx]);
#else // IS_SRC1_BLOCKED
        float tmp_src0 = CONVERT_FLOAT_T(src0[idx]);
        float tmp_src1
                = CONVERT_FLOAT_T(src1[local_channel + sub_group_size * idx]);
#endif
#if WITH_SRC0_SCALE
        tmp_src0 = tmp_src0 * src0_scale[0];
#endif
#if WITH_SRC1_SCALE
        tmp_src1 = tmp_src1 * src1_scale[0];
#endif

        d = get_eltwise_op(tmp_src0, tmp_src1);

#if WITH_SUM
        dst_data = CONVERT_FLOAT_T(
                DST_BLOCK_READ(&dst[local_channel + sub_group_size * idx]));
#endif
        const int po_mb = dims0[0];
        const int po_oc = dims0[1] + get_sub_group_local_id();
        APPLY_POST_OPS_SERIAL(d, float, dst_data, float, po_mb, 1, po_oc, 1,
                dims0[2], 1, dims0[3], 1, dims0[4], 1, dims0[5], 1);
        ++dims0[NDIMS - 1];

        DST_BLOCK_WRITE(&dst[local_channel + sub_group_size * idx], TO_DST(d));
    }

#endif // IS_SRC1_BRAODCAST
}
#endif // IS_PLAIN_LAYOUT
