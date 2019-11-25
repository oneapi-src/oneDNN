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

#if IN_TYPE_F16 || OUT_TYPE_F16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define IN_OFF(x0, x1, x2, x3, x4, x5) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4 \
            + ((x5) % SRC_B5) * SRC_SB5 + ((x5) / SRC_B5) * SRC_S5)

#define OUT_OFF(x0, x1, x2, x3, x4, x5) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4 \
            + ((x5) % DST_B5) * DST_SB5 + ((x5) / DST_B5) * DST_S5)

#if IN_TYPE_F32
#define DT_IN float
#else
#error Unimplemented
#endif // IN_TYPE_F32

#if OUT_TYPE_S8
#define DT_OUT char
#else
#error Unimplemented
#endif // OUT_TYPE_S8

#if OUT_TYPE_S8
#define CONVERT_F32_TO_OUT convert_char_sat_rte
#define CONVERT_F32_TO_OUT8 convert_char8_sat_rte
#else
#error Unimplemented
#endif

#define CONVERT_IN_TO_OUT(x) CONVERT_F32_TO_OUT(x)
#define REORDER(_out, _in, _a, _b) \
    do { \
        _out = CONVERT_IN_TO_OUT(_in); \
    } while (0)

// About compensation
#define COMP_DT float
#define COMP_DST_OFFSET_EL (DST_D0 * DST_S0)
#define COMP_OFF(i0, i1, i2, i3) \
    ((((i0) * (DST_D1) + (i1)) * (DST_D3) + (i2)) * (DST_D4) + (i3))

#define QZ_B0(v, scale) CONVERT_F32_TO_OUT(v *scale)

#if SUB_GROUP_SIZE != 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#endif
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
wei_reorder(__global DT_IN *input, __global DT_IN *scales,
        __global DT_OUT *output, float alpha, float beta) {

    __global char *temp = (__global char *)(output + COMP_DST_OFFSET_EL);
    __global COMP_DT *comp
            = (__global COMP_DT *)(((unsigned long)temp + (sizeof(COMP_DT) - 1))
                    & -sizeof(COMP_DT));

#if REF_REORDER

    const int d0 = get_global_id(0) / DST_D1;
    const int d1 = get_global_id(0) % DST_D1;
    const int d3 = get_global_id(1) / DST_D4;
    const int d4 = get_global_id(1) % DST_D4;

    int reduction = 0;
    for (int d2 = 0; d2 < SRC_D2; ++d2) {
        const int in_off = IN_OFF(d0, d1, d2, d3, d4, 0);
        const int out_off = OUT_OFF(d0, d1, d2, d3, d4, 0);
        REORDER(output[out_off], input[in_off], alpha, beta);

        // calculate compensation
#if MASK
        float s = scales[d3 * d4];
#else
        float s = scales[0];
#endif
        reduction += convert_int(QZ_B0(input[in_off], s));
    }
    comp[COMP_OFF(d0, d1, d3, d4)] = convert_float(reduction);
#else
#error Unimplemented
#endif
}
