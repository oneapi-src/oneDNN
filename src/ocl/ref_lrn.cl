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

#include "ocl/ocl_types.h"

#if !defined(ACCUMULATOR_TYPE)
#    define ACCUMULATOR_TYPE float
#    define TO_ACCUMULATOR_TYPE(v) (float)(v)
#    define ACCUMULATOR_TYPE_ZERO 0.0f
#endif

#define SRC_W_STRIDE SRC_S3
#define SRC_H_STRIDE SRC_S2

#define DST_W_STRIDE DST_S3
#define DST_H_STRIDE DST_S2

#if LRN_FWD == 1
__kernel void ref_lrn_fwd(__global const DATA_T *src,
#if IS_TRAINING == 1
        __global DATA_T *ws,
#endif
        __global DATA_T *dst) {
    const uint mb = get_global_id(GWS_MB);
    const uint ic = get_global_id(GWS_IC);
    const uint ih = get_global_id(GWS_HW) / IW;
    const uint iw = get_global_id(GWS_HW) % IW;

    const uint src_index = SRC_OFF(mb, ic, 0, ih, iw);
    const uint dst_index = DST_OFF(mb, ic, 0, ih, iw);

    ACCUMULATOR_TYPE sum = 0.0f;

#if ACROSS_CHANNEL

    for (int j = 0; j < LOCAL_SIZE; j++) {
        const int z_idx = (j + ic - PADDING);
        bool zero = (z_idx < 0 || z_idx >= IC);
        DATA_T val = zero ? 0.0f : src[SRC_OFF(mb, z_idx, 0, ih, iw)];
        sum += val * val;
    }

#else

    const int w_start = ((int)iw - PADDING);
    const int h_start = ((int)ih - PADDING);
    int src_offset = SRC_OFF(mb, ic, 0, h_start, w_start);

    for (int j = 0; j < LOCAL_SIZE; ++j) {
        for (int i = 0; i < LOCAL_SIZE; ++i) {
            int src_offset_w = w_start + i;
            int src_offset_h = h_start + j;
            bool zero = false;
            zero = src_offset_w < 0 ? true : zero;
            zero = src_offset_h < 0 ? true : zero;
            zero = src_offset_w >= IW ? true : zero;
            zero = src_offset_h >= IH ? true : zero;

            DATA_T val = zero ? DATA_ZERO : src[src_offset];

            sum += val * val;
            src_offset += SRC_W_STRIDE;
        }
        src_offset += SRC_H_STRIDE - LOCAL_SIZE * SRC_W_STRIDE;
    }
#endif

    const DATA_T num_elements_div = NUM_ELEMENTS_DIV;
    const DATA_T base = TO_DATA_T(LRN_K)
            + TO_DATA_T((ACCUMULATOR_TYPE)LRN_ALPHA * sum * num_elements_div);
    const DATA_T normalization_factor = native_powr(base, TO_DATA_T(-LRN_BETA));

    const DATA_T val = src[src_index];
    const DATA_T normres = val * normalization_factor;
#if IS_TRAINING == 1
    ws[dst_index] = base;
#endif
    dst[dst_index] = normres;
}
#endif

#if LRN_BWD == 1
__kernel void ref_lrn_bwd(__global const DATA_T *src,
        __global const DATA_T *diff_dst,
        __global DATA_T *ws,
        __global DATA_T *diff_src) {
    const uint mb = get_global_id(GWS_MB);
    const uint ic = get_global_id(GWS_IC);
    const uint ih = get_global_id(GWS_HW) / IW;
    const uint iw = get_global_id(GWS_HW) % IW;

    const uint src_index = SRC_OFF(mb, ic, 0, ih, iw);
    const uint dst_index = DST_OFF(mb, ic, 0, ih, iw);
    const DATA_T num_elements_div = NUM_ELEMENTS_DIV;
    ACCUMULATOR_TYPE B = 0;

#if ACROSS_CHANNEL

    for (int j = 0; j < LOCAL_SIZE; j++) {
        const int z_idx = (j + ic - PADDING);
        bool zero = (z_idx < 0 || z_idx >= IC);
        if (!zero) {
            DATA_T val = src[SRC_OFF(mb, z_idx, 0, ih, iw)];
            DATA_T omega =  ws[SRC_OFF(mb, z_idx, 0, ih, iw)];
            DATA_T tmp = 1.0f / native_powr(omega, TO_DATA_T(LRN_BETA + 1));
            B += tmp * val * diff_dst[DST_OFF(mb, z_idx, 0, ih, iw)];
        }
    }
#else

    const int w_start = ((int)iw - PADDING);
    const int h_start = ((int)ih - PADDING);

    for (int j = 0; j < LOCAL_SIZE; ++j) {
        for (int i = 0; i < LOCAL_SIZE; ++i) {
            int src_offset_w = w_start + i;
            int src_offset_h = h_start + j;
            bool zero = false;
            zero = src_offset_w < 0 ? true : zero;
            zero = src_offset_h < 0 ? true : zero;
            zero = src_offset_w >= IW ? true : zero;
            zero = src_offset_h >= IH ? true : zero;

            if (!zero) {
                int data_off = SRC_OFF(mb, ic, 0, src_offset_h, src_offset_w);
                DATA_T val = src[data_off];
                DATA_T omega = ws[data_off];
                DATA_T tmp = 1.0f / native_powr(omega, TO_DATA_T(LRN_BETA + 1));
                B += tmp * val * diff_dst[data_off];
            }
        }
    }
#endif
    const ACCUMULATOR_TYPE A = native_powr(ws[src_index], TO_DATA_T(-LRN_BETA)) * diff_dst[dst_index];

    diff_src[src_index] = A - src[src_index] * 2 * (ACCUMULATOR_TYPE)LRN_ALPHA * (ACCUMULATOR_TYPE)LRN_BETA * num_elements_div * B;
}
#endif
