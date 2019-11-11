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

#define SRC_W_STRIDE SRC_S3
#define SRC_H_STRIDE SRC_S2

#define DST_W_STRIDE DST_S3
#define DST_H_STRIDE DST_S2

#if IS_FWD == 1
KERNEL_ATTR
__kernel void ref_lrn_fwd(__global const DATA_T *src,
#if IS_TRAINING == 1
        __global DEF_ACC_DATA_T *ws,
#endif
        __global DATA_T *dst) {
    const uint mb = GWS_GET_MB();
    const uint ic = GWS_GET_IC();
    const uint ih = GWS_GET_IH();
    const uint iw = GWS_GET_IW();

    const uint src_index = SRC_OFF(mb, ic, 0, ih, iw);
    const uint dst_index = DST_OFF(mb, ic, 0, ih, iw);

    DEF_ACC_DATA_T sum = 0.0f;

#if ACROSS_CHANNEL

    for (int j = 0; j < LOCAL_SIZE; j++) {
        const int z_idx = (j + ic - PADDING);
        bool zero = (z_idx < 0 || z_idx >= IC);
        DEF_ACC_DATA_T val = zero
                ? 0.0f
                : TO_DEF_ACC_DATA_T(src[SRC_OFF(mb, z_idx, 0, ih, iw)]);
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

            DEF_ACC_DATA_T val
                    = zero ? DATA_ZERO : TO_DEF_ACC_DATA_T(src[src_offset]);

            sum += val * val;
            src_offset += SRC_W_STRIDE;
        }
        src_offset += SRC_H_STRIDE - LOCAL_SIZE * SRC_W_STRIDE;
    }
#endif

    const DEF_ACC_DATA_T num_elements_div = NUM_ELEMENTS_DIV;
    const DEF_ACC_DATA_T base = (DEF_ACC_DATA_T)LRN_K
            + (DEF_ACC_DATA_T)LRN_ALPHA * sum * num_elements_div;
    const DEF_ACC_DATA_T normalization_factor
            = native_powr(base, (DEF_ACC_DATA_T)(-LRN_BETA));

    const DEF_ACC_DATA_T val = TO_DEF_ACC_DATA_T(src[src_index]);
    const DEF_ACC_DATA_T normres = val * normalization_factor;
#if IS_TRAINING == 1
    ws[dst_index] = base;
#endif
    dst[dst_index] = TO_DATA_T(normres);
}
#endif

#if IS_BWD == 1
KERNEL_ATTR
__kernel void ref_lrn_bwd(__global const DATA_T *src,
        __global const DATA_T *diff_dst, __global DEF_ACC_DATA_T *ws,
        __global DATA_T *diff_src) {
    const uint mb = GWS_GET_MB();
    const uint ic = GWS_GET_IC();
    const uint ih = GWS_GET_IH();
    const uint iw = GWS_GET_IW();

    const uint src_index = SRC_OFF(mb, ic, 0, ih, iw);
    const uint dst_index = DST_OFF(mb, ic, 0, ih, iw);
    const DEF_ACC_DATA_T num_elements_div = NUM_ELEMENTS_DIV;
    DEF_ACC_DATA_T B = 0;

#if ACROSS_CHANNEL

    for (int j = 0; j < LOCAL_SIZE; j++) {
        const int z_idx = (j + ic - PADDING);
        bool zero = (z_idx < 0 || z_idx >= IC);
        if (!zero) {
            DEF_ACC_DATA_T val
                    = TO_DEF_ACC_DATA_T(src[SRC_OFF(mb, z_idx, 0, ih, iw)]);
            DEF_ACC_DATA_T omega = ws[SRC_OFF(mb, z_idx, 0, ih, iw)];
            DEF_ACC_DATA_T tmp = (DEF_ACC_DATA_T)1.0f
                    / native_powr(omega, (DEF_ACC_DATA_T)LRN_BETA + 1);
            B += tmp * val
                    * TO_DEF_ACC_DATA_T(
                            diff_dst[DST_OFF(mb, z_idx, 0, ih, iw)]);
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
                DEF_ACC_DATA_T val = TO_DEF_ACC_DATA_T(src[data_off]);
                DEF_ACC_DATA_T omega = ws[data_off];
                DEF_ACC_DATA_T tmp = (DEF_ACC_DATA_T)1.0f
                        / native_powr(omega, (DEF_ACC_DATA_T)(LRN_BETA + 1));
                B += tmp * val * TO_DEF_ACC_DATA_T(diff_dst[data_off]);
            }
        }
    }
#endif
    const DEF_ACC_DATA_T A
            = native_powr(ws[src_index], (DEF_ACC_DATA_T)-LRN_BETA)
            * TO_DEF_ACC_DATA_T(diff_dst[dst_index]);

    diff_src[src_index] = TO_DATA_T(A
            - TO_DEF_ACC_DATA_T(src[src_index]) * 2 * (DEF_ACC_DATA_T)LRN_ALPHA
                    * (DEF_ACC_DATA_T)LRN_BETA * num_elements_div * B);
}
#endif
