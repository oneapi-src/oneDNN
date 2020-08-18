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

#if DT_S8 || DT_U8
#define RINT rint
#else
#define RINT
#endif

// Read functions.
inline VECT_DATA_T read_vect_c_block(
        int idx, const __global DATA_T *ptr, int c, int stride);
inline VECT_INT_T read_vect_c_block_int(
        int idx, const __global int *ptr, int c, int stride);

// Write functions.
inline void write_vect_c_block(
        int idx, __global DATA_T *ptr, int c, int stride, VECT_DATA_T block);
inline void write_vect_c_block_int(
        int idx, __global int *ptr, int c, int stride, VECT_INT_T block);

#if IS_FWD
KERNEL_ATTR
__kernel void gen9_pooling_fwd(
        __global DATA_T *src, __global int *ws, __global DATA_T *dst) {
    const int mb = GWS_GET_MB();
    const int c = GWS_GET_C();
    const int od = GWS_GET_OD();
    const int oh = GWS_GET_OH();

    // Stride between subgroup elements within an MB/C block.
#if USE_MB_BLOCK
    const int src_stride = (SRC_SB0 > 1) ? SRC_SB0 : SRC_S0;
    const int dst_stride = (DST_SB0 > 1) ? DST_SB0 : DST_S0;
#elif USE_C_BLOCK
    const int src_stride = (SRC_B1 > 1) ? SRC_S1 : SUB_GROUP_SIZE;
    const int dst_stride = (DST_B1 > 1) ? DST_S1 : SUB_GROUP_SIZE;
#endif
    const int ws_stride = dst_stride;

    for (int ow = 0; ow < OW; ++ow) {
        const int id = od * SD - PD;
        const int ih = oh * SH - PH;
        const int iw = ow * SW - PW;

#if ALG_AVG_P || ALG_AVG_NP
        VECT_FLOAT_T A0 = DATA_ZERO;
        VECT_FLOAT_T A1 = DATA_ZERO;
#endif
        VECT_DATA_T D0 = ALG_MAX ? DATA_MIN : DATA_ZERO;
        VECT_DATA_T D1 = ALG_MAX ? DATA_MIN : DATA_ZERO;
        VECT_INT_T WS0 = 0, WS1 = 0;

        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    if (id + kd < 0 || id + kd >= ID) continue;
                    if (ih + kh < 0 || ih + kh >= IH) continue;
                    if (iw + kw < 0 || iw + kw >= IW) continue;

                    int src_off = SRC_OFF(mb, c, id + kd, ih + kh, iw + kw);

                    VECT_DATA_T S0 = read_vect_c_block(
                            0, &src[src_off], c, src_stride);
                    VECT_DATA_T S1 = read_vect_c_block(
                            1, &src[src_off], c, src_stride);

#if ALG_MAX
#if IS_TRAINING
                    VECT_INT_T CMP0 = isless(D0, S0);
                    WS0 = select(WS0, kd * KH * KW + kh * KW + kw, CMP0);
                    D0 = select(D0, S0, CMP0);

                    VECT_INT_T CMP1 = isless(D1, S1);
                    WS1 = select(WS1, kd * KH * KW + kh * KW + kw, CMP1);
                    D1 = select(D1, S1, CMP1);

#else // TRAINING
                    D0 = max(D0, S0);
                    D1 = max(D1, S1);
#endif // TRAINING
#else // ALG_MAX
                    A0 += CONVERT_VECT_FLOAT_T(S0);
                    A1 += CONVERT_VECT_FLOAT_T(S1);
#endif // ALG_MAX
                }
            }

#if ALG_AVG_P
        D0 = CONVERT_VECTOR_DATA_T(RINT(A0 / (KD * KH * KW)));
        D1 = CONVERT_VECTOR_DATA_T(RINT(A1 / (KD * KH * KW)));
#endif // ALG_AVG_P

#if ALG_AVG_NP
        const int id_start = max(od * SD - PD, 0);
        const int ih_start = max(oh * SH - PH, 0);
        const int iw_start = max(ow * SW - PW, 0);
        const int id_end = min(od * SD - PD + KD, ID);
        const int ih_end = min(oh * SH - PH + KH, IH);
        const int iw_end = min(ow * SW - PW + KW, IW);
        const DATA_T num_summands = (ih_end - ih_start) * (iw_end - iw_start)
                * (id_end - id_start);
        D0 = CONVERT_VECTOR_DATA_T(RINT(A0 / num_summands));
        D1 = CONVERT_VECTOR_DATA_T(RINT(A1 / num_summands));
#endif // ALG_AVG_NP

        int dst_off = DST_OFF(mb, c, od, oh, ow);
        write_vect_c_block(0, &dst[dst_off], c, dst_stride, D0);
        write_vect_c_block(1, &dst[dst_off], c, dst_stride, D1);

#if ALG_MAX && IS_TRAINING
        int ws_off = dst_off;
        write_vect_c_block_int(0, &ws[ws_off], c, ws_stride, WS0);
        write_vect_c_block_int(1, &ws[ws_off], c, ws_stride, WS1);
#endif // ALG_MAX && IS_TRAINING
    }
}
#endif

#if IS_BWD
KERNEL_ATTR
__kernel void gen9_pooling_bwd(__global DATA_T *diff_src, __global int *ws,
        __global DATA_T *diff_dst) {

    const int mb = GWS_GET_MB();
    const int c = GWS_GET_C();
    const int id = GWS_GET_ID();
    const int ih = GWS_GET_IH();
    const int iw = GWS_GET_IW();

    // Stride between subgroup elements within an MB/C block.
#if USE_MB_BLOCK
    const int src_stride = (SRC_SB0 > 1) ? SRC_SB0 : SRC_S0;
    const int dst_stride = (DST_SB0 > 1) ? DST_SB0 : DST_S0;
#elif USE_C_BLOCK
    const int src_stride = (SRC_B1 > 1) ? SRC_S1 : SUB_GROUP_SIZE;
    const int dst_stride = (DST_B1 > 1) ? DST_S1 : SUB_GROUP_SIZE;
#endif
    const int ws_stride = dst_stride;

    VECT_DATA_T S0 = 0, S1 = 0;
    for (int kd = 0; kd < KD; kd++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int od = (id + PD - kd);
                int oh = (ih + PH - kh);
                int ow = (iw + PW - kw);
                if (od % SD != 0 || oh % SH != 0 || ow % SW != 0) continue;
                od /= SD;
                oh /= SH;
                ow /= SW;
                if (od < 0 || od >= OD) continue;
                if (oh < 0 || oh >= OH) continue;
                if (ow < 0 || ow >= OW) continue;

                const int dst_off = DST_OFF(mb, c, od, oh, ow);
                VECT_DATA_T D0 = read_vect_c_block(
                        0, &diff_dst[dst_off], c, dst_stride);
                VECT_DATA_T D1 = read_vect_c_block(
                        1, &diff_dst[dst_off], c, dst_stride);

#if ALG_MAX
                VECT_INT_T WS0
                        = read_vect_c_block_int(0, &ws[dst_off], c, ws_stride);
                VECT_INT_T WS1
                        = read_vect_c_block_int(1, &ws[dst_off], c, ws_stride);

                VECT_INT_T CMP0 = isnotequal(
                        AS_VECT_DATA_T(WS0 - kd * KH * KW - kh * KW - kw),
                        (VECT_DATA_T)0);
                D0 = select(D0, (VECT_DATA_T)0, CMP0);

                VECT_INT_T CMP1 = isnotequal(
                        AS_VECT_DATA_T(WS1 - kd * KH * KW - kh * KW - kw),
                        (VECT_DATA_T)0);
                D1 = select(D1, (VECT_DATA_T)0, CMP1);
#endif
#if ALG_AVG_NP
                const int id_start = max(id - kd, 0);
                const int ih_start = max(ih - kh, 0);
                const int iw_start = max(iw - kw, 0);
                const int id_end = min(id - kd + KD, ID);
                const int ih_end = min(ih - kh + KH, IH);
                const int iw_end = min(iw - kw + KW, IW);
                const DATA_T num_summands = (ih_end - ih_start)
                        * (iw_end - iw_start) * (id_end - id_start);
                D0 /= num_summands;
                D1 /= num_summands;
#endif
                S0 += D0;
                S1 += D1;
            }
        }
    }
#if ALG_AVG_P
    S0 /= KD * KH * KW;
    S1 /= KD * KH * KW;
#endif

    int src_off = SRC_OFF(mb, c, id, ih, iw);
    write_vect_c_block(0, &diff_src[src_off], c, src_stride, S0);
    write_vect_c_block(1, &diff_src[src_off], c, src_stride, S1);
}
#endif

inline DATA_T read_c_block(const __global DATA_T *ptr, int c) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    int tail = C_WO_PADDING - c;
    return (local_id < tail) ? ptr[local_id] : 0;
#else
    return AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)ptr));
#endif
}

inline VECT_DATA_T read_vect_c_block(
        int idx, const __global DATA_T *ptr, int c, int stride) {
    if (idx >= NVECT) return 0;

    if ((stride == SUB_GROUP_SIZE) && (C_WO_PADDING % SUB_GROUP_SIZE == 0)) {
        return AS_VECT_DATA_T(VECT_BLOCK_READ((const __global BLOCK_DATA_T *)ptr
                + idx * VECT_DT_N * SUB_GROUP_SIZE));
    } else {
        VECT_DATA_T ret;
        for (int i = 0; i < VECT_DT_N; i++) {
            int c_off = (USE_C_BLOCK ? (idx * VECT_DT_N + i) * SUB_GROUP_SIZE
                                     : 0);
#if VECT_DT_N == 1
            ret = read_c_block(ptr + (idx * VECT_DT_N + i) * stride, c + c_off);
#else
            ret[i] = read_c_block(
                    ptr + (idx * VECT_DT_N + i) * stride, c + c_off);
#endif
        }
        return ret;
    }
}

inline int read_c_block_int(const __global int *ptr, int c) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    int tail = C_WO_PADDING - c;
    return (local_id < tail) ? ptr[local_id] : 0;
#else
    return as_int(intel_sub_group_block_read((const __global uint *)ptr));
#endif
}

inline VECT_INT_T read_vect_c_block_int(
        int idx, const __global int *ptr, int c, int stride) {
    if (idx >= NVECT) return 0;

    if ((stride == SUB_GROUP_SIZE) && (C_WO_PADDING % SUB_GROUP_SIZE == 0)) {
        return AS_VECT_INT_T(VECT_UINT_READ(
                (const __global uint *)ptr + idx * VECT_DT_N * SUB_GROUP_SIZE));
    } else {
        VECT_INT_T ret;
        for (int i = 0; i < VECT_DT_N; i++) {
            int c_off = (USE_C_BLOCK ? (idx * VECT_DT_N + i) * SUB_GROUP_SIZE
                                     : 0);
#if VECT_DT_N == 1
            ret = read_c_block_int(
                    ptr + (idx * VECT_DT_N + i) * stride, c + c_off);
#else
            ret[i] = read_c_block_int(
                    ptr + (idx * VECT_DT_N + i) * stride, c + c_off);
#endif
        }
        return ret;
    }
}

inline void write_c_block(__global DATA_T *ptr, int c, DATA_T value) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    int tail = C_WO_PADDING - c;
    if (local_id < tail) ptr[local_id] = value;
#else
    BLOCK_WRITE((__global BLOCK_DATA_T *)ptr, AS_BLOCK_DATA_T(value));
#endif
}

inline void write_vect_c_block(
        int idx, __global DATA_T *ptr, int c, int stride, VECT_DATA_T block) {
    if (idx >= NVECT) return;

    if ((stride == SUB_GROUP_SIZE) && (C_WO_PADDING % SUB_GROUP_SIZE == 0)) {
        VECT_BLOCK_WRITE(
                (__global BLOCK_DATA_T *)ptr + idx * VECT_DT_N * SUB_GROUP_SIZE,
                AS_VECT_BLOCK_DATA_T(block));
    } else {
        for (int i = 0; i < VECT_DT_N; i++) {
            int c_off = (USE_C_BLOCK ? (idx * VECT_DT_N + i) * SUB_GROUP_SIZE
                                     : 0);
#if VECT_DT_N == 1
            write_c_block(
                    ptr + (idx * VECT_DT_N + i) * stride, c + c_off, block);
#else
            write_c_block(
                    ptr + (idx * VECT_DT_N + i) * stride, c + c_off, block[i]);
#endif
        }
    }
}

inline void write_c_block_int(__global int *ptr, int c, int value) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    int tail = C_WO_PADDING - c;
    if (local_id < tail) ptr[local_id] = value;
#else
    intel_sub_group_block_write((__global uint *)ptr, as_uint(value));
#endif
}

inline void write_vect_c_block_int(
        int idx, __global int *ptr, int c, int stride, VECT_INT_T block) {
    if (idx >= NVECT) return;

    if ((stride == SUB_GROUP_SIZE) && (C_WO_PADDING % SUB_GROUP_SIZE == 0)) {
        VECT_UINT_WRITE((__global uint *)ptr + idx * VECT_DT_N * SUB_GROUP_SIZE,
                AS_VECT_UINT_T(block));
    } else {
        for (int i = 0; i < VECT_DT_N; i++) {
            int c_off = (USE_C_BLOCK ? (idx * VECT_DT_N + i) * SUB_GROUP_SIZE
                                     : 0);
#if VECT_DT_N == 1
            write_c_block_int(
                    ptr + (idx * VECT_DT_N + i) * stride, c + c_off, block);
#else
            write_c_block_int(
                    ptr + (idx * VECT_DT_N + i) * stride, c + c_off, block[i]);
#endif
        }
    }
}
