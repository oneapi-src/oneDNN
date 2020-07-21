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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#define BLOCK_SIZE OC_BLOCK
#define BLOCKED_DATA_T CONCAT2(DATA_T, BLOCK_SIZE)

#define OC_OUTER_BLOCK OC_BLOCK
#define IC_OUTER_BLOCK IC_BLOCK

#define WINO_D (WINO_M + WINO_R - 1)

static inline int off_nCdhw16c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += n * (C / 16) * D * H * W * 16;
    off += (c / 16) * D * H * W * 16;
    off += d * H * W * 16;
    off += h * W * 16;
    off += w * 16;
    off += c % 16;
    return off;
}

static inline int off_NCdhw16n16c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += (n / 16) * (C / 16) * D * H * W * 16 * 16;
    off += (c / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (n % 16) * 16;
    off += (c % 16);
    return off;
}

static inline int off_gOIdhw16i16o(int g, int o, int i, int d, int h, int w,
        int O, int I, int D, int H, int W) {
    int off = 0;
    off += g * (O / 16) * (I / 16) * D * H * W * 16 * 16;
    off += (o / 16) * (I / 16) * D * H * W * 16 * 16;
    off += (i / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (i % 16) * 16;
    off += (o % 16);
    return off;
}

static inline int src_off(int n, int c, int d, int h, int w) {
    if (SRC_W16C) return off_nCdhw16c(n, c, d, h, w, G * IC, 1, IH, IW);
    if (SRC_16N16C) return off_NCdhw16n16c(n, c, d, h, w, G * IC, 1, IH, IW);
    return 0;
}

static inline int wei_off(int g, int o, int i, int d, int h, int w) {
    return off_gOIdhw16i16o(g, o, i, d, h, w, OC, IC, 1, KH, KW);
}

static inline int dst_off(int n, int c, int d, int h, int w) {
    if (DST_W16C) return off_nCdhw16c(n, c, d, h, w, G * OC, 1, OH, OW);
    if (DST_16N16C) return off_NCdhw16n16c(n, c, d, h, w, G * OC, 1, OH, OW);
    return 0;
}

#define gemm_Xx1(m, n, p, block_size, result, A, B) \
    do { \
        for (long i = 0; i < m; i++) { \
            for (long j = 0; j < p; j++) { \
                for (long c = 0; c < block_size; c++) { \
                    result[i][j][c] = 0; \
                } \
                for (long k = 0; k < n; k++) { \
                    for (long c = 0; c < block_size; c++) { \
                        result[i][j][c] += A[i][k][c] * B[k][j]; \
                    } \
                } \
            } \
        } \
    } while (0)

#define gemm_1xX(m, n, p, block_size, result, A, B) \
    do { \
        for (long i = 0; i < m; i++) { \
            for (long j = 0; j < p; j++) { \
                for (long c = 0; c < block_size; c++) { \
                    result[i][j][c] = 0; \
                } \
                for (long k = 0; k < n; k++) { \
                    for (long c = 0; c < block_size; c++) { \
                        result[i][j][c] += A[i][k] * B[k][j][c]; \
                    } \
                } \
            } \
        } \
    } while (0)

/* wei_transform U_(ic, oc) = G * g_(ic, oc) * G_t */
static inline void transform_wei_to_U(__global DATA_T *U_param,
        const __global DATA_T *wei, int i_oc, int i_ic) {
    // U needs to be 4x4xIC_BLOCKxOC_BLOCK
    const DATA_T G_2x2[WINO_D][WINO_R]
            = {{1, 0, 0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0, 0, 1}};
    const DATA_T GT_2x2[WINO_R][WINO_D]
            = {{1, 0.5, 0.5, 0}, {0, 0.5, -0.5, 0}, {0, 0.5, 0.5, 1}};
    __global BLOCKED_DATA_T(*U)[WINO_D][WINO_D][IC_BLOCK]
            = (__global BLOCKED_DATA_T(*)[WINO_D][WINO_D][IC_BLOCK])U_param;
    BLOCKED_DATA_T g[WINO_R][WINO_R][IC_BLOCK];
    BLOCKED_DATA_T U_tmp[WINO_D][WINO_D][IC_BLOCK];

    for (long kh = 0; kh < WINO_R; kh++) {
        for (long kw = 0; kw < WINO_R; kw++) {
            long g_offset = (kh * WINO_R + kw) * IC_BLOCK * OC_BLOCK;
            long wei_offset = wei_off(0, i_oc, i_ic, 0, kh, kw);
            for (long ic = 0; ic < IC_BLOCK; ic++) {
                for (long oc = 0; oc < IC_BLOCK; oc++) {
                    g[kh][kw][ic][oc] = wei[wei_offset + ic * OC_BLOCK + oc];
                }
            }
        }
    }
    long U_offset = (i_ic / IC_BLOCK) * (OC / OC_BLOCK) + (i_oc / OC_BLOCK);
    gemm_1xX(WINO_D, WINO_R, WINO_R, IC_BLOCK, U_tmp, G_2x2, g);
    gemm_Xx1(WINO_D, WINO_R, WINO_D, IC_BLOCK, U[U_offset], U_tmp, GT_2x2);
}

/* static inline void src_transform V_(n, ic) = B^T * d_(n, ic) * B */
static inline void transform_src_to_V(DATA_T (*V)[WINO_D][WINO_D][IC_BLOCK],
        const __global DATA_T *src, int i_mb, int i_icb, int ih, int iw) {
    const DATA_T BT_2x2[WINO_D][WINO_D]
            = {{1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
    const DATA_T B_2x2[WINO_D][WINO_D]
            = {{1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};

    DATA_T d[WINO_D][WINO_D][IC_BLOCK];
    DATA_T V_TMP[WINO_D][WINO_D][IC_BLOCK]; // B^Td
    for (long i = 0; i < WINO_D; i++) {
        long i_ih = ih + i;
        for (long j = 0; j < WINO_D; j++) {
            long i_iw = iw + j;
            if (i_iw < 0 || i_iw >= IW || i_ih < 0 || i_ih >= IH) {
                for (long ic = 0; ic < IC_BLOCK; ic++) {
                    d[i][j][ic] = 0;
                }
                continue;
            };
            for (long ic = 0; ic < IC_BLOCK; ic++) {
                long offset = src_off(i_mb, i_icb, 0, i_ih, i_iw);
                d[i][j][ic] = src[src_off(i_mb, i_icb + ic, 0, i_ih, i_iw)];
            }
        }
    }

    gemm_1xX(4, 4, 4, IC_BLOCK, V_TMP, BT_2x2, d);
    gemm_Xx1(4, 4, 4, IC_BLOCK, (*V), V_TMP, B_2x2);
}

static inline void transform_M_to_dst(__global DATA_T *dst,
        DATA_T M[WINO_D][WINO_D][OC_BLOCK], int i_mb, int i_ocb, int i_oh,
        int i_ow) {

    const DATA_T AT_2x2[WINO_M][WINO_D] = {{1, 1, 1, 0}, {0, 1, -1, -1}};
    const DATA_T A_2x2[WINO_D][WINO_M] = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};

    /* Inverse Transform Y=A^T M A (mb_block x oc_block x 4 x 2) */
    DATA_T Y_TMP[WINO_M][WINO_D][OC_BLOCK];
    DATA_T Y[WINO_M][WINO_M][OC_BLOCK];
    gemm_1xX(2, 4, 4, OC_BLOCK, Y_TMP, AT_2x2, M);
    gemm_Xx1(2, 4, 2, OC_BLOCK, Y, Y_TMP, A_2x2);

    // Accumulate into dst
    for (long i = 0; i < WINO_M; i++) {
        for (long j = 0; j < WINO_M; j++) {
            if (i_oh + i >= OH || i_ow + j >= OW) continue;
            for (long oc = 0; oc < OC_BLOCK; oc++) {
                dst[dst_off(i_mb, i_ocb + oc, 0, i_oh + i, i_ow + j)]
                        = Y[i][j][oc];
            }
        }
    }
}

__kernel void gen9_wino_wei_transform(
        const __global DATA_T *wei, __global DATA_T *U) {
    int gid0 = get_global_id(0);

    long i_oc_start = (gid0 % OCB) * OC_OUTER_BLOCK;
    long i_oc_end = i_oc_start + OC_BLOCK;
    long i_ic_start = (gid0 / OCB) * IC_OUTER_BLOCK;
    long i_ic_end = i_ic_start + IC_BLOCK;

    /* wei_transform U = G * g * G_t */
    for (long i_icb = i_ic_start; i_icb < i_ic_end; i_icb += IC_BLOCK) {
        for (long i_ocb = i_oc_start; i_ocb < i_oc_end; i_ocb += OC_BLOCK) {
            transform_wei_to_U(U, wei, i_ocb, i_icb);
        }
    }
}

__kernel void gen9_wino_conv_fwd(const __global DATA_T *src,
        const __global DATA_T *U_param, const __global DATA_T *bia,
        __global DATA_T *dst POST_OP_ARGS) {
    //cldnn 2x3, 6x3
    //cpu 2x3, 4x3?
    //Limitations of cldnn, contained in layout_optimizer.cpp
    // 3x3 kernel, stride=1 dilate=0, max_size from memory constraints, poor performance small spatial, ofm/ifm multiple of 64 (is that batch size)?
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int lid = get_local_id(0);

    const __global DATA_T(*U)[WINO_D][WINO_D][IC_BLOCK][OC_BLOCK]
            = (const __global DATA_T(*)[WINO_D][WINO_D][IC_BLOCK][OC_BLOCK])
                    U_param;

    /* Trivially parallel on mb and oc, oh/WINO_M and ow/WINO_M */
    long i_oc_start = (gid0 % OCB) * OC_OUTER_BLOCK;
    long i_oc_end = i_oc_start + OC_BLOCK;
    long i_mb_start = (gid0 / OCB) * MB_BLOCK;
    long i_mb_end = i_mb_start + MB_BLOCK;
    if (i_mb_end > MB) i_mb_end = MB;

    long i_ow_start = (gid1 % OWB) * OW_BLOCK;
    long i_ow_end = i_ow_start + OW_BLOCK;
    if (i_ow_end > OW) i_ow_end = OW;
    long i_oh_start = (gid1 / OWB) * OH_BLOCK;
    long i_oh_end = i_oh_start + OH_BLOCK;
    if (i_oh_end > OH) i_oh_end = OH;

    for_(long i_mb = i_mb_start; i_mb < i_mb_end; i_mb++)
    for_(long i_ocb = i_oc_start; i_ocb < i_oc_end; i_ocb += OC_BLOCK)
    for_(long i_oh = i_oh_start; i_oh < i_oh_end; i_oh += WINO_M)
    for (long i_ow = i_ow_start; i_ow < i_ow_end; i_ow += WINO_M) {
        DATA_T M[WINO_D][WINO_D][OC_BLOCK]; // Blocked on oc
        for_(long i = 0; i < WINO_D; i++)
        for (long j = 0; j < WINO_D; j++) {
            for (long oc = 0; oc < OC_BLOCK; oc++) {
                M[i][j][oc] = 0;
            }
        }

        for (long i_icb = 0; i_icb < IC; i_icb += IC_BLOCK) {
            // Src Transform to V;
            DATA_T V[WINO_D][WINO_D][IC_BLOCK]; // B^TdB
            transform_src_to_V(&V, src, i_mb, i_icb, i_oh - PH, i_ow - PW);

            long U_offset
                    = (i_icb / IC_BLOCK) * (OC / OC_BLOCK) + (i_ocb / OC_BLOCK);
            for (long i = 0; i < WINO_D; i++) {
                for (long j = 0; j < WINO_D; j++) {
                    for (long ic = 0; ic < IC_BLOCK; ic++) {
                        for (long oc = 0; oc < OC_BLOCK; oc++) {
                            M[i][j][oc]
                                    += U[U_offset][i][j][ic][oc] * V[i][j][ic];
                        }
                    }
                }
            }
        }

        transform_M_to_dst(dst, M, i_mb, i_ocb, i_oh, i_ow);
    }
}
