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
#if WITH_ELTWISE == 1
#include "ocl/ocl_post_ops.h"
#endif

#undef GRX
#define GRX 8

#if defined(S8S8)
#define FLOATA char
#define FLOATA2 char2
#define FLOATA4 char4
#define FLOATB char
#define FLOATB4 char4
#define SHUFFLE(X, Y) as_char4(intel_sub_group_shuffle(as_int(X), Y))
#endif

#if defined(U8S8)
#define FLOATA uchar
#define FLOATA2 uchar2
#define FLOATA4 uchar4
#define FLOATB char
#define FLOATB4 char4
#define SHUFFLE(X, Y) as_char4(intel_sub_group_shuffle(as_int(X), Y))
#endif

#if defined(S8U8)
#define FLOATA char
#define FLOATA2 char2
#define FLOATA4 char4
#define FLOATB uchar
#define FLOATB4 uchar4
#define SHUFFLE(X, Y) as_uchar4(intel_sub_group_shuffle(as_int(X), Y))
#endif

#if defined(U8U8)
#define FLOATA uchar
#define FLOATA2 uchar2
#define FLOATA4 uchar4
#define FLOATB uchar
#define FLOATB4 uchar4
#define SHUFFLE(X, Y) as_uchar4(intel_sub_group_shuffle(as_int(X), Y))
#endif

#define FLOATC int
#define FLOATC4 int4

#if WITH_ELTWISE == 1
#define POST_OP(val) \
    do { \
        if (apply_eltwise) \
            val = fwd_eltwise(val, eltwise_alpha, eltwise_beta); \
    } while (0)
#else
#define POST_OP(val)
#endif

#define UPDATE_C_EACH(X, OFF) \
    do { \
        if (n > X + OFF) { \
            if (m > 0) { \
                float val = c[0]; \
                POST_OP(val); \
                c[0] = val; \
            } \
            if (m > 1) { \
                float val = c[1]; \
                POST_OP(val); \
                c[1] = val; \
            } \
            if (m > 2) { \
                float val = c[2]; \
                POST_OP(val); \
                c[2] = val; \
            } \
            if (m > 3) { \
                float val = c[3]; \
                POST_OP(val); \
                c[3] = val; \
            } \
            c += ldc; \
        } \
    } while (0)

#define UPDATE_C(X) \
    do { \
        UPDATE_C_EACH(X, 0); \
        UPDATE_C_EACH(X, 1); \
        UPDATE_C_EACH(X, 2); \
        UPDATE_C_EACH(X, 3); \
    } while (0)

#ifdef FF
#define ADD_EACH(X, OFF) \
    do { \
        if (n > X + OFF) { \
            if (m > 0) \
                c[0] = ((!beta) ? 0 : c[0]) + sc[X / 4 + 0].s##OFF \
                        + ((!apply_co) ? 0 : co[0]) + xa[0] + xb[0]; \
            if (m > 1) \
                c[1] = ((!beta) ? 0 : c[1]) + sc[X / 4 + 4].s##OFF \
                        + ((!apply_co) ? 0 : co[0]) + xa[1] + xb[0]; \
            if (m > 2) \
                c[2] = ((!beta) ? 0 : c[2]) + sc[X / 4 + 8].s##OFF \
                        + ((!apply_co) ? 0 : co[0]) + xa[2] + xb[0]; \
            if (m > 3) \
                c[3] = ((!beta) ? 0 : c[3]) + sc[X / 4 + 12].s##OFF \
                        + ((!apply_co) ? 0 : co[0]) + xa[3] + xb[0]; \
            xb++; \
            c += ldc; \
        } \
    } while (0)
#elif defined CC
#define ADD_EACH(X, OFF) \
    do { \
        if (n > X + OFF) { \
            if (m > 0) \
                c[0] = ((!beta) ? 0 : c[0]) + sc[X / 4 + 0].s##OFF \
                        + ((!apply_co) ? 0 : co[0]) + xa[0] + xb[0]; \
            if (m > 1) \
                c[1] = ((!beta) ? 0 : c[1]) + sc[X / 4 + 4].s##OFF \
                        + ((!apply_co) ? 0 : co[1]) + xa[1] + xb[0]; \
            if (m > 2) \
                c[2] = ((!beta) ? 0 : c[2]) + sc[X / 4 + 8].s##OFF \
                        + ((!apply_co) ? 0 : co[2]) + xa[2] + xb[0]; \
            if (m > 3) \
                c[3] = ((!beta) ? 0 : c[3]) + sc[X / 4 + 12].s##OFF \
                        + ((!apply_co) ? 0 : co[3]) + xa[3] + xb[0]; \
            xb++; \
            c += ldc; \
        } \
    } while (0)
#else
#define ADD_EACH(X, OFF) \
    do { \
        if (n > X + OFF) { \
            if (m > 0) \
                c[0] = ((!beta) ? 0 : c[0]) + sc[X / 4 + 0].s##OFF \
                        + ((!apply_co) ? 0 : co[0]) + xa[0] + xb[0]; \
            if (m > 1) \
                c[1] = ((!beta) ? 0 : c[1]) + sc[X / 4 + 4].s##OFF \
                        + ((!apply_co) ? 0 : co[0]) + xa[1] + xb[0]; \
            if (m > 2) \
                c[2] = ((!beta) ? 0 : c[2]) + sc[X / 4 + 8].s##OFF \
                        + ((!apply_co) ? 0 : co[0]) + xa[2] + xb[0]; \
            if (m > 3) \
                c[3] = ((!beta) ? 0 : c[3]) + sc[X / 4 + 12].s##OFF \
                        + ((!apply_co) ? 0 : co[0]) + xa[3] + xb[0]; \
            xb++; \
            c += ldc; \
            co++; \
        } \
    } while (0)
#endif

#define ADD_SCALE(X) \
    do { \
        ADD_EACH(X, 0); \
        ADD_EACH(X, 1); \
        ADD_EACH(X, 2); \
        ADD_EACH(X, 3); \
    } while (0)

#define ACCUMULATE_1(a, b) \
    ((FLOATC)a.s0 * (FLOATC)b.s0) + ((FLOATC)a.s1 * (FLOATC)b.s1) \
            + ((FLOATC)a.s2 * (FLOATC)b.s2) + ((FLOATC)a.s3 * (FLOATC)b.s3)

#define ACCUMULATE(a, b0, b1, b2, b3) \
    (FLOATC4)(ACCUMULATE_1(a, b0), ACCUMULATE_1(a, b1), ACCUMULATE_1(a, b2), \
            ACCUMULATE_1(a, b3))

#define GROUPSIZE_M (6 * UNROLL_M)
#define GROUPSIZE_N (4 * UNROLL_N)

__attribute__((intel_reqd_sub_group_size(GRX))) kernel void
gen9_gemm_compute_x8x8s32(global FLOATA *a, global FLOATB *b, global FLOATC *c,
        long offsetA, long offsetB, long offsetC, long lda, long ldb, long ldc,
        long m, long n, long k, int beta, FLOATA ao, FLOATB bo,
        global FLOATC *co, long offsetCO, int apply_co, local FLOATA *sa,
        local FLOATB *sb, int apply_eltwise, float eltwise_alpha,
        float eltwise_beta) {

    long kk = (k + UNROLL_K - 1) & ~(UNROLL_K - 1);
    long i, j, l, ll;
    global FLOATC *c_ori;

    long lid = get_local_id(0); // local ID
    long idx = get_local_id(1);
    long idy = get_local_id(2);
    long gdx = get_group_id(1);
    long gdy = get_group_id(2);

    long ctotal = get_local_size(0) * get_local_size(1) * get_local_size(2);
    long cid = (idy * get_local_size(1) + idx) * get_local_size(0) + lid;
    long mwidth = (m + ctotal - 1) / ctotal;
    long nwidth = (n + ctotal - 1) / ctotal;
    long moffset = ((cid * mwidth) & ~(UNROLL_M - 1)) * kk
            + ((cid * mwidth) & (UNROLL_M - 1)) * UNROLL_K;
    long noffset = ((cid * nwidth) & ~(UNROLL_N - 1)) * kk
            + ((cid * nwidth) & (UNROLL_N - 1)) * UNROLL_K;

    // Accumulation array for A and B
    __local FLOATC *xa = (__local FLOATC *)sa;
    sa += UNROLL_M * get_local_size(1) * sizeof(FLOATC);

    __local FLOATC *xb = (__local FLOATC *)sb;
    sb += UNROLL_N * get_local_size(2) * sizeof(FLOATC);

    FLOATC sumA = 0, sumB = (FLOATC)bo * k;

#if defined(NN) || defined(NT)
    a += offsetA + (mwidth * cid + GROUPSIZE_M * gdx);
#else
    a += offsetA + (mwidth * cid + GROUPSIZE_M * gdx) * lda;
#endif

    for (l = 0; l < kk; l += UNROLL_K) {
        for (ll = 0; ll < UNROLL_K; ll++) {
            sa[moffset + l * UNROLL_M + ll]
                    = (((cid < m) && (l + ll < k)) ? *a : 0);
            sumA -= (FLOATC)sa[moffset + l * UNROLL_M + ll];
#if defined(NN) || defined(NT)
            a += lda;
#else
            a++;
#endif
        }
    }

    if (mwidth * cid < UNROLL_M * get_local_size(1))
        xa[mwidth * cid] = (FLOATC)bo * sumA;

#if defined(NN) || defined(TN)
    b += offsetB + (nwidth * cid + GROUPSIZE_N * gdy) * ldb;
#else
    b += offsetB + (nwidth * cid + GROUPSIZE_N * gdy);
#endif

    for (l = 0; l < kk; l += UNROLL_K) {
        for (ll = 0; ll < UNROLL_K; ll++) {
            sb[noffset + l * UNROLL_N + ll]
                    = (((cid < n) && (l + ll < k)) ? *b : 0);
            sumB -= (FLOATC)sb[noffset + l * UNROLL_N + ll];

#if defined(NN) || defined(TN)
            b++;
#else
            b += ldb;
#endif
        }
    }

    if (nwidth * cid < UNROLL_N * get_local_size(2))
        xb[nwidth * cid] = (FLOATC)ao * sumB;

    m -= GROUPSIZE_M * gdx + UNROLL_M * idx;
    if (m > UNROLL_M) m = UNROLL_M;
    n -= GROUPSIZE_N * gdy + UNROLL_N * idy;
    if (n > UNROLL_N) n = UNROLL_N;

    c += offsetC + UNROLL_M * idx + GROUPSIZE_M * gdx + UNROLL_M * lid / GRX
            + (UNROLL_N * idy + GROUPSIZE_N * gdy) * ldc;

    c_ori = c;

    if (apply_co) {
        co += offsetCO;
#ifdef CC
        co += GROUPSIZE_M * gdx + UNROLL_M * idx + UNROLL_M * lid / GRX;
#endif
#ifdef RR
        co += GROUPSIZE_N * gdy + UNROLL_N * idy;
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((m <= 0) || (n <= 0)) return;
    m -= UNROLL_M * lid / GRX;

    sa += UNROLL_M * kk * idx + UNROLL_M * UNROLL_K * lid / GRX;
    sb += UNROLL_N * kk * idy + UNROLL_K * lid;

    xa += UNROLL_M * idx + UNROLL_M * lid / GRX;
    xb += UNROLL_N * idy;

    FLOATC4 sc[UNROLL_M * UNROLL_N / GRX / 4] = {0};

    for (l = 0; l < kk; l += UNROLL_K) {
        FLOATA4 a0, a1, a2, a3;
        FLOATB4 bb, b0, b1, b2, b3;

        a0 = ((__local FLOATA4 *)sa)[0];
        a1 = ((__local FLOATA4 *)sa)[1];
        a2 = ((__local FLOATA4 *)sa)[2];
        a3 = ((__local FLOATA4 *)sa)[3];

        for (ll = 0; ll < GRX / 4; ll++) {
            bb = ((__local FLOATB4 *)sb)[0];
            b0 = SHUFFLE(bb, 0);
            b1 = SHUFFLE(bb, 1);
            b2 = SHUFFLE(bb, 2);
            b3 = SHUFFLE(bb, 3);

            sc[ll * 2 + 0] += ACCUMULATE(a0, b0, b1, b2, b3);
            sc[ll * 2 + 4] += ACCUMULATE(a1, b0, b1, b2, b3);
            sc[ll * 2 + 8] += ACCUMULATE(a2, b0, b1, b2, b3);
            sc[ll * 2 + 12] += ACCUMULATE(a3, b0, b1, b2, b3);
            b0 = SHUFFLE(bb, 4);
            b1 = SHUFFLE(bb, 5);
            b2 = SHUFFLE(bb, 6);
            b3 = SHUFFLE(bb, 7);

            sc[ll * 2 + 1] += ACCUMULATE(a0, b0, b1, b2, b3);
            sc[ll * 2 + 5] += ACCUMULATE(a1, b0, b1, b2, b3);
            sc[ll * 2 + 9] += ACCUMULATE(a2, b0, b1, b2, b3);
            sc[ll * 2 + 13] += ACCUMULATE(a3, b0, b1, b2, b3);

            sb += UNROLL_N * GRX / 4;
        }
        sa += UNROLL_M * UNROLL_K;
    }

    ADD_SCALE(0);
    ADD_SCALE(4);
    ADD_SCALE(8);
    ADD_SCALE(12);

    // Update C with POST_OP
    c = c_ori;
    if (apply_eltwise) {
        UPDATE_C(0);
        UPDATE_C(4);
        UPDATE_C(8);
        UPDATE_C(12);
    }
}
