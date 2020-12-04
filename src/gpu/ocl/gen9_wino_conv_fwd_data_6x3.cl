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

#define IW_BLOCK (OW_BLOCK + KW - 1)

#define WINO_D (WINO_M + WINO_R - 1)

#define TO_TYPE(value) ((DATA_T)value)

#define UTRANS_BLOCK 8
#define UTRANS_DATA_T CONCAT2(DATA_T, UTRANS_BLOCK)
#define AS_UTRANS_DATA_T CONCAT2(as_, COMP_DATA_T)
#define UTRANS_BLOCK_READ(ptr) \
    AS_UTRANS_DATA_T(BLOCK_READ8((const __global BLOCK_DATA_T *)ptr))
#define UTRANS_BLOCK_WRITE(data, ptr) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)ptr, AS_BLOCK_DATA8_T(data))

#define TRANS_BLOCK 4 // = (WINO_IC_BLOCK / (LWS_0 * LWS_1 / WINO_IW_BLOCK))
#define TRANS_DATA_T CONCAT2(DATA_T, TRANS_BLOCK)

#define COMP_BLOCK 8
#define COMP_DATA_T CONCAT2(DATA_T, COMP_BLOCK)
#define AS_COMP_DATA_T CONCAT2(as_, COMP_DATA_T)

#define COMP_UNROLL 2

#define OUT_TYPE_BLOCK 2 // = (WINO_OW_BLOCK / 7)
#define OUT_BLOCK_DATA_T CONCAT2(DATA_T, OUT_TYPE_BLOCK)

#define OUT_BLOCK_READ(ptr) CONCAT2(vload, OUT_TYPE_BLOCK)(0, ptr)
#define OUT_BLOCK_WRITE(data, ptr) \
    do { \
        OUT_BLOCK_DATA_T result = data; \
        unroll_for(int _i = 0; _i < OUT_TYPE_BLOCK; _i++) { \
            (ptr)[_i] = result[_i]; \
        } \
    } while (0)

#define UCOMP_BLOCK_READ(ptr) \
    AS_COMP_DATA_T(BLOCK_READ8((const __global BLOCK_DATA_T *)ptr))

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

static inline int off_gIOdhw16i16o(int g, int o, int i, int d, int h, int w,
        int O, int I, int D, int H, int W) {
    int off = 0;
    off += g * (I / 16) * (O / 16) * D * H * W * 16 * 16;
    off += (i / 16) * (O / 16) * D * H * W * 16 * 16;
    off += (o / 16) * D * H * W * 16 * 16;
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
    return off_gIOdhw16i16o(g, o, i, d, h, w, OC, IC, 1, KH, KW);
}

static inline int U_off(int o, int i, int z, int w) {

    //  OIw8h16i16o
    const int ic_internal_block = 16;
    const int oc_internal_block = 16;
    int icb = i / ic_internal_block;
    int ic = i % ic_internal_block;
    int ocb = o / oc_internal_block;
    int oc = o % oc_internal_block;

    int off = ocb * (WINO_IC / ic_internal_block) * KW * ic_internal_block
            * WINO_D * oc_internal_block;
    off += icb * KW * ic_internal_block * WINO_D * oc_internal_block;
    off += w * ic_internal_block * WINO_D * oc_internal_block;
    off += z * ic_internal_block * oc_internal_block;
    off += ic * oc_internal_block;
    off += oc;

    return off;
}

static inline int V_off(int i, int z, int w, int block_size) {

    //V data format is 2C8h16w16c
    const int ic_internal_block = 16;
    const int iw_block = 16;

    int icb = i / ic_internal_block;
    int ic = i % ic_internal_block;
    int off = icb * WINO_D * iw_block * ic_internal_block;
    off += z * iw_block * ic_internal_block;
    off += w * ic_internal_block;
    off += ic;
    return off / block_size;
}

static inline int M_off(int o, int z, int w, int block_size) {

    //M data format is 8h16W16c2w
    const int ow_internal_block = 2;
    int owb = w / ow_internal_block;
    int ow = w % ow_internal_block;
    int off = z * OW_BLOCK / ow_internal_block * OC_BLOCK * ow_internal_block;
    off += owb * OC_BLOCK * ow_internal_block;
    off += o * ow_internal_block;
    off += ow;
    return off / block_size;
}

static inline int dst_off(int n, int c, int d, int h, int w) {
    if (DST_W16C) return off_nCdhw16c(n, c, d, h, w, G * OC, 1, OH, OW);
    if (DST_16N16C) return off_NCdhw16n16c(n, c, d, h, w, G * OC, 1, OH, OW);
    return 0;
}

__attribute__((reqd_work_group_size(OC_BLOCK, 1, 1)))
__attribute__((intel_reqd_sub_group_size(OC_BLOCK))) __kernel void
gen9_wino_wei_transform_6x3(
        __global DATA_T *U, const __global DATA_T *weights) {
    const uint weights_tile_width = 1;
    const uint weights_tile_height = WINO_M;
    const uint in_kw = get_global_id(1) * weights_tile_width;
    const uint in_kh = get_global_id(2) * weights_tile_height;

    const uint U_tile_width = 1;
    const uint U_tile_height = WINO_D;

    const uint out_kw = get_global_id(1) * U_tile_width;
    const uint out_kh = get_global_id(2) * U_tile_height;
    const uint oc0 = (get_group_id(0) % (WINO_OC / OC_BLOCK)) * OC_BLOCK;
    const uint ic = (get_group_id(0) / (WINO_OC / OC_BLOCK)) * 8;

    uint in_idx = wei_off(0, oc0, ic, 0, in_kh, in_kw);
    bool is_valid = ic < IC && oc0 < OC;

    UTRANS_DATA_T g0, g1, g2;
    g0 = is_valid ? UTRANS_BLOCK_READ(&weights[in_idx]) : 0;
    in_idx += wei_off(0, 0, 0, 0, 1, 0);
    g1 = is_valid ? UTRANS_BLOCK_READ(&weights[in_idx]) : 0;
    in_idx += wei_off(0, 0, 0, 0, 1, 0);
    g2 = is_valid ? UTRANS_BLOCK_READ(&weights[in_idx]) : 0;

    uint out_idx = U_off(oc0, ic, out_kh, out_kw);

    UTRANS_BLOCK_WRITE(g0, &U[out_idx]);
    out_idx += U_off(0, 0, 1, 0);
    UTRANS_BLOCK_WRITE(TO_TYPE(-2.0 / 9) * (g0 + g1 + g2), &U[out_idx]);
    out_idx += U_off(0, 0, 1, 0);
    UTRANS_BLOCK_WRITE(TO_TYPE(2.0 / 9) * (-g0 + g1 - g2), &U[out_idx]);
    out_idx += U_off(0, 0, 1, 0);
    UTRANS_BLOCK_WRITE(TO_TYPE(1.0 / 90) * g0 + TO_TYPE(2.0 / 90) * g1
                    + TO_TYPE(4.0 / 90) * g2,
            &U[out_idx]);
    out_idx += U_off(0, 0, 1, 0);
    UTRANS_BLOCK_WRITE(TO_TYPE(1.0 / 90) * g0 - TO_TYPE(2.0 / 90) * g1
                    + TO_TYPE(4.0 / 90) * g2,
            &U[out_idx]);
    out_idx += U_off(0, 0, 1, 0);
    UTRANS_BLOCK_WRITE(TO_TYPE(64.0 / 90) * g0 + TO_TYPE(32.0 / 90) * g1
                    + TO_TYPE(16.0 / 90) * g2,
            &U[out_idx]);
    out_idx += U_off(0, 0, 1, 0);
    UTRANS_BLOCK_WRITE(TO_TYPE(64.0 / 90) * g0 - TO_TYPE(32.0 / 90) * g1
                    + TO_TYPE(16.0 / 90) * g2,
            &U[out_idx]);
    out_idx += U_off(0, 0, 1, 0);
    UTRANS_BLOCK_WRITE(g2, &U[out_idx]);
}

#define DOT8i_0(_result, _A, _B, i) \
    { _result = mad(_A.s0, sub_group_broadcast(_B.s0, i), _result); }
#define DOT8i_1(_result, _A, _B, i) \
    { _result = mad(_A.s1, sub_group_broadcast(_B.s1, i), _result); }
#define DOT8i_2(_result, _A, _B, i) \
    { _result = mad(_A.s2, sub_group_broadcast(_B.s2, i), _result); }
#define DOT8i_3(_result, _A, _B, i) \
    { _result = mad(_A.s3, sub_group_broadcast(_B.s3, i), _result); }
#define DOT8i_4(_result, _A, _B, i) \
    { _result = mad(_A.s4, sub_group_broadcast(_B.s4, i), _result); }
#define DOT8i_5(_result, _A, _B, i) \
    { _result = mad(_A.s5, sub_group_broadcast(_B.s5, i), _result); }
#define DOT8i_6(_result, _A, _B, i) \
    { _result = mad(_A.s6, sub_group_broadcast(_B.s6, i), _result); }
#define DOT8i_7(_result, _A, _B, i) \
    { _result = mad(_A.s7, sub_group_broadcast(_B.s7, i), _result); }

__attribute__((reqd_work_group_size(16, 8, 1)))
__attribute__((intel_reqd_sub_group_size(16))) __kernel void
gen9_wino_conv_fwd_6x3(__global DATA_T *dst, const __global DATA_T *src,
        const __global DATA_T *U_param,
        const __global DATA_T *bias POST_OP_ARGS) {
    //               (DxC2)x(UxWx8c)
    const uint slm_size = (WINO_IC_BLOCK * WINO_D * IW_BLOCK) / TRANS_BLOCK;
    __local TRANS_DATA_T V[slm_size]; // 8 KB

    const DATA_T sc = TO_TYPE(0.1);
    const DATA_T scl = TO_TYPE(1.0) / sc;
    const TRANS_DATA_T scl_vec = (TRANS_DATA_T)(sc, sc, sc, sc);

    const int ow0 = get_group_id(0) * OW_BLOCK;
    const int oh = get_group_id(1) * OH_BLOCK;
    const int gid2 = get_group_id(2);
    const int oc0 = (gid2 % (OC / OC_BLOCK)) * OC_BLOCK;
    const int mb = gid2 / (OC / OC_BLOCK);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    uint lxd8 = lx / 8;
    uint lxm8 = lx % 8;
    uint lxd2 = lx / 2;
    uint lxm2 = lx % 2;

    const int oc = oc0 + lx;
    const int ow = ow0 + 2 * ly;

    // Load ic32ih8iw16 input tile, with 2 pixel overlap in ih and iw.
    // Compute oc16oh6ow14 output tile.

    int iw0_write = ly * 2 + lxd8;
    int iw0_read = lxd2;
    int iw = ow0 + iw0_write - PW;
    int ih = oh - PH;
    int ic0_write = lxm8 * TRANS_BLOCK;
    int ic0_read = 8 * lxm2;

    // Initialize variables to accumulate intermediate output tile
    OUT_BLOCK_DATA_T M0 = (OUT_BLOCK_DATA_T)(0, 0);
    OUT_BLOCK_DATA_T M1 = (OUT_BLOCK_DATA_T)(0, 0);
    OUT_BLOCK_DATA_T M2 = (OUT_BLOCK_DATA_T)(0, 0);
    OUT_BLOCK_DATA_T M3 = (OUT_BLOCK_DATA_T)(0, 0);
    OUT_BLOCK_DATA_T M4 = (OUT_BLOCK_DATA_T)(0, 0);
    OUT_BLOCK_DATA_T M5 = (OUT_BLOCK_DATA_T)(0, 0);
    OUT_BLOCK_DATA_T M6 = (OUT_BLOCK_DATA_T)(0, 0);

    // Computation is separated into three main stages, load/transform input,
    // compute intermediate output block, and transform/store final output.
    // Between these stages, the dimensions handled by local work groups
    // changes.

    // Buffers used to load and transform ic32ih8iw16 src tile into V
    // Each local thread transforms a block with dimensions c4h8w1
    // For the computation, src_i traverses ih dimension, ly * 2 + lx/8
    // traverses iw dimension, and lx % 8 traverses ic dimension
    const __global DATA_T *src_load = src + src_off(mb, ic0_write, 0, ih, iw);
    const int V_write_idx = V_off(ic0_write, 0, iw0_write, TRANS_BLOCK);
    __local TRANS_DATA_T *V_write = &V[V_write_idx];

    // Buffers used to compute oc16oh8ow14 intermediate output tile. Each
    // local thread transforms a block with dimensions c1h1w14. For the
    // computed output, M_i traverses ow dimension, ly traverses oh
    // dimension, and lx traverses oc dimension.
    const __global DATA_T *U = U_param + U_off(oc, 0, ly, 0);
    const int V_read_idx = V_off(ic0_read, ly, iw0_read, TRANS_BLOCK);
    __local const COMP_DATA_T *V_read
            = (__local const COMP_DATA_T *)&V[V_read_idx]; // ly * 64 + lx * 2;

    __attribute__((opencl_unroll_hint(1))) for (uint c = 0; c < IC;
                                                c += WINO_IC_BLOCK) {
        // Load and transform ic32ih8iw16 src tile into V
        {
            bool x_in = 0 <= iw && iw < IW && ic0_read + c < IC;
            bool y0_in = 0 <= (ih + 0) && (ih + 0) < IH && x_in;
            bool y1_in = 0 <= (ih + 1) && (ih + 1) < IH && x_in;
            bool y2_in = 0 <= (ih + 2) && (ih + 2) < IH && x_in;
            bool y3_in = 0 <= (ih + 3) && (ih + 3) < IH && x_in;
            bool y4_in = 0 <= (ih + 4) && (ih + 4) < IH && x_in;
            bool y5_in = 0 <= (ih + 5) && (ih + 5) < IH && x_in;
            bool y6_in = 0 <= (ih + 6) && (ih + 6) < IH && x_in;
            bool y7_in = 0 <= (ih + 7) && (ih + 7) < IH && x_in;

            TRANS_DATA_T src0 = y0_in
                    ? *((const __global TRANS_DATA_T *)(src_load
                            + src_off(0, 0, 0, 0, 0)))
                    : 0;
            TRANS_DATA_T src1 = y1_in
                    ? *((const __global TRANS_DATA_T *)(src_load
                            + src_off(0, 0, 0, 1, 0)))
                    : 0;
            TRANS_DATA_T src2 = y2_in
                    ? *((const __global TRANS_DATA_T *)(src_load
                            + src_off(0, 0, 0, 2, 0)))
                    : 0;
            TRANS_DATA_T src3 = y3_in
                    ? *((const __global TRANS_DATA_T *)(src_load
                            + src_off(0, 0, 0, 3, 0)))
                    : 0;
            TRANS_DATA_T src4 = y4_in
                    ? *((const __global TRANS_DATA_T *)(src_load
                            + src_off(0, 0, 0, 4, 0)))
                    : 0;
            TRANS_DATA_T src5 = y5_in
                    ? *((const __global TRANS_DATA_T *)(src_load
                            + src_off(0, 0, 0, 5, 0)))
                    : 0;
            TRANS_DATA_T src6 = y6_in
                    ? *((const __global TRANS_DATA_T *)(src_load
                            + src_off(0, 0, 0, 6, 0)))
                    : 0;
            TRANS_DATA_T src7 = y7_in
                    ? *((const __global TRANS_DATA_T *)(src_load
                            + src_off(0, 0, 0, 7, 0)))
                    : 0;

            //Scale input to prevent intermediate computations overflow in some
            //cases, output is adjusted with the same scale factor after main
            //computation
            src0 = src0 * scl_vec;
            src1 = src1 * scl_vec;
            src2 = src2 * scl_vec;
            src3 = src3 * scl_vec;
            src4 = src4 * scl_vec;
            src5 = src5 * scl_vec;
            src6 = src6 * scl_vec;
            src7 = src7 * scl_vec;

            // Compute Winograd f6x3 data transform and store components in SLM.
            V_write[V_off(0, 0, 0, TRANS_BLOCK)]
                    = src0 - TO_TYPE(5.25) * src2 + TO_TYPE(5.25) * src4 - src6;

            TRANS_DATA_T x0 = src1 - TO_TYPE(4.25) * src3 + src5;
            TRANS_DATA_T x1 = src2 - TO_TYPE(4.25) * src4 + src6;

            V_write[V_off(0, 1, 0, TRANS_BLOCK)] = x1 + x0;
            V_write[V_off(0, 2, 0, TRANS_BLOCK)] = x1 - x0;

            TRANS_DATA_T x2 = TO_TYPE(-5) * src3 + src1;
            TRANS_DATA_T x3 = TO_TYPE(4) * src5 + x2;
            TRANS_DATA_T x4 = TO_TYPE(0.25) * src2 + src6;
            TRANS_DATA_T x5 = TO_TYPE(-1.25) * src4 + x4;

            V_write[V_off(0, 3, 0, TRANS_BLOCK)] = TO_TYPE(0.5) * x3 + x5;
            V_write[V_off(0, 4, 0, TRANS_BLOCK)] = TO_TYPE(-0.5) * x3 + x5;

            TRANS_DATA_T x6 = TO_TYPE(4) * src1 + src5;
            TRANS_DATA_T x7 = TO_TYPE(-5) * src3 + x6;
            TRANS_DATA_T x8 = TO_TYPE(4) * src2 + src6;
            TRANS_DATA_T x9 = TO_TYPE(-5) * src4 + x8;

            V_write[V_off(0, 5, 0, TRANS_BLOCK)] = TO_TYPE(+0.5) * x7 + x9;
            V_write[V_off(0, 6, 0, TRANS_BLOCK)] = TO_TYPE(-0.5) * x7 + x9;

            V_write[V_off(0, 7, 0, TRANS_BLOCK)] = -src1 + TO_TYPE(5.25) * src3
                    - TO_TYPE(5.25) * src5 + src7;
        }

        src_load += src_off(0, WINO_IC_BLOCK, 0, 0, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate oc16oh8ow14 intermediate output tile stored in the M_i
        __local const COMP_DATA_T *V_read_outer = V_read;

        __attribute__((opencl_unroll_hint(
                1))) for (uint c_outer = 0; c_outer < WINO_IC_BLOCK;
                          c_outer += COMP_UNROLL * COMP_BLOCK) {
            // Fetch 16 input components, spread across subgroup.
            COMP_DATA_T V_block0 = V_read_outer[V_off(0, 0, 0, COMP_BLOCK)];
            COMP_DATA_T V_block1
                    = V_read_outer[V_off(0, 0, COMP_BLOCK, COMP_BLOCK)];
            V_read_outer += V_off(COMP_UNROLL * COMP_BLOCK, 0, 0, COMP_BLOCK);

            __attribute__((opencl_unroll_hint(
                    COMP_UNROLL))) for (int c_inner = 0; c_inner < COMP_UNROLL;
                                        ++c_inner) {
                // 2*14 * 3 * 4 = 336 MADs
                // Fetch 8 channels of Winograd components from f(k,s)
                const COMP_DATA_T f00 = UCOMP_BLOCK_READ(
                        (const __global ushort *)&U[U_off(0, 0, 0, 0)]);

                // f0 x v[0 .. 14]
                DOT8i_0(M0.s0, f00, V_block0, 0 + c_inner);
                DOT8i_0(M0.s1, f00, V_block0, 2 + c_inner);
                DOT8i_0(M1.s0, f00, V_block0, 4 + c_inner);
                DOT8i_0(M1.s1, f00, V_block0, 6 + c_inner);

                DOT8i_0(M2.s0, f00, V_block0, 8 + c_inner);
                DOT8i_0(M2.s1, f00, V_block0, 10 + c_inner);
                DOT8i_0(M3.s0, f00, V_block0, 12 + c_inner);
                DOT8i_0(M3.s1, f00, V_block0, 14 + c_inner);

                DOT8i_0(M4.s0, f00, V_block1, 0 + c_inner);
                DOT8i_0(M4.s1, f00, V_block1, 2 + c_inner);
                DOT8i_0(M5.s0, f00, V_block1, 4 + c_inner);
                DOT8i_0(M5.s1, f00, V_block1, 6 + c_inner);

                DOT8i_0(M6.s0, f00, V_block1, 8 + c_inner);
                DOT8i_0(M6.s1, f00, V_block1, 10 + c_inner);

                // f0 x v[0 .. 14]
                DOT8i_1(M0.s0, f00, V_block0, 0 + c_inner);
                DOT8i_1(M0.s1, f00, V_block0, 2 + c_inner);
                DOT8i_1(M1.s0, f00, V_block0, 4 + c_inner);
                DOT8i_1(M1.s1, f00, V_block0, 6 + c_inner);

                DOT8i_1(M2.s0, f00, V_block0, 8 + c_inner);
                DOT8i_1(M2.s1, f00, V_block0, 10 + c_inner);
                DOT8i_1(M3.s0, f00, V_block0, 12 + c_inner);
                DOT8i_1(M3.s1, f00, V_block0, 14 + c_inner);

                DOT8i_1(M4.s0, f00, V_block1, 0 + c_inner);
                DOT8i_1(M4.s1, f00, V_block1, 2 + c_inner);
                DOT8i_1(M5.s0, f00, V_block1, 4 + c_inner);
                DOT8i_1(M5.s1, f00, V_block1, 6 + c_inner);

                DOT8i_1(M6.s0, f00, V_block1, 8 + c_inner);
                DOT8i_1(M6.s1, f00, V_block1, 10 + c_inner);

                // f0 x v[0 .. 14]
                DOT8i_2(M0.s0, f00, V_block0, 0 + c_inner);
                DOT8i_2(M0.s1, f00, V_block0, 2 + c_inner);
                DOT8i_2(M1.s0, f00, V_block0, 4 + c_inner);
                DOT8i_2(M1.s1, f00, V_block0, 6 + c_inner);

                DOT8i_2(M2.s0, f00, V_block0, 8 + c_inner);
                DOT8i_2(M2.s1, f00, V_block0, 10 + c_inner);
                DOT8i_2(M3.s0, f00, V_block0, 12 + c_inner);
                DOT8i_2(M3.s1, f00, V_block0, 14 + c_inner);

                DOT8i_2(M4.s0, f00, V_block1, 0 + c_inner);
                DOT8i_2(M4.s1, f00, V_block1, 2 + c_inner);
                DOT8i_2(M5.s0, f00, V_block1, 4 + c_inner);
                DOT8i_2(M5.s1, f00, V_block1, 6 + c_inner);

                DOT8i_2(M6.s0, f00, V_block1, 8 + c_inner);
                DOT8i_2(M6.s1, f00, V_block1, 10 + c_inner);

                // f0 x v[0 .. 14]
                DOT8i_3(M0.s0, f00, V_block0, 0 + c_inner);
                DOT8i_3(M0.s1, f00, V_block0, 2 + c_inner);
                DOT8i_3(M1.s0, f00, V_block0, 4 + c_inner);
                DOT8i_3(M1.s1, f00, V_block0, 6 + c_inner);

                DOT8i_3(M2.s0, f00, V_block0, 8 + c_inner);
                DOT8i_3(M2.s1, f00, V_block0, 10 + c_inner);
                DOT8i_3(M3.s0, f00, V_block0, 12 + c_inner);
                DOT8i_3(M3.s1, f00, V_block0, 14 + c_inner);

                DOT8i_3(M4.s0, f00, V_block1, 0 + c_inner);
                DOT8i_3(M4.s1, f00, V_block1, 2 + c_inner);
                DOT8i_3(M5.s0, f00, V_block1, 4 + c_inner);
                DOT8i_3(M5.s1, f00, V_block1, 6 + c_inner);

                DOT8i_3(M6.s0, f00, V_block1, 8 + c_inner);
                DOT8i_3(M6.s1, f00, V_block1, 10 + c_inner);

                // f0 x v[0 .. 14]
                DOT8i_4(M0.s0, f00, V_block0, 0 + c_inner);
                DOT8i_4(M0.s1, f00, V_block0, 2 + c_inner);
                DOT8i_4(M1.s0, f00, V_block0, 4 + c_inner);
                DOT8i_4(M1.s1, f00, V_block0, 6 + c_inner);

                DOT8i_4(M2.s0, f00, V_block0, 8 + c_inner);
                DOT8i_4(M2.s1, f00, V_block0, 10 + c_inner);
                DOT8i_4(M3.s0, f00, V_block0, 12 + c_inner);
                DOT8i_4(M3.s1, f00, V_block0, 14 + c_inner);

                DOT8i_4(M4.s0, f00, V_block1, 0 + c_inner);
                DOT8i_4(M4.s1, f00, V_block1, 2 + c_inner);
                DOT8i_4(M5.s0, f00, V_block1, 4 + c_inner);
                DOT8i_4(M5.s1, f00, V_block1, 6 + c_inner);

                DOT8i_4(M6.s0, f00, V_block1, 8 + c_inner);
                DOT8i_4(M6.s1, f00, V_block1, 10 + c_inner);

                // f0 x v[0 .. 14]
                DOT8i_5(M0.s0, f00, V_block0, 0 + c_inner);
                DOT8i_5(M0.s1, f00, V_block0, 2 + c_inner);
                DOT8i_5(M1.s0, f00, V_block0, 4 + c_inner);
                DOT8i_5(M1.s1, f00, V_block0, 6 + c_inner);

                DOT8i_5(M2.s0, f00, V_block0, 8 + c_inner);
                DOT8i_5(M2.s1, f00, V_block0, 10 + c_inner);
                DOT8i_5(M3.s0, f00, V_block0, 12 + c_inner);
                DOT8i_5(M3.s1, f00, V_block0, 14 + c_inner);

                DOT8i_5(M4.s0, f00, V_block1, 0 + c_inner);
                DOT8i_5(M4.s1, f00, V_block1, 2 + c_inner);
                DOT8i_5(M5.s0, f00, V_block1, 4 + c_inner);
                DOT8i_5(M5.s1, f00, V_block1, 6 + c_inner);

                DOT8i_5(M6.s0, f00, V_block1, 8 + c_inner);
                DOT8i_5(M6.s1, f00, V_block1, 10 + c_inner);

                // f0 x v[0 .. 14]
                DOT8i_6(M0.s0, f00, V_block0, 0 + c_inner);
                DOT8i_6(M0.s1, f00, V_block0, 2 + c_inner);
                DOT8i_6(M1.s0, f00, V_block0, 4 + c_inner);
                DOT8i_6(M1.s1, f00, V_block0, 6 + c_inner);

                DOT8i_6(M2.s0, f00, V_block0, 8 + c_inner);
                DOT8i_6(M2.s1, f00, V_block0, 10 + c_inner);
                DOT8i_6(M3.s0, f00, V_block0, 12 + c_inner);
                DOT8i_6(M3.s1, f00, V_block0, 14 + c_inner);

                DOT8i_6(M4.s0, f00, V_block1, 0 + c_inner);
                DOT8i_6(M4.s1, f00, V_block1, 2 + c_inner);
                DOT8i_6(M5.s0, f00, V_block1, 4 + c_inner);
                DOT8i_6(M5.s1, f00, V_block1, 6 + c_inner);

                DOT8i_6(M6.s0, f00, V_block1, 8 + c_inner);
                DOT8i_6(M6.s1, f00, V_block1, 10 + c_inner);

                // f0 x v[0 .. 14]
                DOT8i_7(M0.s0, f00, V_block0, 0 + c_inner);
                DOT8i_7(M0.s1, f00, V_block0, 2 + c_inner);
                DOT8i_7(M1.s0, f00, V_block0, 4 + c_inner);
                DOT8i_7(M1.s1, f00, V_block0, 6 + c_inner);

                DOT8i_7(M2.s0, f00, V_block0, 8 + c_inner);
                DOT8i_7(M2.s1, f00, V_block0, 10 + c_inner);
                DOT8i_7(M3.s0, f00, V_block0, 12 + c_inner);
                DOT8i_7(M3.s1, f00, V_block0, 14 + c_inner);

                DOT8i_7(M4.s0, f00, V_block1, 0 + c_inner);
                DOT8i_7(M4.s1, f00, V_block1, 2 + c_inner);
                DOT8i_7(M5.s0, f00, V_block1, 4 + c_inner);
                DOT8i_7(M5.s1, f00, V_block1, 6 + c_inner);

                DOT8i_7(M6.s0, f00, V_block1, 8 + c_inner);
                DOT8i_7(M6.s1, f00, V_block1, 10 + c_inner);

                const COMP_DATA_T f01 = UCOMP_BLOCK_READ(
                        (const __global ushort *)&U[U_off(0, 0, 0, 1)]);

                // f1[c_inner] x v[1 .. 15]
                DOT8i_0(M0.s0, f01, V_block0, 2 + c_inner);
                DOT8i_0(M0.s1, f01, V_block0, 4 + c_inner);
                DOT8i_0(M1.s0, f01, V_block0, 6 + c_inner);
                DOT8i_0(M1.s1, f01, V_block0, 8 + c_inner);

                DOT8i_0(M2.s0, f01, V_block0, 10 + c_inner);
                DOT8i_0(M2.s1, f01, V_block0, 12 + c_inner);
                DOT8i_0(M3.s0, f01, V_block0, 14 + c_inner);
                DOT8i_0(M3.s1, f01, V_block1, 0 + c_inner);

                DOT8i_0(M4.s0, f01, V_block1, 2 + c_inner);
                DOT8i_0(M4.s1, f01, V_block1, 4 + c_inner);
                DOT8i_0(M5.s0, f01, V_block1, 6 + c_inner);
                DOT8i_0(M5.s1, f01, V_block1, 8 + c_inner);

                DOT8i_0(M6.s0, f01, V_block1, 10 + c_inner);
                DOT8i_0(M6.s1, f01, V_block1, 12 + c_inner);

                // f1[c_inner] x v[1 .. 15]
                DOT8i_1(M0.s0, f01, V_block0, 2 + c_inner);
                DOT8i_1(M0.s1, f01, V_block0, 4 + c_inner);
                DOT8i_1(M1.s0, f01, V_block0, 6 + c_inner);
                DOT8i_1(M1.s1, f01, V_block0, 8 + c_inner);

                DOT8i_1(M2.s0, f01, V_block0, 10 + c_inner);
                DOT8i_1(M2.s1, f01, V_block0, 12 + c_inner);
                DOT8i_1(M3.s0, f01, V_block0, 14 + c_inner);
                DOT8i_1(M3.s1, f01, V_block1, 0 + c_inner);

                DOT8i_1(M4.s0, f01, V_block1, 2 + c_inner);
                DOT8i_1(M4.s1, f01, V_block1, 4 + c_inner);
                DOT8i_1(M5.s0, f01, V_block1, 6 + c_inner);
                DOT8i_1(M5.s1, f01, V_block1, 8 + c_inner);

                DOT8i_1(M6.s0, f01, V_block1, 10 + c_inner);
                DOT8i_1(M6.s1, f01, V_block1, 12 + c_inner);

                // f1[c_inner] x v[1 .. 15]
                DOT8i_2(M0.s0, f01, V_block0, 2 + c_inner);
                DOT8i_2(M0.s1, f01, V_block0, 4 + c_inner);
                DOT8i_2(M1.s0, f01, V_block0, 6 + c_inner);
                DOT8i_2(M1.s1, f01, V_block0, 8 + c_inner);

                DOT8i_2(M2.s0, f01, V_block0, 10 + c_inner);
                DOT8i_2(M2.s1, f01, V_block0, 12 + c_inner);
                DOT8i_2(M3.s0, f01, V_block0, 14 + c_inner);
                DOT8i_2(M3.s1, f01, V_block1, 0 + c_inner);

                DOT8i_2(M4.s0, f01, V_block1, 2 + c_inner);
                DOT8i_2(M4.s1, f01, V_block1, 4 + c_inner);
                DOT8i_2(M5.s0, f01, V_block1, 6 + c_inner);
                DOT8i_2(M5.s1, f01, V_block1, 8 + c_inner);

                DOT8i_2(M6.s0, f01, V_block1, 10 + c_inner);
                DOT8i_2(M6.s1, f01, V_block1, 12 + c_inner);

                // f1[c_inner] x v[1 .. 15]
                DOT8i_3(M0.s0, f01, V_block0, 2 + c_inner);
                DOT8i_3(M0.s1, f01, V_block0, 4 + c_inner);
                DOT8i_3(M1.s0, f01, V_block0, 6 + c_inner);
                DOT8i_3(M1.s1, f01, V_block0, 8 + c_inner);

                DOT8i_3(M2.s0, f01, V_block0, 10 + c_inner);
                DOT8i_3(M2.s1, f01, V_block0, 12 + c_inner);
                DOT8i_3(M3.s0, f01, V_block0, 14 + c_inner);
                DOT8i_3(M3.s1, f01, V_block1, 0 + c_inner);

                DOT8i_3(M4.s0, f01, V_block1, 2 + c_inner);
                DOT8i_3(M4.s1, f01, V_block1, 4 + c_inner);
                DOT8i_3(M5.s0, f01, V_block1, 6 + c_inner);
                DOT8i_3(M5.s1, f01, V_block1, 8 + c_inner);

                DOT8i_3(M6.s0, f01, V_block1, 10 + c_inner);
                DOT8i_3(M6.s1, f01, V_block1, 12 + c_inner);

                // f1[c_inner] x v[1 .. 15]
                DOT8i_4(M0.s0, f01, V_block0, 2 + c_inner);
                DOT8i_4(M0.s1, f01, V_block0, 4 + c_inner);
                DOT8i_4(M1.s0, f01, V_block0, 6 + c_inner);
                DOT8i_4(M1.s1, f01, V_block0, 8 + c_inner);

                DOT8i_4(M2.s0, f01, V_block0, 10 + c_inner);
                DOT8i_4(M2.s1, f01, V_block0, 12 + c_inner);
                DOT8i_4(M3.s0, f01, V_block0, 14 + c_inner);
                DOT8i_4(M3.s1, f01, V_block1, 0 + c_inner);

                DOT8i_4(M4.s0, f01, V_block1, 2 + c_inner);
                DOT8i_4(M4.s1, f01, V_block1, 4 + c_inner);
                DOT8i_4(M5.s0, f01, V_block1, 6 + c_inner);
                DOT8i_4(M5.s1, f01, V_block1, 8 + c_inner);

                DOT8i_4(M6.s0, f01, V_block1, 10 + c_inner);
                DOT8i_4(M6.s1, f01, V_block1, 12 + c_inner);

                // f1[c_inner] x v[1 .. 15]
                DOT8i_5(M0.s0, f01, V_block0, 2 + c_inner);
                DOT8i_5(M0.s1, f01, V_block0, 4 + c_inner);
                DOT8i_5(M1.s0, f01, V_block0, 6 + c_inner);
                DOT8i_5(M1.s1, f01, V_block0, 8 + c_inner);

                DOT8i_5(M2.s0, f01, V_block0, 10 + c_inner);
                DOT8i_5(M2.s1, f01, V_block0, 12 + c_inner);
                DOT8i_5(M3.s0, f01, V_block0, 14 + c_inner);
                DOT8i_5(M3.s1, f01, V_block1, 0 + c_inner);

                DOT8i_5(M4.s0, f01, V_block1, 2 + c_inner);
                DOT8i_5(M4.s1, f01, V_block1, 4 + c_inner);
                DOT8i_5(M5.s0, f01, V_block1, 6 + c_inner);
                DOT8i_5(M5.s1, f01, V_block1, 8 + c_inner);

                DOT8i_5(M6.s0, f01, V_block1, 10 + c_inner);
                DOT8i_5(M6.s1, f01, V_block1, 12 + c_inner);

                // f1[c_inner] x v[1 .. 15]
                DOT8i_6(M0.s0, f01, V_block0, 2 + c_inner);
                DOT8i_6(M0.s1, f01, V_block0, 4 + c_inner);
                DOT8i_6(M1.s0, f01, V_block0, 6 + c_inner);
                DOT8i_6(M1.s1, f01, V_block0, 8 + c_inner);

                DOT8i_6(M2.s0, f01, V_block0, 10 + c_inner);
                DOT8i_6(M2.s1, f01, V_block0, 12 + c_inner);
                DOT8i_6(M3.s0, f01, V_block0, 14 + c_inner);
                DOT8i_6(M3.s1, f01, V_block1, 0 + c_inner);

                DOT8i_6(M4.s0, f01, V_block1, 2 + c_inner);
                DOT8i_6(M4.s1, f01, V_block1, 4 + c_inner);
                DOT8i_6(M5.s0, f01, V_block1, 6 + c_inner);
                DOT8i_6(M5.s1, f01, V_block1, 8 + c_inner);

                DOT8i_6(M6.s0, f01, V_block1, 10 + c_inner);
                DOT8i_6(M6.s1, f01, V_block1, 12 + c_inner);

                // f1[c_inner] x v[1 .. 15]
                DOT8i_7(M0.s0, f01, V_block0, 2 + c_inner);
                DOT8i_7(M0.s1, f01, V_block0, 4 + c_inner);
                DOT8i_7(M1.s0, f01, V_block0, 6 + c_inner);
                DOT8i_7(M1.s1, f01, V_block0, 8 + c_inner);

                DOT8i_7(M2.s0, f01, V_block0, 10 + c_inner);
                DOT8i_7(M2.s1, f01, V_block0, 12 + c_inner);
                DOT8i_7(M3.s0, f01, V_block0, 14 + c_inner);
                DOT8i_7(M3.s1, f01, V_block1, 0 + c_inner);

                DOT8i_7(M4.s0, f01, V_block1, 2 + c_inner);
                DOT8i_7(M4.s1, f01, V_block1, 4 + c_inner);
                DOT8i_7(M5.s0, f01, V_block1, 6 + c_inner);
                DOT8i_7(M5.s1, f01, V_block1, 8 + c_inner);

                DOT8i_7(M6.s0, f01, V_block1, 10 + c_inner);
                DOT8i_7(M6.s1, f01, V_block1, 12 + c_inner);

                const COMP_DATA_T f02 = UCOMP_BLOCK_READ(
                        (const __global ushort *)&U[U_off(0, 0, 0, 2)]);
                U += U_off(0, COMP_BLOCK, 0, 0);

                // f2[c_inner] x v[2 .. 16]
                DOT8i_0(M0.s0, f02, V_block0, 4 + c_inner);
                DOT8i_0(M0.s1, f02, V_block0, 6 + c_inner);
                DOT8i_0(M1.s0, f02, V_block0, 8 + c_inner);
                DOT8i_0(M1.s1, f02, V_block0, 10 + c_inner);

                DOT8i_0(M2.s0, f02, V_block0, 12 + c_inner);
                DOT8i_0(M2.s1, f02, V_block0, 14 + c_inner);
                DOT8i_0(M3.s0, f02, V_block1, 0 + c_inner);
                DOT8i_0(M3.s1, f02, V_block1, 2 + c_inner);

                DOT8i_0(M4.s0, f02, V_block1, 4 + c_inner);
                DOT8i_0(M4.s1, f02, V_block1, 6 + c_inner);
                DOT8i_0(M5.s0, f02, V_block1, 8 + c_inner);
                DOT8i_0(M5.s1, f02, V_block1, 10 + c_inner);

                DOT8i_0(M6.s0, f02, V_block1, 12 + c_inner);
                DOT8i_0(M6.s1, f02, V_block1, 14 + c_inner);

                // f2[c_inner] x v[2 .. 16]
                DOT8i_1(M0.s0, f02, V_block0, 4 + c_inner);
                DOT8i_1(M0.s1, f02, V_block0, 6 + c_inner);
                DOT8i_1(M1.s0, f02, V_block0, 8 + c_inner);
                DOT8i_1(M1.s1, f02, V_block0, 10 + c_inner);

                DOT8i_1(M2.s0, f02, V_block0, 12 + c_inner);
                DOT8i_1(M2.s1, f02, V_block0, 14 + c_inner);
                DOT8i_1(M3.s0, f02, V_block1, 0 + c_inner);
                DOT8i_1(M3.s1, f02, V_block1, 2 + c_inner);

                DOT8i_1(M4.s0, f02, V_block1, 4 + c_inner);
                DOT8i_1(M4.s1, f02, V_block1, 6 + c_inner);
                DOT8i_1(M5.s0, f02, V_block1, 8 + c_inner);
                DOT8i_1(M5.s1, f02, V_block1, 10 + c_inner);

                DOT8i_1(M6.s0, f02, V_block1, 12 + c_inner);
                DOT8i_1(M6.s1, f02, V_block1, 14 + c_inner);

                // f2[c_inner] x v[2 .. 16]
                DOT8i_2(M0.s0, f02, V_block0, 4 + c_inner);
                DOT8i_2(M0.s1, f02, V_block0, 6 + c_inner);
                DOT8i_2(M1.s0, f02, V_block0, 8 + c_inner);
                DOT8i_2(M1.s1, f02, V_block0, 10 + c_inner);

                DOT8i_2(M2.s0, f02, V_block0, 12 + c_inner);
                DOT8i_2(M2.s1, f02, V_block0, 14 + c_inner);
                DOT8i_2(M3.s0, f02, V_block1, 0 + c_inner);
                DOT8i_2(M3.s1, f02, V_block1, 2 + c_inner);

                DOT8i_2(M4.s0, f02, V_block1, 4 + c_inner);
                DOT8i_2(M4.s1, f02, V_block1, 6 + c_inner);
                DOT8i_2(M5.s0, f02, V_block1, 8 + c_inner);
                DOT8i_2(M5.s1, f02, V_block1, 10 + c_inner);

                DOT8i_2(M6.s0, f02, V_block1, 12 + c_inner);
                DOT8i_2(M6.s1, f02, V_block1, 14 + c_inner);

                // f2[c_inner] x v[2 .. 16]
                DOT8i_3(M0.s0, f02, V_block0, 4 + c_inner);
                DOT8i_3(M0.s1, f02, V_block0, 6 + c_inner);
                DOT8i_3(M1.s0, f02, V_block0, 8 + c_inner);
                DOT8i_3(M1.s1, f02, V_block0, 10 + c_inner);

                DOT8i_3(M2.s0, f02, V_block0, 12 + c_inner);
                DOT8i_3(M2.s1, f02, V_block0, 14 + c_inner);
                DOT8i_3(M3.s0, f02, V_block1, 0 + c_inner);
                DOT8i_3(M3.s1, f02, V_block1, 2 + c_inner);

                DOT8i_3(M4.s0, f02, V_block1, 4 + c_inner);
                DOT8i_3(M4.s1, f02, V_block1, 6 + c_inner);
                DOT8i_3(M5.s0, f02, V_block1, 8 + c_inner);
                DOT8i_3(M5.s1, f02, V_block1, 10 + c_inner);

                DOT8i_3(M6.s0, f02, V_block1, 12 + c_inner);
                DOT8i_3(M6.s1, f02, V_block1, 14 + c_inner);

                // f2[c_inner] x v[2 .. 16]
                DOT8i_4(M0.s0, f02, V_block0, 4 + c_inner);
                DOT8i_4(M0.s1, f02, V_block0, 6 + c_inner);
                DOT8i_4(M1.s0, f02, V_block0, 8 + c_inner);
                DOT8i_4(M1.s1, f02, V_block0, 10 + c_inner);

                DOT8i_4(M2.s0, f02, V_block0, 12 + c_inner);
                DOT8i_4(M2.s1, f02, V_block0, 14 + c_inner);
                DOT8i_4(M3.s0, f02, V_block1, 0 + c_inner);
                DOT8i_4(M3.s1, f02, V_block1, 2 + c_inner);

                DOT8i_4(M4.s0, f02, V_block1, 4 + c_inner);
                DOT8i_4(M4.s1, f02, V_block1, 6 + c_inner);
                DOT8i_4(M5.s0, f02, V_block1, 8 + c_inner);
                DOT8i_4(M5.s1, f02, V_block1, 10 + c_inner);

                DOT8i_4(M6.s0, f02, V_block1, 12 + c_inner);
                DOT8i_4(M6.s1, f02, V_block1, 14 + c_inner);

                // f2[c_inner] x v[2 .. 16]
                DOT8i_5(M0.s0, f02, V_block0, 4 + c_inner);
                DOT8i_5(M0.s1, f02, V_block0, 6 + c_inner);
                DOT8i_5(M1.s0, f02, V_block0, 8 + c_inner);
                DOT8i_5(M1.s1, f02, V_block0, 10 + c_inner);

                DOT8i_5(M2.s0, f02, V_block0, 12 + c_inner);
                DOT8i_5(M2.s1, f02, V_block0, 14 + c_inner);
                DOT8i_5(M3.s0, f02, V_block1, 0 + c_inner);
                DOT8i_5(M3.s1, f02, V_block1, 2 + c_inner);

                DOT8i_5(M4.s0, f02, V_block1, 4 + c_inner);
                DOT8i_5(M4.s1, f02, V_block1, 6 + c_inner);
                DOT8i_5(M5.s0, f02, V_block1, 8 + c_inner);
                DOT8i_5(M5.s1, f02, V_block1, 10 + c_inner);

                DOT8i_5(M6.s0, f02, V_block1, 12 + c_inner);
                DOT8i_5(M6.s1, f02, V_block1, 14 + c_inner);

                // f2[c_inner] x v[2 .. 16]
                DOT8i_6(M0.s0, f02, V_block0, 4 + c_inner);
                DOT8i_6(M0.s1, f02, V_block0, 6 + c_inner);
                DOT8i_6(M1.s0, f02, V_block0, 8 + c_inner);
                DOT8i_6(M1.s1, f02, V_block0, 10 + c_inner);

                DOT8i_6(M2.s0, f02, V_block0, 12 + c_inner);
                DOT8i_6(M2.s1, f02, V_block0, 14 + c_inner);
                DOT8i_6(M3.s0, f02, V_block1, 0 + c_inner);
                DOT8i_6(M3.s1, f02, V_block1, 2 + c_inner);

                DOT8i_6(M4.s0, f02, V_block1, 4 + c_inner);
                DOT8i_6(M4.s1, f02, V_block1, 6 + c_inner);
                DOT8i_6(M5.s0, f02, V_block1, 8 + c_inner);
                DOT8i_6(M5.s1, f02, V_block1, 10 + c_inner);

                DOT8i_6(M6.s0, f02, V_block1, 12 + c_inner);
                DOT8i_6(M6.s1, f02, V_block1, 14 + c_inner);

                // f2[c_inner] x v[2 .. 16]
                DOT8i_7(M0.s0, f02, V_block0, 4 + c_inner);
                DOT8i_7(M0.s1, f02, V_block0, 6 + c_inner);
                DOT8i_7(M1.s0, f02, V_block0, 8 + c_inner);
                DOT8i_7(M1.s1, f02, V_block0, 10 + c_inner);

                DOT8i_7(M2.s0, f02, V_block0, 12 + c_inner);
                DOT8i_7(M2.s1, f02, V_block0, 14 + c_inner);
                DOT8i_7(M3.s0, f02, V_block1, 0 + c_inner);
                DOT8i_7(M3.s1, f02, V_block1, 2 + c_inner);

                DOT8i_7(M4.s0, f02, V_block1, 4 + c_inner);
                DOT8i_7(M4.s1, f02, V_block1, 6 + c_inner);
                DOT8i_7(M5.s0, f02, V_block1, 8 + c_inner);
                DOT8i_7(M5.s1, f02, V_block1, 10 + c_inner);

                DOT8i_7(M6.s0, f02, V_block1, 12 + c_inner);
                DOT8i_7(M6.s1, f02, V_block1, 14 + c_inner);
            }
            U += U_off(0, 2 * COMP_BLOCK, 0, 0)
                    - 2 * U_off(0, COMP_BLOCK, 0, 0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store intermediate output tile to SLM.
    {
        __local OUT_BLOCK_DATA_T *M_write
                = (__local OUT_BLOCK_DATA_T *)&V[M_off(0, ly, 0, 4)];
        M_write += M_off(lx, 0, 0, OUT_TYPE_BLOCK);

        M_write[M_off(0, 0, 0, OUT_TYPE_BLOCK)] = M0;
        M_write[M_off(0, 0, 2, OUT_TYPE_BLOCK)] = M1;
        M_write[M_off(0, 0, 4, OUT_TYPE_BLOCK)] = M2;
        M_write[M_off(0, 0, 6, OUT_TYPE_BLOCK)] = M3;
        M_write[M_off(0, 0, 8, OUT_TYPE_BLOCK)] = M4;
        M_write[M_off(0, 0, 10, OUT_TYPE_BLOCK)] = M5;
        M_write[M_off(0, 0, 12, OUT_TYPE_BLOCK)] = M6;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Transform and store final oc16oh6ow14 output tile.
    if (ly < 7) {
        // Load multiplies from SLM.
        __local const OUT_BLOCK_DATA_T *M_read
                = (__local OUT_BLOCK_DATA_T *)&V[M_off(0, 0, ly * 2, 4)];
        M_read += M_off(lx, 0, 0, OUT_TYPE_BLOCK);

        OUT_BLOCK_DATA_T M0 = M_read[M_off(0, 0, 0, OUT_TYPE_BLOCK)];
        OUT_BLOCK_DATA_T M1 = M_read[M_off(0, 1, 0, OUT_TYPE_BLOCK)];
        OUT_BLOCK_DATA_T M2 = M_read[M_off(0, 2, 0, OUT_TYPE_BLOCK)];
        OUT_BLOCK_DATA_T M3 = M_read[M_off(0, 3, 0, OUT_TYPE_BLOCK)];
        OUT_BLOCK_DATA_T M4 = M_read[M_off(0, 4, 0, OUT_TYPE_BLOCK)];
        OUT_BLOCK_DATA_T M5 = M_read[M_off(0, 5, 0, OUT_TYPE_BLOCK)];
        OUT_BLOCK_DATA_T M6 = M_read[M_off(0, 6, 0, OUT_TYPE_BLOCK)];
        OUT_BLOCK_DATA_T M7 = M_read[M_off(0, 7, 0, OUT_TYPE_BLOCK)];

        // Inverse Transform.
        OUT_BLOCK_DATA_T x0 = M1 + M2;
        OUT_BLOCK_DATA_T x1 = M1 - M2;

        OUT_BLOCK_DATA_T x2 = M3 + M4;
        OUT_BLOCK_DATA_T x3 = M3 - M4;

        OUT_BLOCK_DATA_T x4 = M5 + M6;
        OUT_BLOCK_DATA_T x5 = M5 - M6;

        OUT_BLOCK_DATA_T C0 = M0 + x0 + x2 + x4;
        OUT_BLOCK_DATA_T C1 = x1 + TO_TYPE(2) * x3 + TO_TYPE(0.5f) * x5;
        OUT_BLOCK_DATA_T C2 = x0 + TO_TYPE(4.f) * x2 + TO_TYPE(0.25f) * x4;
        OUT_BLOCK_DATA_T C3 = x1 + TO_TYPE(8.f) * x3 + TO_TYPE(0.125f) * x5;
        OUT_BLOCK_DATA_T C4 = x0 + TO_TYPE(16.f) * x2 + TO_TYPE(0.0625f) * x4;
        OUT_BLOCK_DATA_T C5
                = x1 + TO_TYPE(32.f) * x3 + TO_TYPE(0.03125f) * x5 + M7;

        C0 = C0 * scl;
        C1 = C1 * scl;
        C2 = C2 * scl;
        C3 = C3 * scl;
        C4 = C4 * scl;
        C5 = C5 * scl;

        // Write data
        int dst_idx = dst_off(mb, oc, 0, oh, ow);
        const int w_size = dst_off(0, 0, 0, 0, 1);
        const int h_size = dst_off(0, 0, 0, 1, 0);

        const bool ow0 = OW % OW_BLOCK == 0 || ow < OW;
        const bool ow1 = OW % OW_BLOCK == 0 || ow + 1 < OW;

        const bool oh1 = OH % OH_BLOCK == 0 || oh + 1 < OH;
        const bool oh2 = OH % OH_BLOCK == 0 || oh + 2 < OH;
        const bool oh3 = OH % OH_BLOCK == 0 || oh + 3 < OH;
        const bool oh4 = OH % OH_BLOCK == 0 || oh + 4 < OH;
        const bool oh5 = OH % OH_BLOCK == 0 || oh + 5 < OH;

        if (WITH_BIAS || WITH_POST_OP) {
            const int c_size = WINO_M * OUT_TYPE_BLOCK;
            DATA_T C[c_size];
            OUT_BLOCK_WRITE(C0, &C[0 * OUT_TYPE_BLOCK]);
            OUT_BLOCK_WRITE(C1, &C[1 * OUT_TYPE_BLOCK]);
            OUT_BLOCK_WRITE(C2, &C[2 * OUT_TYPE_BLOCK]);
            OUT_BLOCK_WRITE(C3, &C[3 * OUT_TYPE_BLOCK]);
            OUT_BLOCK_WRITE(C4, &C[4 * OUT_TYPE_BLOCK]);
            OUT_BLOCK_WRITE(C5, &C[5 * OUT_TYPE_BLOCK]);
            if (WITH_BIAS) {
                for (int oh_block = 0; oh_block < WINO_M; oh_block++) {
                    for (int ow_block = 0; ow_block < OUT_TYPE_BLOCK;
                            ow_block++) {
                        const int c_off = oh_block * OUT_TYPE_BLOCK + ow_block;
                        C[c_off] += (OC_WO_PADDING % OC_BLOCK == 0
                                            || oc < OC_WO_PADDING)
                                ? bias[oc]
                                : DATA_ZERO;
                    }
                }
            }

            DATA_T S[c_size];
            if (WITH_SUM) {
                for (int oh_block = 0; oh_block < WINO_M; oh_block++) {
                    bool valid_oh = OH % OH_BLOCK == 0 || oh + oh_block < OH;
                    for (int ow_block = 0; ow_block < OUT_TYPE_BLOCK;
                            ow_block++) {
                        const int s_off = oh_block * OUT_TYPE_BLOCK + ow_block;
                        const int dst_off = dst_idx + oh_block * h_size
                                + ow_block * w_size;
                        bool valid_ow
                                = OW % OW_BLOCK == 0 || ow + ow_block < OW;
                        S[s_off] = valid_oh && valid_ow
                                ? dst[dst_idx + oh_block * h_size
                                        + ow_block * w_size]
                                : 0;
                    }
                }
            }

            for (int didx = 0; didx < c_size; ++didx) {
                float accum = CONVERT_FLOAT_T(C[didx]);
                float sum = CONVERT_FLOAT_T(S[didx]);
                int po_oc = oc;

                APPLY_POST_OPS_SERIAL_BINARY_2D(
                        C, DATA_T, S, DATA_T, mb, 1, po_oc, 1);
                C[didx] = TO_DATA_T(accum);
            }
            C0 = OUT_BLOCK_READ(&C[0 * OUT_TYPE_BLOCK]);
            C1 = OUT_BLOCK_READ(&C[1 * OUT_TYPE_BLOCK]);
            C2 = OUT_BLOCK_READ(&C[2 * OUT_TYPE_BLOCK]);
            C3 = OUT_BLOCK_READ(&C[3 * OUT_TYPE_BLOCK]);
            C4 = OUT_BLOCK_READ(&C[4 * OUT_TYPE_BLOCK]);
            C5 = OUT_BLOCK_READ(&C[5 * OUT_TYPE_BLOCK]);
        }

        if (ow0) dst[dst_idx + 0 * h_size + 0 * w_size] = C0.s0;
        if (ow1) dst[dst_idx + 0 * h_size + 1 * w_size] = C0.s1;

        if (oh1) {
            if (ow0) dst[dst_idx + 1 * h_size + 0 * w_size] = C1.s0;
            if (ow1) dst[dst_idx + 1 * h_size + 1 * w_size] = C1.s1;
        }

        if (oh2) {
            if (ow0) dst[dst_idx + 2 * h_size + 0 * w_size] = C2.s0;
            if (ow1) dst[dst_idx + 2 * h_size + 1 * w_size] = C2.s1;
        }

        if (oh3) {
            if (ow0) dst[dst_idx + 3 * h_size + 0 * w_size] = C3.s0;
            if (ow1) dst[dst_idx + 3 * h_size + 1 * w_size] = C3.s1;
        }

        if (oh4) {
            if (ow0) dst[dst_idx + 4 * h_size + 0 * w_size] = C4.s0;
            if (ow1) dst[dst_idx + 4 * h_size + 1 * w_size] = C4.s1;
        }

        if (oh5) {
            if (ow0) dst[dst_idx + 5 * h_size + 0 * w_size] = C5.s0;
            if (ow1) dst[dst_idx + 5 * h_size + 1 * w_size] = C5.s1;
        }
    }
}
