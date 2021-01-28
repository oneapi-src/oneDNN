/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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
#include "gpu/ocl/offsets.h"

#define IW_BLOCK (OW_BLOCK + KW - 1)

#define WINO_D (WINO_M + WINO_R - 1)

#define TO_TYPE(value) ((DATA_T)value)

#define UTRANS_BLOCK VECT_DT_N
#define UTRANS_DATA_T VECT_DATA_T
#define AS_UTRANS_DATA_T AS_VECT_DATA_T
#define UTRANS_BLOCK_READ(ptr) \
    AS_UTRANS_DATA_T(VECT_BLOCK_READ((const __global BLOCK_DATA_T *)ptr))
#define UTRANS_BLOCK_WRITE(data, ptr) \
    VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)ptr, AS_VECT_BLOCK_DATA_T(data))

#define VTRANS_BLOCK 4 // = (WINO_IC_BLOCK / (LWS_0 * LWS_1 / WINO_IW_BLOCK))
#define VTRANS_DATA_T CONCAT2(DATA_T, VTRANS_BLOCK)

#define COMP_BLOCK VECT_DT_N
#define COMP_DATA_T VECT_DATA_T
#define AS_COMP_DATA_T AS_VECT_DATA_T
#define COMP_READ(ptr) CONCAT2(vload, COMP_BLOCK)(0, ptr)
#define COMP_WRITE(data, ptr) CONCAT2(vstore, COMP_BLOCK)(data, 0, ptr)
#define COMP_BLOCK_READ(ptr) \
    AS_COMP_DATA_T(VECT_BLOCK_READ((const __global VECT_BLOCK_DATA_T *)ptr))

#define COMP_UNROLL (IC_BLOCK / COMP_BLOCK)

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

    //V data format is 2C8h16w16c (when IW_BLOCK = 16)
    const int ic_internal_block = 16;

    int icb = i / ic_internal_block;
    int ic = i % ic_internal_block;
    int off = icb * WINO_D * IW_BLOCK * ic_internal_block;
    off += z * IW_BLOCK * ic_internal_block;
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

static inline int get_Vtrans_ic0(int lx, int ly) {
    return VTRANS_BLOCK * (lx % 8);
}
static inline int get_Vtrans_ih0(int lx, int ly) {
    // Must be zero (without wino tile blocking) to perform the V transform
    // since the transformation uses a linear combination of the height values;
    return 0;
}
static inline int get_Vtrans_iw0(int lx, int ly) {
    return 8 * (lx / 8) + ly;
}

static inline int get_Vcomp_ic0(int lx, int ly) {
    return 8 * (lx % 2);
}
static inline int get_Vcomp_ih0(int lx, int ly) {
    // Relies on the fact that WINO_D = 8 to get full utilization of the local
    // workgroup.
    return ly;
}
static inline int get_Vcomp_iw0(int lx, int ly) {
    return lx / 2;
}

static inline int get_Ucomp_ic0(int lx, int ly) {
    // Must be zero as M is accumulated with product over ic. Could be
    // parallelized for blocking if a reduction over M is implemented.
    return 0;
}
static inline int get_Ucomp_oc0(int lx, int ly) {
    return lx;
}
static inline int get_Ucomp_kh0(int lx, int ly) {
    // Relies on the fact that WINO_D = 8
    return get_Vcomp_ih0(lx, ly);
}
static inline int get_Ucomp_kw0(int lx, int ly) {
    //Must be zero as product of kw is accumulated into M. Could be parallelized
    //if a reduction over M is implemented.
    return 0;
}

static inline int get_Mcomp_oc0(int lx, int ly) {
    return get_Ucomp_oc0(lx, ly);
}
static inline int get_Mcomp_oh0(int lx, int ly) {
    // Relies on the fact that WINO_D = 8
    return get_Vcomp_ih0(lx, ly);
}
static inline int get_Mcomp_ow0(int lx, int ly) {
    return 0;
}

static inline int get_out_oh0(int lx, int ly) {
    // Must be zero (without wino tile blocking) to perform the dst transform
    // since the transformation uses a linear combination of the height value;
    return 0;
}
static inline int get_out_ow0(int lx, int ly) {
    return OUT_TYPE_BLOCK * ly;
}
static inline int get_out_oc0(int lx, int ly) {
    return lx;
}

#if WINO_M == 6
static inline void wino_U_transform(
        UTRANS_DATA_T U[WINO_D], UTRANS_DATA_T wei[WINO_R]) {
    U[0] = wei[0];
    U[1] = TO_TYPE(-2.0 / 9) * (wei[0] + wei[1] + wei[2]);
    U[2] = TO_TYPE(2.0 / 9) * (-wei[0] + wei[1] - wei[2]);
    U[3] = TO_TYPE(1.0 / 90) * wei[0] + TO_TYPE(2.0 / 90) * wei[1]
            + TO_TYPE(4.0 / 90) * wei[2];
    U[4] = TO_TYPE(1.0 / 90) * wei[0] - TO_TYPE(2.0 / 90) * wei[1]
            + TO_TYPE(4.0 / 90) * wei[2];
    U[5] = TO_TYPE(64.0 / 90) * wei[0] + TO_TYPE(32.0 / 90) * wei[1]
            + TO_TYPE(16.0 / 90) * wei[2];
    U[6] = TO_TYPE(64.0 / 90) * wei[0] - TO_TYPE(32.0 / 90) * wei[1]
            + TO_TYPE(16.0 / 90) * wei[2];
    U[7] = wei[2];
}

// The API on this function is different from the other transform functions
// because interleaving the transform with writing the data out gives a small
// performance boost
static inline void wino_V_transform(
        __local VTRANS_DATA_T *V, const VTRANS_DATA_T src[WINO_D]) {
    // Compute Winograd f6x3 data transform and store components in SLM.
    V[V_off(0, 0, 0, VTRANS_BLOCK)]
            = src[0] - TO_TYPE(5.25) * src[2] + TO_TYPE(5.25) * src[4] - src[6];

    VTRANS_DATA_T x0 = src[1] - TO_TYPE(4.25) * src[3] + src[5];
    VTRANS_DATA_T x1 = src[2] - TO_TYPE(4.25) * src[4] + src[6];

    V[V_off(0, 1, 0, VTRANS_BLOCK)] = x1 + x0;
    V[V_off(0, 2, 0, VTRANS_BLOCK)] = x1 - x0;

    VTRANS_DATA_T x2 = TO_TYPE(-5) * src[3] + src[1];
    VTRANS_DATA_T x3 = TO_TYPE(4) * src[5] + x2;
    VTRANS_DATA_T x4 = TO_TYPE(0.25) * src[2] + src[6];
    VTRANS_DATA_T x5 = TO_TYPE(-1.25) * src[4] + x4;

    V[V_off(0, 3, 0, VTRANS_BLOCK)] = TO_TYPE(0.5) * x3 + x5;
    V[V_off(0, 4, 0, VTRANS_BLOCK)] = TO_TYPE(-0.5) * x3 + x5;

    VTRANS_DATA_T x6 = TO_TYPE(4) * src[1] + src[5];
    VTRANS_DATA_T x7 = TO_TYPE(-5) * src[3] + x6;
    VTRANS_DATA_T x8 = TO_TYPE(4) * src[2] + src[6];
    VTRANS_DATA_T x9 = TO_TYPE(-5) * src[4] + x8;

    V[V_off(0, 5, 0, VTRANS_BLOCK)] = TO_TYPE(+0.5) * x7 + x9;
    V[V_off(0, 6, 0, VTRANS_BLOCK)] = TO_TYPE(-0.5) * x7 + x9;

    V[V_off(0, 7, 0, VTRANS_BLOCK)] = -src[1] + TO_TYPE(5.25) * src[3]
            - TO_TYPE(5.25) * src[5] + src[7];
}
static inline void wino_m_transform(
        OUT_BLOCK_DATA_T C[WINO_M], OUT_BLOCK_DATA_T M[WINO_D]) {
    // Inverse Transform.
    OUT_BLOCK_DATA_T x0 = M[1] + M[2];
    OUT_BLOCK_DATA_T x1 = M[1] - M[2];

    OUT_BLOCK_DATA_T x2 = M[3] + M[4];
    OUT_BLOCK_DATA_T x3 = M[3] - M[4];

    OUT_BLOCK_DATA_T x4 = M[5] + M[6];
    OUT_BLOCK_DATA_T x5 = M[5] - M[6];

    C[0] = M[0] + x0 + x2 + x4;
    C[1] = x1 + TO_TYPE(2) * x3 + TO_TYPE(0.5f) * x5;
    C[2] = x0 + TO_TYPE(4.f) * x2 + TO_TYPE(0.25f) * x4;
    C[3] = x1 + TO_TYPE(8.f) * x3 + TO_TYPE(0.125f) * x5;
    C[4] = x0 + TO_TYPE(16.f) * x2 + TO_TYPE(0.0625f) * x4;
    C[5] = x1 + TO_TYPE(32.f) * x3 + TO_TYPE(0.03125f) * x5 + M[7];
}
#elif WINO_M == 4
static inline void wino_U_transform(
        UTRANS_DATA_T U[WINO_D], UTRANS_DATA_T wei[WINO_R]) {
    U[0] = wei[0] / 4;
    U[1] = (wei[0] + wei[1] + wei[2]) / (-6);
    U[2] = (wei[0] - wei[1] + wei[2]) / (-6);
    U[3] = (wei[0] + 2 * wei[1] + 4 * wei[2]) / 24;
    U[4] = (wei[0] - 2 * wei[1] + 4 * wei[2]) / 24;
    U[5] = wei[2];
}

static inline void wino_V_transform(
        __local VTRANS_DATA_T *V, const VTRANS_DATA_T src[WINO_D]) {
    // Compute Winograd f4x3 data transform and store components in SLM.
    V[V_off(0, 0, 0, VTRANS_BLOCK)] = 4 * src[0] - 5 * src[2] + src[4];
    V[V_off(0, 1, 0, VTRANS_BLOCK)] = -4 * (src[1] + src[2]) + src[3] + src[4];
    V[V_off(0, 2, 0, VTRANS_BLOCK)] = 4 * (src[1] - src[2]) - src[3] + src[4];
    V[V_off(0, 3, 0, VTRANS_BLOCK)]
            = -2 * src[1] - src[2] + 2 * src[3] + src[4];
    V[V_off(0, 4, 0, VTRANS_BLOCK)] = 2 * src[1] - src[2] - 2 * src[3] + src[4];
    V[V_off(0, 5, 0, VTRANS_BLOCK)] = 4 * src[1] - 5 * src[3] + src[5];
}

static inline void wino_m_transform(
        OUT_BLOCK_DATA_T C[WINO_M], OUT_BLOCK_DATA_T M[WINO_D]) {
    OUT_BLOCK_DATA_T x0 = M[1] + M[2];
    OUT_BLOCK_DATA_T x1 = M[1] - M[2];
    OUT_BLOCK_DATA_T x2 = M[3] + M[4];
    OUT_BLOCK_DATA_T x3 = M[3] - M[4];

    C[0] = M[0] + x0 + x2;
    C[1] = x1 + 2 * x3;
    C[2] = x0 + 4 * x2;
    C[3] = x1 + 8 * x3 + M[5];
}
#elif WINO_M == 2
static inline void wino_U_transform(
        UTRANS_DATA_T U[WINO_D], UTRANS_DATA_T wei[WINO_R]) {
    U[0] = wei[0];
    U[1] = (wei[0] + wei[1] + wei[2]) / 2;
    U[2] = (wei[0] - wei[1] + wei[2]) / 2;
    U[3] = wei[2];
}

static inline void wino_V_transform(
        __local VTRANS_DATA_T *V, const VTRANS_DATA_T src[WINO_D]) {
    // Compute Winograd f2x3 data transform and store components in SLM.
    V[V_off(0, 0, 0, VTRANS_BLOCK)] = src[0] - src[2];
    V[V_off(0, 1, 0, VTRANS_BLOCK)] = src[1] + src[2];
    V[V_off(0, 2, 0, VTRANS_BLOCK)] = -src[1] + src[2];
    V[V_off(0, 3, 0, VTRANS_BLOCK)] = src[1] - src[3];
}

static inline void wino_m_transform(
        OUT_BLOCK_DATA_T C[WINO_M], OUT_BLOCK_DATA_T M[WINO_D]) {
    C[0] = M[0] + M[1] + M[2];
    C[1] = M[1] - M[2] - M[3];
}
#else
#error "Unsupported Winograd Tile Size"
#endif

__attribute__((reqd_work_group_size(OC_BLOCK, 1, 1)))
__attribute__((intel_reqd_sub_group_size(OC_BLOCK))) __kernel void
gen9_wino_wei_transform(__global DATA_T *U, const __global DATA_T *weights) {
    const uint weights_tile_width = 1;
    const uint weights_tile_height = WINO_M;
    const uint in_kw = get_global_id(1) * weights_tile_width;
    const uint in_kh = get_global_id(2) * weights_tile_height;

    const uint U_tile_width = 1;
    const uint U_tile_height = WINO_D;

    const uint out_kw = get_global_id(1) * U_tile_width;
    const uint out_kh = get_global_id(2) * U_tile_height;
    const uint oc0 = (get_group_id(0) % (WINO_OC / OC_BLOCK)) * OC_BLOCK;
    const uint ic = (get_group_id(0) / (WINO_OC / OC_BLOCK)) * UTRANS_BLOCK;

    uint in_idx = wei_off(0, oc0, ic, 0, in_kh, in_kw);
    bool is_valid = ic < IC && oc0 < OC;

    UTRANS_DATA_T g[WINO_R];
    for (int i = 0; i < WINO_R; i++) {
        g[i] = is_valid ? UTRANS_BLOCK_READ(&weights[in_idx]) : 0;
        in_idx += wei_off(0, 0, 0, 0, 1, 0);
    }

    UTRANS_DATA_T out_tile[WINO_D];
    wino_U_transform(out_tile, g);

    uint out_idx = U_off(oc0, ic, out_kh, out_kw);

    unroll_for(int i = 0; i < WINO_D; i++) {
        UTRANS_BLOCK_WRITE(out_tile[i], &U[out_idx]);
        out_idx += U_off(0, 0, 1, 0);
    }
}

#define DOTi(_result, _A, _B) \
    { _result = mad(_A, _B, _result); }

__attribute__((reqd_work_group_size(16, 8, 1)))
__attribute__((intel_reqd_sub_group_size(16))) __kernel void
gen9_wino_conv_fwd(__global DATA_T *dst, const __global DATA_T *src,
        const __global DATA_T *U_param,
        const __global DATA_T *bias POST_OP_ARGS) {
    const uint slm_size = (WINO_IC_BLOCK * WINO_D * IW_BLOCK) / VTRANS_BLOCK;
    __local VTRANS_DATA_T V[slm_size]; // 8 KB

    const DATA_T scl = TO_TYPE(16);
    const DATA_T sc = TO_TYPE(1) / scl;
    const VTRANS_DATA_T scl_vec = (VTRANS_DATA_T)(sc, sc, sc, sc);

    const int ow0 = get_group_id(0) * OW_BLOCK;
    const int oh0 = get_group_id(1) * OH_BLOCK;
    const int gid2 = get_group_id(2);
    const int oc0 = (gid2 % (OC / OC_BLOCK)) * OC_BLOCK;
    const int mb = gid2 / (OC / OC_BLOCK);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // Load ic32ih'WINO_D'iw'IW_BLOCK' input tile, with 2 pixel overlap in ih
    // and iw. Compute oc'OC_BLOCK'oh'WINO_M'ow'OW_BLOCK' output tile.

    // Initialize variables to accumulate intermediate output tile
    const int M_size = OW_BLOCK;
    DATA_T M[M_size];

    for (int i = 0; i < M_size; i++) {
        M[i] = 0;
    }

    // Computation is separated into three main stages, load/transform input,
    // compute intermediate output block, and transform/store final output.
    // Between these stages, the dimensions handled by local work groups
    // changes.

    // Buffers used to load and transform ic32ih'WINO_D'iw16 src tile into V
    // Each local thread transforms a block with dimensions c4h8w1
    const int Vtrans_ic = get_Vtrans_ic0(lx, ly);
    const int Vtrans_ih = get_Vtrans_ih0(lx, ly);
    const int Vtrans_iw = get_Vtrans_iw0(lx, ly);
    const int src_ic = Vtrans_ic;
    const int src_ih = oh0 - PH + Vtrans_ih;
    const int src_iw = ow0 - PW + Vtrans_iw;
    const __global DATA_T *src_load
            = src + src_off(mb, src_ic, 0, src_ih, src_iw);
    const int V_write_idx
            = V_off(Vtrans_ic, Vtrans_ih, Vtrans_iw, VTRANS_BLOCK);
    __local VTRANS_DATA_T *V_write = &V[V_write_idx];

    // Buffers used to compute oc'OC_BLOCK'oh'WINO_D'ow'OW_BLOCK' intermediate
    // output tile. Each local thread transforms a block with dimensions
    // c1h1w`OW_BLOCK`.
    const int U_oc = oc0 + get_Ucomp_oc0(lx, ly);
    const int U_ic = get_Ucomp_ic0(lx, ly);
    const int U_kh = get_Ucomp_kh0(lx, ly);
    const int U_kw = get_Ucomp_kw0(lx, ly);
    const __global DATA_T *U = U_param + U_off(U_oc, U_ic, U_kh, U_kw);
    const int Vcomp_ic = get_Vcomp_ic0(lx, ly);
    const int Vcomp_ih = get_Vcomp_ih0(lx, ly);
    const int Vcomp_iw = get_Vcomp_iw0(lx, ly);
    const int V_read_idx = V_off(Vcomp_ic, Vcomp_ih, Vcomp_iw, VTRANS_BLOCK);
    __local const COMP_DATA_T *V_read
            = (__local const COMP_DATA_T *)&V[V_read_idx];

    __attribute__((opencl_unroll_hint(1))) for (uint c = 0; c < IC;
                                                c += WINO_IC_BLOCK) {
        // Load and transform ic32ih'WINO_D'iw'IW_BLOCK' src tile into V
        if (IW_BLOCK == 16 || Vtrans_iw < IW_BLOCK) {
            bool x_in = 0 <= src_iw && src_iw < IW && src_ic + c < IC;
            VTRANS_DATA_T src[WINO_D];
            for (int index = 0; index < WINO_D; index++) {
                bool y_in = 0 <= (src_ih + index) && (src_ih + index) < IH
                        && x_in;
                src[index] = y_in ? *((const __global VTRANS_DATA_T *)(src_load
                                     + src_off(0, 0, 0, index, 0)))
                                  : 0;

                //Scale input to prevent intermediate computations overflow in
                //some cases, output is adjusted with the same scale factor
                //after main computation
                src[index] = src[index] * scl_vec;
            }
            wino_V_transform(V_write, src);
        }

        src_load += src_off(0, WINO_IC_BLOCK, 0, 0, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate oc'OC_BLOCK'oh'WINO_D'ow'OW_BLOCK' intermediate output
        // tile stored in the M_i
        __local const COMP_DATA_T *V_read_outer = V_read;

        const int outer_c_blocking = COMP_UNROLL * COMP_BLOCK;
        __attribute__((opencl_unroll_hint(
                1))) for (uint c_outer = 0; c_outer < WINO_IC_BLOCK
                          && (WINO_D == 8 || ly < WINO_D);
                          c_outer += outer_c_blocking) {
            // Fetch input components, spread across subgroup.
            DATA_T V_block[outer_c_blocking];

            // Blocking/Stride parameters for how elements are loaded from V
            // into V_block
            const int c_block = 8;
            const int w_count = outer_c_blocking / c_block;
            const int w_stride = outer_c_blocking / w_count;

            unroll_for(int w_load = 0; w_load < w_count; w_load++) {
                unroll_for(int c_load = 0; c_load < c_block;
                           c_load += COMP_BLOCK) {
                    COMP_WRITE(V_read_outer[V_off(c_load, 0, w_load * w_stride,
                                       COMP_BLOCK)],
                            &V_block[w_load * c_block + c_load]);
                }
            }
            V_read_outer += V_off(outer_c_blocking, 0, 0, COMP_BLOCK);

#define V_BLOCK(_ic, _iw) \
    sub_group_broadcast( \
            V_block[(_ic) % c_block + c_block * ((_iw) / w_stride)], \
            (IC_BLOCK / c_block) * ((_iw) % w_stride) + ((_ic) / c_block))

            unroll_for(int c_inner = 0; c_inner < outer_c_blocking;
                       c_inner += COMP_BLOCK) {
                unroll_for(int kw_in = 0; kw_in < KW; kw_in++) {
                    const COMP_DATA_T f0
                            = COMP_BLOCK_READ(&U[U_off(0, 0, 0, kw_in)]);

                    unroll_for(int c_in = 0; c_in < COMP_BLOCK; c_in++) {
                        unroll_for(int ow_in = 0; ow_in < OW_BLOCK; ow_in++) {
                            DOTi(M[ow_in], f0[c_in],
                                    V_BLOCK(c_in + c_inner, kw_in + ow_in));
                        }
                    }
                }

                U += U_off(0, COMP_BLOCK, 0, 0);
            }
            U += U_off(0, COMP_UNROLL * COMP_BLOCK, 0, 0)
                    - COMP_UNROLL * U_off(0, COMP_BLOCK, 0, 0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store intermediate output tile to SLM.
    {
        const int M_oc = get_Mcomp_oc0(lx, ly);
        const int M_oh = get_Mcomp_oh0(lx, ly);
        const int M_ow = get_Mcomp_ow0(lx, ly);
        __local DATA_T *M_write = (__local DATA_T *)&V[M_off(0, M_oh, 0, 4)];
        M_write += M_off(M_oc, 0, 0, 1);

        for (int i = 0; i < M_size; i++) {
            M_write[M_off(0, 0, M_ow + i, 1)] = M[i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Transform and store final oc'OC_BLOCK'oh'WINO_M'ow'OW_BLOCK' output tile.
    // Each local thread transforms a block with dimensions
    // c1h`WINO_D`w`OUT_TYPE_BLOCK` to the final output with dimensions
    // c1h`WINO_M`w`OUT_TYPE_BLOCK`.
    if (get_out_ow0(lx, ly) < OW_BLOCK) {
        // Load multiplies from SLM.
        const int M_oc = get_out_oc0(lx, ly);
        const int M_oh = get_out_oh0(lx, ly);
        const int M_ow = get_out_ow0(lx, ly);
        __local const OUT_BLOCK_DATA_T *M_read
                = (__local OUT_BLOCK_DATA_T *)&V[M_off(0, 0, M_ow, 4)];
        M_read += M_off(M_oc, 0, 0, OUT_TYPE_BLOCK);

        OUT_BLOCK_DATA_T M[WINO_D];
        for (int i = 0; i < WINO_D; i++) {
            M[i] = M_read[M_off(0, M_oh + i, 0, OUT_TYPE_BLOCK)];
        }
        OUT_BLOCK_DATA_T C[WINO_M];
        DATA_T *C_dat = C;

        wino_m_transform(C, M);
        unroll_for(int i = 0; i < WINO_M; i++) { C[i] = C[i] * scl; }

        // Write data
        const int oc = oc0 + M_oc;
        const int ow = ow0 + M_ow;
        const int oh = oh0 + M_oh;
        int dst_idx = dst_off(mb, oc, 0, oh, ow);
        const int w_size = dst_off(0, 0, 0, 0, 1);
        const int h_size = dst_off(0, 0, 0, 1, 0);

        if (WITH_BIAS || WITH_POST_OP) {
            const int c_size = WINO_M * OUT_TYPE_BLOCK;
            if (WITH_BIAS) {
                for (int oh_block = 0; oh_block < WINO_M; oh_block++) {
                    for (int ow_block = 0; ow_block < OUT_TYPE_BLOCK;
                            ow_block++) {
                        const int c_off = oh_block * OUT_TYPE_BLOCK + ow_block;
                        C_dat[c_off] += (OC_WO_PADDING % OC_BLOCK == 0
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
                float accum = CONVERT_FLOAT_T(C_dat[didx]);
                float sum = CONVERT_FLOAT_T(S[didx]);
                int po_oc = oc;

                APPLY_POST_OPS_SERIAL_BINARY_2D(
                        C_dat, DATA_T, S, DATA_T, mb, 1, po_oc, 1);
                C_dat[didx] = TO_DATA_T(accum);
            }
        }

        unroll_for(int h_off = 0; h_off < WINO_M; h_off++) {
            if (h_off == 0 || OH % OH_BLOCK == 0 || oh + h_off < OH) {
                unroll_for(int w_off = 0; w_off < OUT_TYPE_BLOCK; w_off++) {
                    int c_off = 2 * h_off + w_off;
                    if (OW % OW_BLOCK == 0 || ow + w_off < OW)
                        dst[dst_idx + h_off * h_size + w_off * w_size]
                                = C_dat[c_off];
                }
            }
        }
    }
}
