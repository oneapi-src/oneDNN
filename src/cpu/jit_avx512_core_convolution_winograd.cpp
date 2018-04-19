/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifdef __INTEL_COMPILER
#include <immintrin.h>
#endif

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_common_convolution_winograd.hpp"
#include "jit_avx512_core_convolution_winograd.hpp"

#ifndef _MSC_VER
#define pragma_unroll _Pragma("unroll")
#else
#define pragma_unroll
#endif


namespace mkldnn {
namespace impl {
namespace cpu {

namespace {

void inline load_ps(float *dest, const float *src_mem) {
#ifdef __INTEL_COMPILER
    __m512 *Iv512 = (__m512 *)dest;
    Iv512[0] = _mm512_load_ps(src_mem);
#else
#pragma omp simd
    for (int v = 0; v < simd_w; v++) dest[v] = src_mem[v];
#endif
}

void inline store_output(float *dest, const float *data, bool streamout) {
#ifdef __INTEL_COMPILER
    if (streamout)
        _mm512_stream_ps(dest, *((__m512 *)data));
    else
        _mm512_store_ps(dest, *((__m512 *)data));
#else
#pragma omp simd
    for (int v = 0; v < simd_w; v++)
        dest[v] = data[v];
#endif
}

}

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <bool is_fwd>
void _jit_avx512_core_convolution_winograd_t<is_fwd>
::weight_transform_data(const jit_conv_winograd_conf_t &jcp,
        float *wp, float *twp)
{
    float G[] = {0.26890756302521f, 0.688403361344538f, 0.119514472455649f,
                 1.13777777777778f, 0.430252100840336f, 0.179271708683473f};
    const int kh = 3;
    const int kw = 3;
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[kh][kw][simd_w][simd_w];
    float T[alpha][3][simd_w];
    jit_wino_transform_call_s p = {0};

    p.src = wp;
    p.dst = twp;
    p.G = G;
    p.M = F;
    p.Mw = Fw;
    p.T = T;

    kernel_->weights_transform_data_ker(&p);
}

template<bool is_fwd>
void _jit_avx512_core_convolution_winograd_t<is_fwd>::output_transform_data
(int image, const jit_conv_winograd_conf_t &jcp,
    const post_ops_t &p_ops, float *toutp, float *pout_b, float *bias) {

    float G[] = {0.625f, 1.5f, 0.390625f, 2.25f, 0.244140625f, 3.375f};
    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];
    float T[tile_size][alpha][simd_w];

    jit_wino_transform_call_s p = {0};
    p.src = toutp;
    p.dst = pout_b;
    p.G = G;
    p.M = O;
    p.Mw = Ow;
    p.T = T;
    p.bias = bias;

    int tile_base_index = image * jcp.itiles * jcp.jtiles;
    int tile_block_ur = tile_base_index % jcp.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / jcp.tile_block_ur) % jcp.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / jcp.tile_block_ur) / jcp.nb_tile_block_ur;

    for (int tj = 0; tj < jcp.jtiles; tj++) {
        for (int ti = 0; ti < jcp.itiles; ti++) {

            p.tile_block_ur = tile_block_ur;
            p.nb_tile_block_ur = nb_tile_block_ur;
            p.tile_block = tile_block;
            p.tj = tj;
            p.ti = ti;

            kernel_->output_transform_data_ker(&p);

            tile_block_ur++;
            if (tile_block_ur >= jcp.tile_block_ur) {
                tile_block_ur = 0;
                nb_tile_block_ur++;
            }
            if (nb_tile_block_ur >= jcp.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }
}

template<bool is_fwd>
void _jit_avx512_core_convolution_winograd_t<is_fwd>
::output_transform_tileblock_data(int tile_block,
    const jit_conv_winograd_conf_t &jcp, const post_ops_t &p_ops,
    float *toutp, float *outp, float *bias) {

    float G[] = {0.625f, 1.5f, 0.390625f, 2.25f, 0.244140625f, 3.375f};
    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];
    float T[tile_size][alpha][simd_w];

    jit_wino_transform_call_s p = {0};
    p.src = toutp;
    p.dst = outp;
    p.G = G;
    p.M = O;
    p.Mw = Ow;
    p.T = T;
    p.bias = bias;

    int outw = is_fwd ? jcp.ow : jcp.iw;
    int outh = is_fwd ? jcp.oh : jcp.ih;

    int tile_index = tile_block * jcp.nb_tile_block_ur * jcp.tile_block_ur;

    for (int nb_tile_block_ur = 0;
        nb_tile_block_ur < jcp.nb_tile_block_ur;
        nb_tile_block_ur++) {

        for (int tile_block_ur = 0; tile_block_ur < jcp.tile_block_ur;
            tile_block_ur++) {
            int img = tile_index / (jcp.jtiles * jcp.itiles);
            int ti = tile_index % jcp.itiles;
            int tj = (tile_index / jcp.itiles) % jcp.jtiles;

            p.tile_block_ur = tile_block_ur;
            p.nb_tile_block_ur = nb_tile_block_ur;
            p.tile_block = tile_block;
            p.tj = tj;
            p.ti = ti;
            p.dst = outp + img * (jcp.dimM / jcp.dimM_simd_block)
                               * outh * outw * jcp.dimM_simd_block;

            kernel_->output_transform_data_ker(&p);

            tile_index++;
        }
    }
}


template<bool is_fwd>
void _jit_avx512_core_convolution_winograd_t<is_fwd>
    ::input_transform_data(int image, const jit_conv_winograd_conf_t &jcp,
        float *inp, float *tinp)
{
    float G[] = {-2.25f, -0.390625f, 0.87890625f, -2.640625f,
                 0.625f, -0.625f, 1.5f, -1.5f, -2.640625f};

    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];
    float T[alpha][alpha][simd_w];

    jit_wino_transform_call_s p = {0};

    p.src = inp;
    p.dst = tinp;
    p.G = G;
    p.M = I;
    p.Mw = Iw;
    p.T = T;

    int tile_base_index = image * jcp.itiles * jcp.jtiles;
    int tile_block_ur = tile_base_index % jcp.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / jcp.tile_block_ur) % jcp.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / jcp.tile_block_ur) / jcp.nb_tile_block_ur;

    for (int tj = 0; tj < jcp.jtiles; tj++) {
        for (int ti = 0; ti < jcp.itiles; ti++) {

            p.tile_block_ur = tile_block_ur;
            p.nb_tile_block_ur = nb_tile_block_ur;
            p.tile_block = tile_block;
            p.tj = tj;
            p.ti = ti;

            kernel_->input_transform_data_ker(&p);

            tile_block_ur++;
            if (tile_block_ur >= jcp.tile_block_ur) {
                tile_block_ur = 0;
                nb_tile_block_ur++;
            }
            if (nb_tile_block_ur >= jcp.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }
}

template <bool is_fwd>
void _jit_avx512_core_convolution_winograd_t<is_fwd>
    ::input_transform_tileblock_data(int tile_block,
        const jit_conv_winograd_conf_t &jcp,
        float *inp, float *tinp)
{
    float G[] = {-2.25f, -0.390625f, 0.87890625f, -2.640625f,
               0.625f, -0.625f, 1.5f, -1.5f, -2.640625f};
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];
    float T[alpha][alpha][simd_w];

    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int inpw = is_fwd ? jcp.iw : jcp.ow;

    array_offset_calculator<float, 5> input(inp,
        jcp.mb, jcp.dimK / simd_w, inph, inpw, simd_w);
    array_offset_calculator<float, 7> output(tinp,
        alpha, alpha,
        jcp.dimN_block, jcp.dimK_nb_block, jcp.dimK_block,
        jcp.dimN_reg_block, jcp.dimK_reg_block);

    jit_wino_transform_call_s p = {0};

    p.dst = tinp;
    p.G = G;
    p.M = I;
    p.Mw = Iw;
    p.T = T;


    int tile_index = tile_block * jcp.nb_tile_block_ur * jcp.tile_block_ur;

    for (int nb_tile_block_ur = 0;
            nb_tile_block_ur < jcp.nb_tile_block_ur;
            nb_tile_block_ur++) {

        for (int tile_block_ur = 0; tile_block_ur < jcp.tile_block_ur;
                tile_block_ur++) {

            int img = tile_index / (jcp.jtiles * jcp.itiles);
            int ti = tile_index % jcp.itiles;
            int tj = (tile_index / jcp.itiles) % jcp.jtiles;
            float *pinp_b = &(input(img, 0, 0, 0, 0));

            p.src = pinp_b;
            p.tile_block_ur = tile_block_ur;
            p.nb_tile_block_ur = nb_tile_block_ur;
            p.tj = tj;
            p.ti = ti;

            kernel_->input_transform_data_ker(&p);

            tile_index++;
        }
    }
}

template <bool is_fwd>
void _jit_avx512_core_convolution_winograd_t<is_fwd>::_execute_data_W_S_G_D(
        float *inp_ptr, float *out_ptr, float *wei_ptr, float *bias_ptr) {
    const auto &jcp = kernel_->jcp;
    const auto &p_ops = attr_->post_ops_;

    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int inpw = is_fwd ? jcp.iw : jcp.ow;
    const int outh = is_fwd ? jcp.oh : jcp.ih;
    const int outw = is_fwd ? jcp.ow : jcp.iw;

    /* Notation:
       FWD: dimM:oc, dimN:ntiles, dimK:ic,
       BWD: dimM:ic, dimN:ntiles, dimK:oc,
       FWD/BWD: V: src/diff_dst transform, U:weight transform,
                M:dst/diff_src transform  */
    array_offset_calculator<float, 5> input(inp_ptr,
            jcp.mb, jcp.dimK/jcp.dimK_reg_block, inph, inpw,
            jcp.dimK_reg_block);
    array_offset_calculator<float, 5> output(out_ptr,
            jcp.mb, jcp.dimM/jcp.dimM_simd_block, outh, outw,
            jcp.dimM_simd_block);
    array_offset_calculator<float, 6> weights(wei_ptr,
            jcp.oc/jcp.oc_simd_block, jcp.ic/jcp.ic_simd_block, jcp.kh, jcp.kw,
            jcp.ic_simd_block, jcp.oc_simd_block);
    array_offset_calculator<float, 2> bias(bias_ptr,
            jcp.dimM/jcp.dimM_simd_block, jcp.dimM_simd_block);

    array_offset_calculator<float, 8> M(
            (float *)((is_fwd
                    ? (this->scratchpad_)->M_ptr()
                    : (this->scratchpad_)->V_ptr())),
            jcp.dimN_nb_block, jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimN_block, jcp.dimM_block * jcp.dimM_reg_block,
            jcp.dimN_reg_block, jcp.dimM_simd_block);
    array_offset_calculator<float, 8> U((float *)((this->scratchpad_)->U_ptr()),
            jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimK_nb_block,
            jcp.dimM_block * jcp.dimM_reg_block, jcp.dimK_block,
            jcp.dimK_reg_block, jcp.dimM_simd_block);
    array_offset_calculator<float, 8> V(
            (float *)((is_fwd
                    ? (this->scratchpad_)->V_ptr()
                    : (this->scratchpad_)->M_ptr())),
            jcp.dimN_nb_block, alpha, alpha,
            jcp.dimN_block, jcp.dimK_nb_block,
            jcp.dimK_block, jcp.dimN_reg_block, jcp.dimK_reg_block);

#pragma omp parallel
    {
#pragma omp for nowait collapse(3)
        for (int img = 0; img < jcp.mb; img++){
            for (int K_blk1 = 0; K_blk1 < jcp.dimK_nb_block; K_blk1++){
                for (int K_blk2 = 0; K_blk2 < jcp.dimK_block; K_blk2++){

                    input_transform_data(img, jcp,
                        &(input(img, K_blk1 * jcp.dimK_block + K_blk2,
                                0, 0, 0)),
                        &(V(0, 0, 0, 0, K_blk1, K_blk2, 0, 0)));

                }
            }
        }

#pragma omp for nowait collapse(4) schedule(static)
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++){
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++){
                for (int ofm2 = 0; ofm2 < jcp.oc_block * jcp.oc_reg_block;
                     ofm2++){
                    for (int ifm2 = 0; ifm2 < jcp.ic_block * jcp.ic_reg_block;
                         ifm2++){
                        float *U_base_ptr = is_fwd
                        ? &(U(ofm1, 0, 0, ifm1, ofm2, ifm2, 0, 0))
                        : &(U(ifm1, 0, 0, ofm1, ifm2, ofm2, 0, 0));
                        weight_transform_data(jcp,
                            &(weights(
                                ofm1 * jcp.oc_block * jcp.oc_reg_block + ofm2,
                                ifm1 * jcp.ic_block * jcp.ic_reg_block + ifm2,
                                0, 0, 0, 0)),
                            U_base_ptr);
                    }
                }
            }
        }

#pragma omp barrier

#pragma omp for collapse(4) nowait schedule(static)
        for (int N_blk1 = 0; N_blk1 < jcp.dimN_nb_block; N_blk1++){
            for (int oj = 0; oj < alpha; oj++){
                for (int oi = 0; oi < alpha; oi++){
                    for (int M_blk1 = 0; M_blk1 < jcp.dimM_nb_block; M_blk1++){
                        for (int K_blk1 = 0; K_blk1 < jcp.dimK_nb_block;
                             K_blk1++)
                        for (int N_blk2 = 0; N_blk2 < jcp.dimN_block; N_blk2++)
                            kernel_->gemm_loop_ker(
                                    (float *)&(M(N_blk1, M_blk1, oj, oi,
                                        N_blk2, 0, 0, 0)),
                                    (const float *)&(U(M_blk1, oj, oi,
                                        K_blk1, 0, 0, 0, 0)),
                                    (const float *)&(V(N_blk1, oj, oi,
                                        N_blk2, K_blk1, 0, 0, 0)), K_blk1);
                    }
                }
            }
        }

#pragma omp barrier

#pragma omp for collapse(3)
        for (int img = 0; img < jcp.mb; img++){
            for (int M_blk1 = 0; M_blk1 < jcp.dimM_nb_block; M_blk1++){
                for (int M_blk2 = 0;
                     M_blk2 < jcp.dimM_block * jcp.dimM_reg_block; M_blk2++){
                      output_transform_data(img, jcp, p_ops,
                        &(M(0, M_blk1, 0, 0, 0, M_blk2, 0, 0)),
                        &(output(img,M_blk1 * jcp.dimM_block
                            * jcp.dimM_reg_block + M_blk2, 0, 0, 0)),
                        &(bias(M_blk1 * jcp.dimM_block * jcp.dimM_reg_block
                            + M_blk2, 0)));
                }
            }
        }
    }
}

template void
_jit_avx512_core_convolution_winograd_t<true>::_execute_data_W_S_G_D(
        float *, float *, float *, float *);
template void
_jit_avx512_core_convolution_winograd_t<false>::_execute_data_W_S_G_D(
        float *, float *, float *, float *);

template <bool is_fwd>
void _jit_avx512_core_convolution_winograd_t<is_fwd>::_execute_data_W_SGD(
        float *inp_ptr, float *out_ptr, float *wei_ptr, float *bias_ptr) {
    const auto &jcp = kernel_->jcp;
    const auto &p_ops = attr_->post_ops_;

    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int inpw = is_fwd ? jcp.iw : jcp.ow;
    const int outh = is_fwd ? jcp.oh : jcp.ih;
    const int outw = is_fwd ? jcp.ow : jcp.iw;

    array_offset_calculator<float, 5> input(inp_ptr,
        jcp.mb, jcp.dimK/jcp.dimK_reg_block, inph, inpw, jcp.dimK_reg_block);
    array_offset_calculator<float, 5> output(out_ptr,
        jcp.mb, jcp.dimM/jcp.dimM_simd_block, outh, outw, jcp.dimM_simd_block);
    array_offset_calculator<float, 6> weights(wei_ptr,
        jcp.oc/jcp.oc_simd_block, jcp.ic/jcp.ic_simd_block, jcp.kh, jcp.kw,
        jcp.ic_simd_block, jcp.oc_simd_block);
    array_offset_calculator<float, 2> bias(bias_ptr,
        jcp.oc/jcp.oc_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 8> U((float *)((this->scratchpad_)->U_ptr()),
            jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimK_nb_block,
            jcp.dimM_block  * jcp.dimM_reg_block, jcp.dimK_block,
            jcp.dimK_reg_block, jcp.dimM_simd_block);

    array_offset_calculator<float, 8> M(
            (float *)((is_fwd
                    ? (this->scratchpad_)->M_ptr()
                    : (this->scratchpad_)->V_ptr())),
            0, jcp.dimM_nb_block, alpha, alpha,
            jcp.dimN_block, jcp.dimM_block * jcp.dimM_reg_block,
            jcp.dimN_reg_block, jcp.dimM_simd_block);
    array_offset_calculator<float, 8> V(
            (float *)((is_fwd
                    ? (this->scratchpad_)->V_ptr()
                    : (this->scratchpad_)->M_ptr())),
            0, alpha, alpha, jcp.dimN_block,
            jcp.dimK_nb_block, jcp.dimK_block,
            jcp.dimN_reg_block, jcp.dimK_reg_block);


#pragma omp parallel
    {
#pragma omp for collapse(4) schedule(static)
    for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
        for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
            for (int ofm2 = 0; ofm2 < jcp.oc_block * jcp.oc_reg_block; ofm2++) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block * jcp.ic_reg_block;
                      ifm2++) {
                    float *U_base_ptr = is_fwd
                                      ? &(U(ofm1, 0, 0, ifm1, ofm2, ifm2, 0, 0))
                                      : &(U(ifm1, 0, 0, ofm1, ifm2, ofm2, 0, 0));
                    weight_transform_data(jcp,
                            &(weights(
                                ofm1 * jcp.oc_block * jcp.oc_reg_block + ofm2,
                                ifm1 * jcp.ic_block * jcp.ic_reg_block + ifm2,
                                0, 0, 0, 0)),
                            U_base_ptr);
                }
            }
        }
    }

    int ithr = omp_get_thread_num();

    #pragma omp for schedule(static)
    for (int tile_block = 0; tile_block < jcp.tile_block; tile_block++) {
        for (int K_blk1 = 0; K_blk1 < jcp.dimK_nb_block; K_blk1++) {
            for (int K_blk2 = 0; K_blk2 < jcp.dimK_block; K_blk2++) {

                input_transform_tileblock_data(
                        tile_block, jcp,
                        &(input(0, K_blk1 * jcp.dimK_block + K_blk2, 0, 0, 0)),
                        &(V(ithr, 0, 0, 0, K_blk1, K_blk2, 0, 0)));
            }
        }

        for (int oj = 0; oj < alpha; oj++) {
            for (int oi = 0; oi < alpha; oi++) {
                for (int M_blk1 = 0; M_blk1 < jcp.dimM_nb_block; M_blk1++)
                for (int K_blk1 = 0; K_blk1 < jcp.dimK_nb_block; K_blk1++)
                for (int N_blk = 0; N_blk < jcp.dimN_block; N_blk++)
                    kernel_->gemm_loop_ker(
                            (float *)&(M(ithr, M_blk1, oj, oi,
                                    N_blk, 0, 0, 0)),
                            (const float *)&(U(M_blk1, oj, oi, K_blk1,
                                    0, 0, 0, 0)),
                            (const float *)&(V(ithr, oj, oi,
                                    N_blk, K_blk1, 0, 0, 0)), K_blk1);
            }
        }

        for (int M_blk1 = 0; M_blk1 < jcp.dimM_nb_block; M_blk1++) {
            for (int M_blk2 = 0; M_blk2 < jcp.dimM_block * jcp.dimM_reg_block;
                  M_blk2++) {
                float *bias_ptr = is_fwd
                    ? &(bias(M_blk1 * jcp.dimM_block * jcp.dimM_reg_block
                        + M_blk2, 0))
                    : NULL;
                  output_transform_tileblock_data(tile_block, jcp, p_ops,
                        &(M(ithr, M_blk1, 0, 0, 0, M_blk2, 0, 0)),
                        &(output(0, M_blk1 * jcp.dimM_block
                            * jcp.dimM_reg_block + M_blk2, 0, 0, 0)),
                        bias_ptr);
            }
        }
    }
    }
}

template void
_jit_avx512_core_convolution_winograd_t<true>::_execute_data_W_SGD(
        float *, float *, float *, float *);
template void
_jit_avx512_core_convolution_winograd_t<false>::_execute_data_W_SGD(
        float *, float *, float *, float *);

namespace {

void src_transform_bwd_weights(int image, jit_conv_winograd_conf_t jcp,
        float *inp, float *tinp) {
    const int ifwp = jcp.iw + jcp.l_pad;
    const int ifhp = jcp.ih + jcp.t_pad;
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
            jcp.mb, jcp.ic/simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 7> output(tinp,
            jcp.tile_block,
            alpha, alpha,
            jcp.ic_block,
            jcp.nb_tile_block_ur,
            jcp.tile_block_ur,
            jcp.ic_simd_block);

    int tile_base_index = image * (jcp.itiles * jcp.jtiles);
    int tblk3 = tile_base_index  % jcp.tile_block_ur;
    int tblk2 = (tile_base_index / jcp.tile_block_ur) % jcp.nb_tile_block_ur;
    int tblk1 = (tile_base_index / jcp.tile_block_ur) / jcp.nb_tile_block_ur;

    for (int tj = 0; tj < jcp.jtiles; tj++) {
        for (int ti = 0; ti < jcp.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((jcp.t_pad <= ydim) && ydim < ifhp) {
                    float *pinp_j = inp + (ydim - jcp.t_pad) * jcp.iw * simd_w;
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((jcp.l_pad <= xdim) && xdim < ifwp) {
                            float *pinp_i = pinp_j + (xdim - jcp.l_pad) * simd_w;
                            load_ps(I[j][i], pinp_i);
                        } else {
#pragma omp simd
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < alpha; i++) {
#pragma omp simd
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_I_4x4_3x3(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    store_output(&(output(tblk1,
                                    j, i, 0,
                                    tblk2, tblk3, 0)),
                                 Iw[j][i], true);
                }
            }

            tblk3++;

            if (tblk3 == jcp.tile_block_ur) {
                tblk3 = 0;
                ++tblk2;
            }
            if (tblk2 == jcp.nb_tile_block_ur) {
                tblk2 = 0;
                ++tblk1;
            }
        }
    }
}

template <bool with_bias>
void diff_dst_transform_bwd_weights(int image, jit_conv_winograd_conf_t jcp,
        float *inp, float *tinp, float *dbias) {
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
            jcp.mb, jcp.oc/simd_w, jcp.oh, jcp.ow, jcp.oc_simd_block);
    array_offset_calculator<float, 8> output(tinp,
            jcp.tile_block,
            alpha, alpha,
            jcp.oc_block,
            jcp.nb_tile_block_ur,
            jcp.tile_block_ur,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    int tile_base_index = image * jcp.itiles * jcp.jtiles;
    int tile_block_ur_st = tile_base_index % jcp.tile_block_ur;
    int nb_tile_block_ur_st = (tile_base_index / jcp.tile_block_ur)
            % jcp.nb_tile_block_ur;
    int tile_block_st = (tile_base_index / jcp.tile_block_ur)
            / jcp.nb_tile_block_ur;

    for (int ofm3 = 0; ofm3 < jcp.oc_reg_block; ofm3++) {
        int tblk3 = tile_block_ur_st;
        int tblk2 = nb_tile_block_ur_st;
        int tblk1 = tile_block_st;
        float *pinp = inp + ofm3 * jcp.oh * jcp.ow * simd_w;
        for (int tj = 0; tj < jcp.jtiles; tj++) {
            for (int ti = 0; ti < jcp.itiles; ti++) {
                for (int j = 0; j < alpha; j++) {
                    int ydim = tj * tile_size + j;
                    if (ydim < jcp.oh) {
                        float *pinp_j = pinp  + ydim * jcp.ow * simd_w;
                        for (int i = 0; i < alpha; i++) {
                            int xdim = ti * tile_size + i;
                            if (xdim < jcp.ow) {
                                float *pinp_i = pinp_j + xdim * simd_w;
                                load_ps(I[j][i], pinp_i);

                                if (with_bias && j < tile_size
                                    && i < tile_size) {
#pragma omp simd
                                    for (int v = 0; v < simd_w; v++) {
                                        dbias[ofm3 * simd_w + v] += pinp_i[v];
                                    };

                                }
                            } else {
#pragma omp simd
                                for (int v = 0; v < simd_w; v++) {
                                    I[j][i][v] = 0.0f;
                                }
                            }
                        }
                    } else {
                        for (int i = 0; i < alpha; i++) {
#pragma omp simd
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                }

                trans_W_3x3_4x4_wu(Iw, I);

                for (int j = 0; j < alpha; j++) {
                    for (int i = 0; i < alpha; i++) {
                        store_output(&(output(tblk1,
                                        j, i, 0,
                                        tblk2, tblk3,
                                        ofm3, 0)),
                                     Iw[j][i], true);
                    }
                }
                tblk3++;
                if (tblk3 >= jcp.tile_block_ur) {
                    tblk3 = 0;
                    tblk2++;
                }
                if (tblk2 >= jcp.nb_tile_block_ur) {
                    tblk2 = 0;
                    tblk1++;
                }
            }
        }
    }
}

void diff_weights_transform_bwd_weights(jit_conv_winograd_conf_t jcp,
        float *wp, float *twp, int first_K_block=0) {
    const int kh = 3;
    const int kw = 3;
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[kh][kw][simd_w][simd_w];
    float Fr[kh][kw][simd_w][simd_w];

    array_offset_calculator<float, 8> input(twp,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block,
            jcp.oc_reg_block,
            jcp.oc_simd_block);
    array_offset_calculator<float, 4> output(wp,
            jcp.kh, jcp.kw,
            jcp.ic_simd_block, jcp.oc_simd_block);

    for (int j = 0; j < alpha; j++) {
        for (int i = 0; i < alpha; i++) {
            for (int v = 0; v < jcp.ic_simd_block; v++) {
                float *inp_ptr = &(input(j, i, 0, 0, v, 0, 0));
                load_ps(Fw[j][i][v], inp_ptr);
            }
        }
    }

    trans_O_3x3_4x4_wu(Fw, F);

    if (!first_K_block) {
        for (int j = 0; j < kh; j++) {
            for (int i = 0; i < kw; i++) {
                for (int v = 0; v < jcp.ic_simd_block; v++) {
                    store_output(&(output(j, i, v, 0)), F[j][i][v], true);
                }
            }
        }
    } else {
        for (int j = 0; j < kh; j++) {
            for (int i = 0; i < kw; i++) {
                float *pout = &(output(j, i, 0, 0));
pragma_unroll
                for (int v = 0; v < simd_w; v++) {
                    load_ps(Fr[j][i][v], pout);
                    pout += simd_w;
                }
            }
        }
        for (int j = 0; j < kh; j++) {
            for (int i = 0; i < kw; i++) {
pragma_unroll
                for (int v = 0; v < simd_w; v++) {
#pragma omp simd
                    for (int k = 0; k < simd_w; k++) {
                        F[j][i][v][k] += Fr[j][i][v][k];
                    }
                }
            }
        }
        for (int j = 0; j < kh; j++) {
            for (int i = 0; i < kw; i++) {
                float *pout = &(output(j, i, 0, 0));
pragma_unroll
                for (int v = 0; v < simd_w; v++) {
                    store_output(pout, F[j][i][v], true);
                    pout += simd_w;
                }
            }
        }
    }
}

void subarray_sum(size_t num_arrs, float *output, size_t nelems,
        float *input_ptrs[], size_t input_starts[], size_t input_ends[]) {
    using namespace nstl;
    const size_t block_size = 16 * 1024 / sizeof(float);
    const size_t blocks_number = nelems / block_size;
    const size_t tail = nelems % block_size;

#pragma omp parallel
    {
        const int ithr = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        size_t start{ 0 }, end{ 0 };
        balance211(blocks_number, nthr, ithr, start, end);

        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * block_size;
            size_t end_e = start_e + block_size;
            size_t input_start = max(start_e, min(input_starts[0], end_e));
            size_t input_end = max(start_e, min(input_ends[0], end_e));
#pragma omp simd
            for (size_t e = start_e; e < input_start; e++) {
                output[e] = 0.f;
            }
#pragma omp simd
            for (size_t e = input_start; e < input_end; e++) {
                output[e] = input_ptrs[0][e];
            }
#pragma omp simd
            for (size_t e = input_end; e < end_e; e++) {
                output[e] = 0.f;
            }
            for (size_t a = 1; a < num_arrs; a++) {
                input_start = max(start_e, input_starts[a]);
                input_end = min(input_ends[a], end_e);
#pragma omp simd
                for (size_t e = input_start; e < input_end; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            size_t end_e = nelems;
            size_t input_start = max(start_e, min(input_starts[0], end_e));
            size_t input_end = max(start_e, min(input_ends[0], end_e));
#pragma omp simd
            for (size_t e = start_e; e < input_start; e++) {
                output[e] = 0.f;
            }
#pragma omp simd
            for (size_t e = input_start; e < input_end; e++) {
                output[e] = input_ptrs[0][e];
            }
#pragma omp simd
            for (size_t e = input_end; e < end_e; e++) {
                output[e] = 0.f;
            }
            for (size_t a = 1; a < num_arrs; a++) {
                input_start = max(start_e, input_starts[a]);
                input_end = min(input_ends[a], end_e);
#pragma omp simd
                for (size_t e = input_start; e < input_end; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }
    }
}

const int max_threads_number = 1024;

void src_transform_bwd_weights_tile(int tile_block,
    jit_conv_winograd_conf_t jcp, float *inp, float *tinp) {
    const int ifwp = jcp.iw + jcp.l_pad;
    const int ifhp = jcp.ih + jcp.t_pad;
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
        jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 6> output(tinp,
        alpha, alpha,
        jcp.ic_block,
        jcp.nb_tile_block_ur,
        jcp.tile_block_ur,
        jcp.ic_simd_block);

    int tile_index = tile_block * jcp.nb_tile_block_ur * jcp.tile_block_ur;
    for (int tblk2 = 0; tblk2 < jcp.nb_tile_block_ur; ++tblk2) {
        for (int tblk3 = 0; tblk3 < jcp.tile_block_ur; ++tblk3) {

            int img = tile_index / (jcp.jtiles * jcp.itiles);
            int ti = tile_index % jcp.itiles;
            int tj = (tile_index / jcp.itiles) % jcp.jtiles;

            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((jcp.t_pad <= ydim) && ydim < ifhp) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((jcp.l_pad <= xdim) && xdim < ifwp) {
                            float *pinp = &(input(img, 0, ydim - jcp.t_pad,
                                    xdim - jcp.l_pad, 0));
                            load_ps(I[j][i], pinp);
                        }
                        else {
#pragma omp simd
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                }
                else {
                    for (int i = 0; i < alpha; i++) {
#pragma omp simd
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_I_4x4_3x3(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    float *pout = &(output(j, i, 0, tblk2, tblk3, 0));
                    store_output(pout, Iw[j][i], false);
                }
            }
            tile_index++;
        }
    }
}

template <bool with_bias>
void diff_dst_transform_bwd_weights_tile(int tile_block, bool biasu,
    jit_conv_winograd_conf_t jcp, float *inp, float *tinp, float *dbias) {
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
        jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, jcp.oc_simd_block);
    array_offset_calculator<float, 7> output(tinp,
        alpha, alpha,
        jcp.oc_block,
        jcp.nb_tile_block_ur,
        jcp.tile_block_ur,
        jcp.oc_reg_block,
        jcp.oc_simd_block);

    int tile_index = tile_block * jcp.nb_tile_block_ur * jcp.tile_block_ur;
    for (int tblk2 = 0; tblk2 < jcp.nb_tile_block_ur; ++tblk2) {
        for (int tblk3 = 0; tblk3 < jcp.tile_block_ur; tblk3++) {

            int img = tile_index / (jcp.jtiles * jcp.itiles);
            int ti = tile_index % jcp.itiles;
            int tj = (tile_index / jcp.itiles) % jcp.jtiles;

            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < jcp.oh) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < jcp.ow) {
                            float *pinp = &input(img, 0, ydim, xdim, 0);
                            load_ps(I[j][i], pinp);
                            if (with_bias && biasu
                                && j < tile_size && i < tile_size) {
#pragma omp simd
                                for (int v = 0; v < simd_w; v++) {
                                    dbias[v] += pinp[v];
                                }
                            }
                        } else {
#pragma omp simd
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                }
                else {
                   for (int i = 0; i < alpha; i++) {
#pragma omp simd
                       for (int v = 0; v < simd_w; v++) {
                           I[j][i][v] = 0.0f;
                       }
                   }
                }
            }

            trans_W_3x3_4x4_wu(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    float *pout = &(output(j, i, 0,
                                tblk2, tblk3, 0, 0));
                    store_output(pout, Iw[j][i], false);
                }
            }
            tile_index++;
        }
    }
}

// Sum to the first buffer array
void array_sum(size_t num_arrs, float *output,
    size_t nelems, float *input_ptrs[], bool reduce_to_first = true) {
    const size_t block_size = 16 * 1024 / sizeof(float);
    const size_t blocks_number = nelems / block_size;
    const size_t tail = nelems % block_size;

#pragma omp parallel
    {
        const size_t ithr = omp_get_thread_num();
        const size_t nthr = omp_get_num_threads();
        size_t start{ 0 }, end{ 0 };
        balance211(blocks_number, nthr, ithr, start, end);

        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * block_size;
            size_t end_e = start_e + block_size;
            if (!reduce_to_first) {
#               pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] = input_ptrs[0][e];
                }
            }
            for (size_t a = 1; a < num_arrs; a++) {
#               pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            size_t end_e = nelems;
            if (!reduce_to_first) {
#               pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] = input_ptrs[0][e];
                }
            }
            for (size_t a = 1; a < num_arrs; a++) {
#               pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }
    }
}
} //bwdw namespace

void jit_avx512_core_convolution_winograd_bwd_weights_t::
_execute_backward_weights_SDGtWo() {
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();

    auto diff_dst_transform_bwdw_b = jcp.with_bias
                                   ? diff_dst_transform_bwd_weights_tile<true>
                                   : diff_dst_transform_bwd_weights_tile<false>;

    array_offset_calculator<float, 5> src((float *)this->input_memory(0),
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 1> diff_bias((float *)this->memory(1), jcp.oc);

    array_offset_calculator<float, 8> Us((float *)(scratchpad_->U_ptr()),
            0, alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    int U_sz = nthreads * alpha * alpha * jcp.oc / jcp.nb_oc
        * jcp.ic / jcp.nb_ic * sizeof(float);
    array_offset_calculator<float, 7>diff_weights_prv(
            (float *)(scratchpad_->U_ptr() + U_sz),
            0, jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);

    array_offset_calculator<float, 8> M((float *)(scratchpad_->M_ptr()),
            0, alpha, alpha,
            jcp.oc_block,
            jcp.nb_tile_block_ur,
            jcp.tile_block_ur,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    array_offset_calculator<float, 7> V((float *)(scratchpad_->V_ptr()),
            0, alpha, alpha,
            jcp.ic_block,
            jcp.nb_tile_block_ur,
            jcp.tile_block_ur,
            jcp.ic_simd_block);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()), nthreads, jcp.oc);

#pragma omp parallel
{
    if (jcp.with_bias) {
#pragma omp for nowait collapse(2)
        for (int ithr = 0; ithr < nthreads; ithr++) {
            for (int ofm = 0; ofm < jcp.oc / simd_w; ofm++) {
                float *pdbias = &(diff_bias_prv(ithr, ofm * simd_w));
#pragma omp simd
                for (int v = 0; v < simd_w; v++) {
                    pdbias[v] = 0.0f;
                }
            }
        }
    }

    int ithr = omp_get_thread_num();
    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
        int first_tblk = 0;
#pragma omp for
        for (int tblk1 = 0; tblk1 < jcp.tile_block; ++tblk1) {
            for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                int ifm = ifm1 * jcp.ic_block + ifm2;
                src_transform_bwd_weights_tile(tblk1, jcp,
                        &(src(0, ifm, 0, 0, 0)),
                        &(V(ithr, 0, 0, ifm2, 0, 0, 0)));
            }

            for (int ofm1 = 0; ofm1 < jcp.nb_oc; ++ofm1) {
                for (int ofm2 = 0; ofm2 < jcp.oc_block; ++ofm2) {
                    for (int ofm3 = 0; ofm3 < jcp.oc_reg_block; ++ofm3) {
                        int ofm = (ofm1 * jcp.oc_block + ofm2) * jcp.oc_reg_block
                                + ofm3;
                        float *dbias = jcp.with_bias
                                     ? &(diff_bias_prv(ithr, ofm * simd_w))
                                     : NULL;
                        bool biasu = ifm1 == 0;
                        diff_dst_transform_bwdw_b(tblk1, biasu, jcp,
                                &(diff_dst(0, ofm, 0, 0, 0)),
                                &(M(ithr, 0, 0, ofm2, 0, 0, ofm3, 0)),
                                dbias);
                    }
                }

                for (int oj = 0; oj < alpha; ++oj) {
                    for (int oi = 0; oi < alpha; ++oi) {
                        kernel_->gemm_loop_ker_first_iter(
                                &(Us(ithr, oj, oi, 0, 0, 0, 0, 0)),
                                &(M(ithr, oj, oi, 0, 0, 0, 0, 0)),
                                &(V(ithr, oj, oi, 0, 0, 0, 0)));
                    }
                }

                for (int ofm2 = 0; ofm2 < jcp.oc_block; ++ofm2) {
                    for (int ofm3 = 0; ofm3 < jcp.oc_reg_block; ++ofm3) {
                        int ofm = (ofm1 * jcp.oc_block + ofm2) * jcp.oc_reg_block
                                + ofm3;
                        for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                            int ifm = ifm1 * jcp.ic_block + ifm2;
                            diff_weights_transform_bwd_weights(jcp,
                                    &(diff_weights_prv(ithr, ofm, ifm,
                                            0, 0, 0, 0)),
                                    &(Us(ithr, 0, 0, ofm2, ifm2, 0, ofm3, 0)),
                                    first_tblk);
                        }
                    }
                }
            }
            ++first_tblk;
        }
    }
}

    // Reduce diff-weights
    {
        float *output = (float *)(this->memory(0));
        float *input_base = (float *)(scratchpad_->U_ptr() + U_sz);
        int nelems = jcp.oc * jcp.ic * jcp.kh * jcp.kw;
        float *input_ptrs[max_threads_number];
        for (int i = 0; i < nthreads; ++i) {
            input_ptrs[i] = input_base + nelems * i;
        }
        array_sum(nthreads, output, nelems, input_ptrs, false);

        if (jcp.with_bias) {
            output = (float *)(this->memory(1));
            input_base = (float *)(scratchpad_->bias_ptr());
            for (int i = 0; i < nthreads; ++i) {
                input_ptrs[i] = input_base + jcp.oc * i;
            }
            array_sum(nthreads, output, jcp.oc, input_ptrs, false);
        }
    }
}

void jit_avx512_core_convolution_winograd_bwd_weights_t::
_execute_backward_weights_S_D_Giot_W() {
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();

    auto diff_dst_transform_bwdw_b = jcp.with_bias
                                   ? diff_dst_transform_bwd_weights<true>
                                   : diff_dst_transform_bwd_weights<false>;

    array_offset_calculator<float, 5> src((float *)this->input_memory(0),
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 1> diff_bias((float *)this->memory(1), jcp.oc);

    array_offset_calculator<float, 9> U((float *)(scratchpad_->U_ptr()),
            jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    int U_size = jcp.oc * jcp.ic * alpha * alpha * sizeof(float);
    array_offset_calculator<float, 10> Us(
            (float *)(scratchpad_->U_ptr() + U_size),
            0, jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    array_offset_calculator<float, 9> M((float *)(scratchpad_->M_ptr()),
            jcp.nb_oc,
            jcp.tile_block,
            alpha, alpha,
            jcp.oc_block,
            jcp.nb_tile_block_ur,
            jcp.tile_block_ur ,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    array_offset_calculator<float, 8> V((float *)(scratchpad_->V_ptr()),
            jcp.nb_ic,
            jcp.tile_block,
            alpha, alpha,
            jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()), nthreads, jcp.oc);

    size_t input_starts[max_threads_number];
    size_t input_ends[max_threads_number];
    size_t first_tblk = 0;

#pragma omp parallel firstprivate(first_tblk)
{
    if (jcp.with_bias) {
#pragma omp for nowait collapse(2)
        for (int ithr = 0; ithr < nthreads; ++ithr) {
            for (int ofm = 0; ofm < jcp.oc; ++ofm) {
                diff_bias_prv(ithr, ofm) = 0.0f;
            }
        }
    }

#pragma omp for collapse(3) nowait
    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
         for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
             for (int img = 0; img < jcp.mb; img++) {
                 int ifm = ifm1 * jcp.ic_block + ifm2;
                 src_transform_bwd_weights(img, jcp,
                         &(src(img, ifm, 0, 0, 0)),
                        &(V(ifm1, 0, 0, 0, ifm2, 0, 0, 0)));
             }
         }
    }

    int ithr = omp_get_thread_num();
#pragma omp for collapse(3)
    for (int ofm1 = 0; ofm1 < jcp.nb_oc; ++ofm1) {
        for (int ofm2 = 0; ofm2 < jcp.oc_block; ++ofm2) {
            for (int img = 0; img < jcp.mb; ++img) {
                int ofm = (ofm1 * jcp.oc_block + ofm2) * jcp.oc_reg_block;
                float *dbias = jcp.with_bias
                             ? &(diff_bias_prv(ithr, ofm * simd_w))
                             : NULL;
                diff_dst_transform_bwdw_b(img, jcp,
                        &(diff_dst(img, ofm, 0, 0, 0)),
                        &(M(ofm1, 0, 0, 0, ofm2, 0, 0, 0, 0)),
                        dbias);
            }
        }
    }

#pragma omp for collapse(5) nowait schedule(static)
    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ++ofm1) {
            for (int oj = 0; oj < alpha; ++oj) {
                for (int oi = 0; oi < alpha; ++oi) {
                    for (int tblk1 = 0; tblk1 < jcp.tile_block; ++tblk1) {
                        if (first_tblk == 0) {
                            input_starts[ithr] =
                                (float *)&(Us(ithr, ifm1, ofm1, oj, oi, 0, 0, 0,
                                            0, 0))
                                - (float *)&(Us(ithr, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0));
                            input_ends[ithr] = input_starts[ithr]
                                    + jcp.oc_block * jcp.ic_block
                                      * jcp.ic_simd_block * jcp.oc_reg_block * jcp.oc_simd_block;
                        }
                        else if (tblk1 == 0) {
                            input_ends[ithr] += jcp.oc_block * jcp.ic_block
                                * jcp.ic_simd_block * jcp.oc_reg_block * jcp.oc_simd_block;
                        }

                        if (first_tblk == 0 || tblk1 == 0) {
                            kernel_->gemm_loop_ker_first_iter(
                                    &(Us(ithr, ifm1, ofm1, oj, oi,
                                            0, 0, 0, 0, 0)),
                                    &(M(ofm1, tblk1, oj, oi, 0, 0, 0, 0, 0)),
                                    &(V(ifm1, tblk1, oj, oi, 0, 0, 0, 0)));
                        } else {
                            kernel_->gemm_loop_ker(
                                    &(Us(ithr, ifm1, ofm1, oj, oi,
                                            0, 0, 0, 0, 0)),
                                    &(M(ofm1, tblk1, oj, oi, 0, 0, 0, 0, 0)),
                                    &(V(ifm1, tblk1, oj, oi, 0, 0, 0, 0)));
                        }
                        ++first_tblk;
                    }
                }
            }
        }
    }
}

    // Reduce diff-weights
    {
        float *output = &(U(0, 0, 0, 0, 0, 0, 0, 0, 0));
        size_t nelems = jcp.ic * jcp.oc * alpha * alpha;
        float *input_ptrs[max_threads_number];
        for (int i = 0; i < nthreads; ++i)
            input_ptrs[i] = output + nelems * (i + 1);
        subarray_sum(nthreads, output, nelems, input_ptrs,
                input_starts, input_ends);
    }

#pragma omp parallel for collapse(5)
    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ++ofm1) {
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ++ofm2) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                    for (int ofm3 = 0;  ofm3 < jcp.oc_reg_block; ++ofm3) {
                        int ofm = (ofm1 * jcp.oc_block + ofm2)
                            * jcp.oc_reg_block + ofm3;
                        int ifm = ifm1 * jcp.ic_block + ifm2;
                        diff_weights_transform_bwd_weights(jcp,
                            &(diff_weights(ofm, ifm, 0, 0, 0, 0)),
                            &(U(ifm1, ofm1, 0, 0, ofm2, ifm2, 0, ofm3, 0)));
                    }
                }
            }
        }
    }

    if (jcp.with_bias) {
#pragma omp parallel for
        for (int ofm1 = 0; ofm1 < jcp.oc / simd_w; ++ofm1) {
            float* pbias = &(diff_bias(ofm1 * simd_w));
            float *pbias_prv = &(diff_bias_prv(0, ofm1 * simd_w));
#pragma omp simd
            for (int ofm2 = 0; ofm2 < simd_w; ++ofm2) {
                pbias[ofm2] = pbias_prv[ofm2];
            }
            for (int ithr = 1; ithr < nthreads; ++ithr) {
                pbias_prv = &(diff_bias_prv(ithr, ofm1 * simd_w));
#pragma omp simd
                for (int ofm2 = 0; ofm2 < simd_w; ++ofm2) {
                    pbias[ofm2] += pbias_prv[ofm2];
                }
            }
        }
    }
}

}
}
}
// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
