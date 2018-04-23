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

namespace {

template <bool ver_4fma>
void diff_src_transform_bwd_weights(int image, jit_conv_winograd_conf_t conv,
        float *inp, float *tinp, float *Iw_temp,
        void (*transpose_4fma_ker)(float *, float *))
{

    const int ifwp = conv.iw + conv.l_pad;
    const int ifhp = conv.ih + conv.t_pad;
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 4> Iw_trans_temp(Iw_temp,
            alpha, alpha, conv.tile_4fma, simd_w);
    array_offset_calculator<float, 5> input(inp,
            conv.mb, conv.ic/simd_w, conv.ih, conv.iw, simd_w);
    array_offset_calculator<float, 8> output(tinp,
            conv.nb_ic, alpha, alpha,
            conv.tile_block, conv.ic_block,
            conv.nb_tile_block_ur, conv.tile_block_ur,
            conv.ic_simd_block * conv.tile_4fma);

    int tile_base_index =
        image * (conv.itiles * conv.jtiles + conv.tile_4fma_padding);
    int tile_4fma = 0;
    int tile_block_ur = (tile_base_index / conv.tile_4fma) % conv.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / conv.tile_4fma / conv.tile_block_ur)
        % conv.nb_tile_block_ur;
    int tile_block = (tile_base_index / conv.tile_4fma / conv.tile_block_ur)
        / conv.nb_tile_block_ur;

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((conv.t_pad <= ydim) && ydim < ifhp) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((conv.l_pad <= xdim) && xdim < ifwp) {
#pragma omp simd
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = input(0, 0,
                                    ydim - conv.t_pad,
                                    xdim - conv.l_pad, v);
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
            trans_I_4x4_3x3(Iw, I);

            if (ver_4fma) {
                for (int j = 0; j < alpha; j++) {
                    for (int i = 0; i < alpha; i++) {
                        float *Iw_temp_base = &(Iw_trans_temp(j, i,
                                tile_4fma, 0));
#pragma omp simd
                        for (int v = 0; v < simd_w; v++) {
                            Iw_temp_base[v] = Iw[j][i][v];
                        }
                    }
                }
                tile_4fma++;
                if (tile_4fma == conv.tile_4fma) {
                    float *outp = &(output(0, 0, 0,
                            tile_block, 0,
                            nb_tile_block_ur, tile_block_ur, 0));
                    transpose_4fma_ker(outp, (float *)Iw_temp);
                    tile_4fma = 0;
                    tile_block_ur++;
                }
            } else {
                for (int j = 0; j < alpha; j++) {
                    for (int i = 0; i < alpha; i++) {
                        store_output(&(output(0, j, i,
                                    tile_block, 0,
                                    nb_tile_block_ur, tile_block_ur, 0)),
                            Iw[j][i], true);
                    }
                }
                tile_block_ur++;
            }

            if (tile_block_ur == conv.tile_block_ur) {
                tile_block_ur = 0;
                ++nb_tile_block_ur;
            }
            if (nb_tile_block_ur == conv.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }

    if (ver_4fma && tile_4fma < conv.tile_4fma && conv.tile_4fma_padding != 0) {

        for (int j = 0; j < alpha; j++) {
            for (int i = 0; i < alpha; i++) {
                for (int tb = tile_4fma; tb < conv.tile_4fma; tb++) {
                    float *Iw_temp_base = &(Iw_trans_temp(j, i, tb, 0));
#pragma omp simd
                    for (int v = 0; v < simd_w; v++) {
                        Iw_temp_base[v] = 0;
                    }
                }
            }
        }
        float *outp = &(output(0, 0, 0,
                tile_block, 0,
                nb_tile_block_ur, tile_block_ur, 0));
        transpose_4fma_ker(outp, (float *)Iw_temp);
    }
}

template <bool with_bias>
void diff_dst_transform_bwd_weights(int image, jit_conv_winograd_conf_t conv,
        float *inp, float *tinp, float *dbias)
{

    const int total_tiles = conv.itiles * conv.jtiles + conv.tile_4fma_padding;
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
        conv.mb, conv.oc / simd_w, conv.oh, conv.ow, conv.oc_simd_block);
    array_offset_calculator<float, 8> output(tinp,
        conv.nb_oc, alpha, alpha,
        conv.tile_block, conv.oc_block,
        conv.nb_tile_block_ur,
        conv.tile_block_ur * conv.tile_4fma, conv.oc_simd_block);

    int tile_base_index = image * total_tiles;
    int tile_block_ur = tile_base_index % (conv.tile_block_ur * conv.tile_4fma);
    int nb_tile_block_ur =
        (tile_base_index / conv.tile_block_ur / conv.tile_4fma)
        % conv.nb_tile_block_ur;
    int tile_block = (tile_base_index / conv.tile_block_ur / conv.tile_4fma)
        / conv.nb_tile_block_ur;

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < conv.oh) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < conv.ow) {
                            float *input_base = &(input(0, 0, ydim, xdim, 0));
#pragma omp simd
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = input_base[v];
                            }
                            if (with_bias && j < tile_size && i < tile_size) {
#pragma omp simd
                                for (int v = 0; v < simd_w; v++) {
                                    dbias[v] += input_base[v];
                                }
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
                    store_output(&(output(0, j, i,
                                tile_block, 0,
                                nb_tile_block_ur,
                                tile_block_ur, 0)),
                        Iw[j][i], true);
                }
            }
            tile_block_ur++;
            if (tile_block_ur >= conv.tile_block_ur * conv.tile_4fma) {
                tile_block_ur = 0;
                nb_tile_block_ur++;
            }
            if (nb_tile_block_ur >= conv.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }
}

void diff_weights_transform_bwd_weights(jit_conv_winograd_conf_t conv,
        float *wp, float *twp)
{
    const int kh = 3;
    const int kw = 3;
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[kh][kw][simd_w][simd_w];

    array_offset_calculator<float, 8> input(twp,
        conv.nb_ic, conv.nb_oc,
        alpha, alpha,
        conv.oc_block, conv.ic_block,
        conv.ic_simd_block, conv.oc_simd_block);
    array_offset_calculator<float, 6> output(wp,
        conv.oc / simd_w, conv.ic / simd_w,
        conv.kh, conv.kw,
        conv.ic_simd_block, conv.oc_simd_block);

    for (int j = 0; j < alpha; j++) {
        for (int i = 0; i < alpha; i++) {
            for (int v = 0; v < conv.ic_simd_block; v++) {
#pragma omp simd
                for (int k = 0; k < conv.oc_simd_block; k++) {
                    Fw[j][i][v][k] = input(0, 0, j, i, 0, 0, v, k);
                }
            }
        }
    }

    trans_O_3x3_4x4_wu(Fw, F);

    for (int j = 0; j < kh; j++) {
        for (int i = 0; i < kw; i++) {
            for (int v = 0; v < conv.ic_simd_block; v++) {
                store_output(&(output(0, 0, j, i, v, 0)),
                    F[j][i][v], true);
            }
        }
    }
}
}

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

void jit_avx512_core_convolution_winograd_bwd_weights_t::
_execute_backward_weights_S_D_G_W()
{
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();

    auto diff_src_transform_bwd_weights_ver = jcp.ver == ver_4fma ?
            diff_src_transform_bwd_weights<true> :
            diff_src_transform_bwd_weights<false>;
    auto diff_dst_transform_bwd_weights_ver = jcp.with_bias
            ? diff_dst_transform_bwd_weights<true>
            : diff_dst_transform_bwd_weights<false>;

    array_offset_calculator<float, 5> diff_src((float *)this->input_memory(0),
            jcp.mb, jcp.ic/simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc/simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc/simd_w, jcp.ic/simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 2> diff_bias(
            (float *)this->memory(1), jcp.oc/simd_w, simd_w);

    array_offset_calculator<float, 8> U(
            (float *)(scratchpad_->U_ptr()),
            jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 8> M(
            (float *)(scratchpad_->M_ptr()),
            jcp.nb_oc, alpha, alpha,
            jcp.tile_block, jcp.oc_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur * jcp.tile_4fma,
            jcp.oc_simd_block);
    array_offset_calculator<float, 8> V(
            (float *)(scratchpad_->V_ptr()),
            jcp.nb_ic, alpha, alpha,
            jcp.tile_block, jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block * jcp.tile_4fma);

    const int trans_buffer_size = alpha * alpha * jcp.tile_4fma
                                * jcp.ic_simd_block;
    array_offset_calculator<float, 2> trans_buffer(
            (float *)(scratchpad_->src_transpose_ptr()),
            nthreads,
            trans_buffer_size);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()),
            omp_get_max_threads(),
            jcp.oc);

#pragma omp parallel num_threads(nthreads)
    {
        if (jcp.with_bias) {
#pragma omp for nowait collapse(2)
            for (int ithr = 0; ithr < nthreads; ithr++) {
                for (int ofm = 0; ofm < jcp.oc; ofm++) {
                    diff_bias_prv(ithr, ofm) = 0.0f;
                }
            }

#pragma omp for nowait
            for (int bofm = 0; bofm < jcp.oc / simd_w; bofm++) {
#pragma omp simd
                for (int v = 0; v < simd_w; v++)
                    diff_bias(bofm, v) = 0.0f;
            }
        }

        const int ithread = omp_get_thread_num();
#pragma omp for nowait collapse(3)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                    float *transb = jcp.ver == ver_4fma
                                  ? &(trans_buffer(ithread, 0))
                                  : NULL;
                    diff_src_transform_bwd_weights_ver(img, jcp,
                            &(diff_src(img, ifm1 * jcp.ic_block + ifm2,
                                    0, 0, 0)),
                            &(V(ifm1, 0, 0, 0, ifm2, 0, 0, 0)),
                            transb,
                            kernel_->transpose_4fma_ker);
                }
            }
        }

#pragma omp for nowait collapse(3)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                    float *dbias = jcp.with_bias
                           ? &(diff_bias_prv(ithread,
                                       simd_w * (ofm1 * jcp.oc_block + ofm2)))
                           : NULL;
                    diff_dst_transform_bwd_weights_ver(img, jcp,
                            &(diff_dst(img, ofm1 * jcp.oc_block + ofm2,
                                    0, 0, 0)),
                            &(M(ofm1, 0, 0, 0, ofm2, 0, 0, 0)),
                            dbias);
                }
            }
        }

#pragma omp barrier

        for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
#pragma omp for nowait collapse(3) schedule(static)
            for (int oj = 0; oj < alpha; oj++) {
                for (int oi = 0; oi < alpha; oi++) {
                    for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                        kernel_->gemm_loop_ker_first_iter(
                                (float *)&(U(ifm1, ofm1,
                                        oj, oi,
                                        0, 0, 0, 0)),
                                (const float *)&(M(ofm1, oj, oi,
                                        0, 0, 0, 0, 0)),
                                (const float *)&(V(ifm1, oj, oi,
                                        0, 0, 0, 0, 0)));
                        for (int tile_block = 1; tile_block < jcp.tile_block;
                                tile_block++) {
                            kernel_->gemm_loop_ker((float *)&(U(ifm1, ofm1,
                                            oj, oi,
                                            0, 0, 0, 0)),
                                    (const float *)&(M(ofm1, oj, oi, tile_block,
                                            0, 0, 0, 0)),
                                    (const float *)&(V(ifm1, oj, oi, tile_block,
                                            0, 0, 0, 0)));
                        }
                    }
                }
            }
        }

#pragma omp barrier

#pragma omp for nowait collapse(4)
        for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
            for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                    for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++) {
                        diff_weights_transform_bwd_weights(jcp,
                                &(diff_weights(ofm1 * jcp.oc_block + ofm2,
                                        ifm1 * jcp.ic_block + ifm2,
                                        0, 0, 0, 0)),
                                &(U(ifm1, ofm1, 0, 0, ofm2, ifm2, 0, 0)));
                    }
                }
            }
        }

        if (jcp.with_bias) {
#pragma omp for
            for (int ofm1 = 0; ofm1 < jcp.oc / simd_w; ofm1++) {
                for (int ithr = 0; ithr < nthreads; ithr++) {
                    float* base_bias_ptr = &(diff_bias(ofm1, 0));
                    float* base_bias_prv_ptr = &(diff_bias_prv(
                                ithr * jcp.oc + ofm1 * simd_w));
#pragma omp simd
                    for (int ofm2 = 0; ofm2 < simd_w; ofm2++) {
                        base_bias_ptr[ofm2] += base_bias_prv_ptr[ofm2];
                    }
                }
            }
        }
    }
}

namespace {

const int max_threads_number = 1024;

template <bool ver_4fma>
void diff_src_transform_bwd_weights_tile(int tile_block,
    jit_conv_winograd_conf_t conv, float *inp, float *tinp,
    void(*transpose_4fma_ker)(float *, float *))
{
    const int ifwp = conv.iw + conv.l_pad;
    const int ifhp = conv.ih + conv.t_pad;
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    float *Iw_buffer = nullptr;
    if (ver_4fma) {
        Iw_buffer = (float *)malloc(alpha * alpha * conv.tile_4fma
            * simd_w * sizeof(float), 64);
    }
    array_offset_calculator<float, 4> Iw_scratchpad(Iw_buffer,
        alpha, alpha, conv.tile_4fma, simd_w);
    array_offset_calculator<float, 5> input(inp,
        conv.mb, conv.ic / simd_w, conv.ih, conv.iw, simd_w);
    array_offset_calculator<float, 7> output(tinp,
        0, alpha, alpha,
        conv.ic_block,
        conv.nb_tile_block_ur, conv.tile_block_ur,
        conv.ic_simd_block * conv.tile_4fma);

    int tile_4fma = 0;

    int n_tiles = tile_block * conv.nb_tile_block_ur * conv.tile_block_ur;
    for (int nb_tile_block_ur = 0; nb_tile_block_ur < conv.nb_tile_block_ur;
        nb_tile_block_ur++) {
        for (int tile_block_ur = 0; tile_block_ur < conv.tile_block_ur;
            tile_block_ur++) {

            int img = n_tiles / (conv.jtiles * conv.itiles);
            int no_tile = n_tiles % (conv.jtiles * conv.itiles);
            int ti = no_tile % conv.itiles;
            int tj = no_tile / conv.itiles;

            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((conv.t_pad <= ydim) && ydim < ifhp) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((conv.l_pad <= xdim) && xdim < ifwp) {
#pragma omp simd
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = input(img, 0,
                                    ydim - conv.t_pad,
                                    xdim - conv.l_pad, v);
                            }
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

            if (ver_4fma) {
                for (int j = 0; j < alpha; j++) {
                    for (int i = 0; i < alpha; i++) {
#pragma omp simd
                        for (int v = 0; v < simd_w; v++) {
                            Iw_scratchpad(j, i, tile_4fma, v) = Iw[j][i][v];
                        }
                    }
                }
                tile_4fma++;
                if (tile_4fma == conv.tile_4fma) {
                    float *outp = &(output(0, 0, 0, 0,
                        nb_tile_block_ur, tile_block_ur, 0));
                    transpose_4fma_ker(outp, (float *)Iw_buffer);
                    tile_4fma = 0;
                }
            }
            else {
                for (int j = 0; j < alpha; j++) {
                    for (int i = 0; i < alpha; i++) {
                        store_output(
                            &(output(0, j, i, 0,
                                nb_tile_block_ur, tile_block_ur, 0)),
                            Iw[j][i], false);

                    }
                }
            }
            n_tiles++;
        }
    }
}

template <bool with_bias>
void diff_dst_transform_bwd_weights_tile(int tile_block,
    jit_conv_winograd_conf_t conv, float *inp, float *tinp, float *dbias)
{
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
        conv.mb, conv.oc / simd_w, conv.oh, conv.ow, conv.oc_simd_block);
    array_offset_calculator<float, 7> output(tinp,
        conv.nb_oc, alpha, alpha,
        conv.oc_block,
        conv.nb_tile_block_ur,
        conv.tile_block_ur * conv.tile_4fma, conv.oc_simd_block);

    int n_tiles = tile_block * conv.nb_tile_block_ur * conv.tile_block_ur;
    for (int nb_tile_block_ur = 0; nb_tile_block_ur < conv.nb_tile_block_ur;
        nb_tile_block_ur++) {
        for (int tile_block_ur = 0; tile_block_ur < conv.tile_block_ur;
            tile_block_ur++) {

            int img = n_tiles / (conv.jtiles * conv.itiles);
            int no_tile = n_tiles % (conv.jtiles * conv.itiles);
            int ti = no_tile % conv.itiles;
            int tj = no_tile / conv.itiles;

            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < conv.oh) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < conv.ow) {
                            float *input_base = &input(img, 0, ydim, xdim, 0);
#pragma omp simd
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = input_base[v];
                            }
                            if (with_bias && j < tile_size && i < tile_size) {
#pragma omp simd
                                for (int v = 0; v < simd_w; v++) {
                                    dbias[v] += input_base[v];
                                }
                            }
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

            trans_W_3x3_4x4_wu(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    /*TODO: Try instrinsic for casting into __m512*/
                    store_output(&(output(0, j, i, 0,
                        nb_tile_block_ur, tile_block_ur, 0)),
                        Iw[j][i], false);
                }
            }
            n_tiles++;
        }
    }
}

// Sum to the first buffer array
void array_sum(int num_arrs, float *output,
    size_t nelems, float *input_ptrs[], bool reduce_to_first = true)
{
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
            if (!reduce_to_first) {
#               pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] = input_ptrs[0][e];
                }
            }
            for (int a = 1; a < num_arrs; a++) {
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
            for (int a = 1; a < num_arrs; a++) {
#               pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }
    }
}

void subarray_sum(int num_arrs, float *output, size_t nelems,
        float *input_ptrs[], size_t input_starts[], size_t input_ends[])
{
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
            for (int a = 1; a < num_arrs; a++) {
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
            for (int a = 1; a < num_arrs; a++) {
                input_start = max(start_e, input_starts[a]);
                input_end = min(input_ends[a], end_e);
#pragma omp simd
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }
    }
}
}

void jit_avx512_core_convolution_winograd_bwd_weights_t::
_execute_backward_weights_S_D_Giot_W()
{
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();
    int U_size = jcp.oc * jcp.ic * alpha * alpha * sizeof(float);

    auto diff_src_transform_bwd_weights_ver = jcp.ver == ver_4fma ?
            diff_src_transform_bwd_weights<true> :
            diff_src_transform_bwd_weights<false>;
    auto diff_dst_transform_bwd_weights_ver = jcp.with_bias
                                        ? diff_dst_transform_bwd_weights<true>
                                        : diff_dst_transform_bwd_weights<false>;

    array_offset_calculator<float, 5> diff_src((float *)this->input_memory(0),
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 2> diff_bias(
            (float *)this->memory(1), jcp.oc / simd_w, simd_w);

    array_offset_calculator<float, 8> U((float *)(scratchpad_->U_ptr()),
            jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 9> Us(
            (float *)(scratchpad_->U_ptr() + U_size),
            0, jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 8> M((float *)(scratchpad_->M_ptr()),
            jcp.nb_oc, alpha, alpha,
            jcp.tile_block, jcp.oc_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur * jcp.tile_4fma,
            jcp.oc_simd_block);

    array_offset_calculator<float, 8> V((float *)(scratchpad_->V_ptr()),
            jcp.nb_ic, alpha, alpha,
            jcp.tile_block, jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block * jcp.tile_4fma);

    const int trans_buffer_size = alpha * alpha * jcp.tile_4fma
        * jcp.ic_simd_block;
    array_offset_calculator<float, 2> trans_buffer(
        (float *)(scratchpad_->src_transpose_ptr()),
        nthreads,
        trans_buffer_size);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()), nthreads, jcp.oc);

#pragma omp parallel
    {
        if (jcp.with_bias) {
#pragma omp for nowait collapse(2)
            for (int ithr = 0; ithr < nthreads; ithr++) {
                for (int ofm = 0; ofm < jcp.oc; ofm++) {
                    diff_bias_prv(ithr, ofm) = 0.0f;
                }
            }
#pragma omp for nowait
            for (int bofm = 0; bofm < jcp.oc / simd_w; bofm++) {
#pragma omp simd
                for (int v = 0; v < simd_w; v++)
                    diff_bias(bofm, v) = 0.0f;
            }
        }
    }

#pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
#pragma omp for nowait collapse(3)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                    float *transb = jcp.ver == ver_4fma
                        ? &(trans_buffer(ithread, 0))
                        : NULL;
                    diff_src_transform_bwd_weights_ver(img, jcp,
                        &(diff_src(img, ifm1 * jcp.ic_block + ifm2,
                            0, 0, 0)),
                        &(V(ifm1, 0, 0, 0, ifm2, 0, 0, 0)),
                        transb,
                        kernel_->transpose_4fma_ker);
                }
            }
        }
    }

#pragma omp parallel num_threads(nthreads)
#pragma omp for nowait collapse(3)
    for (int img = 0; img < jcp.mb; img++) {
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                const int ithread = omp_get_thread_num();
                float *dbias = jcp.with_bias
                    ? &(diff_bias_prv(ithread,
                                simd_w * (ofm1 * jcp.oc_block + ofm2)))
                    : NULL;
                diff_dst_transform_bwd_weights_ver(img, jcp,
                        &(diff_dst(img, ofm1 * jcp.oc_block + ofm2, 0, 0, 0)),
                        &(M(ofm1, 0, 0, 0, ofm2, 0, 0, 0)), dbias);
            }
        }
    }

    size_t input_starts[max_threads_number];
    size_t input_ends[max_threads_number];
    int th_counter = 0;
#pragma omp parallel firstprivate(th_counter) \
    num_threads(nthreads)
#pragma omp for nowait collapse(5) schedule(static)
    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int oj = 0; oj < alpha; oj++) {
                for (int oi = 0; oi < alpha; oi++) {
                    for (int tile_block = 0; tile_block < jcp.tile_block;
                            tile_block++) {
                        int ithr = omp_get_thread_num();
                        if (th_counter == 0) {
                            input_starts[ithr] = (float *)&(Us(ithr, ifm1, ofm1,
                                oj, oi, 0, 0, 0, 0)) - (float *)&(Us(ithr, 0, 0,
                                    0, 0, 0, 0, 0, 0));
                            input_ends[ithr] = input_starts[ithr]
                                    + jcp.oc_block * jcp.ic_block
                                      * jcp.ic_simd_block * jcp.oc_simd_block;
                        }
                        else if (tile_block == 0) {
                            input_ends[ithr] += jcp.oc_block * jcp.ic_block
                                * jcp.ic_simd_block * jcp.oc_simd_block;
                        }
                        if (th_counter == 0 || tile_block == 0) {
                            kernel_->gemm_loop_ker_first_iter(
                                    &(Us(ithr, ifm1, ofm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ofm1, oj, oi, tile_block, 0, 0, 0, 0)),
                                    &(V(ifm1, oj, oi, tile_block, 0, 0, 0, 0)));
                        } else {
                            kernel_->gemm_loop_ker(
                                    &(Us(ithr, ifm1, ofm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ofm1, oj, oi, tile_block, 0, 0, 0, 0)),
                                    &(V(ifm1, oj, oi, tile_block, 0, 0, 0, 0)));
                        }
                        th_counter++;
                    }
                }
            }
        }
    }

    // Reduce diff-weights
    {
        float *output = &(U(0, 0, 0, 0, 0, 0, 0, 0));
        size_t nelems = jcp.ic * jcp.oc * alpha * alpha;
        float *input_ptrs[max_threads_number];
        for (int i = 0; i < nthreads; i++)
            input_ptrs[i] = output + nelems * (i + 1);
        subarray_sum(
                nthreads, output, nelems, input_ptrs, input_starts, input_ends);
    }

#pragma omp parallel
#pragma omp for collapse(4)
    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++) {
                    diff_weights_transform_bwd_weights(jcp,
                            &(diff_weights(ofm1 * jcp.oc_block + ofm2,
                                    ifm1 * jcp.ic_block + ifm2,
                                    0, 0, 0, 0)),
                            &(U(ifm1, ofm1, 0, 0, ofm2, ifm2, 0, 0)));
                }
            }
        }
    }

#pragma omp parallel
    if (jcp.with_bias) {
#pragma omp for
        for (int ofm1 = 0; ofm1 < jcp.oc / simd_w; ofm1++) {
            for (int ithr = 0; ithr < nthreads; ithr++) {
                float* base_bias_ptr = &(diff_bias(ofm1, 0));
                float* base_bias_prv_ptr = &(diff_bias_prv(
                            ithr * jcp.oc + ofm1 * simd_w));
#pragma omp simd
                for (int ofm2 = 0; ofm2 < simd_w; ofm2++) {
                    base_bias_ptr[ofm2] += base_bias_prv_ptr[ofm2];
                }
            }
        }
    }
}

void jit_avx512_core_convolution_winograd_bwd_weights_t::
_execute_backward_weights_SDGtWo()
{
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();

    auto diff_src_transform_bwd_weights_ver_tile = jcp.ver == ver_4fma ?
            diff_src_transform_bwd_weights_tile<true> :
            diff_src_transform_bwd_weights_tile<false>;
    auto diff_dst_transform_bwd_weights_ver = jcp.with_bias
                                  ? diff_dst_transform_bwd_weights_tile<true>
                                  : diff_dst_transform_bwd_weights_tile<false>;

    array_offset_calculator<float, 5> diff_src((float *)this->input_memory(0),
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 3> diff_bias(
            (float *)this->memory(1), jcp.nb_oc, jcp.oc_block, simd_w);

    array_offset_calculator<float, 8> Us((float *)(scratchpad_->U_ptr()),
            0, jcp.nb_ic, alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 7> M((float *)(scratchpad_->M_ptr()),
            0, alpha, alpha,
            jcp.oc_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur * jcp.tile_4fma,
            jcp.oc_simd_block);

    array_offset_calculator<float, 8> V((float *)(scratchpad_->V_ptr()),
            0, jcp.nb_ic, alpha, alpha,
            jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block * jcp.tile_4fma);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()),
            nthreads, jcp.oc / jcp.nb_oc);

    for (int ofm1 = 0; ofm1 < jcp.nb_oc; ++ofm1) {
        int th_counter = 0;

#pragma omp parallel
        {
            if (jcp.with_bias) {
#pragma omp for nowait collapse(2)
                for (int ithr = 0; ithr < nthreads; ithr++) {
                    for (int ofm = 0; ofm < jcp.oc / jcp.nb_oc; ofm++) {
                        diff_bias_prv(ithr, ofm) = 0.0f;
                    }
                }
#pragma omp for nowait
                for (int bofm = 0; bofm < jcp.oc_block; bofm++) {
#pragma omp simd
                    for (int v = 0; v < simd_w; v++)
                        diff_bias(ofm1, bofm, v) = 0.0f;
                }
            }
        }

#pragma omp parallel firstprivate(th_counter) num_threads(nthreads)
#pragma omp for nowait
        for (int tile_block = 0; tile_block < jcp.tile_block; tile_block++) {
            int ithr = omp_get_thread_num();
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                    diff_src_transform_bwd_weights_ver_tile(tile_block, jcp,
                            &(diff_src(0, ifm1 * jcp.ic_block + ifm2, 0, 0, 0)),
                            &(V(ithr, ifm1, 0, 0, ifm2, 0, 0, 0)),
                            kernel_->transpose_4fma_ker);
                }
            }

            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                float *dbias = jcp.with_bias
                    ? &(diff_bias_prv(ithr, simd_w * ofm2))
                    : NULL;
                diff_dst_transform_bwd_weights_ver(tile_block, jcp,
                        &(diff_dst(0, ofm1 * jcp.oc_block + ofm2, 0, 0, 0)),
                        &(M(ithr, 0, 0, ofm2, 0, 0, 0)),
                        dbias);
            }

            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                for (int oj = 0; oj < alpha; oj++) {
                    for (int oi = 0; oi < alpha; oi++) {
                        if (th_counter == 0)
                            kernel_->gemm_loop_ker_first_iter(
                                    &(Us(ithr, ifm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ithr, oj, oi, 0, 0, 0, 0)),
                                    &(V(ithr, ifm1, oj, oi, 0, 0, 0, 0)));
                        else
                            kernel_->gemm_loop_ker(
                                    &(Us(ithr, ifm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ithr, oj, oi, 0, 0, 0, 0)),
                                    &(V(ithr, ifm1, oj, oi, 0, 0, 0, 0)));
                    }
                }
            }
            th_counter++;
        }
        // Reduce diff-weights
        {
            float *output = (float *)(scratchpad_->U_ptr());
            size_t nelems
                    = jcp.ic * (jcp.oc / jcp.nb_oc) * alpha * alpha;
            float *input_ptrs[max_threads_number];
            for (int i = 0; i < nthreads; i++) {
                input_ptrs[i] = output + nelems * i;
            }
            array_sum(nthreads, output, nelems, input_ptrs);
        }

#pragma omp parallel
#pragma omp for collapse(3)
        for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++) {
                    diff_weights_transform_bwd_weights(jcp,
                            &(diff_weights(ofm1 * jcp.oc_block + ofm2,
                                    ifm1 * jcp.ic_block + ifm2,
                                    0, 0, 0, 0)),
                            &(Us(0, ifm1, 0, 0, ofm2, ifm2, 0, 0)));
                }
            }
        }

#pragma omp parallel
        if (jcp.with_bias) {
#pragma omp for
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                for (int ithr = 0; ithr < nthreads; ithr++) {
                    float* base_bias_ptr = &(diff_bias(ofm1, ofm2, 0));
                    float* base_bias_prv_ptr = &(diff_bias_prv(
                                ithr * jcp.oc_block * simd_w + ofm2 * simd_w));
#pragma omp simd
                    for (int ofm3 = 0; ofm3 < simd_w; ofm3++) {
                        base_bias_ptr[ofm3] += base_bias_prv_ptr[ofm3];
                    }
                }
            }
        }
    }
}

void jit_avx512_core_convolution_winograd_bwd_weights_t::
_execute_backward_weights_SDGt_W()
{
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();

    auto diff_src_transform_bwd_weights_ver_tile = jcp.ver == ver_4fma ?
            diff_src_transform_bwd_weights_tile<true> :
            diff_src_transform_bwd_weights_tile<false>;
    auto diff_dst_transform_bwd_weights_ver = jcp.with_bias
                                  ? diff_dst_transform_bwd_weights_tile<true>
                                  : diff_dst_transform_bwd_weights_tile<false>;

    array_offset_calculator<float, 5> diff_src((float *)this->input_memory(0),
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 2> diff_bias(
            (float *)this->memory(1), jcp.oc / simd_w, simd_w);

    array_offset_calculator<float, 8> U((float *)(scratchpad_->U_ptr()),
            jcp.nb_oc, jcp.nb_ic,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 9> Us((float *)(scratchpad_->U_ptr()),
            0, jcp.nb_oc, jcp.nb_ic,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 8> M((float *)(scratchpad_->M_ptr()),
            0, jcp.nb_oc, alpha, alpha, jcp.oc_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur * jcp.tile_4fma,
            jcp.oc_simd_block);

    array_offset_calculator<float, 8> V((float *)(scratchpad_->V_ptr()),
            0, jcp.nb_ic, alpha, alpha, jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block * jcp.tile_4fma);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()),
            nthreads, jcp.oc);

#pragma omp parallel
    {
        if (jcp.with_bias) {
#pragma omp for nowait collapse(2)
            for (int ithr = 0; ithr < nthreads; ithr++) {
                for (int ofm = 0; ofm < jcp.oc; ofm++) {
                    diff_bias_prv(ithr, ofm) = 0.0f;
                }
            }
#pragma omp for nowait
            for (int bofm = 0; bofm < jcp.oc / simd_w; bofm++) {
#pragma omp simd
                for (int v = 0; v < simd_w; v++)
                    diff_bias(bofm, v) = 0.0f;
            }
        }
    }

    int th_counter = 0;
#pragma omp parallel firstprivate(th_counter) num_threads(nthreads)
#pragma omp for nowait
    for (int tile_block = 0; tile_block < jcp.tile_block; tile_block++) {
        int ithr = omp_get_thread_num();

        for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
            for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                diff_src_transform_bwd_weights_ver_tile(tile_block, jcp,
                        &(diff_src(0, ifm1 * jcp.ic_block + ifm2,
                                0, 0, 0)),
                        &(V(ithr, ifm1, 0, 0, ifm2, 0, 0, 0)),
                        kernel_->transpose_4fma_ker);
            }
        }

        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                float *dbias = jcp.with_bias
                    ? &(diff_bias_prv(ithr,
                                simd_w * (ofm1 * jcp.oc_block + ofm2)))
                    : NULL;
                diff_dst_transform_bwd_weights_ver(tile_block, jcp,
                        &(diff_dst(0, ofm1 * jcp.oc_block + ofm2,
                                0, 0, 0)),
                        &(M(ithr, ofm1, 0, 0, ofm2, 0, 0, 0)),
                        dbias);
            }
        }

        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int oj = 0; oj < alpha; oj++) {
                for (int oi = 0; oi < alpha; oi++) {
                    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                        if (th_counter == 0)
                            kernel_->gemm_loop_ker_first_iter(
                                    &(Us(ithr, ofm1, ifm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ithr, ofm1, oj, oi, 0, 0, 0, 0)),
                                    &(V(ithr, ifm1, oj, oi, 0, 0, 0, 0)));
                        else
                            kernel_->gemm_loop_ker(
                                    &(Us(ithr, ofm1, ifm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ithr, ofm1, oj, oi, 0, 0, 0, 0)),
                                    &(V(ithr, ifm1, oj, oi, 0, 0, 0, 0)));
                    }
                }
            }
        }
        th_counter++;
    }

    // Reduce diff-weights
    {
        float *output = (float *)(scratchpad_->U_ptr());
        size_t nelems = jcp.ic * jcp.oc * alpha * alpha;
        float *input_ptrs[max_threads_number];
        for (int i = 0; i < nthreads; i++) {
            input_ptrs[i] = output + nelems * i;
        }
        array_sum(nthreads, output, nelems, input_ptrs);
    }

#pragma omp parallel
#pragma omp for collapse(4)
    for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
        for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++) {
                    diff_weights_transform_bwd_weights(jcp,
                            &(diff_weights(ofm1 * jcp.oc_block + ofm2,
                                    ifm1 * jcp.ic_block + ifm2,
                                    0, 0, 0, 0)),
                            &(U(ofm1, ifm1, 0, 0, ofm2, ifm2, 0, 0)));
                }
            }
        }
    }

#pragma omp parallel
    if (jcp.with_bias) {
#pragma omp for
        for (int ofm1 = 0; ofm1 < jcp.oc / simd_w; ofm1++) {
            for (int ithr = 0; ithr < nthreads; ithr++) {
                float* base_bias_ptr = &(diff_bias(ofm1, 0));
                float* base_bias_prv_ptr = &(diff_bias_prv(
                            ithr * jcp.oc + ofm1 * simd_w));
#pragma omp simd
                for (int ofm2 = 0; ofm2 < simd_w; ofm2++) {
                    base_bias_ptr[ofm2] += base_bias_prv_ptr[ofm2];
                }
            }
        }
    }
}
}
}
}
// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
