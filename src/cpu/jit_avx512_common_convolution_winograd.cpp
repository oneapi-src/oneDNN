/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "jit_avx512_common_convolution_winograd.hpp"

#ifndef _MSC_VER
#define pragma_unroll _Pragma("unroll")
#else
#define pragma_unroll
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <typename Telem, size_t Tdims>
struct array_offset_calculator {
    template <typename... Targs>
    array_offset_calculator(Telem *base, Targs... Fargs) : _dims{ Fargs... }
    {
        _base_ptr = base;
    }
    template <typename... Targs>
    inline Telem &operator()(Targs... Fargs)
    {
        return *(_base_ptr + _offset(1, Fargs...));
    }

private:
    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t element)
    {
        return element;
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element)
    {
        return element + (_dims[dimension] * theta);
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element,
            Targs... Fargs)
    {
        size_t t_prime = element + (_dims[dimension] * theta);
        return _offset(dimension + 1, t_prime, Fargs...);
    }

    Telem *_base_ptr;
    const int _dims[Tdims];
};

void trans_W_4x4_3x3(float Fw_[6][6][16][16], float F[3][3][16][16]) {
    float Fw[6][16];
    float T[6][3][16];
    float t0[16];
    float t1[16];
    float t2[16];

    for (int j = 0; j < 16; j++) {
#pragma unroll
        for (int i = 0; i < 3; i++) {
#pragma omp simd
            for (int k = 0; k < 16; k++) {
                t0[k] = 0.26890756302521f * F[2][i][j][k];
                t1[k] = -t0[k] - 0.688403361344538f * F[0][i][j][k];
                t2[k] = t0[k] + 0.119514472455649f * F[0][i][j][k];

                T[0][i][k] = 1.13777777777778f * F[0][i][j][k];
                T[1][i][k] = t1[k] - 0.430252100840336f * F[1][i][j][k];
                T[2][i][k] = t1[k] + 0.430252100840336f * F[1][i][j][k];
                T[3][i][k] = t2[k] + 0.179271708683473f * F[1][i][j][k];
                T[4][i][k] = t2[k] - 0.179271708683473f * F[1][i][j][k];
                T[5][i][k] = F[2][i][j][k];
            }
        }
#pragma unroll
        for (int i = 0; i < 6; i++) {
#pragma omp simd
            for (int k = 0; k < 16; k++) {
                t0[k] = 0.26890756302521f * T[i][2][k];
                t1[k] = -t0[k] - 0.688403361344538f * T[i][0][k];
                t2[k] = t0[k] + 0.119514472455649f * T[i][0][k];

                Fw[0][k] = 1.13777777777778f * T[i][0][k];
                Fw[1][k] = t1[k] - 0.430252100840336f * T[i][1][k];
                Fw[2][k] = t1[k] + 0.430252100840336f * T[i][1][k];
                Fw[3][k] = t2[k] + 0.179271708683473f * T[i][1][k];
                Fw[4][k] = t2[k] - 0.179271708683473f * T[i][1][k];
                Fw[5][k] = T[i][2][k];
#pragma unroll
                for (int l = 0; l < 6; l++) {
                    Fw_[i][l][j][k] = Fw[l][k];
                }
            }
        }
    }
}

void trans_O_4x4_3x3(float Mw[6][6][16], float O[4][4][16]) {
    float T[4][6][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];

#pragma unroll
    for (int i = 0; i < 6; i++) {
#pragma omp simd
        for (int v = 0; v < 16; v++) {
            t0[v] = Mw[1][i][v] + Mw[2][i][v];
            t1[v] = Mw[3][i][v] + Mw[4][i][v];
            t2[v] = Mw[1][i][v] - Mw[2][i][v];
            t3[v] = Mw[3][i][v] - Mw[4][i][v];

            T[0][i][v] = t0[v] + t1[v] + Mw[0][i][v];
            T[1][i][v] = t2[v] * 0.625f + t3[v] * 1.5f;
            T[2][i][v] = t0[v] * 0.390625f + t1[v] * 2.25f;
            T[3][i][v] = t2[v] * 0.244140625f + t3[v] * 3.375f + Mw[5][i][v];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma omp simd
        for (int v = 0; v < 16; v++) {
            t0[v] = T[i][1][v] + T[i][2][v];
            t1[v] = T[i][3][v] + T[i][4][v];
            t2[v] = T[i][1][v] - T[i][2][v];
            t3[v] = T[i][3][v] - T[i][4][v];

            O[i][0][v] = t0[v] + t1[v] + T[i][0][v];
            O[i][1][v] = t2[v] * 0.625f + t3[v] * 1.5f;
            O[i][2][v] = t0[v] * 0.390625f + t1[v] * 2.25f;
            O[i][3][v] = t2[v] * 0.244140625f + t3[v] * 3.375f + T[i][5][v];
        }
    }
}


void trans_W_3x3_4x4(float Fw[6][6][16], float F[4][6][16])
{
    const float rcp3 = 1.0f / 3.0f;
    const float rcp4 = 1.0f / 4.0f;
    const float rcp6 = 1.0f / 6.0f;
    const float rcp12 = 1.0f / 12.0f;
    const float rcp24 = 1.0f / 24.0f;
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];
    float t4[16];
    float T[6][4][16];

pragma_unroll
    for (int i = 0; i < 4; i++) {
#pragma omp simd
        for (int j = 0; j < 16; j++) {
            t0[j] = F[2][i][j] * rcp6;
            t1[j] = F[0][i][j] * -rcp6 - t0[j];
            t2[j] = F[0][i][j] * rcp24 + t0[j];
            t3[j] = (F[1][i][j] + F[3][i][j]) * rcp6;
            t4[j] = F[1][i][j] * rcp12 + F[3][i][j] * rcp3;

            T[0][i][j] = F[0][i][j] * rcp4;
            T[1][i][j] = t1[j] - t3[j];
            T[2][i][j] = t1[j] + t3[j];
            T[3][i][j] = t2[j] + t4[j];
            T[4][i][j] = t2[j] - t4[j];
            T[5][i][j] = F[3][i][j];
        }
    }
pragma_unroll
    for (int i = 0; i < 6; i++) {
#pragma omp simd
        for (int j = 0; j < 16; j++) {
            t0[j] = T[i][2][j] * rcp6;
            t1[j] = T[i][0][j] * -rcp6 - t0[j];
            t2[j] = T[i][0][j] * rcp24 + t0[j];
            t3[j] = (T[i][1][j] + T[i][3][j]) * rcp6;
            t4[j] = T[i][1][j] * rcp12 + T[i][3][j] * rcp3;

            Fw[i][0][j] = T[i][0][j] * rcp4;
            Fw[i][1][j] = t1[j] - t3[j];
            Fw[i][2][j] = t1[j] + t3[j];
            Fw[i][3][j] = t2[j] + t4[j];
            Fw[i][4][j] = t2[j] - t4[j];
            Fw[i][5][j] = T[i][3][j];
        }
    }
}

void trans_O_3x3_4x4(float Mw[6][6][16][16], float M[3][3][16][16])
{
    float T[4][6][16];
    float M_[3][16];
    float t0[16];
    float t1[16];
    float t2[16];

    for (int j = 0; j < 16; j++) {
pragma_unroll
        for (int i = 0; i < 6; i++) {
#pragma omp simd
            for (int l = 0; l < 16; l++) {
                t0[l] = Mw[1][i][j][l] + Mw[2][i][j][l];
                t1[l] = Mw[3][i][j][l] + Mw[4][i][j][l];
                t2[l] = t1[l] * 4.0f + Mw[5][i][j][l];

                T[0][i][l] = Mw[0][i][j][l] + t0[l] + t1[l];
                T[1][i][l] = (Mw[1][i][j][l] - Mw[2][i][j][l]) +
                             2.0f * (Mw[3][i][j][l] - Mw[4][i][j][l]);
                T[2][i][l] = t0[l] + t2[l];
            }
        }
pragma_unroll
        for (int i = 0; i < 3; i++) {
#pragma omp simd
            for (int l = 0; l < 16; l++) {
                t0[l] = T[i][1][l] + T[i][2][l];
                t1[l] = T[i][3][l] + T[i][4][l];
                t2[l] = t1[l] * 4.0f + T[i][5][l];

                M_[0][l] = T[i][0][l] + t0[l] + t1[l];
                M_[1][l] = (T[i][1][l] - T[i][2][l]) +
                           2.0f * (T[i][3][l] - T[i][4][l]);
                M_[2][l] = t0[l] + t2[l];

                for (int k = 0; k < 3; k++) {
                    M[i][k][j][l] = M_[k][l];
                }
            }
        }
    }
}

void trans_I_4x4_3x3(float Iw[6][6][16], float I[6][6][16])
{
    float T[6][6][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];
    float t4[16];
    float t5[16];

pragma_unroll
    for (int i = 0; i < 6; i++) {
#pragma omp simd
        for (int v = 0; v < 16; v++) {
            t0[v] = I[2][i][v] * -2.25f + I[4][i][v];
            t1[v] = I[1][i][v] * -2.25f + I[3][i][v];
            t2[v] = I[2][i][v] * -0.390625f + I[4][i][v];
            t3[v] = I[1][i][v] * -0.390625f + I[3][i][v];
            t4[v] = I[0][i][v] * 0.87890625f + I[4][i][v];
            t5[v] = I[1][i][v] * 0.87890625f + I[5][i][v];

            T[0][i][v] = I[2][i][v] * -2.640625f + t4[v];
            T[1][i][v] = t1[v] * 0.625f + t0[v];
            T[2][i][v] = t1[v] * -0.625f + t0[v];
            T[3][i][v] = t3[v] * 1.5f + t2[v];
            T[4][i][v] = t3[v] * -1.5f + t2[v];
            T[5][i][v] = I[3][i][v] * -2.640625f + t5[v];
        }
    }

pragma_unroll
    for (int i = 0; i < 6; i++) {
#pragma omp simd
        for (int v = 0; v < 16; v++) {
            t0[v] = T[i][2][v] * -2.25f + T[i][4][v];
            t1[v] = T[i][1][v] * -2.25f + T[i][3][v];
            t2[v] = T[i][2][v] * -0.390625f + T[i][4][v];
            t3[v] = T[i][1][v] * -0.390625f + T[i][3][v];
            t4[v] = T[i][0][v] * 0.87890625f + T[i][4][v];
            t5[v] = T[i][1][v] * 0.87890625f + T[i][5][v];

            Iw[i][0][v] = T[i][2][v] * -2.640625f + t4[v];
            Iw[i][1][v] = t1[v] * 0.625f + t0[v];
            Iw[i][2][v] = t1[v] * -0.625f + t0[v];
            Iw[i][3][v] = t3[v] * 1.5f + t2[v];
            Iw[i][4][v] = t3[v] * -1.5f + t2[v];
            Iw[i][5][v] = T[i][3][v] * -2.640625f + t5[v];
        }
    }
}

void trans_W_3x3_4x4_wu(float Fw[6][6][16], float F[4][6][16])
{
    float T[6][4][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];
    float t4[16];

pragma_unroll
    for (int i = 0; i < 4; i++) {
#pragma omp simd
        for (int v = 0; v < 16; v++) {
            t0[v] = F[2][i][v] * 0.26890756302521f;
            t1[v] = F[0][i][v] * -0.688403361344538f - t0[v];
            t2[v] = F[0][i][v] * 0.119514472455649f + t0[v];
            t3[v] = F[1][i][v] * 0.430252100840336f +
                    F[3][i][v] * 0.168067226890756f;
            t4[v] = F[1][i][v] * 0.179271708683473f +
                    F[3][i][v] * 0.403361344537815f;

            T[0][i][v] = F[0][i][v] * 1.13777777777778f;
            T[1][i][v] = t1[v] - t3[v];
            T[2][i][v] = t1[v] + t3[v];
            T[3][i][v] = t2[v] + t4[v];
            T[4][i][v] = t2[v] - t4[v];
            T[5][i][v] = F[3][i][v];
        }
    }
pragma_unroll
    for (int i = 0; i < 6; i++) {
        for (int v = 0; v < 16; v++) {
            t0[v] = T[i][2][v] * 0.26890756302521f;
            t1[v] = T[i][0][v] * -0.688403361344538f - t0[v];
            t2[v] = T[i][0][v] * 0.119514472455649f + t0[v];
            t3[v] = T[i][1][v] * 0.430252100840336f +
                    T[i][3][v] * 0.168067226890756f;
            t4[v] = T[i][1][v] * 0.179271708683473f +
                    T[i][3][v] * 0.403361344537815f;

            Fw[i][0][v] = T[i][0][v] * 1.13777777777778f;
            Fw[i][1][v] = t1[v] - t3[v];
            Fw[i][2][v] = t1[v] + t3[v];
            Fw[i][3][v] = t2[v] + t4[v];
            Fw[i][4][v] = t2[v] - t4[v];
            Fw[i][5][v] = T[i][3][v];
        }
    }
}

void trans_O_3x3_4x4_wu(float Mw[6][6][16][16], float M[3][3][16][16])
{
    float T[3][6][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float M_[3][16];

    for (int j = 0; j < 16; j++) {
pragma_unroll
        for (int i = 0; i < 6; i++) {
#pragma omp simd
            for (int v = 0; v < 16; v++) {
                t0[v] = Mw[1][i][j][v] + Mw[2][i][j][v];
                t1[v] = Mw[3][i][j][v] + Mw[4][i][j][v];
                t2[v] = t1[v] * 2.25f + Mw[5][i][j][v];

                T[0][i][v] = Mw[0][i][j][v] + t0[v] + t1[v];
                T[1][i][v] = 0.625f * (Mw[1][i][j][v] - Mw[2][i][j][v]) +
                             1.5f * (Mw[3][i][j][v] - Mw[4][i][j][v]);
                T[2][i][v] = t0[v] * 0.390625f + t2[v];
            }
        }
pragma_unroll
        for (int i = 0; i < 3; i++) {
#pragma omp simd
            for (int v = 0; v < 16; v++) {
                t0[v] = T[i][1][v] + T[i][2][v];
                t1[v] = T[i][3][v] + T[i][4][v];
                t2[v] = t1[v] * 2.25f + T[i][5][v];

                M_[0][v] = T[i][0][v] + t0[v] + t1[v];
                M_[1][v] = 0.625f * (T[i][1][v] - T[i][2][v]) +
                           1.5f * (T[i][3][v] - T[i][4][v]);
                M_[2][v] = t0[v] * 0.390625f + t2[v];
            }

pragma_unroll
            for (int k = 0; k < 3; k++) {
#pragma omp simd
                for (int v = 0; v < 16; v++) {
                    M[i][k][j][v] = M_[k][v];
                }
            }
        }
    }
}

void inline stream_ps(float *dest, float *data) {
    const int simd_w = 16;
#ifdef __INTEL_COMPILER
    _mm512_stream_ps(dest, *((__m512 *)data));
#else
# pragma omp simd
    for (int v=0; v < simd_w; v++) dest[v] = data[v];
#endif
}

void src_transform_fwd(int image, jit_conv_winograd_conf_t conv,
        float *inp, float *tinp)
{
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;
    const int ifwp = conv.iw + conv.l_pad;
    const int ifhp = conv.ih + conv.t_pad;
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
            conv.mb, conv.ic/simd_w, conv.ih, conv.iw, simd_w);
    array_offset_calculator<float, 8> output(tinp,
            conv.tile_block, alpha, alpha,
            conv.nb_tile_block_ur, conv.nb_ic, conv.ic_block,
            conv.tile_block_ur, simd_w);

    int tile_base_index = image * conv.itiles * conv.jtiles;
    int tile_block_ur = tile_base_index % conv.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / conv.tile_block_ur) % conv.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / conv.tile_block_ur) / conv.nb_tile_block_ur;

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((conv.t_pad <= ydim) && (ydim < ifhp)) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((conv.l_pad <= xdim) && (xdim < ifwp)) {
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

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    stream_ps(
                            &(output(
                                    tile_block, j, i,
                                    nb_tile_block_ur, 0, 0,
                                    tile_block_ur, 0)),
                            Iw[j][i]);
                }
            }
            tile_block_ur++;
            if (tile_block_ur >= conv.tile_block_ur) {
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

void weight_transform_fwd(jit_conv_winograd_conf_t conv, float *wp, float *twp)
{
    const int simd_w = 16;
    const int alpha = 6;
    const int ic_simd_block = 16;
    const int oc_simd_block = 16;
    const int kh = 3;
    const int kw = 3;
    array_offset_calculator<float, 6> input(wp,
            conv.nb_oc * conv.oc_block,
            conv.nb_ic * conv.ic_block,
            conv.kh, conv.kw,
            ic_simd_block, oc_simd_block);
    array_offset_calculator<float, 8> output(twp,
            alpha, alpha,
            conv.nb_oc, conv.nb_ic,
            conv.oc_block, conv.ic_block,
            ic_simd_block, oc_simd_block);
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[kh][kw][simd_w][simd_w];

    for (int j = 0; j < kh; j++) {
        for (int i = 0; i < kw; i++) {
            for (int v1 = 0; v1 < simd_w; v1++) {
#pragma omp simd
                for (int v2 = 0; v2 < simd_w; v2++) {
                    F[j][i][v1][v2] = input(0, 0, j, i, v1, v2);
                }
            }
        }
    }

    trans_W_4x4_3x3(Fw, F);

    for (int j = 0; j < alpha; j++) {
        for (int i = 0; i < alpha; i++) {
            for (int v1 = 0; v1 < simd_w; v1++) {
#pragma omp simd
                for (int v2 = 0; v2 < simd_w; v2++) {
                    output(j, i, 0, 0, 0, 0, v1, v2) = Fw[j][i][v1][v2];
                }
            }
        }
    }
}

template <bool with_bias, bool with_relu>
void dst_transform_fwd(int image, jit_conv_winograd_conf_t conv, float *toutp,
        float *outp, float *bias)
{
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;
    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];

    array_offset_calculator<float, 8> input(toutp,
            conv.tile_block, conv.nb_oc,
            alpha, alpha,
            conv.nb_tile_block_ur, conv.oc_block,
            conv.tile_block_ur, simd_w);
    array_offset_calculator<float, 4> output(outp,
            conv.mb, conv.oh, conv.ow, simd_w);

    int tile_base_index = image * conv.itiles * conv.jtiles;
    int tile_block_ur = tile_base_index % conv.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / conv.tile_block_ur) % conv.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / conv.tile_block_ur) / conv.nb_tile_block_ur;

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
#pragma omp simd
                    for (int v = 0; v < simd_w; v++) {
                        Ow[j][i][v] = input(tile_block, 0,
                                j, i,
                                nb_tile_block_ur, 0,
                                tile_block_ur, v);
                    }
                }
            }

            trans_O_4x4_3x3(Ow, O);

            for (int j = 0; j < tile_size; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < conv.oh) {
                    for (int i = 0; i < tile_size; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < conv.ow) {
#pragma omp simd
                            for (int v = 0; v < simd_w; v++) {
                                O[j][i][v] += with_bias ? bias[v] : 0.0f;
                                O[j][i][v] = (with_relu && O[j][i][v] < 0.0f)
                                             ? O[j][i][v] * conv.relu_negative_slope
                                             : O[j][i][v];

                            }
                            stream_ps(&(output(0, ydim, xdim, 0)), O[j][i]);
                        }
                    }
                }
            }
            tile_block_ur++;
            if (tile_block_ur >= conv.tile_block_ur) {
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
void diff_dst_transform_bwd_data( int image, jit_conv_winograd_conf_t conv,
        float *inp, float *tinp)
{
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;
    const int l_pad_winograd = conv.iw + conv.r_pad - conv.ow;
    const int t_pad_winograd = conv.ih + conv.b_pad - conv.oh;
    const int ofwp = conv.ow + l_pad_winograd;
    const int ofhp = conv.oh + t_pad_winograd;
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
            conv.mb, conv.oc / simd_w, conv.oh, conv.ow, simd_w);
    array_offset_calculator<float, 8> output(tinp,
            conv.tile_block, alpha, alpha, conv.nb_tile_block_ur,
            conv.nb_oc, conv.oc_block, conv.tile_block_ur, simd_w);

    int tile_base_index = image * conv.itiles * conv.jtiles;
    int tile_block_ur = tile_base_index % conv.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / conv.tile_block_ur) % conv.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / conv.tile_block_ur) / conv.nb_tile_block_ur;

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            float *base = &(input(0, 0,
                        tj * tile_size - conv.t_pad,
                        ti * tile_size - conv.l_pad, 0));
            float *tmp = base;
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((t_pad_winograd <= ydim) && (ydim < ofhp)) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((l_pad_winograd <= xdim) && (xdim < ofwp)) {
#pragma omp simd
                            for (int v = 0; v < 16; v++) {
                                I[j][i][v] = *(tmp + v);
                            }
                        } else {
#pragma omp simd
                            for (int v = 0; v < 16; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                        tmp += simd_w;
                    }
                } else {
                    for (int i = 0; i < alpha; i++) {
#pragma omp simd
                        for (int v = 0; v < 16; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
                base += (conv.iw * simd_w);
                tmp = base;
            }

            trans_I_4x4_3x3(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    stream_ps(
                            &(output(tile_block, j, i, nb_tile_block_ur,
                                    0, 0, tile_block_ur, 0)),
                            Iw[j][i]);
                }
            }
            tile_block_ur++;
            if (tile_block_ur >= conv.tile_block_ur) {
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

void weight_transform_bwd_data(jit_conv_winograd_conf_t conv,
        float *wp, float *twp)
{
    const int simd_w = 16;
    const int alpha = 6;

    const int ic_simd_block = 16;
    const int oc_simd_block = 16;

    array_offset_calculator<float, 5> input(wp,
            conv.ic/simd_w, conv.kh, conv.kw, simd_w, simd_w);
    array_offset_calculator<float, 8> output(twp,
            alpha, alpha, conv.nb_ic, conv.nb_oc,
            conv.ic_block, conv.oc_block,
            oc_simd_block, ic_simd_block);

    float Fw[alpha][alpha][simd_w][simd_w];
    float F[3][3][simd_w][simd_w];

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            for (int v = 0; v < 16; v++) {
#pragma omp simd
                for (int k = 0; k < 16; k++) {
                    F[j][i][k][v] = input(0, 2 - j, 2 - i, v, k);
                }
            }
        }
    }

    trans_W_4x4_3x3(Fw, F);

    for (int j = 0; j < alpha; j++) {
        for (int i = 0; i < alpha; i++) {
            for (int v = 0; v < 16; v++) {
#pragma omp simd
                for (int k = 0; k < 16; k++) {
                    output(j, i, 0, 0, 0, 0, v, k) = Fw[j][i][v][k];
                }
            }
        }
    }
}

void diff_src_transform_bwd_data(int image, jit_conv_winograd_conf_t conv,
        float *toutp, float *outp)
{
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;

    array_offset_calculator<float, 8> input(toutp,
            conv.tile_block, conv.nb_ic,
            alpha, alpha,
            conv.nb_tile_block_ur, conv.ic_block,
            conv.tile_block_ur, simd_w);
    array_offset_calculator<float, 5> output(outp,
            conv.mb, conv.ic/simd_w, conv.ih, conv.iw, simd_w);

    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];

    int tile_base_index = image * conv.itiles * conv.jtiles;
    int tile_block_ur = tile_base_index % conv.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / conv.tile_block_ur) % conv.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / conv.tile_block_ur) / conv.nb_tile_block_ur;

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
#pragma omp simd
                    for (int v = 0; v < 16; v++) {
                        Ow[j][i][v] = input(tile_block, 0,
                                j, i,
                                nb_tile_block_ur, 0,
                                tile_block_ur, v);
                    }
                }
            }

            trans_O_4x4_3x3(Ow, O);

            for (int j = 0; j < tile_size; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < conv.ih) {
                    for (int i = 0; i < tile_size; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < conv.iw) {
                            stream_ps(
                                    &(output(0, 0, ydim, xdim, 0)),
                                    O[j][i]);
                        }
                    }
                }
            }
            tile_block_ur++;
            if (tile_block_ur >= conv.tile_block_ur) {
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

template <bool ver_4fma>
void diff_src_transform_bwd_weights(int image, jit_conv_winograd_conf_t conv,
        float *inp, float *tinp, float *Iw_temp,
        void (*transpose_4fma_ker)(float *, float *))
{
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = conv.alpha - 2;
    const int ifwp = conv.iw + conv.l_pad;
    const int ifhp = conv.ih + conv.t_pad;
    float I[conv.alpha][conv.alpha][simd_w];
    float Iw[conv.alpha][conv.alpha][simd_w];

    array_offset_calculator<float, 4> Iw_trans_temp(Iw_temp,
            alpha, alpha, conv.tile_4fma, simd_w);
    array_offset_calculator<float, 5> input(inp,
            conv.mb, conv.ic/simd_w, conv.ih, conv.iw, simd_w);
    array_offset_calculator<float, 8> output(tinp,
            conv.nb_ic, conv.alpha, conv.alpha,
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
            for (int j = 0; j < conv.alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((conv.t_pad <= ydim) && ydim < ifhp) {
                    for (int i = 0; i < conv.alpha; i++) {
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
                    for (int i = 0; i < conv.alpha; i++) {
#pragma omp simd
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }
            trans_I_4x4_3x3(Iw, I);

            if (ver_4fma) {
                for (int j = 0; j < conv.alpha; j++) {
                    for (int i = 0; i < conv.alpha; i++) {
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
                for (int j = 0; j < conv.alpha; j++) {
                    for (int i = 0; i < conv.alpha; i++) {
                        stream_ps(
                                &(output(0, j, i,
                                        tile_block, 0,
                                        nb_tile_block_ur, tile_block_ur, 0)),
                                Iw[j][i]);
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

        for (int j = 0; j < conv.alpha; j++) {
            for (int i = 0; i < conv.alpha; i++) {
                for (int tb = tile_4fma; tb < conv.tile_4fma; tb++) {
                    float *Iw_temp_base = &(Iw_trans_temp(j, i, tb, 0));
#                   pragma omp simd
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
    const int simd_w = 16;
    const int tile_size = conv.alpha - 2;
    const int total_tiles = conv.itiles * conv.jtiles + conv.tile_4fma_padding;
    float I[conv.alpha][conv.alpha][simd_w];
    float Iw[conv.alpha][conv.alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
            conv.mb, conv.oc/simd_w, conv.oh, conv.ow, conv.oc_simd_block);
    array_offset_calculator<float, 8> output(tinp,
            conv.nb_oc, conv.alpha, conv.alpha,
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
            for (int j = 0; j < conv.alpha; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < conv.oh) {
                    for (int i = 0; i < conv.alpha; i++) {
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
                    for (int i = 0; i < conv.alpha; i++) {
#pragma omp simd
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_W_3x3_4x4_wu(Iw, I);

            for (int j = 0; j < conv.alpha; j++) {
                for (int i = 0; i < conv.alpha; i++) {
                    /*TODO: Try instrinsic for casting into __m512*/
                    stream_ps(
                            &(output(0, j, i,
                                    tile_block, 0,
                                    nb_tile_block_ur,
                                    tile_block_ur, 0)),
                            Iw[j][i]);
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
    const int simd_w = 16;
    const int kh = 3;
    const int kw = 3;
    float Fw[conv.alpha][conv.alpha][simd_w][simd_w];
    float F[kh][kw][simd_w][simd_w];

    array_offset_calculator<float, 8> input(twp,
            conv.nb_ic, conv.nb_oc,
            conv.alpha, conv.alpha,
            conv.oc_block, conv.ic_block,
            conv.ic_simd_block, conv.oc_simd_block);
    array_offset_calculator<float, 6> output(wp,
            conv.oc/simd_w, conv.ic/simd_w,
            conv.kh, conv.kw,
            conv.ic_simd_block, conv.oc_simd_block);

    for (int j = 0; j < conv.alpha; j++) {
        for (int i = 0; i < conv.alpha; i++) {
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
                stream_ps(
                        &(output(0, 0, j, i, v, 0)),
                        F[j][i][v]);
            }
        }
    }
}


template <bool with_relu>
void _jit_avx512_common_convolution_winograd_fwd_t<with_relu>::execute_forward()
{
    const int simd_w = 16;
    const int alpha = 6;
    const auto &jcp = kernel_->jcp;

    auto output_transform = jcp.with_bias ? dst_transform_fwd<true, with_relu> :
                                            dst_transform_fwd<false, with_relu>;

    array_offset_calculator<float, 5> src((float *)this->input_memory(0),
            jcp.mb, jcp.ic/simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> dst((float *)this->memory(),
            jcp.mb, jcp.oc/simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> weights((float *)this->input_memory(1),
            jcp.oc/simd_w, jcp.ic/simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 2> bias((float *)this->input_memory(2),
            jcp.oc/simd_w, simd_w);

    char *base_ptr = scratchpad_buffer_->get();
    array_offset_calculator<float, 8> M(
            (float *)(base_ptr + sp_offsets_.mp_offset_),
            jcp.tile_block, jcp.nb_oc,
            alpha, alpha,
            jcp.nb_tile_block_ur, jcp.oc_block,
            jcp.tile_block_ur, simd_w);
    array_offset_calculator<float, 8> U(
            (float *)(base_ptr + sp_offsets_.up_offset_),
            alpha, alpha,
            jcp.nb_oc, jcp.nb_ic,
            jcp.oc_block, jcp.ic_block,
            simd_w, simd_w);
    array_offset_calculator<float, 8> V(
            (float *)(base_ptr + sp_offsets_.vp_offset_),
            jcp.tile_block, alpha, alpha,
            jcp.nb_tile_block_ur, jcp.nb_ic,
            jcp.ic_block, jcp.tile_block_ur, simd_w);

#pragma omp parallel
    {
////////////////////  New src_transform //////////////////////
#pragma omp for nowait collapse(3)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++) {
                    src_transform_fwd(img, jcp,
                            &(src(img, ifm1 * jcp.ic_block + ifm2, 0, 0, 0)),
                            &(V(0, 0, 0, 0, ifm1, ifm2, 0, 0)));
                }
            }
        }

//////////////////// New weight_transform ////////////////////
#pragma omp for nowait collapse(4)
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                    for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++) {
                        weight_transform_fwd(jcp,
                                &(weights(ofm1 * jcp.oc_block + ofm2,
                                        ifm1 * jcp.ic_block + ifm2,
                                        0, 0, 0, 0)),
                                &(U(0, 0, ofm1, ifm1, ofm2, ifm2, 0, 0)));
                    }
                }
            }
        }

#pragma omp barrier
        ////////////////////////// New GEMM //////////////////////////
        // We want the treads on the same tile to work on the same
        // image chunk to help data locality. We parallelize on
        // alpha^2 and nb_oc to share the input image in L2 as the
        // input image is generally larger than the filter and causes
        // many L2 misses,
        for (int tile_block = 0; tile_block < jcp.tile_block; tile_block++) {
#pragma omp for collapse(3) nowait schedule(static)
            for (int oj = 0; oj < alpha; oj++) {
                for (int oi = 0; oi < alpha; oi++) {
                    for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                        for (int nb_tile_block_ur = 0;
                                nb_tile_block_ur < jcp.nb_tile_block_ur;
                                nb_tile_block_ur++) {
                            kernel_->gemm_loop_ker_first_iter(
                                    (float *)&(M(tile_block, ofm1,
                                            oj, oi,
                                            nb_tile_block_ur, 0,
                                            0, 0)),
                                    (const float *)&(U(oj, oi,
                                            ofm1, 0,
                                            0, 0, 0, 0)),
                                    (const float *)&(V(tile_block, oj, oi,
                                            nb_tile_block_ur, 0,
                                            0, 0, 0)));
                            for (int ifm1 = 1; ifm1 < jcp.nb_ic; ifm1++) {
                                kernel_->gemm_loop_ker(
                                        (float *)&(M(tile_block, ofm1,
                                                oj, oi,
                                                nb_tile_block_ur, 0,
                                                0, 0)),
                                        (const float *)&(U(oj, oi,
                                                ofm1, ifm1,
                                                0, 0, 0, 0)),
                                        (const float *)&(V(tile_block, oj, oi,
                                                nb_tile_block_ur, ifm1,
                                                0, 0, 0)));
                            }
                        }
                    }
                }
            }
        }

#pragma omp barrier
//////////////////// New output_transform ////////////////////
#pragma omp for collapse(3)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                    output_transform(img, jcp,
                            &(M(0, ofm1, 0, 0, 0, ofm2, 0, 0)),
                            &(dst(img, ofm1 * jcp.oc_block + ofm2, 0, 0, 0)),
                            &(bias(ofm1 * jcp.oc_block + ofm2, 0)));
                }
            }
        }
    }
}

template void
_jit_avx512_common_convolution_winograd_fwd_t<true>::execute_forward();
template void
_jit_avx512_common_convolution_winograd_fwd_t<false>::execute_forward();

void jit_avx512_common_convolution_winograd_bwd_data_t::execute_backward_data()
{
    const int simd_w = 16;
    const auto &jcp = kernel_->jcp;

    array_offset_calculator<float, 5> diff_src((float *)this->memory(),
            jcp.mb, jcp.ic/simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(0),
            jcp.mb, jcp.oc/simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> weights((float *)this->input_memory(1),
            jcp.oc/simd_w, jcp.ic/simd_w, jcp.kh, jcp.kw, simd_w, simd_w);

    char *base_ptr = scratchpad_buffer_->get();
    array_offset_calculator<float, 8> U(
            (float *)(base_ptr + sp_offsets_.up_offset_),
            jcp.alpha, jcp.alpha,
            jcp.nb_ic, jcp.nb_oc,
            jcp.ic_block, jcp.oc_block,
            simd_w, simd_w);
    array_offset_calculator<float, 8> V(
            (float *)(base_ptr + sp_offsets_.vp_offset_),
            jcp.tile_block, jcp.nb_ic,
            jcp.alpha, jcp.alpha,
            jcp.nb_tile_block_ur, jcp.ic_block,
            jcp.tile_block_ur, simd_w);
    array_offset_calculator<float, 8> M(
            (float *)(base_ptr + sp_offsets_.mp_offset_),
            jcp.tile_block, jcp.alpha, jcp.alpha,
            jcp.nb_tile_block_ur, jcp.nb_oc,
            jcp.oc_block, jcp.tile_block_ur, simd_w);

#pragma omp parallel
    {
//******************** New dst_transform ********************//
#pragma omp for collapse(3) nowait
        for (int img = 0; img < jcp.mb; img++) {
            for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                    diff_dst_transform_bwd_data(img, jcp,
                            &(diff_dst(img, ofm1 * jcp.oc_block + ofm2,
                                    0, 0, 0)),
                            &(M(0, 0, 0, 0, ofm1, ofm2, 0, 0)));
                }
            }
        }

#pragma omp for collapse(4) nowait
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                    for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++) {
                        weight_transform_bwd_data(jcp,
                                &(weights(ofm1 * jcp.oc_block + ofm2,
                                        ifm1 * jcp.ic_block + ifm2,
                                        0, 0, 0, 0)),
                                &(U(0, 0, ifm1, ofm1, ifm2, ofm2, 0, 0)));
                    }
                }
            }
        }

#pragma omp barrier
        for (int tile_block = 0; tile_block < jcp.tile_block; tile_block++) {
#pragma omp for collapse(3) nowait schedule(static)
            for (int oj = 0; oj < jcp.alpha; oj++) {
                for (int oi = 0; oi < jcp.alpha; oi++) {
                    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                        for (int nb_tile_block_ur = 0;
                                nb_tile_block_ur < jcp.nb_tile_block_ur;
                                nb_tile_block_ur++) {
                            kernel_->gemm_loop_ker_first_iter(
                                    (float *)&(V(tile_block, ifm1,
                                            oj, oi,
                                            nb_tile_block_ur, 0,
                                            0, 0)),
                                    (const float *)&(U(oj, oi,
                                            ifm1, 0,
                                            0, 0, 0, 0)),
                                    (const float *)&(M(tile_block, oj, oi,
                                            nb_tile_block_ur, 0,
                                            0, 0, 0)));
                            for (int ofm1 = 1; ofm1 < jcp.nb_oc; ofm1++) {
                                kernel_->gemm_loop_ker(
                                        (float *)&(V(tile_block, ifm1,
                                                oj, oi,
                                                nb_tile_block_ur, 0,
                                                0, 0)),
                                        (const float *)&(U(oj, oi,
                                                ifm1, ofm1,
                                                0, 0, 0, 0)),
                                        (const float *)&(M(tile_block, oj, oi,
                                                nb_tile_block_ur, ofm1,
                                                0, 0, 0)));
                            }
                        }
                    }
                }
            }
        }

#pragma omp barrier
#pragma omp for collapse(3)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++) {
                    diff_src_transform_bwd_data(img, jcp,
                            &(V(0, ifm1, 0, 0, 0, ifm2, 0, 0)),
                            &(diff_src(img, ifm1 * jcp.ic_block + ifm2,
                                    0, 0, 0)));
                }
            }
        }
    }
}

void jit_avx512_common_convolution_winograd_bwd_weights_t::
       execute_backward_weights()
{
    const int simd_w = 16;
    const auto &jcp = kernel_->jcp;
    int nthreads;

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

    char *base_ptr = scratchpad_buffer_->get();

    array_offset_calculator<float, 8> U(
            (float *)(base_ptr + sp_offsets_.up_offset_),
            jcp.nb_ic, jcp.nb_oc,
            jcp.alpha, jcp.alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 8> M(
            (float *)(base_ptr + sp_offsets_.mp_offset_),
            jcp.nb_oc, jcp.alpha, jcp.alpha,
            jcp.tile_block, jcp.oc_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur * jcp.tile_4fma,
            jcp.oc_simd_block);
    array_offset_calculator<float, 8> V(
            (float *)(base_ptr + sp_offsets_.vp_offset_),
            jcp.nb_ic, jcp.alpha, jcp.alpha,
            jcp.tile_block, jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block * jcp.tile_4fma);

    const int trans_buffer_size = jcp.alpha * jcp.alpha * jcp.tile_4fma
                                * jcp.ic_simd_block;
    array_offset_calculator<float, 2> trans_buffer(
            (float *)(base_ptr + sp_offsets_.diff_src_trans_offset_),
            omp_get_max_threads(),
            trans_buffer_size);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(base_ptr + sp_offsets_.biasu_offset_),
            omp_get_max_threads(),
            jcp.oc);

#pragma omp parallel
    {
        if (jcp.with_bias) {
            nthreads = omp_get_num_threads();
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
            for (int oj = 0; oj < jcp.alpha; oj++) {
                for (int oi = 0; oi < jcp.alpha; oi++) {
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
    } // end omp parallel
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
