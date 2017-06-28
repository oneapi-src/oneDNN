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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

#include "jit_avx512_common_convolution_winograd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

void trans_I_4x4_3x3(float Iw[6][6][16], float I[6][6][16]) {
    float T[6][6][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];
    float t4[16];
    float t5[16];

#pragma unroll
    for (int i = 0; i < 6; i++) {
#pragma omp simd
        for (int v = 0; v < 16; v++) {
            t0[v] = -4.0f * I[2][i][v] + I[4][i][v];
            t1[v] = -4.0f * I[1][i][v] + I[3][i][v];
            t2[v] = I[4][i][v] - I[2][i][v];
            t3[v] = I[3][i][v] - I[1][i][v];
            t4[v] = -5.0f * I[2][i][v] + I[4][i][v];
            t5[v] = -5.0f * I[3][i][v] + I[5][i][v];

            T[0][i][v] = 4.0f * I[0][i][v] + t4[v];
            T[1][i][v] = t0[v] + t1[v];
            T[2][i][v] = t0[v] - t1[v];
            T[3][i][v] = 2.0f * t3[v] + t2[v];
            T[4][i][v] = -2.0f * t3[v] + t2[v];
            T[5][i][v] = 4.0f * I[1][i][v] + t5[v];
        }
    }
#pragma unroll
    for (int i = 0; i < 6; i++) {
#pragma omp simd
        for (int v = 0; v < 16; v++) {
            t0[v] = -4.0f * T[i][2][v] + T[i][4][v];
            t1[v] = -4.0f * T[i][1][v] + T[i][3][v];
            t2[v] = T[i][4][v] - T[i][2][v];
            t3[v] = T[i][3][v] - T[i][1][v];
            t4[v] = -5.0f * T[i][2][v] + T[i][4][v];
            t5[v] = -5.0f * T[i][3][v] + T[i][5][v];

            Iw[i][0][v] = 4.0f * T[i][0][v] + t4[v];
            Iw[i][1][v] = t0[v] + t1[v];
            Iw[i][2][v] = t0[v] - t1[v];
            Iw[i][3][v] = 2.0f * t3[v] + t2[v];
            Iw[i][4][v] = -2.0f * t3[v] + t2[v];
            Iw[i][5][v] = 4.0f * T[i][1][v] + t5[v];
        }
    }
}

void trans_W_4x4_3x3(float Fw_[6][6][16][16], float F[3][3][16][16]) {
    const float rcp4 = 1.0f / 4.0f;
    const float rcp6 = 1.0f / 6.0f;
    const float rcp12 = 1.0f / 12.0f;
    const float rcp24 = 1.0f / 24.0f;
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
                t0[k] = rcp6 * F[2][i][j][k];
                t1[k] = -t0[k] - rcp6 * F[0][i][j][k];
                t2[k] = t0[k] + rcp24 * F[0][i][j][k];
                T[0][i][k] = rcp4 * F[0][i][j][k];
                T[1][i][k] = t1[k] - rcp6 * F[1][i][j][k];
                T[2][i][k] = t1[k] + rcp6 * F[1][i][j][k];
                T[3][i][k] = t2[k] + rcp12 * F[1][i][j][k];
                T[4][i][k] = t2[k] - rcp12 * F[1][i][j][k];
                T[5][i][k] = F[2][i][j][k];
            }
        }
#pragma unroll
        for (int i = 0; i < 6; i++) {
#pragma omp simd
            for (int k = 0; k < 16; k++) {
                t0[k] = rcp6 * T[i][2][k];
                t1[k] = -t0[k] - rcp6 * T[i][0][k];
                t2[k] = t0[k] + rcp24 * T[i][0][k];
                Fw[0][k] = rcp4 * T[i][0][k];
                Fw[1][k] = t1[k] - rcp6 * T[i][1][k];
                Fw[2][k] = t1[k] + rcp6 * T[i][1][k];
                Fw[3][k] = t2[k] + rcp12 * T[i][1][k];
                Fw[4][k] = t2[k] - rcp12 * T[i][1][k];
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
            T[1][i][v] = t2[v] + t3[v] * 2.0f;
            T[2][i][v] = t0[v] + t1[v] * 4.0f;
            T[3][i][v] = t2[v] + t3[v] * 8.0f + Mw[5][i][v];
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
            O[i][1][v] = t2[v] + t3[v] * 2.0f;
            O[i][2][v] = t0[v] + t1[v] * 4.0f;
            O[i][3][v] = t2[v] + t3[v] * 8.0f + T[i][5][v];
        }
    }
}

void trans_W_3x3_4x4(float Fw[6][6][16], float F[4][6][16]) {
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

#pragma unroll
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
#pragma unroll
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

void trans_O_3x3_4x4(float Mw[6][6][16][16], float M[3][3][16][16]) {
    float T[4][6][16];
    float M_[3][16];
    float t0[16];
    float t1[16];
    float t2[16];

    for (int j = 0; j < 16; j++) {
#pragma unroll
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
#pragma unroll
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

void trans_I_4x4_3x3_wu(float Iw[6][6][16], float I[6][6][16]) {
    float T[6][6][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];
    float t4[16];
    float t5[16];

#pragma unroll
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

#pragma unroll
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

void trans_W_3x3_4x4_wu(float Fw[6][6][16], float F[4][6][16]) {
    float T[6][4][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];
    float t4[16];

#pragma unroll
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
#pragma unroll
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

void trans_O_3x3_4x4_wu(float Mw[6][6][16][16], float M[3][3][16][16]) {
    float T[3][6][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float M_[3][16];

    for (int j = 0; j < 16; j++) {
#pragma unroll
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
#pragma unroll
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

#pragma unroll
            for (int k = 0; k < 3; k++) {
#pragma omp simd
                for (int v = 0; v < 16; v++) {
                    M[i][k][j][v] = M_[k][v];
                }
            }
        }
    }
}

void src_transform_fwd(jit_conv_winograd_conf_t conv, float *inp, float *tinp)
{
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;
    const int ifwp = conv.iw + conv.l_pad;
    const int ifhp = conv.ih + conv.t_pad;
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];
    float(*input)[conv.ih][conv.iw][simd_w] = (decltype(input))(inp);
    float(*output)[alpha][conv.nb_ic * conv.bimg][conv.itiles * conv.jtiles]
                  [simd_w] = (decltype(output))(tinp);

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((conv.t_pad <= ydim) && (ydim < ifhp)) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((conv.l_pad <= xdim) && (xdim < ifwp)) {
#pragma omp simd
                            for (int v = 0; v < 16; v++) {
                                I[j][i][v] = input[0][ydim - conv.t_pad]
                                                  [xdim - conv.l_pad][v];
                            }
                        } else {
#pragma omp simd
                            for (int v = 0; v < 16; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < alpha; i++) {
#pragma omp simd
                        for (int v = 0; v < 16; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_I_4x4_3x3(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
#pragma omp simd
                    for (int v = 0; v < 16; v++) {
                        output[j][i][0][tj * conv.itiles + ti][v] = Iw[j][i][v];
                    }
                }
            }
        }
    }
}

void weight_transform_fwd(jit_conv_winograd_conf_t conv, float *wp, float *twp)
{
    const int simd_w = 16;
    const int alpha = 6;
    float(*input)[conv.nb_ic][3][3][simd_w][simd_w] = (decltype(input))(wp);
    float(*output)[alpha][conv.nb_ic * conv.nb_oc][simd_w][simd_w] =
        (decltype(output))(twp);
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[3][3][simd_w][simd_w];

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            for (int v1 = 0; v1 < 16; v1++) {
#pragma omp simd
                for (int v2 = 0; v2 < 16; v2++) {
                    F[j][i][v1][v2] = input[0][0][j][i][v1][v2];
                }
            }
        }
    }

    trans_W_4x4_3x3(Fw, F);

    for (int j = 0; j < alpha; j++) {
        for (int i = 0; i < alpha; i++) {
            for (int v1 = 0; v1 < 16; v1++) {
#pragma omp simd
                for (int v2 = 0; v2 < 16; v2++) {
                    output[j][i][0][v1][v2] = Fw[j][i][v1][v2];
                }
            }
        }
    }
}

template<bool with_bias>
void dst_transform_fwd(jit_conv_winograd_conf_t conv, float *toutp,
        float *outp, float *bias) {
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;
    const int total_tiles = conv.itiles * conv.jtiles;

    float (*output)[conv.ow][simd_w] = (decltype(output))(outp);
    float (*input)[alpha][conv.nb_oc * conv.bimg][total_tiles][simd_w] =
        (decltype(input))(toutp);
    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
#pragma omp simd
                    for (int v = 0; v < 16; v++) {
                        Ow[j][i][v] = input[j][i][0][tj * conv.itiles + ti][v];
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
                            for (int v = 0; v < 16; v++) {
                                output[ydim][xdim][v] = O[j][i][v] +
                                    (with_bias ? bias[v] : 0);
                            }
                        }
                    }
                }
            }
        }
    }
}

void diff_dst_transform_bwd_data(jit_conv_winograd_conf_t conv, float *inp,
        float *tinp) {
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;
    const int total_tiles = conv.itiles * conv.jtiles;
    const int l_pad_winograd = conv.iw + conv.r_pad - conv.ow;
    const int t_pad_winograd = conv.ih + conv.b_pad - conv.oh;
    const int ofwp = conv.ow + l_pad_winograd;
    const int ofhp = conv.oh + t_pad_winograd;
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];
    float(*input)[conv.oh][conv.ow][simd_w] = (decltype(input))(inp);
    float(*output)[alpha][conv.nb_oc * conv.bimg][total_tiles][simd_w] =
        (decltype(output))(tinp);

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    int xdim = ti * tile_size + i;
                    int ydim = tj * tile_size + j;
                    if ((l_pad_winograd <= xdim) && (xdim < ofwp) &&
                        (t_pad_winograd <= ydim) && (ydim < ofhp)) {
#pragma omp simd
                        for (int v = 0; v < 16; v++) {
                            I[j][i][v] = input[0][ydim - t_pad_winograd]
                                              [xdim - l_pad_winograd][v];
                        }
                    } else {
#pragma omp simd
                        for (int v = 0; v < 16; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_I_4x4_3x3(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
#pragma omp simd
                    for (int v = 0; v < 16; v++) {
                        output[j][i][0][tj * conv.itiles + ti][v]
                            = Iw[j][i][v];
                    }
                }
            }
        }
    }
}

void weight_transform_bwd_data(jit_conv_winograd_conf_t conv, float *wp,
        float *twp) {
    const int simd_w = 16;
    const int alpha = 6;
    float(*input)[conv.nb_ic][3][3][simd_w][simd_w] = (decltype(input))(wp);
    float(*output)[alpha][conv.nb_ic * conv.nb_oc][simd_w][simd_w] =
        (decltype(output))(twp);
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[3][3][simd_w][simd_w];

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            for (int v = 0; v < 16; v++) {
#pragma omp simd
                for (int k = 0; k < 16; k++) {
                    F[j][i][k][v] = input[0][0][2 - j][2 - i][v][k];
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
                    output[j][i][0][v][k] = Fw[j][i][v][k];
                }
            }
        }
    }
}

void diff_src_transform_bwd_data(jit_conv_winograd_conf_t conv, float *toutp,
                         float *outp) {
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;
    const int total_tiles = conv.itiles * conv.jtiles;

    float(*output)[conv.ih][conv.iw][simd_w] = (decltype(output))(outp);
    float(*input)[alpha][conv.nb_ic * conv.bimg][total_tiles][simd_w] =
        (decltype(input))(toutp);
    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
#pragma omp simd
                    for (int v = 0; v < 16; v++) {
                        Ow[j][i][v] = input[j][i][0][tj * conv.itiles + ti][v];
                    }
                }
            }

            trans_O_4x4_3x3(Ow, O);

            for (int j = 0; j < tile_size; j++) {
                for (int i = 0; i < tile_size; i++) {
                    int xdim = ti * tile_size + i;
                    int ydim = tj * tile_size + j;
                    if ((xdim < conv.iw) && (ydim < conv.ih)) {
#pragma omp simd
                        for (int v = 0; v < 16; v++) {
                            output[0][ydim][xdim][v] = O[j][i][v];
                        }
                    }
                }
            }
        }
    }
}

void diff_src_transform_bwd_weights(jit_conv_winograd_conf_t conv, float *inp,
        float *tinp) {
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;
    const int total_tiles = conv.itiles * conv.jtiles;
    const int ifwp = conv.iw + conv.l_pad;
    const int ifhp = conv.ih + conv.t_pad;
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];
    float(*input)[conv.ih][conv.iw][simd_w] = (decltype(input))(inp);
    float(*output)[alpha][conv.nb_ic * conv.bimg][total_tiles][simd_w] =
        (decltype(output))(tinp);

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((conv.t_pad <= ydim) && (ydim < ifhp)) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((conv.l_pad <= xdim) && (xdim < ifwp)) {
#pragma omp simd
                            for (int v = 0; v < 16; v++) {
                                I[j][i][v] = input[0][ydim - conv.t_pad]
                                                  [xdim - conv.l_pad][v];
                            }
                        } else {
#pragma omp simd
                            for (int v = 0; v < 16; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < alpha; i++) {
#pragma omp simd
                        for (int v = 0; v < 16; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_I_4x4_3x3_wu(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
#pragma omp simd
                    for (int v = 0; v < 16; v++) {
                        output[j][i][0][tj * conv.itiles + ti][v]
                            = Iw[j][i][v];
                    }
                }
            }
        }
    }
}

void diff_dst_transform_bwd_weights(jit_conv_winograd_conf_t conv, float *inp,
        float *tinp) {
    const int simd_w = 16;
    const int alpha = 6;
    const int tile_size = alpha - 2;
    const int total_tiles = conv.itiles * conv.jtiles;
    float(*input)[conv.oh][conv.ow][simd_w] = (decltype(input))(inp);
    float(*output)[alpha][conv.nb_oc * conv.bimg][total_tiles][simd_w] =
        (decltype(output))(tinp);
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < conv.oh) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < conv.ow) {
#pragma omp simd
                            for (int v = 0; v < 16; v++) {
                                I[j][i][v] = input[0][ydim][xdim][v];
                            }
                        } else {
#pragma omp simd
                            for (int v = 0; v < 16; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < alpha; i++) {
#pragma omp simd
                        for (int v = 0; v < 16; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_W_3x3_4x4_wu(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
#pragma omp simd
                    for (int v = 0; v < 16; v++) {
                        output[j][i][0][tj * conv.itiles + ti][v] = Iw[j][i][v];
                    }
                }
            }
        }
    }
}

void diff_weights_transform_bwd_weights(jit_conv_winograd_conf_t conv,
        float *wp, float *twp) {
    const int simd_w = 16;
    const int alpha = 6;
    float(*output)[conv.nb_ic][3][3][simd_w][simd_w] = (decltype(output))(wp);
    float(*input)[alpha][conv.nb_ic * conv.nb_oc][simd_w][simd_w] =
        (decltype(input))(twp);
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[3][3][simd_w][simd_w];

    for (int j = 0; j < alpha; j++) {
        for (int i = 0; i < alpha; i++) {
            for (int v = 0; v < 16; v++) {
#pragma omp simd
                for (int k = 0; k < 16; k++) {
                    Fw[j][i][v][k] = input[j][i][0][v][k];
                }
            }
        }
    }

    trans_O_3x3_4x4_wu(Fw, F);

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 16; k++) {
#pragma omp simd
                for (int l = 0; l < 16; l++) {
                    output[0][0][j][i][k][l] = F[j][i][k][l];
                }
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

    float(*src)[jcp.nb_ic][jcp.ih][jcp.iw][simd_w] =
        (decltype(src))(this->input_memory(0));
    float(*dst)[jcp.nb_oc][jcp.oh][jcp.ow][simd_w] =
        (decltype(dst))(this->memory());
    float(*weights)[jcp.nb_ic][jcp.kh][jcp.kw][simd_w][simd_w] =
        (decltype(weights))(this->input_memory(1));
    float (*bias)[simd_w] = (decltype(bias))(this->input_memory(2));

    float(*U)[alpha][jcp.nb_oc][jcp.nb_ic][simd_w][simd_w] = (decltype(U))(up_);
    float(*V)[alpha][alpha][jcp.nb_ic][jcp.bimg][jcp.jtiles][jcp.itiles]
             [simd_w] = (decltype(V))(vp_);
    float(*M)[alpha][alpha][jcp.nb_oc][jcp.bimg][jcp.jtiles][jcp.itiles]
             [simd_w] = (decltype(M))(mp_);

#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                src_transform_fwd(jcp, &(src[img][ifm1][0][0][0]),
                    &(V[img / jcp.bimg][0][0][ifm1][img % jcp.bimg][0][0][0]));
            }
        }

#pragma omp for collapse(2)
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                weight_transform_fwd(jcp, &(weights[ofm1][ifm1][0][0][0][0]),
                        &(U[0][0][ofm1][ifm1][0][0]));
            }
        }

#pragma omp for collapse(3)
        for (int img = 0; img < jcp.mb / jcp.bimg; img++) {
            for (int oj = 0; oj < alpha; oj++) {
                for (int oi = 0; oi < alpha; oi++) {
                    for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                        kernel_->gemm_loop_ker(
                            (float *)&(M[img][oj][oi][ofm1][0][0][0][0]),
                            (const float *)&(U[oj][oi][ofm1][0][0][0]),
                            (const float *)&(V[img][oj][oi][0][0][0][0][0]));
                    }
                }
            }
        }
        auto output_transform = jcp.with_bias
            ? dst_transform_fwd<true> : dst_transform_fwd<false>;
#pragma omp for collapse(2)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                output_transform(jcp, &M[img / jcp.bimg][0][0][ofm1]
                                 [img % jcp.bimg][0][0][0],
                                 &dst[img][ofm1][0][0][0], bias[ofm1]);
            }
        }
    } // end omp parallel
}

template void
_jit_avx512_common_convolution_winograd_fwd_t<true>::execute_forward();
template void
_jit_avx512_common_convolution_winograd_fwd_t<false>::execute_forward();

void jit_avx512_common_convolution_winograd_bwd_data_t::
    execute_backward_data() {
    const int simd_w = 16;
    const int alpha = 6;
    const auto &jcp = kernel_->jcp;

    float(*diff_src)[jcp.nb_ic][jcp.ih][jcp.iw][simd_w] =
        (decltype(diff_src))(this->memory());
    float(*diff_dst)[jcp.nb_oc][jcp.oh][jcp.ow][simd_w] =
        (decltype(diff_dst))(this->input_memory(0));
    float(*weights)[jcp.nb_ic][jcp.kh][jcp.kw][simd_w][simd_w] =
        (decltype(weights))(this->input_memory(1));

    float(*U)[alpha][jcp.nb_ic][jcp.nb_oc][simd_w][simd_w] = (decltype(U))(up_);
    float(*V)[alpha][alpha][jcp.nb_ic][jcp.bimg][jcp.jtiles][jcp.itiles]
             [simd_w] = (decltype(V))(vp_);
    float(*M)[alpha][alpha][jcp.nb_oc][jcp.bimg][jcp.jtiles][jcp.itiles]
             [simd_w] = (decltype(M))(mp_);

#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                diff_dst_transform_bwd_data(
                    jcp, &(diff_dst[img][ofm1][0][0][0]),
                    &(M[img / jcp.bimg][0][0][ofm1][img % jcp.bimg][0][0][0]));
            }
        }

#pragma omp for collapse(2)
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                weight_transform_bwd_data(jcp,
                        &(weights[ofm1][ifm1][0][0][0][0]),
                        &(U[0][0][ifm1][ofm1][0][0]));
            }
        }

#pragma omp for collapse(3)
        for (int img = 0; img < jcp.mb / jcp.bimg; img++) {
            for (int oj = 0; oj < alpha; oj++) {
                for (int oi = 0; oi < alpha; oi++) {
                    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                        kernel_->gemm_loop_ker(
                            (float *)&(V[img][oj][oi][ifm1][0][0][0][0]),
                            (const float *)&(U[oj][oi][ifm1][0][0][0]),
                            (const float *)&(M[img][oj][oi][0][0][0][0][0]));
                    }
                }
            }
        }

#pragma omp for collapse(2)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                diff_src_transform_bwd_data(jcp,
                        &(V[img / jcp.bimg][0][0][ifm1]
                            [img % jcp.bimg][0][0][0]),
                        &(diff_src[img][ifm1][0][0][0]));
            }
        }
    } // end omp parallel
}

void jit_avx512_common_convolution_winograd_bwd_weights_t::
    execute_backward_weights() {
    const int simd_w = 16;
    const int alpha = 6;
    const auto &jcp = kernel_->jcp;

    float(*diff_src)[jcp.nb_ic][jcp.ih][jcp.iw][simd_w]
        = (decltype(diff_src))(this->input_memory(0));
    float(*diff_dst)[jcp.nb_oc][jcp.oh][jcp.ow][simd_w]
        = (decltype(diff_dst))(this->input_memory(1));
    float(*diff_weights)[jcp.nb_ic][jcp.kh][jcp.kw][simd_w][simd_w]
        = (decltype(diff_weights))(this->memory(0));
    float(*diff_bias)[simd_w] = (decltype(diff_bias))(this->memory(1));

    float(*U)[alpha][jcp.nb_oc][jcp.nb_ic][simd_w][simd_w] = (decltype(U))(up_);
    float(*V)[alpha][alpha][jcp.nb_ic][jcp.bimg][jcp.jtiles][jcp.itiles]
             [simd_w] = (decltype(V))(vp_);
    float(*M)[alpha][alpha][jcp.nb_oc][jcp.bimg][jcp.jtiles][jcp.itiles]
             [simd_w] = (decltype(M))(mp_);

#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                diff_src_transform_bwd_weights(jcp,
                        &(diff_src[img][ifm1][0][0][0]),
                        &(V[img / jcp.bimg][0][0][ifm1]
                            [img % jcp.bimg][0][0][0]));
            }
        }

#pragma omp for collapse(2)
        for (int img = 0; img < jcp.mb; img++) {
            for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                diff_dst_transform_bwd_weights(jcp,
                        &(diff_dst[img][ofm1][0][0][0]),
                        &(M[img / jcp.bimg][0][0][ofm1]
                            [img % jcp.bimg][0][0][0]));
            }
        }

#pragma omp for collapse(3)
        for (int oj = 0; oj < alpha; oj++) {
            for (int oi = 0; oi < alpha; oi++) {
                for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
                    kernel_->gemm_loop_ker_first_img(
                        (float *)&(U[oj][oi][ofm1][0][0][0]),
                        (const float *)&(M[0][oj][oi][ofm1][0][0][0][0]),
                        (const float *)&(V[0][oj][oi][0][0][0][0][0]));

                    for (int img = 1; img < jcp.mb / jcp.bimg; img++) {
                        kernel_->gemm_loop_ker(
                            (float *)&(U[oj][oi][ofm1][0][0][0]),
                            (const float *)&(M[img][oj][oi][ofm1][0][0][0][0]),
                            (const float *)&(V[img][oj][oi][0][0][0][0][0]));
                    }
                }
            }
        }

#pragma omp for collapse(2)
        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                diff_weights_transform_bwd_weights(jcp,
                        &(diff_weights[ofm1][ifm1][0][0][0][0]),
                        &(U[0][0][ofm1][ifm1][0][0]));
            }
        }

        if (conf_.with_bias()) {
#pragma omp for
            for (int bofm = 0; bofm < jcp.nb_oc; bofm++) {
                for (int v = 0; v < 16; v++)
                    diff_bias[bofm][v] = 0.0f;
                for (int img = 0; img < jcp.mb; img++) {
                    for (int h = 0; h < jcp.oh; h++) {
                        for (int w = 0; w < jcp.ow; w++) {
#pragma omp simd
                            for (int v = 0; v < 16; v++) {
                                diff_bias[bofm][v] +=
                                    diff_dst[img][bofm][h][w][v];
                            }
                        }
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
