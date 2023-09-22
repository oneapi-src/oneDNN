/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_CONV_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_CONV_REF_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "../test_utils_arr_fill.hpp"
#include <compiler/ir/graph/graph.hpp>
#include <test_utils.hpp>

enum dir_t {
    DIR_UNDEF = 0,
    FLAG_DAT = 1,
    FLAG_WEI = 2,
    FLAG_BIA = 4,
    FLAG_FWD = 32,
    FLAG_BWD = 64,
    FLAG_INF = 128,
    FWD_D = FLAG_FWD + FLAG_DAT,
    FWD_I = FLAG_FWD + FLAG_DAT + FLAG_INF,
    FWD_B = FLAG_FWD + FLAG_DAT + FLAG_BIA,
    BWD_D = FLAG_BWD + FLAG_DAT,
    BWD_DW = FLAG_BWD + FLAG_DAT + FLAG_WEI,
    BWD_W = FLAG_BWD + FLAG_WEI,
    BWD_WB = FLAG_BWD + FLAG_WEI + FLAG_BIA,
};

namespace sc = dnnl::impl::graph::gc;

inline int64_t src_off_f(const int64_t G, const int64_t IC, const int64_t ID,
        const int64_t IH, const int64_t IW, int64_t mb, int64_t g, int64_t ic,
        int64_t id, int64_t ih, int64_t iw) {
    return (((mb * IC + g * IC / G + ic) * ID + id) * IH + ih) * IW + iw;
}
inline int64_t wei_off_f(const int64_t G, const int64_t OC, const int64_t IC,
        const int64_t KD, const int64_t KH, const int64_t KW, int64_t g,
        int64_t oc, int64_t ic, int64_t kd, int64_t kh, int64_t kw) {
    return ((((g * OC / G + oc) * IC / G + ic) * KD + kd) * KH + kh) * KW + kw;
}
inline int64_t bia_off_f(
        const int64_t G, const int64_t OC, int64_t g, int64_t oc) {
    return g * OC / G + oc;
}
inline int64_t bn_off_f(
        const int64_t G, const int64_t OC, int64_t g, int64_t oc) {
    return g * OC / G + oc;
}
inline int64_t dst_off_f(const int64_t G, const int64_t OC, const int64_t OD,
        const int64_t OH, const int64_t OW, int64_t mb, int64_t g, int64_t oc,
        int64_t od, int64_t oh, int64_t ow) {
    return (((mb * OC + g * OC / G + oc) * OD + od) * OH + oh) * OW + ow;
}

template <typename src_type, typename wei_type, typename dst_type,
        typename bias_type = float>
void compute_ref_direct_fwd(const int64_t MB, const int64_t G, const int64_t OC,
        const int64_t IC, const int64_t IH, const int64_t IW, const int64_t OH,
        const int64_t OW, const int64_t KH, const int64_t KW, const int64_t SH,
        const int64_t SW, const int64_t PH, const int64_t PW, src_type *src_m,
        wei_type *wei_m, bias_type *bia_m, dst_type *dst_m, dir_t dir,
        float *mul_m = nullptr, float *add_m = nullptr, bool bn_relu = false,
        const int64_t OD = 1, const int64_t ID = 1, const int64_t SD = 1,
        const int64_t PD = 0, const int64_t KD = 1, const int64_t DD = 1,
        const int64_t DH = 1, const int64_t DW = 1, bool qconv = false) {
    /* help compiler optimize the code */

    const int64_t OCG = OC / G, ICG = IC / G;

    auto ker = [&](dst_type &d, int64_t g, int64_t mb, int64_t oc, int64_t od,
                       int64_t oh, int64_t ow) {
        const src_type *__restrict src_loc
                = (const src_type *)src_m + (mb * IC + g * ICG) * ID * IH * IW;
        const wei_type *__restrict wei_loc
                = (const wei_type *)wei_m + (g * OCG + oc) * ICG * KD * KH * KW;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t id = od * SD - PD + kd * DD;
            if (id < 0 || id >= ID) continue;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH - PH + kh * DH;
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW - PW + kw * DW;
                    if (iw < 0 || iw >= IW) continue;
                    for (int64_t ic = 0; ic < ICG; ++ic) {
                        int64_t src_off = ((ic * ID + id) * IH + ih) * IW + iw;
                        int64_t wei_off = ((ic * KD + kd) * KH + kh) * KW + kw;
                        d += src_loc[src_off] * wei_loc[wei_off];
                    }
                }
            }
        }
    };

    sc::utils::parallel_for(0, G * MB * OCG * OD * OH * OW, 1, [&](int64_t i) {
        int64_t g = i / (MB * OCG * OD * OH * OW),
                mb = i / (OCG * OD * OH * OW) % MB,
                oc = i / (OD * OH * OW) % OCG;
        int64_t od = i / (OH * OW) % OD, oh = i / OW % OH, ow = i % OW;
        const size_t dst_off
                = dst_off_f(G, OC, OD, OH, OW, mb, g, oc, od, oh, ow);
        dst_type &dst = ((dst_type *)dst_m)[dst_off];

        dst_type conv_res = 0;
        ker(conv_res, g, mb, oc, od, oh, ow);

        if (dir & FLAG_BIA) {
            const size_t bia_off = bia_off_f(G, OC, g, oc);
            if (qconv) {
                conv_res += ((int32_t *)bia_m)[bia_off];
            } else {
                conv_res += ((float *)bia_m)[bia_off];
            }
        }

        if (bn_relu) {
            // y = max((a*x+b),0)
            const size_t bn_off = bn_off_f(G, OC, g, oc);
            conv_res *= ((float *)mul_m)[bn_off];
            conv_res += ((float *)add_m)[bn_off];
            conv_res = std::max(conv_res, static_cast<dst_type>(0));
        }

        dst = conv_res;
    });
}

inline void compute_ref_direct_bwd_d(const int64_t MB, const int64_t G,
        const int64_t OC, const int64_t IC, const int64_t IH, const int64_t IW,
        const int64_t OH, const int64_t OW, const int64_t KH, const int64_t KW,
        const int64_t SH, const int64_t SW, const int64_t PH, const int64_t PW,
        float *diff_src_m, float *wei_m, float *bia_m, float *diff_dst_m,
        dir_t dir = dir_t::BWD_D, const int64_t OD = 1, const int64_t ID = 1,
        const int64_t SD = 1, const int64_t PD = 0, const int64_t KD = 1,
        const int64_t DD = 1, const int64_t DH = 1, const int64_t DW = 1) {
    /* help compiler optimize the code */

    const int64_t OCG = OC / G, ICG = IC / G;

    enum { precompute_size = 16 };
    const bool fast = false;

    /* pre-computes arrays of oh(ow) and kh(kw) for traversing in kernel */
    auto precompute_ok
            = [](int64_t i, int64_t O, int64_t K, int64_t S, int64_t P,
                      int64_t D, int64_t &num, int64_t *_o, int64_t *_k) {
                  assert(K <= precompute_size);
                  num = 0;
                  for (int64_t k = 0; k < K; ++k) {
                      int64_t o = i - k * D + P;
                      if (o < 0 || o % S) continue;
                      o /= S;
                      if (o >= O) continue;
                      _k[num] = k;
                      _o[num] = o;
                      ++num;
                  }
              };

    auto ker_fast = [&](float &ds, int64_t g, int64_t mb, int64_t ic,
                            int64_t id, int64_t ih, int64_t iw) {
        int64_t kd[precompute_size], od[precompute_size], num_d;
        int64_t kh[precompute_size], oh[precompute_size], num_h;
        int64_t kw[precompute_size], ow[precompute_size], num_w;
        precompute_ok(id, OD, KD, SD, PD, DD, num_d, od, kd);
        precompute_ok(ih, OH, KH, SH, PH, DH, num_h, oh, kh);
        precompute_ok(iw, OW, KW, SW, PW, DW, num_w, ow, kw);

        const float *__restrict diff_dst_loc = (const float *)diff_dst_m
                + (mb * OC + g * OCG) * OD * OH * OW;
        const float *__restrict wei_loc
                = (const float *)wei_m + ((g * OCG) * ICG + ic) * KD * KH * KW;

        for (int64_t d = 0; d < num_d; ++d)
            for (int64_t h = 0; h < num_h; ++h)
                for (int64_t w = 0; w < num_w; ++w) {
                    for (int64_t oc = 0; oc < OCG; ++oc) {
                        const int64_t diff_dst_off
                                = ((oc * OD + od[d]) * OH + oh[h]) * OW + ow[w];
                        const int64_t wei_off
                                = ((oc * ICG * KD + kd[d]) * KH + kh[h]) * KW
                                + kw[w];
                        ds += diff_dst_loc[diff_dst_off] * wei_loc[wei_off];
                    }
                }
    };

    auto ker = [&](float &ds, int64_t g, int64_t mb, int64_t ic, int64_t id,
                       int64_t ih, int64_t iw) {
        const float *__restrict diff_dst_loc = (const float *)diff_dst_m
                + (mb * OC + g * OCG) * OD * OH * OW;
        const float *__restrict wei_loc
                = (const float *)wei_m + ((g * OCG) * ICG + ic) * KD * KH * KW;

        for (int64_t kd = 0; kd < KD; ++kd) {
            int64_t od = id - kd * DD + PD;
            if (od < 0 || od % SD || od >= OD * SD) continue;
            od /= SD;
            for (int64_t kh = 0; kh < KH; ++kh) {
                int64_t oh = ih - kh * DH + PH;
                if (oh < 0 || oh % SH || oh >= OH * SH) continue;
                oh /= SH;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    int64_t ow = iw - kw * DW + PW;
                    if (ow < 0 || ow % SW || ow >= OW * SW) continue;
                    ow /= SW;
                    for (int64_t oc = 0; oc < OCG; ++oc) {
                        const int64_t diff_dst_off
                                = ((oc * OD + od) * OH + oh) * OW + ow;
                        const int64_t wei_off
                                = ((oc * ICG * KD + kd) * KH + kh) * KW + kw;
                        ds += diff_dst_loc[diff_dst_off] * wei_loc[wei_off];
                    }
                }
            }
        }
    };
    sc::utils::parallel_for(0, G * MB * ICG * ID * IH * IW, 1, [&](int64_t i) {
        int64_t g = i / (MB * ICG * ID * IH * IW),
                mb = i / (ICG * ID * IH * IW) % MB,
                ic = i / (ID * IH * IW) % ICG;
        int64_t id = i / (IH * IW) % ID, ih = i / IW % IH, iw = i % IW;
        size_t src_off = src_off_f(G, IC, ID, IH, IW, mb, g, ic, id, ih, iw);
        float &ds = ((float *)diff_src_m)[src_off];
        float conv_res = 0;
        if (fast)
            ker_fast(conv_res, g, mb, ic, id, ih, iw);
        else
            ker(conv_res, g, mb, ic, id, ih, iw);

        if (dir & FLAG_BIA) {
            const size_t bia_off = (size_t)g * ICG + ic;
            conv_res += ((float *)bia_m)[bia_off];
        }

        ds = conv_res;
    });
}

inline void compute_ref_bwd_weights(const int64_t MB, const int64_t G,
        const int64_t OC, const int64_t IC, const int64_t IH, const int64_t IW,
        const int64_t OH, const int64_t OW, const int64_t KH, const int64_t KW,
        const int64_t SH, const int64_t SW, const int64_t PH, const int64_t PW,
        float *src_m, float *diff_wei_m, float *diff_dst_m,
        dir_t dir = dir_t::BWD_W, const int64_t OD = 1, const int64_t ID = 1,
        const int64_t SD = 1, const int64_t PD = 0, const int64_t KD = 1,
        const int64_t DD = 1, const int64_t DH = 1, const int64_t DW = 1) {
    /* help compiler optimize the code */

    const int64_t OCG = OC / G, ICG = IC / G;

    auto ker = [&](float &dw, int64_t g, int64_t oc, int64_t ic, int64_t kd,
                       int64_t kh, int64_t kw) {
        for (int64_t mb = 0; mb < MB; ++mb) {
            const float *__restrict diff_dst_loc = (const float *)diff_dst_m
                    + (mb * OC + g * OCG + oc) * OD * OH * OW;
            const float *__restrict src_loc = (const float *)src_m
                    + (mb * IC + g * ICG + ic) * ID * IH * IW;
            for (int64_t od = 0; od < OD; ++od) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        const int64_t id = od * SD + kd * DD - PD;
                        const int64_t ih = oh * SH + kh * DH - PH;
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                                || iw >= IW) {
                            continue;
                        }
                        size_t diff_dst_off = (od * OH + oh) * OW + ow;
                        size_t src_off = (id * IH + ih) * IW + iw;
                        dw += diff_dst_loc[diff_dst_off] * src_loc[src_off];
                    }
                }
            }
        }
    };

    // (OC, IC, KD, KH, KW)
    // TODO(xxx): fix grouped conv here
    sc::utils::parallel_for(0, G * OCG * ICG * KD * KH * KW, 1, [&](int64_t i) {
        int64_t g = i / (OCG * ICG * KD * KH * KW),
                oc = i / (ICG * KD * KH * KW) % OCG,
                ic = i / (KD * KH * KW) % ICG;
        int64_t kd = i / (KH * KW) % KD, kh = i / KW % KH, kw = i % KW;
        size_t wei_off
                = wei_off_f(G, OC, IC, KD, KH, KW, g, oc, ic, kd, kh, kw);
        float &dw = ((float *)diff_wei_m)[wei_off];
        dw = 0;
        ker(dw, g, oc, ic, kd, kh, kw);
    });
}

template <typename T>
T NCHW2NCHWc(T &input, int N, int C, int H, int W, int c) {
    size_t dim3 = C * H * W * c, dim2 = H * W, dim1 = W;
    T output(input.size());
    sc::utils::parallel_for(0, N * C * H, 1, [&](int64_t fuse) {
        auto n = fuse / (C * H), c_o = fuse / H % C, h = fuse % H;
        for (auto w = 0; w < W; ++w) {
            for (auto c_i = 0; c_i < c; ++c_i) {
                output[n * dim3 + c_o * H * W * c + h * W * c + w * c + c_i]
                        = input[n * dim3 + (c_o * c + c_i) * dim2 + h * dim1
                                + w];
            }
        }
    });
    return output;
}

template <typename T>
T NCHWc2NCHW(T &input, int N, int C, int H, int W, int c) {
    size_t dim3 = C * H * W * c, dim2 = H * W, dim1 = W;
    T output(input.size());
    sc::utils::parallel_for(0, N * C * H, 1, [&](int64_t fuse) {
        auto n = fuse / (C * H), c_o = fuse / H % C, h = fuse % H;
        for (auto w = 0; w < W; ++w) {
            for (auto c_i = 0; c_i < c; ++c_i) {
                output[n * dim3 + (c_o * c + c_i) * dim2 + h * dim1 + w]
                        = input[n * dim3 + c_o * H * W * c + h * W * c + w * c
                                + c_i];
            }
        }
    });
    return output;
}

template <typename T>
T NHWC2NCHW(T &input, int N, int H, int W, int C) {
    size_t dim3 = C * H * W, dim2 = H * W, dim1 = W;
    T output(input.size());
    sc::utils::parallel_for(0, N * C * H, 1, [&](int64_t fuse) {
        auto n = fuse / (C * H), c = fuse / H % C, h = fuse % H;
        for (auto w = 0; w < W; ++w) {
            output[n * dim3 + c * dim2 + h * dim1 + w]
                    = input[n * dim3 + h * W * C + w * C + c];
        }
    });
    return output;
}

template <typename T>
T NHWC2NCHWcn(T &input, int N, int C, int H, int W, int c, int n,
        int origin_n = 0, int origin_c = 0) {
    size_t dim1 = n, dim2 = c * dim1, dim3 = W * dim2, dim4 = H * dim3,
           dim5 = C * dim4;
    if (!origin_n) { origin_n = N * n; }
    if (!origin_c) { origin_c = C * c; }
    T output(N * dim5);
    sc::utils::parallel_for(0, N * C * H, 1, [&](int64_t fuse) {
        auto out_n = fuse / (C * H), out_c = fuse / H % C, h = fuse % H;
        for (auto w = 0; w < W; ++w) {
            for (auto in_c = 0; in_c < c; in_c++) {
                for (auto in_n = 0; in_n < n; in_n++) {
                    if (out_n * n + in_n < origin_n
                            && out_c * c + in_c < origin_c) {
                        output[out_n * dim5 + out_c * dim4 + h * dim3 + w * dim2
                                + in_c * dim1 + in_n]
                                = input[(out_n * n + in_n) * origin_c * H * W
                                        + h * W * origin_c + w * origin_c
                                        + out_c * c + in_c];
                    } else {
                        output[out_n * dim5 + out_c * dim4 + h * dim3 + w * dim2
                                + in_c * dim1 + in_n]
                                = 0;
                    }
                }
            }
        }
    });
    return output;
}

template <typename T>
T any2NCHW(sc::sc_data_format_t input_format, T &input, int N, int C, int H,
        int W, int c) {
    if (input_format == sc::sc_data_format_t::NCHWc(c)) {
        if (c <= 0) { COMPILE_ASSERT(0, "Invalid blocking dim NCHWc: c <= 0"); }
        return NCHWc2NCHW(input, N, C / c, H, W, c);
    } else if (input_format == sc::sc_data_format_t::NHWC()) {
        return NHWC2NCHW(input, N, H, W, C);
    } else {
        COMPILE_ASSERT(0,
                "Unsupported input format, only NCHWc and NHWC are supported")
    }
}

template <typename T>
T KCRS2KCRSck(T &input, int K, int C, int R, int S, int c, int k) {
    size_t dim3 = C * R * S * c, dim2 = R * S, dim1 = S;
    T output(input.size());
    sc::utils::parallel_for(0, K * C * R, 1, [&](int64_t fuse) {
        auto k_o = fuse / (C * R), c_o = fuse / R % C, r = fuse % R;
        for (auto s = 0; s < S; ++s) {
            for (auto c_i = 0; c_i < c; ++c_i) {
                for (auto k_i = 0; k_i < k; ++k_i) {
                    output[k_o * C * R * S * c * k + c_o * R * S * c * k
                            + r * S * c * k + s * c * k + c_i * k + k_i]
                            = input[(k_o * k + k_i) * dim3
                                    + (c_o * c + c_i) * dim2 + r * dim1 + s];
                }
            }
        }
    });
    return output;
}

template <typename T>
T KCRSck2KCRS(T &input, int K, int C, int R, int S, int c, int k) {
    size_t dim3 = C * R * S * c, dim2 = R * S, dim1 = S;
    T output(input.size());
    sc::utils::parallel_for(0, K * C * R, 1, [&](int64_t fuse) {
        auto k_o = fuse / (C * R), c_o = fuse / R % C, r = fuse % R;
        for (auto s = 0; s < S; ++s) {
            for (auto c_i = 0; c_i < c; ++c_i) {
                for (auto k_i = 0; k_i < k; ++k_i) {
                    output[(k_o * k + k_i) * dim3 + (c_o * c + c_i) * dim2
                            + r * dim1 + s]
                            = input[k_o * C * R * S * c * k
                                    + c_o * R * S * c * k + r * S * c * k
                                    + s * c * k + c_i * k + k_i];
                }
            }
        }
    });
    return output;
}

// for vnni blocking format
template <typename T>
T KCRSckc2KCRS(T &input, int K_num_blk, int C_num_blk, int R, int S, int C_blk,
        int K_blk, int C_pack = 4) {
    COMPILE_ASSERT((C_pack == 2 || C_pack == 4),
            "Invalid C_pack (" << C_pack
                               << "), which should be either 2 or 4!");
    const int K = K_num_blk * K_blk;
    const int C = C_num_blk * C_blk * C_pack;
    const int C_full_blk = C_blk * C_pack;
    const std::vector<int> out_strides = {C * R * S, R * S, S, 1};
    const std::vector<int> in_strides
            = {C_num_blk * R * S * C_blk * K_blk * C_pack,
                    R * S * C_blk * K_blk * C_pack, S * C_blk * K_blk * C_pack,
                    C_blk * K_blk * C_pack, K_blk * C_pack, C_pack, 1};
    T output(input.size());

    sc::utils::parallel_for(0, K, 1, [&](int64_t k) {
        auto k_o = k / K_blk, k_i = k % K_blk;
        for (int c = 0; c < C; ++c) {
            auto c_o = c / C_full_blk, c_i = c % C_full_blk / C_pack,
                 c_ii = c % C_full_blk % C_pack;
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    output[k * out_strides[0] + c * out_strides[1]
                            + r * out_strides[2] + s]
                            = input[k_o * in_strides[0] + c_o * in_strides[1]
                                    + r * in_strides[2] + s * in_strides[3]
                                    + c_i * in_strides[4] + k_i * in_strides[5]
                                    + c_ii];
                }
            }
        }
    });
    return output;
}

// for vnni blocking format
template <typename T>
T KCRS2KCRSckc(T &input, int K_num_blk, int C_num_blk, int R, int S, int C_blk,
        int K_blk, int C_pack = 4) {
    COMPILE_ASSERT((C_pack == 2 || C_pack == 4),
            "Invalid C_pack (" << C_pack
                               << "), which should be either 2 or 4!");
    const int K = K_num_blk * K_blk;
    const int C = C_num_blk * C_blk * C_pack;
    const int C_full_blk = C_blk * C_pack;
    const std::vector<int> out_strides = {C * R * S, R * S, S, 1};
    const std::vector<int> in_strides
            = {C_num_blk * R * S * C_blk * K_blk * C_pack,
                    R * S * C_blk * K_blk * C_pack, S * C_blk * K_blk * C_pack,
                    C_blk * K_blk * C_pack, K_blk * C_pack, C_pack, 1};
    T output(input.size());

    sc::utils::parallel_for(0, K, 1, [&](int64_t k) {
        auto k_o = k / K_blk, k_i = k % K_blk;
        for (int c = 0; c < C; ++c) {
            auto c_o = c / C_full_blk, c_i = c % C_full_blk / C_pack,
                 c_ii = c % C_full_blk % C_pack;
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    output[k_o * in_strides[0] + c_o * in_strides[1]
                            + r * in_strides[2] + s * in_strides[3]
                            + c_i * in_strides[4] + k_i * in_strides[5] + c_ii]
                            = input[k * out_strides[0] + c * out_strides[1]
                                    + r * out_strides[2] + s];
                }
            }
        }
    });
    return output;
}

// for 3d convolution
template <typename T>
T NCDHWc2NCDHW(T &input, int N, int C_num_blk, int D, int H, int W, int C_blk) {
    const std::vector<int> in_strides = {C_num_blk * D * H * W * C_blk,
            D * H * W * C_blk, H * W * C_blk, W * C_blk, C_blk, 1};
    const std::vector<int> out_strides
            = {C_num_blk * C_blk * D * H * W, D * H * W, H * W, W, 1};
    T output(input.size());
    // SC_OMP_CLAUSE("omp parallel for collapse(6)")
    sc::utils::parallel_for(0, N, 1, [&](int64_t n) {
        for (auto c_o = 0; c_o < C_num_blk; ++c_o) {
            for (auto d = 0; d < D; ++d) {
                for (auto h = 0; h < H; ++h) {
                    for (auto w = 0; w < W; ++w) {
                        for (auto c_i = 0; c_i < C_blk; ++c_i) {
                            output[n * out_strides[0]
                                    + (c_o * C_blk + c_i) * out_strides[1]
                                    + d * out_strides[2] + h * out_strides[3]
                                    + w * out_strides[4]]
                                    = input[n * in_strides[0]
                                            + c_o * in_strides[1]
                                            + d * in_strides[2]
                                            + h * in_strides[3]
                                            + w * in_strides[4]
                                            + c_i * in_strides[5]];
                        }
                    }
                }
            }
        }
    });
    return output;
}

template <typename T>
T NDHWC2NCDHW(T &input, int N, int D, int H, int W, int C) {
    const std::vector<int> in_strides = {D * H * W * C, H * W * C, W * C, C, 1};
    const std::vector<int> out_strides
            = {C * D * H * W, D * H * W, H * W, W, 1};
    T output(input.size());
    sc::utils::parallel_for(0, N, 1, [&](int64_t n) {
        for (auto d = 0; d < D; ++d) {
            for (auto h = 0; h < H; ++h) {
                for (auto w = 0; w < W; ++w) {
                    for (auto c = 0; c < C; ++c) {
                        output[n * out_strides[0] + c * out_strides[1]
                                + d * out_strides[2] + h * out_strides[3]
                                + w * out_strides[4]]
                                = input[n * in_strides[0] + d * in_strides[1]
                                        + h * in_strides[2] + w * in_strides[3]
                                        + c * in_strides[4]];
                    }
                }
            }
        }
    });
    return output;
}

template <typename T>
T any2NCDHW(sc::sc_data_format_t input_format, T &input, int N, int C, int D,
        int H, int W, int c) {
    if (input_format == sc::sc_data_format_t::NCDHWc(c)) {
        if (c <= 0) {
            COMPILE_ASSERT(0, "Invalid blocking dim NCDHWc: c <= 0");
        }
        return NCDHWc2NCDHW(input, N, C / c, D, H, W, c);
    } else if (input_format == sc::sc_data_format_t::NDHWC()) {
        return NDHWC2NCDHW(input, N, D, H, W, C);
    } else {
        COMPILE_ASSERT(0,
                "Unsupported input format, only NCDHWc and NDHWC are supported")
    }
}

template <typename T>
T KCDRSck2KCDRS(T &input, int K_num_blk, int C_num_blk, int D, int R, int S,
        int C_blk, int K_blk) {
    T output(input.size());
    const std::vector<int> in_strides = {C_num_blk * D * R * S * C_blk * K_blk,
            D * R * S * C_blk * K_blk, R * S * C_blk * K_blk, S * C_blk * K_blk,
            C_blk * K_blk, K_blk, 1};
    const std::vector<int> out_strides
            = {C_num_blk * C_blk * D * R * S, D * R * S, R * S, S, 1};
    // SC_OMP_CLAUSE("omp parallel for collapse(7)")
    sc::utils::parallel_for(0, K_num_blk, 1, [&](int64_t k_o) {
        for (auto c_o = 0; c_o < C_num_blk; ++c_o) {
            for (auto d = 0; d < D; ++d) {
                for (auto r = 0; r < R; ++r) {
                    for (auto s = 0; s < S; ++s) {
                        for (auto c_i = 0; c_i < C_blk; ++c_i) {
                            for (auto k_i = 0; k_i < K_blk; ++k_i) {
                                output[(k_o * K_blk + k_i) * out_strides[0]
                                        + (c_o * C_blk + c_i) * out_strides[1]
                                        + d * out_strides[2]
                                        + r * out_strides[3]
                                        + s * out_strides[4]]
                                        = input[k_o * in_strides[0]
                                                + c_o * in_strides[1]
                                                + d * in_strides[2]
                                                + r * in_strides[3]
                                                + s * in_strides[4]
                                                + c_i * in_strides[5]
                                                + k_i * in_strides[6]];
                            }
                        }
                    }
                }
            }
        }
    });

    return output;
}

template <typename T>
T KCDRSckc2KCDRS(T &input, int K_num_blk, int C_num_blk, int D, int R, int S,
        int C_blk, int K_blk, int C_pack = 4) {
    COMPILE_ASSERT((C_pack == 2 || C_pack == 4),
            "Invalid C_pack (" << C_pack
                               << "), which should be either 2 or 4!");

    T output(input.size());
    const std::vector<int> in_strides
            = {C_num_blk * D * R * S * C_blk * K_blk * C_pack,
                    D * R * S * C_blk * K_blk * C_pack,
                    R * S * C_blk * K_blk * C_pack, S * C_blk * K_blk * C_pack,
                    C_blk * K_blk * C_pack, K_blk * C_pack, C_pack, 1};
    const std::vector<int> out_strides
            = {C_num_blk * C_blk * C_pack * D * R * S, D * R * S, R * S, S, 1};
    // SC_OMP_CLAUSE("omp parallel for collapse(8)")
    sc::utils::parallel_for(0, K_num_blk, 1, [&](int64_t k_o) {
        for (auto c_o = 0; c_o < C_num_blk; ++c_o) {
            for (auto d = 0; d < D; ++d) {
                for (auto r = 0; r < R; ++r) {
                    for (auto s = 0; s < S; ++s) {
                        for (auto c_i = 0; c_i < C_blk; ++c_i) {
                            for (auto k_i = 0; k_i < K_blk; ++k_i) {
                                for (auto c_p = 0; c_p < C_pack; ++c_p) {
                                    output[(k_o * K_blk + k_i) * out_strides[0]
                                            + ((c_o * C_blk + c_i) * C_pack
                                                      + c_p)
                                                    * out_strides[1]
                                            + d * out_strides[2]
                                            + r * out_strides[3]
                                            + s * out_strides[4]]
                                            = input[k_o * in_strides[0]
                                                    + c_o * in_strides[1]
                                                    + d * in_strides[2]
                                                    + r * in_strides[3]
                                                    + s * in_strides[4]
                                                    + c_i * in_strides[5]
                                                    + k_i * in_strides[6]
                                                    + c_p * in_strides[7]];
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    return output;
}

#endif
