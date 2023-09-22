/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_POOL_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_POOL_REF_HPP

#include <algorithm>
#include <assert.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include "conv_ref.hpp"
#include <test_utils.hpp>
#include <util/parallel.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

inline void compute_pooling_ref_direct_fwd(std::string &pooling_type,
        const int64_t MB, const int64_t G, const int64_t OC, const int64_t IC,
        const int64_t IH, const int64_t IW, const int64_t OH, const int64_t OW,
        const int64_t KH, const int64_t KW, const int64_t SH, const int64_t SW,
        const int64_t PH, const int64_t PW, float *src_m, float *dst_m,
        float *mul_m = nullptr, float *add_m = nullptr, bool bn_relu = false,
        const int64_t OD = 1, const int64_t ID = 1, const int64_t SD = 1,
        const int64_t PD = 1, const int64_t KD = 1, const int64_t DD = 1,
        const int64_t DH = 1, const int64_t DW = 1) {
    /* help compiler optimize the code */

    const int64_t OCG = OC / G, ICG = IC / G;

    auto ker = [&](float &d, int64_t g, int64_t mb, int64_t oc, int64_t od,
                       int64_t oh, int64_t ow) {
        const float *__restrict src_loc
                = (const float *)src_m + (mb * IC + g * ICG) * ID * IH * IW;

        bool first = false;
        for (int64_t kd = 0; kd < KD; ++kd) {
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH - PH + kh;
                if (ih < 0 || ih >= IH) {
                    if (!first) {
                        d = 0.0f;
                        first = true;
                    } else {
                        if (pooling_type == "max") d = std::max(d, 0.0f);
                    }
                    continue;
                }
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW - PW + kw;
                    if (iw < 0 || iw >= IW) {
                        if (!first) {
                            d = 0.0f;
                            first = true;
                        } else {
                            if (pooling_type == "max") d = std::max(d, 0.0f);
                        }
                        continue;
                    }
                    int64_t src_off = (oc * IH + ih) * IW + iw;
                    if (!first) {
                        d = src_loc[src_off];
                        first = true;
                    } else {
                        if (pooling_type == "max")
                            d = std::max(d, src_loc[src_off]);
                        else if (pooling_type == "avg")
                            d = d + src_loc[src_off];
                    }
                }
            }
        }
        if (pooling_type == "avg") d = d / (KH * KW);
    };
    using namespace dnnl::impl::graph::gc;
    utils::parallel_for(0, G * MB * OCG * OD * OH * OW, 1, [&](int64_t i) {
        int64_t g = i / (MB * OCG * OD * OH * OW),
                mb = i / (OCG * OD * OH * OW) % MB,
                oc = i / (OD * OH * OW) % OCG;
        int64_t od = i / (OH * OW) % OD, oh = i / OW % OH, ow = i % OW;
        const size_t dst_off
                = (((mb * OC + g * OC / G + oc) * OD + od) * OH + oh) * OW + ow;
        float &dst = ((float *)dst_m)[dst_off];

        float pooling_res = 0;
        ker(pooling_res, g, mb, oc, od, oh, ow);

        if (bn_relu) {
            // y = max((a*x+b),0)
            const size_t bn_off = g * OC / G + oc;
            pooling_res *= ((float *)mul_m)[bn_off];
            pooling_res += ((float *)add_m)[bn_off];
            pooling_res = std::max(pooling_res, 0.0f);
        }

        dst = pooling_res;
    });
}

// max_indices shape: n oc oh ow
// which stores i*h + w of max value of this window in input tensor
template <typename Store_type, typename Compute_type = Store_type,
        typename Max_indice_type = int32_t>
inline void compute_pooling_ref_bwd(std::string &pooling_type, const int64_t MB,
        const int64_t IC, const int64_t IH, const int64_t IW, const int64_t OH,
        const int64_t OW, const int64_t KH, const int64_t KW, const int64_t SH,
        const int64_t SW, const int64_t PH, const int64_t PW,
        Store_type *dst_delta, Store_type *src_delta,
        Store_type *input_tensor = nullptr,
        Max_indice_type *max_indices = nullptr, bool exclude_pad = false) {
    assert(pooling_type == "avg"
            || (pooling_type == "max" && input_tensor != nullptr));

    auto ker = [&](const Compute_type s_delta, int64_t mb, int64_t ic,
                       int64_t ih, int64_t iw) {
        int64_t nc_offset = (mb * IC + ic) * OH * OW;
        Store_type *dst = dst_delta + nc_offset;

        int64_t max_offset = -1;
        Store_type max_val = std::numeric_limits<Compute_type>::lowest();

        bool has_max_indices = max_indices != nullptr;
        if (has_max_indices) {
            Max_indice_type max_hw_offset
                    = ((mb * IC + ic) * OH + ih) * OW + iw;
            max_offset = max_indices[max_hw_offset];
        }

        for (int64_t kh = 0; kh < KH; ++kh) {
            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t oh = ih * SH - PH + kh;
                const int64_t ow = iw * SW - PW + kw;
                int64_t cur_offset = oh * OW + ow;

                if (oh < 0 || oh >= OH || ow < 0 || ow >= OW) {
                    if (!has_max_indices && pooling_type == "max"
                            && max_val < 0.f) {
                        max_offset = -1;
                        max_val = static_cast<Compute_type>(0);
                    }
                } else {
                    if (pooling_type == "max") {
                        if (has_max_indices) {
                            if (max_offset == cur_offset)
                                dst[cur_offset] = static_cast<Store_type>(
                                        dst[cur_offset] + s_delta);
                        } else {
                            if (input_tensor[nc_offset + cur_offset]
                                    >= max_val) {
                                max_offset = cur_offset;
                                max_val = input_tensor[nc_offset + cur_offset];
                            }
                        }
                    } else if (pooling_type == "avg") {
                        int64_t k = 0;
                        if (exclude_pad) {
                            int hs = ih * SH - PH;
                            int he = hs + KH;
                            if (hs < 0) hs = 0;
                            if (he > OH) he = OH;
                            int ws = iw * SW - PW;
                            int we = ws + KW;
                            if (ws < 0) ws = 0;
                            if (we > OW) we = OW;
                            k = (he - hs) * (we - ws);
                            assert(k > 0 && k <= (KW * KH));

                        } else {
                            k = int(KW * KH);
                        }
                        dst[cur_offset] = static_cast<Store_type>(
                                dst[cur_offset] + s_delta / k);
                    }
                }
            }
        }
        if (!has_max_indices && pooling_type == "max" && max_offset >= 0)
            dst[max_offset]
                    = static_cast<Store_type>(dst[max_offset] + s_delta);
    };

    sc::utils::parallel_for(
            0, MB * IC * OH * OW, 1, [&](int64_t i) { dst_delta[i] = 0; });

    sc::utils::parallel_for(0, MB * IC * IH * IW, 1, [&](int64_t i) {
        int64_t mb = i / (IC * IH * IW), ic = i % (IC * IH * IW) / (IH * IW),
                ih = i % (IC * IH * IW) % (IH * IW) / IW,
                iw = i % (IC * IH * IW) % (IH * IW) % IW;
        Store_type s_delta = src_delta[i];
        ker(static_cast<Store_type>(s_delta), mb, ic, ih, iw);
    });
}

template <typename Store_type, typename Compute_type = Store_type>
inline void compute_pooling_ref_fwd(std::string &pooling_type, const int64_t MB,
        const int64_t IC, const int64_t IH, const int64_t IW, const int64_t OH,
        const int64_t OW, const int64_t KH, const int64_t KW, const int64_t SH,
        const int64_t SW, const int64_t PH, const int64_t PW, Store_type *src_m,
        Store_type *dst_m, float *mul_m = nullptr, float *add_m = nullptr,
        bool bn_relu = false, bool exclude_pad = false) {
    assert(pooling_type == "avg" || pooling_type == "max");
    Compute_type zero = static_cast<Compute_type>(0);
    auto ker = [&](Compute_type &d, int64_t mb, int64_t oc, int64_t oh,
                       int64_t ow) {
        const Store_type *src_loc = src_m + (mb * IC + oc) * IH * IW;

        int count = 0;
        if (pooling_type == "max")
            d = std::numeric_limits<Compute_type>::lowest();
        else
            d = zero;
        for (int64_t kh = 0; kh < KH; ++kh) {
            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t ih = oh * SH - PH + kh;
                const int64_t iw = ow * SW - PW + kw;
                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                    if (pooling_type == "max") d = std::max(d, zero);
                    continue;
                }
                int64_t src_off = ih * IW + iw;
                if (pooling_type == "max")
                    d = std::max(
                            d, static_cast<Compute_type>(src_loc[src_off]));
                else if (pooling_type == "avg") {
                    d = d + src_loc[src_off];
                    count++;
                }
            }
        }
        if (pooling_type == "avg") {
            if (exclude_pad) {
                int hs = oh * SH - PH;
                int he = hs + KH;
                if (hs < 0) hs = 0;
                if (he > IH) he = IH;
                int ws = ow * SW - PW;
                int we = ws + KW;
                if (ws < 0) ws = 0;
                if (we > IW) we = IW;
                int k = (he - hs) * (we - ws);
                assert(k == count);
                d = d / k;
            } else {
                d = d / (KH * KW);
            }
        }
        if (bn_relu) {
            // y = max((a*x+b),0)
            const size_t bn_off = oc;
            d = d * mul_m[bn_off];
            d = d + add_m[bn_off];
            d = std::max(d, zero);
        }
    };

    sc::utils::parallel_for(0, MB * IC * OH * OW, 1, [&](int64_t i) {
        int64_t mb = i / (IC * OH * OW), oc = i % (IC * OH * OW) / (OH * OW),
                oh = i % (IC * OH * OW) % (OH * OW) / OW,
                ow = i % (IC * OH * OW) % (OH * OW) % OW;
        Compute_type pooling_res = zero;
        ker(pooling_res, mb, oc, oh, ow);
        Store_type &dst = dst_m[i];
        dst = static_cast<Store_type>(pooling_res);
    });
}

template <typename src_type, typename wei_type, typename dst_type>
void compute_conv_pooling_postops_ref(const int64_t MB, const int64_t OC,
        const int64_t IC, const int64_t IH, const int64_t IW, const int64_t OH,
        const int64_t OW, const int64_t KH, const int64_t KW, const int64_t SH,
        const int64_t SW, const int64_t PH, const int64_t PW,
        src_type *conv_data, wei_type *conv_weight, dst_type *conv_ouput,
        std::string &pooling_type, const int64_t p_OH, const int64_t p_OW,
        const int64_t p_KH, const int64_t p_KW, const int64_t p_SH,
        const int64_t p_SW, const int64_t p_PH, const int64_t p_PW,
        dst_type *final_output, bool exclude_pad = false, bool bn_relu = false,
        float *mul_m = nullptr, float *add_m = nullptr) {
    compute_ref_direct_fwd<src_type, wei_type, dst_type, float>(MB, 1, OC, IC,
            IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, conv_data, conv_weight,
            nullptr, conv_ouput, dir_t::FWD_I, mul_m, add_m, false);
    compute_pooling_ref_fwd<src_type, src_type>(pooling_type, MB, OC, OH, OW,
            p_OH, p_OW, p_KH, p_KW, p_SH, p_SW, p_PH, p_PW, conv_ouput,
            final_output, mul_m, add_m, true, exclude_pad);
}

template <typename src_type, typename wei_type, typename dst_type>
void compute_conv_postops_pooling_ref(const int64_t MB, const int64_t OC,
        const int64_t IC, const int64_t IH, const int64_t IW, const int64_t OH,
        const int64_t OW, const int64_t KH, const int64_t KW, const int64_t SH,
        const int64_t SW, const int64_t PH, const int64_t PW,
        src_type *conv_data, wei_type *conv_weight, dst_type *conv_ouput,
        std::string &pooling_type, const int64_t p_OH, const int64_t p_OW,
        const int64_t p_KH, const int64_t p_KW, const int64_t p_SH,
        const int64_t p_SW, const int64_t p_PH, const int64_t p_PW,
        dst_type *final_output, bool exclude_pad = false, bool bn_relu = false,
        float *mul_m = nullptr, float *add_m = nullptr) {
    compute_ref_direct_fwd<src_type, wei_type, dst_type, float>(MB, 1, OC, IC,
            IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, conv_data, conv_weight,
            nullptr, conv_ouput, dir_t::FWD_I, mul_m, add_m, true);
    compute_pooling_ref_fwd<src_type, src_type>(pooling_type, MB, OC, OH, OW,
            p_OH, p_OW, p_KH, p_KW, p_SH, p_SW, p_PH, p_PW, conv_ouput,
            final_output, mul_m, add_m, false, exclude_pad);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
