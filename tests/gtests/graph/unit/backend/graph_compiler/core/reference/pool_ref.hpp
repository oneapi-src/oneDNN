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
#include <memory>
#include <string>
#include <utility>
#include <test_utils.hpp>
#include <util/parallel.hpp>

void compute_pooling_ref_direct_fwd(std::string &pooling_type, const int64_t MB,
        const int64_t G, const int64_t OC, const int64_t IC, const int64_t IH,
        const int64_t IW, const int64_t OH, const int64_t OW, const int64_t KH,
        const int64_t KW, const int64_t SH, const int64_t SW, const int64_t PH,
        const int64_t PW, float *src_m, float *dst_m, float *mul_m = nullptr,
        float *add_m = nullptr, bool bn_relu = false, const int64_t OD = 1,
        const int64_t ID = 1, const int64_t SD = 1, const int64_t PD = 1,
        const int64_t KD = 1, const int64_t DD = 1, const int64_t DH = 1,
        const int64_t DW = 1) {
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

#endif
