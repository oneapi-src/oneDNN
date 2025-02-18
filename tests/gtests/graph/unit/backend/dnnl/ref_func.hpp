/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GRAPH_UNIT_BACKEND_DNNL_REF_FUNC_HPP
#define GRAPH_UNIT_BACKEND_DNNL_REF_FUNC_HPP

#include <cmath>
#include <vector>

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"

static inline std::vector<float> mish_func(const std::vector<float> &ref_dst) {
    std::vector<float> out;
    for (auto &rdst : ref_dst) {
        float ret = std::tanh(std::log(std::exp(rdst) + 1.0f)) * rdst;
        out.emplace_back(ret);
    }
    return out;
}

static inline std::vector<float> hardsigmoid_func(
        const std::vector<float> &src, float alpha, float beta) {
    std::vector<float> dst;
    for (auto &in : src) {
        float ret = in * alpha + beta;
        if (ret > 1.f)
            ret = 1.f;
        else if (ret < 0.f)
            ret = 0.f;

        dst.emplace_back(ret);
    }
    return dst;
}

static inline std::vector<float> hardsigmoidbackward_func(
        const std::vector<float> &src, const std::vector<float> &diff_dst,
        float alpha, float beta) {
    std::vector<float> diff_src;
    for (size_t i = 0; i < src.size(); ++i) {
        const float check = src[i] * alpha + beta;
        float ret = 0.f;
        if (check < 1.f && check > 0.f) { ret = alpha * diff_dst[i]; }

        diff_src.emplace_back(ret);
    }

    return diff_src;
}

static inline std::vector<float> sigmoid_func(
        const std::vector<float> &ref_dst) {
    std::vector<float> out(ref_dst.size());
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        out[i] = static_cast<float>(1 / (exp(-1 * ref_dst[i]) + 1));
    }
    return out;
}

static inline std::vector<float> tanh_func(const std::vector<float> &ref_dst) {
    std::vector<float> out(ref_dst.size());
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        const auto rdst = ref_dst[i];
        out[i] = static_cast<float>(
                (exp(rdst) - exp(-rdst)) / (exp(rdst) + exp(-rdst)));
    }
    return out;
}

static inline std::vector<float> sqrt_func(const std::vector<float> &ref_dst) {
    std::vector<float> out(ref_dst.size());
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        out[i] = static_cast<float>(sqrt(ref_dst[i]));
    }
    return out;
}

static inline std::vector<float> round_func(const std::vector<float> &ref_dst) {
    std::vector<float> out(ref_dst.size());
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        out[i] = static_cast<float>(round(ref_dst[i]));
    }
    return out;
}

#endif
