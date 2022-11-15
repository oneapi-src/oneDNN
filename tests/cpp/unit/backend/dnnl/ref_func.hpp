/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_DNNL_REF_FUNC_HPP
#define BACKEND_DNNL_REF_FUNC_HPP

#include <cmath>
#include <vector>

#include "cpp/unit/backend/dnnl/dnnl_test_common.hpp"
#include "cpp/unit/unit_test_common.hpp"

static inline test::vector<float> mish_func(
        const test::vector<float> &ref_dst) {
    test::vector<float> out;
    for (auto &rdst : ref_dst) {
        float ret = std::tanh(std::log(std::exp(rdst) + 1.0f)) * rdst;
        out.emplace_back(ret);
    }
    return out;
}

static inline test::vector<float> hardsigmoid_func(
        const test::vector<float> &inputs, float alpha, float beta) {
    test::vector<float> out;
    for (auto &in : inputs) {
        float ret = 0.f;
        if (in > 3.f)
            ret = 1.f;
        else if (in <= -3.f)
            ret = 0.f;
        else
            ret = in * alpha + beta;

        out.emplace_back(ret);
    }
    return out;
}

static inline test::vector<float> sigmoid_func(
        const test::vector<float> &ref_dst) {
    test::vector<float> out;
    for (auto &rdst : ref_dst) {
        out.emplace_back(static_cast<float>(1 / (exp(-rdst) + 1)));
    }
    return out;
}

static inline test::vector<float> tanh_func(
        const test::vector<float> &ref_dst) {
    test::vector<float> out;
    for (auto &rdst : ref_dst) {
        out.emplace_back(static_cast<float>(
                (exp(rdst) - exp(-rdst)) / (exp(rdst) + exp(-rdst))));
    }
    return out;
}

static inline test::vector<float> sqrt_func(
        const test::vector<float> &ref_dst) {
    test::vector<float> out;
    for (auto &rdst : ref_dst) {
        out.emplace_back(static_cast<float>(sqrt(rdst)));
    }
    return out;
}

static inline test::vector<float> round_func(
        const test::vector<float> &ref_dst) {
    test::vector<float> out;
    for (auto &rdst : ref_dst) {
        out.emplace_back(static_cast<float>(round(rdst)));
    }
    return out;
}

#endif
