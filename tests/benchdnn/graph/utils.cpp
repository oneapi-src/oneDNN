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

#include <algorithm>
#include <set>
#include <vector>

#include <oneapi/dnnl/dnnl_debug.h>

#include "utils.hpp"

namespace graph {

dnnl::graph::op::kind opstr2kind(const std::string &kind) {
    const std::unordered_map<std::string, dnnl::graph::op::kind> op_map = {
            {"Abs", dnnl::graph::op::kind::Abs},
            {"Add", dnnl::graph::op::kind::Add},
            {"AvgPool", dnnl::graph::op::kind::AvgPool},
            {"AvgPoolBackprop", dnnl::graph::op::kind::AvgPoolBackprop},
            {"BatchNormInference", dnnl::graph::op::kind::BatchNormInference},
            {"BatchNormForwardTraining",
                    dnnl::graph::op::kind::BatchNormForwardTraining},
            {"BatchNormTrainingBackprop",
                    dnnl::graph::op::kind::BatchNormTrainingBackprop},
            {"BiasAddBackprop", dnnl::graph::op::kind::BiasAddBackprop},
            {"Clamp", dnnl::graph::op::kind::Clamp},
            {"ClampBackprop", dnnl::graph::op::kind::ClampBackprop},
            {"Concat", dnnl::graph::op::kind::Concat},
            {"Convolution", dnnl::graph::op::kind::Convolution},
            {"ConvolutionBackpropData",
                    dnnl::graph::op::kind::ConvolutionBackpropData},
            {"ConvolutionBackpropFilters",
                    dnnl::graph::op::kind::ConvolutionBackpropFilters},
            {"ConvTranspose", dnnl::graph::op::kind::ConvTranspose},
            {"ConvTransposeBackpropData",
                    dnnl::graph::op::kind::ConvTransposeBackpropData},
            {"ConvTransposeBackpropFilters",
                    dnnl::graph::op::kind::ConvTransposeBackpropFilters},
            {"Divide", dnnl::graph::op::kind::Divide},
            {"Elu", dnnl::graph::op::kind::Elu},
            {"EluBackprop", dnnl::graph::op::kind::EluBackprop},
            {"Erf", dnnl::graph::op::kind::Erf},
            {"Exp", dnnl::graph::op::kind::Exp},
            {"GELU", dnnl::graph::op::kind::GELU},
            {"GELUBackprop", dnnl::graph::op::kind::GELUBackprop},
            {"HardSwish", dnnl::graph::op::kind::HardSwish},
            {"HardSwishBackprop", dnnl::graph::op::kind::HardSwishBackprop},
            {"LayerNorm", dnnl::graph::op::kind::LayerNorm},
            {"LayerNormBackprop", dnnl::graph::op::kind::LayerNormBackprop},
            {"Log", dnnl::graph::op::kind::Log},
            {"LogSoftmax", dnnl::graph::op::kind::LogSoftmax},
            {"LogSoftmaxBackprop", dnnl::graph::op::kind::LogSoftmaxBackprop},
            {"MatMul", dnnl::graph::op::kind::MatMul},
            {"Maximum", dnnl::graph::op::kind::Maximum},
            {"MaxPool", dnnl::graph::op::kind::MaxPool},
            {"MaxPoolBackprop", dnnl::graph::op::kind::MaxPoolBackprop},
            {"Minimum", dnnl::graph::op::kind::Minimum},
            {"Multiply", dnnl::graph::op::kind::Multiply},
            {"Pow", dnnl::graph::op::kind::Pow},
            {"PowBackprop", dnnl::graph::op::kind::PowBackprop},
            {"PReLU", dnnl::graph::op::kind::PReLU},
            {"PReLUBackprop", dnnl::graph::op::kind::PReLUBackprop},
            {"ReduceL1", dnnl::graph::op::kind::ReduceL1},
            {"ReduceL2", dnnl::graph::op::kind::ReduceL2},
            {"ReduceMax", dnnl::graph::op::kind::ReduceMax},
            {"ReduceMean", dnnl::graph::op::kind::ReduceMean},
            {"ReduceMin", dnnl::graph::op::kind::ReduceMin},
            {"ReduceProd", dnnl::graph::op::kind::ReduceProd},
            {"ReduceSum", dnnl::graph::op::kind::ReduceSum},
            {"ReLU", dnnl::graph::op::kind::ReLU},
            {"ReLUBackprop", dnnl::graph::op::kind::ReLUBackprop},
            {"Round", dnnl::graph::op::kind::Round},
            {"Sigmoid", dnnl::graph::op::kind::Sigmoid},
            {"SigmoidBackprop", dnnl::graph::op::kind::SigmoidBackprop},
            {"SoftMax", dnnl::graph::op::kind::SoftMax},
            {"SoftMaxBackprop", dnnl::graph::op::kind::SoftMaxBackprop},
            {"SoftPlus", dnnl::graph::op::kind::SoftPlus},
            {"SoftPlusBackprop", dnnl::graph::op::kind::SoftPlusBackprop},
            {"Sqrt", dnnl::graph::op::kind::Sqrt},
            {"SqrtBackprop", dnnl::graph::op::kind::SqrtBackprop},
            {"Square", dnnl::graph::op::kind::Square},
            {"SquaredDifference", dnnl::graph::op::kind::SquaredDifference},
            {"Subtract", dnnl::graph::op::kind::Subtract},
            {"Tanh", dnnl::graph::op::kind::Tanh},
            {"TanhBackprop", dnnl::graph::op::kind::TanhBackprop},
            {"Wildcard", dnnl::graph::op::kind::Wildcard},
            {"BiasAdd", dnnl::graph::op::kind::BiasAdd},
            {"Interpolate", dnnl::graph::op::kind::Interpolate},
            {"Index", dnnl::graph::op::kind::Index},
            {"InterpolateBackprop", dnnl::graph::op::kind::InterpolateBackprop},
            {"PowBackpropExponent", dnnl::graph::op::kind::PowBackpropExponent},
            {"End", dnnl::graph::op::kind::End},
            {"Quantize", dnnl::graph::op::kind::Quantize},
            {"Dequantize", dnnl::graph::op::kind::Dequantize},
            {"Reorder", dnnl::graph::op::kind::Reorder},
            {"TypeCast", dnnl::graph::op::kind::TypeCast},
            {"StaticReshape", dnnl::graph::op::kind::StaticReshape},
            {"StaticTranspose", dnnl::graph::op::kind::StaticTranspose},
            {"DynamicReshape", dnnl::graph::op::kind::DynamicReshape},
            {"DynamicTranspose", dnnl::graph::op::kind::DynamicTranspose},
            {"DynamicQuantize", dnnl::graph::op::kind::DynamicQuantize},
            {"DynamicDequantize", dnnl::graph::op::kind::DynamicDequantize},
            {"Sign", dnnl::graph::op::kind::Sign},
            {"Negative", dnnl::graph::op::kind::Negative},
            {"Reciprocal", dnnl::graph::op::kind::Reciprocal}};
    const auto it = op_map.find(kind);
    if (it != op_map.end()) {
        return it->second;
    } else {
        fprintf(stderr, "graph: ERROR: Unsupported opkind: `%s`, exiting...\n",
                kind.c_str());
        exit(2);
    }
}

template <typename T>
void compare_data(
        T *dst, T *ref, size_t size, float rtol, float atol, bool equal_nan) {
    auto cal_error = [&](const float dst, const float ref) -> bool {
        const float diff_f32 = dst - ref;
        const float gap = rtol
                        * (std::abs(ref) > std::abs(dst) ? std::abs(ref)
                                                         : std::abs(dst))
                + atol;
        bool good = std::abs(diff_f32) <= gap;
        return good;
    };

    for (size_t i = 0; i < size; ++i) {
        if (std::isfinite(dst[i]) && std::isfinite(ref[i])) {
            const float ref_f32 = static_cast<float>(ref[i]);
            const float dst_f32 = static_cast<float>(dst[i]);
            if (!cal_error(dst_f32, ref_f32)) {
                printf("expected = %s, actual = %s\n",
                        std::to_string(ref[i]).c_str(),
                        std::to_string(dst[i]).c_str());
                throw std::runtime_error(
                        "output result is not equal to excepted "
                        "results");
            }
        } else {
            bool cond = (dst[i] == ref[i]);
            if (equal_nan) { cond = std::isnan(dst[i]) && std::isnan(ref[i]); }
            if (!cond) {
                printf("expected = %s, actual = %s\n",
                        std::to_string(ref[i]).c_str(),
                        std::to_string(dst[i]).c_str());
                throw std::runtime_error(
                        "output result is not equal to excepted "
                        "results");
            }
        }
    }
}

} // namespace graph
