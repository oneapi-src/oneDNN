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

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_types.h"

#include <gtest/gtest.h>

#include <vector>

TEST(APIOp, CreateAllOps) {
    using namespace dnnl::graph;
    dnnl_graph_op_kind_t first_op = dnnl_graph_op_abs;
    dnnl_graph_op_kind_t last_op = dnnl_graph_op_last_symbol;

    // This list should be the same as the definition of `op::kind` in
    // dnnl_graph.hpp.
    // clang-format off
    std::vector<op::kind> all_kind_enums {
            op::kind::Abs,
            op::kind::AbsBackprop,
            op::kind::Add,
            op::kind::AvgPool,
            op::kind::AvgPoolBackprop,
            op::kind::BatchNormTrainingBackprop,
            op::kind::BatchNormForwardTraining,
            op::kind::BatchNormInference,
            op::kind::BiasAdd,
            op::kind::BiasAddBackprop,
            op::kind::Clamp,
            op::kind::ClampBackprop,
            op::kind::Concat,
            op::kind::Convolution,
            op::kind::ConvolutionBackpropData,
            op::kind::ConvolutionBackpropFilters,
            op::kind::ConvTranspose,
            op::kind::ConvTransposeBackpropData,
            op::kind::ConvTransposeBackpropFilters,
            op::kind::Dequantize,
            op::kind::Divide,
            op::kind::DynamicDequantize,
            op::kind::DynamicQuantize,
            op::kind::DynamicReshape,
            op::kind::DynamicTranspose,
            op::kind::Elu,
            op::kind::EluBackprop,
            op::kind::End,
            op::kind::Erf,
            op::kind::Exp,
            op::kind::GELU,
            op::kind::GELUBackprop,
            op::kind::HardSwish,
            op::kind::HardSwishBackprop,
            op::kind::HardTanh,
            op::kind::HardTanhBackprop,
            op::kind::Index,
            op::kind::Interpolate,
            op::kind::InterpolateBackprop,
            op::kind::LayerNorm,
            op::kind::LayerNormBackprop,
            op::kind::Log,
            op::kind::LogSoftmax,
            op::kind::LogSoftmaxBackprop,
            op::kind::MatMul,
            op::kind::Maximum,
            op::kind::MaxPool,
            op::kind::MaxPoolBackprop,
            op::kind::Minimum,
            op::kind::Mish,
            op::kind::MishBackprop,
            op::kind::Multiply,
            op::kind::Negative,
            op::kind::Pow,
            op::kind::PowBackprop,
            op::kind::PowBackpropExponent,
            op::kind::PReLU,
            op::kind::PReLUBackprop,
            op::kind::Quantize,
            op::kind::Reciprocal,
            op::kind::ReduceL1,
            op::kind::ReduceL2,
            op::kind::ReduceMax,
            op::kind::ReduceMean,
            op::kind::ReduceMin,
            op::kind::ReduceProd,
            op::kind::ReduceSum,
            op::kind::ReLU,
            op::kind::ReLUBackprop,
            op::kind::Reorder,
            op::kind::Round,
            op::kind::Sigmoid,
            op::kind::SigmoidBackprop,
            op::kind::Sign,
            op::kind::SoftMax,
            op::kind::SoftMaxBackprop,
            op::kind::SoftPlus,
            op::kind::SoftPlusBackprop,
            op::kind::Sqrt,
            op::kind::SqrtBackprop,
            op::kind::Square,
            op::kind::SquaredDifference,
            op::kind::StaticReshape,
            op::kind::StaticTranspose,
            op::kind::Subtract,
            op::kind::Tanh,
            op::kind::TanhBackprop,
            op::kind::TypeCast,
            op::kind::Wildcard,
    };
    // clang-format on

    const auto num_ops = all_kind_enums.size();
    for (size_t i = static_cast<size_t>(first_op);
            i < static_cast<size_t>(last_op); ++i) {
        ASSERT_LT(i, num_ops);
        op::kind kind = all_kind_enums[i];
        ASSERT_EQ(i, static_cast<size_t>(kind));

        op aop {0, kind, "test op"};
    }
}

TEST(APIOp, CreateWithInputList) {
    using namespace dnnl::graph;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;

    logical_tensor lt1 {0, data_type::f32, layout_type::strided};
    logical_tensor lt2 {1, data_type::f32, layout_type::strided};

    logical_tensor lt3 {2, data_type::f32, layout_type::strided};

    op conv {0, op::kind::Convolution, {lt1, lt2}, {lt3}, "Convolution_1"};
}

TEST(APIOp, SetInput) {
    using namespace dnnl::graph;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;
    const size_t id = 123;
    op conv {id, op::kind::Convolution, "convolution"};
    logical_tensor data {0, data_type::f32, layout_type::strided};
    logical_tensor weight {1, data_type::f32, layout_type::strided};

    conv.add_input(data);
    conv.add_input(weight);
}

TEST(APIOp, SetOutput) {
    using namespace dnnl::graph;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;
    const size_t id = 123;
    op conv {id, op::kind::Convolution, "convolution"};
    logical_tensor output {2, data_type::f32, layout_type::strided};

    conv.add_output(output);
}

TEST(APIOp, SetAttr) {
    using namespace dnnl::graph;
    const size_t id = 123;
    op conv {id, op::kind::Convolution, "convolution"};

    conv.set_attr<std::vector<int64_t>>("strides", {1, 1});
    conv.set_attr<int64_t>("groups", 1);
    conv.set_attr<std::string>("auto_pad", "VALID");
    // conv.set_attr<std::vector<float>>("float_vec", {1., 1.});
    // conv.set_attr<float>("float_val", 1.);
}

TEST(APIOp, ShallowCopy) {
    using namespace dnnl::graph;
    const size_t id = 123;
    op conv {id, op::kind::Convolution, "convolution"};
    op conv_1(conv);

    ASSERT_EQ(conv.get(), conv_1.get());
}
