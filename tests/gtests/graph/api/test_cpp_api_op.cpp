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
            op::kind::AbsBackward,
            op::kind::Add,
            op::kind::AvgPool,
            op::kind::AvgPoolBackward,
            op::kind::BatchNormTrainingBackward,
            op::kind::BatchNormForwardTraining,
            op::kind::BatchNormInference,
            op::kind::BiasAdd,
            op::kind::BiasAddBackward,
            op::kind::Clamp,
            op::kind::ClampBackward,
            op::kind::Concat,
            op::kind::Convolution,
            op::kind::ConvolutionBackwardData,
            op::kind::ConvolutionBackwardWeights,
            op::kind::ConvTranspose,
            op::kind::ConvTransposeBackwardData,
            op::kind::ConvTransposeBackwardWeights,
            op::kind::Dequantize,
            op::kind::Divide,
            op::kind::DynamicDequantize,
            op::kind::DynamicQuantize,
            op::kind::Elu,
            op::kind::EluBackward,
            op::kind::End,
            op::kind::Exp,
            op::kind::GELU,
            op::kind::GELUBackward,
            op::kind::HardSwish,
            op::kind::HardSwishBackward,
            op::kind::Interpolate,
            op::kind::InterpolateBackward,
            op::kind::LayerNorm,
            op::kind::LayerNormBackward,
            op::kind::LeakyReLU,
            op::kind::Log,
            op::kind::LogSoftmax,
            op::kind::LogSoftmaxBackward,
            op::kind::MatMul,
            op::kind::Maximum,
            op::kind::MaxPool,
            op::kind::MaxPoolBackward,
            op::kind::Minimum,
            op::kind::Mish,
            op::kind::MishBackward,
            op::kind::Multiply,
            op::kind::PReLU,
            op::kind::PReLUBackward,
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
            op::kind::ReLUBackward,
            op::kind::Reorder,
            op::kind::Round,
            op::kind::Sigmoid,
            op::kind::SigmoidBackward,
            op::kind::SoftMax,
            op::kind::SoftMaxBackward,
            op::kind::SoftPlus,
            op::kind::SoftPlusBackward,
            op::kind::Sqrt,
            op::kind::SqrtBackward,
            op::kind::Square,
            op::kind::SquaredDifference,
            op::kind::StaticReshape,
            op::kind::StaticTranspose,
            op::kind::Subtract,
            op::kind::Tanh,
            op::kind::TanhBackward,
            op::kind::TypeCast,
            op::kind::Wildcard,
            op::kind::HardSigmoid,
            op::kind::HardSigmoidBackward,
            op::kind::Select,
            op::kind::Pow,
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

TEST(APIOp, CreateWithDefaultName) {
    using namespace dnnl::graph;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;

    logical_tensor lt1 {0, data_type::f32, layout_type::strided};
    logical_tensor lt2 {1, data_type::f32, layout_type::strided};
    logical_tensor lt3 {2, data_type::f32, layout_type::strided};

    op conv {0, op::kind::Convolution};
    conv.add_inputs({lt1, lt2});
    conv.add_outputs({lt3});
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

    conv.set_attr<std::vector<int64_t>>(op::attr::strides, {1, 1});
    conv.set_attr<int64_t>(op::attr::groups, 1);
    conv.set_attr<std::string>(op::attr::auto_pad, "VALID");
    // conv.set_attr<std::vector<float>>("float_vec", {1., 1.});
    // conv.set_attr<float>("float_val", 1.);
}

TEST(APIOp, ShallowCopy) {
    using namespace dnnl::graph;
    const size_t id = 123;
    op conv {id, op::kind::Convolution, "convolution"};
    op conv_1(conv); // NOLINT

    ASSERT_EQ(conv.get(), conv_1.get());
}

TEST(APIOp, SoftPlusAttr) {
    using namespace dnnl::graph;
    const size_t id = 123;
    op softplus {id, op::kind::SoftPlus, "softplus"};
    softplus.set_attr<float>(op::attr::beta, 2.f);

    logical_tensor in {0, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    logical_tensor out {1, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};

    softplus.add_input({in});
    softplus.add_output({out});

    graph g(engine::kind::cpu);
    g.add_op(softplus);
    g.finalize();
}
