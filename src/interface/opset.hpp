/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef INTERFACE_OPSET_HPP
#define INTERFACE_OPSET_HPP

#include <functional>
#include "op_schema.hpp"

namespace dnnl {
namespace graph {
namespace impl {

class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Add, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(AvgPool, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(AvgPoolBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(BatchNormInference, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(BatchNormForwardTraining, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(BatchNormTrainingBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(BiasAdd, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(BiasAddBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Clamp, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ClampBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Concat, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Convolution, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ConvolutionBackpropData, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ConvolutionBackpropFilters, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Divide, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Elu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(EluBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Erf, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Exp, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(GELU, 2);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(GELUBackprop, 2);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(HardTanh, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(HardTanhBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Index, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Interpolate, 4);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(InterpolateBackprop, 4);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LayerNorm, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LayerNormBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Log, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LogSoftmax, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LogSoftmaxBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MaxPool, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MaxPoolBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Maximum, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Minimum, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Multiply, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Pow, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(PowBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(PowBackpropExponent, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceSum, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReLU, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReLUBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Reshape, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Round, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Sigmoid, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SigmoidBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SoftMax, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SoftMaxBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SoftPlus, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SoftPlusBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Sqrt, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SqrtBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Square, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Tanh, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(TanhBackprop, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Transpose, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Wildcard, 1);

// fusion ops
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_abs, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_add, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_add_elu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_add_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_add_relu6, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_bn, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_bn_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_bn_add_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_elu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_hardtanh, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_relu6, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_sigmoid, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_sqrt, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_square, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_tanh, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_add, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_add_elu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_add_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_add_relu6, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bn, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bn_add, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bn_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bn_add_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(BatchNorm_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias_add, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias_add_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias_bn, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias_elu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias_hardtanh, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias_relu6, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias_sigmoid, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_elu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_sigmoid, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_hardtanh, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_gelu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_add, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_add_gelu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_add_relu, 1);
class DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_add_sigmoid, 1);

class opset_v1 {
public:
    static void for_each_schema(std::function<void(op_schema &&)> fn) {
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(AvgPool, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        AvgPoolBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        BatchNormInference, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        BatchNormForwardTraining, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        BatchNormTrainingBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(BiasAdd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        BiasAddBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Clamp, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ClampBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Concat, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Convolution, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        ConvolutionBackpropData, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        ConvolutionBackpropFilters, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Divide, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(EluBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Erf, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Exp, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(HardTanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        HardTanhBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Index, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LayerNorm, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        LayerNormBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Log, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LogSoftmax, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        LogSoftmaxBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Maximum, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MaxPool, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MaxPoolBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Minimum, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Multiply, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Pow, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(PowBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        PowBackpropExponent, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceSum, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReLU, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReLUBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Reshape, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Round, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        SigmoidBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SoftMax, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        SoftMaxBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SoftPlus, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        SoftPlusBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Sqrt, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SqrtBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Square, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Tanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(TanhBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Transpose, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Wildcard, 1)>());

        // fusion ops
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_abs, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bias_add_elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bias_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bias_add_relu6, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_add_elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_add_relu6, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_bn, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bias_bn_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bias_hardtanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bias_relu6, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bias_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_sqrt, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bias_square, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bias_tanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bias_bn_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bn, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bn_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_bn_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        Conv_bn_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Conv_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(BatchNorm_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_bias_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_bias_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_bias_bn, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_bias_elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_bias_hardtanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_bias_relu6, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_bias_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_hardtanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_add_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MatMul_add_sigmoid, 1)>());
    }
};

class opset_v2 {
public:
    static void for_each_schema(std::function<void(op_schema &&)> fn) {
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(GELU, 2)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(GELUBackprop, 2)>());
    }
};

class opset_v4 {
public:
    static void for_each_schema(std::function<void(op_schema &&)> fn) {
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Interpolate, 4)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        InterpolateBackprop, 4)>());
    }
};

inline void register_opset_schema() {
    register_opset_schema<opset_v1>();
    register_opset_schema<opset_v2>();
    register_opset_schema<opset_v4>();
}

} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
