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

#ifndef GRAPH_INTERFACE_OPSET_HPP
#define GRAPH_INTERFACE_OPSET_HPP

#include <functional>

#include "graph/interface/op_def.hpp"
#include "graph/interface/op_schema.hpp"

namespace dnnl {
namespace impl {
namespace graph {

class opset_v1_t {
public:
    static void for_each_schema(const std::function<void(op_schema_t &&)> &fn) {
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Abs, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(AbsBackprop, 1)>());
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
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ConvTranspose, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        ConvTransposeBackpropData, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        ConvTransposeBackpropFilters, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Dequantize, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Divide, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(EluBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(End, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Erf, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Exp, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(GELU, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(GELUBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(HardSwish, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        HardSwishBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Interpolate, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        InterpolateBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LayerNorm, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        LayerNormBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LeakyReLU, 1)>());
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
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Mish, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MishBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Multiply, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(PReLU, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(PReLUBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Quantize, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceL1, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceL2, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceMax, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceMean, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceMin, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceProd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceSum, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReLU, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReLUBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Reorder, 1)>());
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
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        SquaredDifference, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(StaticReshape, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        StaticTranspose, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Subtract, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Tanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(TanhBackprop, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Wildcard, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(TypeCast, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        DynamicQuantize, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        DynamicDequantize, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Reciprocal, 1)>());
    }
};

inline void register_opset_schema() {
    register_opset_schema<opset_v1_t>();
}

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
