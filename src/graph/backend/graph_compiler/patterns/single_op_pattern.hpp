/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_SINGLE_OP_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_SINGLE_OP_PATTERN_HPP

#include <memory>
#include <utility>

#include "graph/backend/graph_compiler/patterns/fusions.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {
namespace pass {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = graph::utils::pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

#define DEFAULT_PRIORITY 1.f
#define COMPILER_BACKEND_SINGLE_OP_PATTERN(pname, op) \
    COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(compiler, single_op_##pname) \
            .set_priority(DEFAULT_PRIORITY) \
            .set_kind(partition_kind_t::misc_post_ops) \
            .set_attr<FCreatePattern>("FCreatePattern", \
                    [](const std::shared_ptr<pb_graph_t> &pgraph) -> void { \
                        pgraph->append_op(graph::op_kind::op); \
                    });

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(single_op_pattern)

COMPILER_BACKEND_SINGLE_OP_PATTERN(add, Add);
COMPILER_BACKEND_SINGLE_OP_PATTERN(subtract, Subtract);
COMPILER_BACKEND_SINGLE_OP_PATTERN(multiply, Multiply);
COMPILER_BACKEND_SINGLE_OP_PATTERN(divide, Divide);
COMPILER_BACKEND_SINGLE_OP_PATTERN(pow, Pow);
COMPILER_BACKEND_SINGLE_OP_PATTERN(matmul, MatMul);
COMPILER_BACKEND_SINGLE_OP_PATTERN(quantize, Quantize);
COMPILER_BACKEND_SINGLE_OP_PATTERN(dequantize, Dequantize);
COMPILER_BACKEND_SINGLE_OP_PATTERN(dynamic_quantize, DynamicQuantize);
COMPILER_BACKEND_SINGLE_OP_PATTERN(dynamic_dequantize, DynamicDequantize);
COMPILER_BACKEND_SINGLE_OP_PATTERN(static_reshape, StaticReshape);
COMPILER_BACKEND_SINGLE_OP_PATTERN(static_transpose, StaticTranspose);
COMPILER_BACKEND_SINGLE_OP_PATTERN(softmax, SoftMax);
COMPILER_BACKEND_SINGLE_OP_PATTERN(reorder, Reorder);
COMPILER_BACKEND_SINGLE_OP_PATTERN(typecast, TypeCast);
COMPILER_BACKEND_SINGLE_OP_PATTERN(relu, ReLU);
COMPILER_BACKEND_SINGLE_OP_PATTERN(sigmoid, Sigmoid);
COMPILER_BACKEND_SINGLE_OP_PATTERN(gelu, GELU);
COMPILER_BACKEND_SINGLE_OP_PATTERN(relu_backward, ReLUBackward);
COMPILER_BACKEND_SINGLE_OP_PATTERN(sidmoid_backward, SigmoidBackward);
COMPILER_BACKEND_SINGLE_OP_PATTERN(gelu_backward, GELUBackward);
COMPILER_BACKEND_SINGLE_OP_PATTERN(reduce_sum, ReduceSum);
COMPILER_BACKEND_SINGLE_OP_PATTERN(bias_add, BiasAdd);
COMPILER_BACKEND_SINGLE_OP_PATTERN(convolution, Convolution);
COMPILER_BACKEND_SINGLE_OP_PATTERN(
        convolution_backward_data, ConvolutionBackwardData);
COMPILER_BACKEND_SINGLE_OP_PATTERN(
        convolution_backward_weights, ConvolutionBackwardWeights);
COMPILER_BACKEND_SINGLE_OP_PATTERN(
        batchnorm_forward_inference, BatchNormForwardTraining);
COMPILER_BACKEND_SINGLE_OP_PATTERN(
        batchnorm_training_backward, BatchNormTrainingBackward);
COMPILER_BACKEND_SINGLE_OP_PATTERN(maxinum, Maximum);
COMPILER_BACKEND_SINGLE_OP_PATTERN(layernorm, LayerNorm);
COMPILER_BACKEND_SINGLE_OP_PATTERN(select, Select);
COMPILER_BACKEND_SINGLE_OP_PATTERN(tanh, Tanh);
COMPILER_BACKEND_SINGLE_OP_PATTERN(reduce_mean, ReduceMean);
COMPILER_BACKEND_SINGLE_OP_PATTERN(concat, Concat);
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
