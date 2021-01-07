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
#ifndef LLGA_BACKEND_PASS_PASSES_SINGLE_NODE_PASS_HPP
#define LLGA_BACKEND_PASS_PASSES_SINGLE_NODE_PASS_HPP

#include <string>

#include "backend/pass/pass_base.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace pass {

#define DNNL_GRAPH_SINGLE_NODE_TRANSFORM(name, backend, op, p) \
    DNNL_GRAPH_REGISTER_TRANSFORMATION_PASS(backend, name) \
            .set_priority(p) \
            .set_attr<FCreatePattern>("FCreatePattern", \
                    [](pattern *apattern) -> void { \
                        apattern->create_node(op_kind::op); \
                    }) \
            .set_attr<FCreateOptPattern>("FCreateOptPattern", \
                    [](pattern *optimized_pattern) -> void { \
                        node_t *anode \
                                = optimized_pattern->create_node(op_kind::op); \
                        anode->set_attr<std::string>("backend", #backend); \
                    });

DNNL_GRAPH_SINGLE_NODE_TRANSFORM(avg_pool_pass, dnnl, AvgPool, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(avg_pool_bw_pass, dnnl, AvgPoolBackprop, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(bn_pass, dnnl, BatchNormInference, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(ln_pass, dnnl, LayerNorm, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(
        bn_bw_pass, dnnl, BatchNormTrainingBackprop, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(conv_pass, dnnl, Convolution, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(
        conv_data_bw_pass, dnnl, ConvolutionBackpropData, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(
        conv_filter_bw_pass, dnnl, ConvolutionBackpropFilters, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(matmul_pass, dnnl, MatMul, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(max_pool_pass, dnnl, MaxPool, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(max_pool_bw_pass, dnnl, MaxPoolBackprop, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(relu_pass, dnnl, ReLU, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(relu_bw_pass, dnnl, ReLUBackprop, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(gelu_pass, dnnl, GELU, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(gelu_bw_pass, dnnl, GELUBackprop, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(sum_pass, dnnl, Add, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(mul_pass, dnnl, Multiply, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(log_softmax_pass, dnnl, LogSoftmax, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(
        log_softmax_bw_pass, dnnl, LogSoftmaxBackprop, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(softmax_pass, dnnl, SoftMax, 8.f)
DNNL_GRAPH_SINGLE_NODE_TRANSFORM(softmax_bwd_pass, dnnl, SoftMaxBackprop, 8.f)

#undef DNNL_GRAPH_SINGLE_NODE_TRANSFORM

} // namespace pass
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
