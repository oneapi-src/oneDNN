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
#ifndef BACKEND_DNNL_PASSES_SINGLE_OP_PASS_HPP
#define BACKEND_DNNL_PASSES_SINGLE_OP_PASS_HPP

#include <string>

#include "backend/dnnl/transformation_pass.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

using pattern = impl::pass::pattern;
using FCreatePattern = impl::pass::FCreatePattern;
using FCreateOptPattern = impl::pass::FCreateOptPattern;

DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(single_op_pass)

#define DNNL_BACKEND_SINGLE_OP_TRANSFORM(name, backend, op, p) \
    DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(backend, name) \
            .set_priority(p) \
            .set_attr<FCreatePattern>("FCreatePattern", \
                    [](pattern *apattern) -> void { \
                        apattern->create_op(op_kind::op); \
                    }) \
            .set_attr<FCreateOptPattern>("FCreateOptPattern", \
                    [](pattern *optimized_pattern) -> void { \
                        op_t *aop = optimized_pattern->create_op(op_kind::op); \
                        aop->set_attr<std::string>("backend", #backend); \
                    });

// register passes with dnnl backend support
DNNL_BACKEND_SINGLE_OP_TRANSFORM(abs_pass, dnnl, Abs, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(avg_pool_pass, dnnl, AvgPool, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(avg_pool_bw_pass, dnnl, AvgPoolBackprop, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(bn_pass, dnnl, BatchNormInference, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(ln_pass, dnnl, LayerNorm, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        bn_fw_train_pass, dnnl, BatchNormForwardTraining, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        bn_bw_pass, dnnl, BatchNormTrainingBackprop, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(conv_pass, dnnl, Convolution, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        conv_data_bw_pass, dnnl, ConvolutionBackpropData, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        conv_filter_bw_pass, dnnl, ConvolutionBackpropFilters, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(matmul_pass, dnnl, MatMul, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(max_pool_pass, dnnl, MaxPool, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(max_pool_bw_pass, dnnl, MaxPoolBackprop, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(relu_pass, dnnl, ReLU, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(relu_bw_pass, dnnl, ReLUBackprop, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(gelu_pass, dnnl, GELU, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(gelu_bw_pass, dnnl, GELUBackprop, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(elu_pass, dnnl, Elu, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(exp_pass, dnnl, Exp, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(hardtanh_pass, dnnl, HardTanh, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(log_pass, dnnl, Log, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(sum_pass, dnnl, Add, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(mul_pass, dnnl, Multiply, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(max_pass, dnnl, Maximum, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(min_pass, dnnl, Minimum, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(pow_pass, dnnl, Pow, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(sqrt_pass, dnnl, Sqrt, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(square_pass, dnnl, Square, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(tanh_pass, dnnl, Tanh, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(log_softmax_pass, dnnl, LogSoftmax, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        log_softmax_bw_pass, dnnl, LogSoftmaxBackprop, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(softmax_pass, dnnl, SoftMax, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(softmax_bwd_pass, dnnl, SoftMaxBackprop, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(quant_pass, dnnl, Quantize, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(dequant_pass, dnnl, Dequantize, 8.f)

#undef DNNL_BACKEND_SINGLE_OP_TRANSFORM

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
