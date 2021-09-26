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
#ifndef BACKEND_DNNL_PATTERNS_SINGLE_OP_PATTERN_HPP
#define BACKEND_DNNL_PATTERNS_SINGLE_OP_PATTERN_HPP

#include <memory>
#include <string>

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

using pb_graph = impl::utils::pm::pb_graph;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(single_op_pass)

#define DNNL_BACKEND_SINGLE_OP_TRANSFORM(name, backend, op, p) \
    DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(backend, name) \
            .set_priority(p) \
            .set_attr<FCreateV2Pattern>("FCreateV2Pattern", \
                    [](std::shared_ptr<pb_graph> pgraph) -> void { \
                        pgraph->append_op(impl::op_kind::op); \
                    }) \
            .set_attr<FCreateV2FusedOp>( \
                    "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> { \
                        std::shared_ptr<op_t> fused_op \
                                = std::make_shared<op_t>(impl::op_kind::op); \
                        fused_op->set_attr<std::string>("backend", #backend); \
                        return fused_op; \
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
DNNL_BACKEND_SINGLE_OP_TRANSFORM(concat_pass, dnnl, Concat, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(conv_pass, dnnl, Convolution, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        conv_data_bw_pass, dnnl, ConvolutionBackpropData, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        conv_filter_bw_pass, dnnl, ConvolutionBackpropFilters, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(convtranspose_pass, dnnl, ConvTranspose, 8.f)
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

// single quantize/dequantize op doesn't need to check data type,
// because it's checked in opschema
DNNL_BACKEND_SINGLE_OP_TRANSFORM(quant_pass, dnnl, Quantize, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(dequant_pass, dnnl, Dequantize, 8.f)

DNNL_BACKEND_SINGLE_OP_TRANSFORM(reorder_pass, dnnl, Reorder, 8.f)

#undef DNNL_BACKEND_SINGLE_OP_TRANSFORM

// if op is typecast, need to filter out bf16-in-f16-out and
// f16-in-bf16-out and same dtype in/out.
#define SET_BF16_F16_CHECK() \
    append_decision_function([](op_t *graph_op) -> bool { \
        logical_tensor_t inport \
                = graph_op->get_input_value(0)->get_logical_tensor(); \
        logical_tensor_t outport \
                = graph_op->get_output_value(0)->get_logical_tensor(); \
        if (inport.data_type == impl::data_type::bf16 \
                && outport.data_type == impl::data_type::f16) \
            return false; \
        if (inport.data_type == impl::data_type::f16 \
                && outport.data_type == impl::data_type::bf16) \
            return false; \
        if (inport.data_type == outport.data_type) return false; \
        return true; \
    })

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, typecast_pass)
        .set_priority(8.f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](std::shared_ptr<pb_graph> pgraph) -> void {
                    impl::utils::pm::pb_op *p_tc = pgraph->append_op(
                            impl::op_kind::TypeCast, "p-typecast");
                    p_tc->SET_BF16_F16_CHECK();
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(impl::op_kind::TypeCast);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
