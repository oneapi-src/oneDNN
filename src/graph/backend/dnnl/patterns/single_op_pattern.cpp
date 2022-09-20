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

#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/kernels/kernels.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/transformation_pattern.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

using pb_graph_t = graph::utils::pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(single_op_pass)

// pname: pattern name, bname: backend name
#define DNNL_BACKEND_SINGLE_OP_TRANSFORM(pname, op, kernel, p) \
    DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, pname) \
            .set_priority(p) \
            .set_kind(partition_kind_t::misc_post_ops) \
            .set_attr<FCreatePattern>("FCreatePattern", \
                    [](const std::shared_ptr<pb_graph_t> &pgraph) -> void { \
                        pgraph->append_op(graph::op_kind::op); \
                    }) \
            .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr { \
                return std::make_shared<kernel>(); \
            });

// register passes with dnnl backend support
DNNL_BACKEND_SINGLE_OP_TRANSFORM(abs_pass, Abs, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(abs_bw_pass, AbsBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(avg_pool_pass, AvgPool, float_pooling_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        avg_pool_bw_pass, AvgPoolBackward, pooling_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(bias_add_pass, BiasAdd, binary_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        bn_pass, BatchNormInference, batchnorm_fwd_t, 8.f)

#define BATCHNORM_INPUT_NUM_CHECK(n1, n2) \
    append_decision_function([](op_t *graph_op) -> bool { \
        return check_input_num<n1>(graph_op) || check_input_num<n2>(graph_op); \
    })

#define BATCHNORM_OUTPUT_NUM_CHECK(n1, n2) \
    append_decision_function([](op_t *graph_op) -> bool { \
        return check_output_num<n1>(graph_op) \
                || check_output_num<n2>(graph_op); \
    })

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, bn_fw_train_pass)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_batchnorm_fwd_training
                            = pgraph->append_op(
                                    graph::op_kind::BatchNormForwardTraining);
                    p_batchnorm_fwd_training->BATCHNORM_INPUT_NUM_CHECK(3, 5);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<batchnorm_fwd_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, bn_bw_pass)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_batchnorm_backprop
                            = pgraph->append_op(
                                    graph::op_kind::BatchNormTrainingBackward);
                    p_batchnorm_backprop->BATCHNORM_OUTPUT_NUM_CHECK(1, 3);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<batchnorm_bwd_t>();
        });

DNNL_BACKEND_SINGLE_OP_TRANSFORM(ln_pass, LayerNorm, layernorm_fwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        ln_bw_pass, LayerNormBackward, layernorm_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(clamp_pass, Clamp, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        clamp_bw_pass, ClampBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(concat_pass, Concat, float_concat, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(conv_pass, Convolution, float_conv_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        conv_data_bw_pass, ConvolutionBackwardData, conv_bwd_data_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(conv_filter_bw_pass,
        ConvolutionBackwardWeights, conv_bwd_weights_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        convtranspose_pass, ConvTranspose, float_convtranspose_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(convtranspose_data_bwd_pass,
        ConvTransposeBackwardData, convtranspose_bwd_data_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(convtranspose_filter_bwd_pass,
        ConvTransposeBackwardWeights, convtranspose_bwd_weights_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(matmul_pass, MatMul, float_matmul, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(max_pool_pass, MaxPool, float_pooling_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        max_pool_bw_pass, MaxPoolBackward, pooling_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(prelu_pass, PReLU, float_prelu_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        prelu_bwd_pass, PReLUBackward, prelu_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(relu_pass, ReLU, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(relu_bw_pass, ReLUBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(gelu_pass, GELU, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(gelu_bw_pass, GELUBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(elu_pass, Elu, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(elu_bw_pass, EluBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(exp_pass, Exp, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        hardswish_pass, HardSwish, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        hardswish_bw_pass, HardSwishBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        leakyrelu_pass, LeakyReLU, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(log_pass, Log, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(sum_pass, Add, binary_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(mul_pass, Multiply, binary_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(max_pass, Maximum, binary_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(min_pass, Minimum, binary_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(mish_pass, Mish, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(mish_bw_pass, MishBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(div_pass, Divide, binary_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(sub_pass, Subtract, binary_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(round_pass, Round, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(sigmoid_pass, Sigmoid, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        sigmoid_bw_pass, SigmoidBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(sqrt_pass, Sqrt, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(sqrt_bw_pass, SqrtBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(square_pass, Square, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        squareddifference_pass, SquaredDifference, binary_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(tanh_pass, Tanh, float_eltwise_fwd, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(tanh_bw_pass, TanhBackward, eltwise_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        logsoftmax_pass, LogSoftmax, logsoftmax_fwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        logsoftmax_bwd_pass, LogSoftmaxBackward, logsoftmax_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(softmax_pass, SoftMax, softmax_fwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        softmax_bwd_pass, SoftMaxBackward, softmax_bwd_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        quant_pass, Quantize, quantize_dequantize_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        dequant_pass, Dequantize, quantize_dequantize_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        dync_quant_pass, DynamicQuantize, quantize_dequantize_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        dync_dequant_pass, DynamicDequantize, quantize_dequantize_t, 8.f)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(reorder_pass, Reorder, float_reorder, 8.f)

#undef DNNL_BACKEND_SINGLE_OP_TRANSFORM

// if op is interpolate, need to filter out attrs not supported by dnnl
#define INTERPOLATE_ATTR_CHECK() \
    append_decision_function([](op_t *graph_op) -> bool { \
        if (graph_op->get_attr<std::string>( \
                    op_attr::coordinate_transformation_mode) \
                != std::string("half_pixel")) \
            return false; \
        return true; \
    })

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, interpolate_pass)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_interpolate
                            = pgraph->append_op(graph::op_kind::Interpolate,
                                    "p-interpolate");
                    p_interpolate->INTERPOLATE_ATTR_CHECK();
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_resampling_fwd>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, interpolate_bwd_pass)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_interpolate_bwd
                            = pgraph->append_op(
                                    graph::op_kind::InterpolateBackward,
                                    "p-interpolate_bwd");
                    p_interpolate_bwd->INTERPOLATE_ATTR_CHECK();
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<resampling_bwd_t>();
        });

// if op is typecast, need to filter out bf16-in-f16-out and
// f16-in-bf16-out and same dtype in/out.
#define SET_BF16_F16_CHECK() \
    append_decision_function([](op_t *graph_op) -> bool { \
        logical_tensor_t inport \
                = graph_op->get_input_value(0)->get_logical_tensor(); \
        logical_tensor_t outport \
                = graph_op->get_output_value(0)->get_logical_tensor(); \
        if (inport.data_type == graph::data_type::bf16 \
                && outport.data_type == graph::data_type::f16) \
            return false; \
        if (inport.data_type == graph::data_type::f16 \
                && outport.data_type == graph::data_type::bf16) \
            return false; \
        if (inport.data_type == outport.data_type) return false; \
        return true; \
    })

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, typecast_pass)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_tc = pgraph->append_op(
                            graph::op_kind::TypeCast, "p-typecast");
                    p_tc->SET_BF16_F16_CHECK();
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_reorder>();
        });

// pname: pattern name, bname: backend name
#define DNNL_BACKEND_SINGLE_REDUCE_OP_TRANSFORM(pname, bname, op, p) \
    DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(bname, pname) \
            .set_priority(p) \
            .set_attr<FCreatePattern>("FCreatePattern", \
                    [](const std::shared_ptr<pb_graph_t> &pgraph) -> void { \
                        graph::utils::pm::pb_op_t *reduction \
                                = pgraph->append_op(graph::op_kind::op); \
                        reduction->append_decision_function([](op_t *graph_op) \
                                                                    -> bool { \
                            if (graph_op->has_attr(op_attr::axes) \
                                    && graph_op->get_attr< \
                                                       std::vector<int64_t>>( \
                                                       op_attr::axes) \
                                               .empty()) \
                                return false; \
                            return true; \
                        }); \
                    }) \
            .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr { \
                return std::make_shared<float_reduction>(); \
            });

DNNL_BACKEND_SINGLE_REDUCE_OP_TRANSFORM(reduce_pass, dnnl, ReduceL1, 8.f)
DNNL_BACKEND_SINGLE_REDUCE_OP_TRANSFORM(reduce_pass, dnnl, ReduceL2, 8.f)
DNNL_BACKEND_SINGLE_REDUCE_OP_TRANSFORM(reduce_pass, dnnl, ReduceMax, 8.f)
DNNL_BACKEND_SINGLE_REDUCE_OP_TRANSFORM(reduce_pass, dnnl, ReduceMean, 8.f)
DNNL_BACKEND_SINGLE_REDUCE_OP_TRANSFORM(reduce_pass, dnnl, ReduceMin, 8.f)
DNNL_BACKEND_SINGLE_REDUCE_OP_TRANSFORM(reduce_pass, dnnl, ReduceProd, 8.f)
DNNL_BACKEND_SINGLE_REDUCE_OP_TRANSFORM(reduce_pass, dnnl, ReduceSum, 8.f)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, softplus_pass)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pgraph->append_op(graph::op_kind::SoftPlus, "softplus");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<float_eltwise_fwd>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, softplus_bw_pass)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pgraph->append_op(
                            graph::op_kind::SoftPlusBackward, "softplus_bw");
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<eltwise_bwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
