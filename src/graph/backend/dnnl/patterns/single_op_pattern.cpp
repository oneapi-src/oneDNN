/*******************************************************************************
 * Copyright 2020-2025 Intel Corporation
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
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

using pb_graph_t = graph::utils::pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(single_op_pass)

// pname: pattern name. The default priority of single op transformation
// patterns is always 8.f.
#define DEFAULT_P 8.f
#define DNNL_BACKEND_SINGLE_OP_TRANSFORM(pname, op, kernel) \
    DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, pname) \
            .set_priority(DEFAULT_P) \
            .set_kind(partition_kind_t::misc_post_ops) \
            .set_attr<FCreatePattern>("FCreatePattern", \
                    [](const std::shared_ptr<pb_graph_t> &pgraph) -> void { \
                        pgraph->append_op(graph::op_kind::op); \
                    }) \
            .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr { \
                const kernels_ptr kernels = {std::make_shared<kernel>()}; \
                return kernels; \
            });

// register passes with dnnl backend support
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, eltwise_fwd_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_eltwise
                            = pgraph->append_alternation(get_unary_ops());
                    // the round algorithm in eltwise primitive does not
                    // support other data types.
                    p_eltwise->append_decision_function([](op_t *graph_op)
                                                                -> bool {
                        if (graph_op->get_kind() == graph::op_kind::Round)
                            return check_input_dtype<graph::data_type::f32>(
                                    graph_op);
                        return true;
                    });
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<float_eltwise_fwd>()};
            return kernels;
        });

#if BUILD_TRAINING
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, eltwise_bwd_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pgraph->append_alternation(get_unary_bwd_ops());
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<eltwise_bwd_t>()};
            return kernels;
        });
#endif

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, binary_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pgraph->append_alternation({graph::op_kind::BiasAdd,
                            graph::op_kind::Add, graph::op_kind::Multiply,
                            graph::op_kind::Maximum, graph::op_kind::Minimum,
                            graph::op_kind::Divide, graph::op_kind::Subtract,
                            graph::op_kind::SquaredDifference});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<binary_t>()};
            return kernels;
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, quant_dequant_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pgraph->append_alternation({graph::op_kind::Quantize,
                            graph::op_kind::Dequantize,
                            graph::op_kind::DynamicQuantize,
                            graph::op_kind::DynamicDequantize});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels
                    = {std::make_shared<quantize_dequantize_t>()};
            return kernels;
        });

#if BUILD_TRAINING
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, avg_pool_bw_pass)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_avg_pool_backward
                            = pgraph->append_op(
                                    graph::op_kind::AvgPoolBackward);
                    p_avg_pool_backward->append_decision_function(
                            check_input_num<1>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<pooling_bwd_t>()};
            return kernels;
        });
#endif

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, bn_pass)
        .set_priority(8.f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_batchnorm = pgraph->append_op(
                            graph::op_kind::BatchNormInference);
                    p_batchnorm->append_decision_function(
                            check_input_dtype_from_offset<graph::data_type::f32,
                                    1>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<batch_norm_fwd_t>()};
            return kernels;
        });

#define BATCHNORM_INPUT_NUM_CHECK(n1, n2) \
    append_decision_function([](op_t *graph_op) -> bool { \
        return check_input_num<n1>(graph_op) || check_input_num<n2>(graph_op); \
    })

#define BATCHNORM_OUTPUT_NUM_CHECK(n1, n2) \
    append_decision_function([](op_t *graph_op) -> bool { \
        return check_output_num<n1>(graph_op) \
                || check_output_num<n2>(graph_op); \
    })

#if BUILD_TRAINING
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, bn_fw_train_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_batchnorm_fwd_training
                            = pgraph->append_op(
                                    graph::op_kind::BatchNormForwardTraining);
                    p_batchnorm_fwd_training->append_decision_function(
                            check_input_dtype_from_offset<graph::data_type::f32,
                                    1>);
                    p_batchnorm_fwd_training->BATCHNORM_INPUT_NUM_CHECK(3, 5);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<batch_norm_fwd_t>()};
            return kernels;
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, bn_bw_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_batchnorm_backprop
                            = pgraph->append_op(
                                    graph::op_kind::BatchNormTrainingBackward);
                    p_batchnorm_backprop->append_decision_function(
                            check_input_dtype_from_offset<graph::data_type::f32,
                                    2>);
                    p_batchnorm_backprop->BATCHNORM_OUTPUT_NUM_CHECK(1, 3);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<batch_norm_bwd_t>()};
            return kernels;
        });
#endif

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, ln_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_layernorm
                            = pgraph->append_op(graph::op_kind::LayerNorm);
                    p_layernorm->append_decision_function(
                            check_input_dtype_from_offset<graph::data_type::f32,
                                    1>);
                    p_layernorm->append_decision_function(
                            check_begin_norm_axis_attr);
                    // primitive only support 2-5D data tensor for layernorm
                    p_layernorm->append_decision_function(
                            check_input_ndim_from_offset<0, 2, 5>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<layer_norm_fwd_t>()};
            return kernels;
        });

#if BUILD_TRAINING
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, ln_bw_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_layernorm_bwd
                            = pgraph->append_op(
                                    graph::op_kind::LayerNormBackward);
                    p_layernorm_bwd->append_decision_function(
                            check_input_dtype_from_offset<graph::data_type::f32,
                                    2>);
                    p_layernorm_bwd->append_decision_function(
                            check_begin_norm_axis_attr);
                    // primitive only support 2-5D data tensor for layernorm
                    p_layernorm_bwd->append_decision_function(
                            check_input_ndim_from_offset<0, 2, 5>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<layer_norm_bwd_t>()};
            return kernels;
        });
#endif

DNNL_BACKEND_SINGLE_OP_TRANSFORM(concat_pass, Concat, float_concat)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(conv_pass, Convolution, float_conv_fwd)

#if BUILD_TRAINING
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, conv_data_bw_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_conv_backward_data
                            = pgraph->append_op(
                                    graph::op_kind::ConvolutionBackwardData);
                    // Can be removed after shape tensor is supported
                    p_conv_backward_data->append_decision_function(
                            check_input_num<2>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<conv_bwd_data_t>()};
            return kernels;
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, conv_weights_bwd_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_conv_backward_weights
                            = pgraph->append_op(
                                    graph::op_kind::ConvolutionBackwardWeights);
                    // Can be removed after shape tensor is supported
                    p_conv_backward_weights->append_decision_function(
                            check_input_num<2>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels
                    = {std::make_shared<conv_bwd_weights_t>()};
            return kernels;
        });
#endif

DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        convtranspose_pass, ConvTranspose, float_convtranspose_fwd)
#if BUILD_TRAINING
DNNL_BACKEND_SINGLE_OP_TRANSFORM(convtranspose_data_bwd_pass,
        ConvTransposeBackwardData, conv_transpose_bwd_data_t)

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, convtranspose_weights_bwd_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_conv_backward_weights
                            = pgraph->append_op(graph::op_kind::
                                            ConvTransposeBackwardWeights);
                    // Can be removed after shape tensor is supported
                    p_conv_backward_weights->append_decision_function(
                            check_input_num<2>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels
                    = {std::make_shared<conv_transpose_bwd_weights_t>()};
            return kernels;
        });
#endif

DNNL_BACKEND_SINGLE_OP_TRANSFORM(greater_equal_pass, GreaterEqual, binary_t)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(matmul_pass, MatMul, float_matmul)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(max_pool_pass, MaxPool, float_pooling_fwd)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(prelu_pass, PReLU, float_prelu_fwd)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(logsoftmax_pass, LogSoftmax, logsoftmax_fwd_t)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(softmax_pass, SoftMax, softmax_fwd_t)

#if BUILD_TRAINING
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        max_pool_bw_pass, MaxPoolBackward, pooling_bwd_t)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(prelu_bwd_pass, PReLUBackward, prelu_bwd_t)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        logsoftmax_bwd_pass, LogSoftmaxBackward, logsoftmax_bwd_t)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(
        softmax_bwd_pass, SoftMaxBackward, softmax_bwd_t)
#endif

DNNL_BACKEND_SINGLE_OP_TRANSFORM(reorder_pass, Reorder, float_reorder)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(reorder_pass, StaticTranspose, float_reorder)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(reorder_pass, StaticReshape, float_reorder)
DNNL_BACKEND_SINGLE_OP_TRANSFORM(select_pass, Select, select_t)
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, gn_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_groupnorm
                            = pgraph->append_op(graph::op_kind::GroupNorm);
                    p_groupnorm->append_decision_function(
                            check_input_dtype_from_offset<graph::data_type::f32,
                                    1>);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<group_norm_fwd_t>()};
            return kernels;
        });

// if op is interpolate, need to filter out attrs not supported by dnnl
#define INTERPOLATE_ATTR_CHECK() \
    append_decision_function([](op_t *graph_op) -> bool { \
        if (graph_op->get_attr<std::string>( \
                    op_attr::coordinate_transformation_mode) \
                != std::string("half_pixel")) \
            return false; \
        return true; \
    })

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, interpolate_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_interpolate
                            = pgraph->append_op(graph::op_kind::Interpolate);
                    p_interpolate->INTERPOLATE_ATTR_CHECK();
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<resampling_fwd_t>()};
            return kernels;
        });

#if BUILD_TRAINING
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, interpolate_bwd_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_interpolate_bwd
                            = pgraph->append_op(
                                    graph::op_kind::InterpolateBackward);
                    p_interpolate_bwd->INTERPOLATE_ATTR_CHECK();
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<resampling_bwd_t>()};
            return kernels;
        });
#endif

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

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, typecast_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *p_tc
                            = pgraph->append_op(graph::op_kind::TypeCast);
                    p_tc->SET_BF16_F16_CHECK();
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<float_reorder>()};
            return kernels;
        });

DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, reduce_pass)
        .set_priority(DEFAULT_P)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    graph::utils::pm::pb_op_t *reduction
                            = pgraph->append_alternation(
                                    {graph::op_kind::ReduceL1,
                                            graph::op_kind::ReduceL2,
                                            graph::op_kind::ReduceMax,
                                            graph::op_kind::ReduceMean,
                                            graph::op_kind::ReduceMin,
                                            graph::op_kind::ReduceProd,
                                            graph::op_kind::ReduceSum});
                    reduction->append_decision_function([](op_t *graph_op)
                                                                -> bool {
                        if (graph_op->has_attr(op_attr::axes)
                                && graph_op->get_attr<std::vector<int64_t>>(
                                                   op_attr::axes)
                                           .empty())
                            return false;
                        return true;
                    });
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<float_reduction>()};
            return kernels;
        });

// GenIndex currently is CPU only
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, gen_index_pass)
        .set_priority(DEFAULT_P)
        .set_engine_kind(engine_kind::cpu)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pgraph->append_op(graph::op_kind::GenIndex);
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernels_ptr {
            const kernels_ptr kernels = {std::make_shared<genindex_t>()};
            return kernels;
        });

#undef DNNL_BACKEND_SINGLE_OP_TRANSFORM
#undef DEFAULT_P

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
