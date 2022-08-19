/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_CONV_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_CONV_PATTERN_HPP

#include <memory>
#include <utility>

#include "backend/graph_compiler/patterns/fusions.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {
namespace pass {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = impl::utils::pm::pb_graph_t;
using FCreatePattern = impl::pass::FCreatePattern;

pm::pb_op_t *conv_bn_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool has_relu = false, bool is_bf16 = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }

    pm::pb_op_t *conv = pgraph->append_op(impl::op_kind::Convolution, in_edges);
    conv->allow_external_output(0);
    conv->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv->append_decision_function(
                check_input_dtype<impl::data_type::bf16>);
    } else {
        conv->append_decision_function(check_input_dtype<impl::data_type::f32>);
    }

    pm::pb_op_t *bn = pgraph->append_op(impl::op_kind::BatchNormForwardTraining,
            in_edges_t {in_edge(0, conv, 0)});
    bn->allow_external_output(0);
    pm::pb_op_t *output = bn;
    if (has_relu) {
        output = pgraph->append_op(impl::op_kind::ReLU, {in_edge(0, bn, 0)});
        output->allow_external_output(0);
    }
    return output;
}

pm::pb_op_t *conv_bn_relu_bwd(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op_t *input, bool has_relu = false, bool is_bf16 = false) {
    in_edges_t in_edges;
    if (input) {
        // delta is the second input of both bn_bwd and relu_bwd
        in_edges = in_edges_t {in_edge(1, input, 0)};
    }

    pm::pb_op_t *relu_bwd;
    if (has_relu) {
        relu_bwd = pgraph->append_op(impl::op_kind::ReLUBackprop, in_edges);
        relu_bwd->allow_external_output(0);
        in_edges = in_edges_t {in_edge(1, relu_bwd, 0)};
    }
    pm::pb_op_t *bn_bwd = pgraph->append_op(
            impl::op_kind::BatchNormTrainingBackprop, in_edges);
    bn_bwd->allow_external_output(0);
    pm::pb_op_t *conv_bwd_data
            = pgraph->append_op(impl::op_kind::ConvolutionBackpropData,
                    in_edges_t {in_edge(0, bn_bwd, 0)});
    conv_bwd_data->allow_external_output(0);
    conv_bwd_data->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv_bwd_data->append_decision_function(
                check_input_dtype<impl::data_type::bf16>);
    } else {
        conv_bwd_data->append_decision_function(
                check_input_dtype<impl::data_type::f32>);
    }

    pm::pb_op_t *conv_bwd_filter
            = pgraph->append_op(impl::op_kind::ConvolutionBackpropFilters,
                    in_edges_t {in_edge(1, bn_bwd, 0)});
    conv_bwd_filter->allow_external_output(0);
    conv_bwd_filter->append_decision_function(check_conv_attrs);
    if (is_bf16) {
        conv_bwd_filter->append_decision_function(
                check_input_dtype<impl::data_type::bf16>);
    } else {
        conv_bwd_filter->append_decision_function(
                check_input_dtype<impl::data_type::f32>);
    }
    return conv_bwd_data;
}

pm::pb_op_t *convolutional_bottleneck_training_forward(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *dst0 = conv_bn_relu(pgraph, nullptr, true, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu(pgraph, dst1, false, is_bf16);
    pm::pb_op_t *dst3 = conv_bn_relu(pgraph, nullptr, false, is_bf16);
    auto bottleneck_add = pgraph->append_op(impl::op_kind::Add,
            {in_edge(0, dst2, 0), in_edge(0, dst3, 0)}, "bottleneck_add");
    auto relu = pgraph->append_op(
            impl::op_kind::ReLU, {in_edge(0, bottleneck_add, 0)}, "relu_last");
    return relu;
};

pm::pb_op_t *identical_bottleneck_training_forward(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *dst0 = conv_bn_relu(pgraph, input, true, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu(pgraph, dst1, false, is_bf16);
    auto bottleneck_add = pgraph->append_op(
            impl::op_kind::Add, {in_edge(0, dst2, 0)}, "bottleneck_add");
    auto relu = pgraph->append_op(
            impl::op_kind::ReLU, {in_edge(0, bottleneck_add, 0)}, "relu_last");
    return relu;
};

pm::pb_op_t *convolutional_bottleneck_training_backward(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *relu_bwd_top
            = pgraph->append_op(impl::op_kind::ReLUBackprop, "relu_bwd_top");
    pm::pb_op_t *dst0 = conv_bn_relu_bwd(pgraph, relu_bwd_top, false, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu_bwd(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu_bwd(pgraph, dst1, true, is_bf16);
    pm::pb_op_t *dst3 = conv_bn_relu_bwd(pgraph, relu_bwd_top, false, is_bf16);
    pm::pb_op_t *bottleneck_add = pgraph->append_op(impl::op_kind::Add,
            {in_edge(0, dst2, 0), in_edge(1, dst3, 0)}, "bottleneck_add");
    return bottleneck_add;
};

pm::pb_op_t *identical_bottleneck_training_backward(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *relu_bwd_top
            = pgraph->append_op(impl::op_kind::ReLUBackprop, "relu_bwd_top");
    pm::pb_op_t *dst0 = conv_bn_relu_bwd(pgraph, relu_bwd_top, false, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu_bwd(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu_bwd(pgraph, dst1, true, is_bf16);
    pm::pb_op_t *bottleneck_add = pgraph->append_op(impl::op_kind::Add,
            {in_edge(0, dst2, 0), in_edge(1, relu_bwd_top, 0)},
            "bottleneck_add");
    return bottleneck_add;
};

pm::pb_op_t *convolutional_bottleneck_training_backward_v2(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    auto dst0 = conv_bn_relu_bwd(pgraph, nullptr, false, is_bf16);
    auto dst1 = conv_bn_relu_bwd(pgraph, dst0, true, is_bf16);
    auto dst2 = conv_bn_relu_bwd(pgraph, dst1, true, is_bf16);
    auto dst3 = conv_bn_relu_bwd(pgraph, nullptr, false, is_bf16);
    auto bottleneck_add = pgraph->append_op(impl::op_kind::Add,
            {in_edge(0, dst2, 0), in_edge(1, dst3, 0)}, "bottleneck_add");
    auto relu_bwd = pgraph->append_op(impl::op_kind::ReLUBackprop,
            {in_edge(1, bottleneck_add, 0)}, "relu_bwd");
    return relu_bwd;
};

pm::pb_op_t *identical_bottleneck_training_backward_v2(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op_t *input,
        bool is_bf16 = false) {
    pm::pb_op_t *dst0 = conv_bn_relu_bwd(pgraph, nullptr, false, is_bf16);
    pm::pb_op_t *dst1 = conv_bn_relu_bwd(pgraph, dst0, true, is_bf16);
    pm::pb_op_t *dst2 = conv_bn_relu_bwd(pgraph, dst1, true, is_bf16);
    pm::pb_op_t *bottleneck_add = pgraph->append_op(
            impl::op_kind::Add, {in_edge(0, dst2, 0)}, "bottleneck_add");
    pm::pb_op_t *relu_bwd = pgraph->append_op(impl::op_kind::ReLUBackprop,
            {in_edge(1, bottleneck_add, 0)}, "relu_bwd");
    return relu_bwd;
};

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(fp32_conv_pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_identical_bottleneck_forward)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_forward(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_convolutional_bottleneck_forward)
        .set_priority(5.5f)
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_forward(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_identical_bottleneck_backward_v1)
        .set_priority(5.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_backward(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_convolutional_bottleneck_backward_v1)
        .set_priority(5.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_backward(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_identical_bottleneck_backward_v2)
        .set_priority(4.0f) // set to lower priority as backup
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_backward_v2(
                            pgraph, nullptr, false);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, f32_convolutional_bottleneck_backward_v2)
        .set_priority(4.5f) // set to lower priority as backup
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_backward_v2(
                            pgraph, nullptr, false);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(bf16_conv_pattern)
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_identical_bottleneck_forward)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_forward(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_convolutional_bottleneck_forward)
        .set_priority(5.5f)
        .set_kind(impl::partition_kind::residual_conv_blocks)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_forward(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_identical_bottleneck_backward_v1)
        .set_priority(5.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_backward(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_convolutional_bottleneck_backward_v1)
        .set_priority(5.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_backward(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_identical_bottleneck_backward_v2)
        .set_priority(4.0f) // set to lower priority as backup
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_training_backward_v2(
                            pgraph, nullptr, true);
                });

COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_convolutional_bottleneck_backward_v2)
        .set_priority(4.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_training_backward_v2(
                            pgraph, nullptr, true);
                });
COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
