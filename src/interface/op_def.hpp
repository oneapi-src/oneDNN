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
#ifndef INTERFACE_OP_DEF_HPP
#define INTERFACE_OP_DEF_HPP

#include <limits>
#include <set>
#include <vector>

#include "interface/op_schema.hpp"
#include "interface/shape_infer.hpp"
#include "interface/type_constraint.hpp"

namespace dnnl {
namespace graph {
namespace impl {

DNNL_GRAPH_OP_SCHEMA(Abs, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(AbsBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input_forward", "input of forward", "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of Abs", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Add, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(AvgPool, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::strides, "the distance to slide the filter",
                        true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, "top and left padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::pads_end, "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::exclude_pad, "a type of pooling strategy",
                        true, attribute_kind::b)
                .set_attr(op_attr::kernel, "size of each filter", true,
                        attribute_kind::is)
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr(op_attr::rounding_type,
                        "a type of rounding to be applied", false,
                        attribute_kind::s, "floor")
                .set_attr(op_attr::auto_pad, "how the padding is calculated",
                        false, attribute_kind::s, "None")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(AvgPoolBackprop, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "output_delta",
                        "the gradient tensor with respect to output of avg "
                        "pool",
                        "T")
                .set_input(1, "input_shape",
                        "(OPTIONAL) the dimensions of original input", "T1")
                .set_output(0, "input_delta",
                        "the the gradient tensor w.r.t. the input of avg pool",
                        "T")
                .set_attr(op_attr::strides, "the distance to slide the filter",
                        true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, "top and left padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::pads_end, "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::exclude_pad, "a type of pooling strategy",
                        true, attribute_kind::b)
                .set_attr(op_attr::kernel, "size of each filter", true,
                        attribute_kind::is)
                .set_attr(op_attr::auto_pad, "how the padding is calculated",
                        false, attribute_kind::s, "None")
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr(op_attr::input_shape, "describing input shape", false,
                        attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T1", {data_type::s32})
                .set_shape_inference_function(infer_pool_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(BatchNormInference, 1,
        op_schema_t()
                .set_num_inputs(5)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(
                        1, "gamma", "gamma scaling for normalized value", "T2")
                .set_input(2, "beta",
                        "beta added to the scaled normalized value", "T2")
                .set_input(3, "mean", "value for mean normalization", "T2")
                .set_input(
                        4, "variance", "value for variance normalization", "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_attr(op_attr::epsilon,
                        "the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_type_constraint_function(check_bn_fwd_data_type))

DNNL_GRAPH_OP_SCHEMA(BatchNormForwardTraining, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4, 5}))
                .set_num_outputs(5)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(
                        1, "gamma", "gamma scaling for normalized value", "T2")
                .set_input(2, "beta",
                        "beta added to the scaled normalized value", "T2")
                .set_input(3, "mean", "value for mean normalization", "T2")
                .set_input(
                        4, "variance", "value for variance normalization", "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_output(
                        1, "running mean", "the computed running mean", "T2")
                .set_output(2, "running variance",
                        "the computed running variance", "T2")
                .set_output(3, "batch mean", "the computed batch mean", "T2")
                .set_output(4, "batch variance", "the computed batch variance",
                        "T2")
                .set_attr(op_attr::epsilon,
                        "the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr(op_attr::momentum,
                        "used for the computation of running_mean and "
                        "running_var",
                        false, attribute_kind::f)
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_bn_fwd_train_output_shape)
                .set_type_constraint_function(check_bn_fwd_data_type))

DNNL_GRAPH_OP_SCHEMA(BatchNormTrainingBackprop, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 2, 3}))
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "output_delta", "the gradient w.r.t. the output",
                        "T1")
                .set_input(
                        2, "gamma", "gamma scaling for normalized value", "T2")
                .set_input(3, "mean",
                        "if is_training is true, pass batch mean, otherwise "
                        "running mean",
                        "T2")
                .set_input(4, "variance",
                        "if is_training is true, pass batch variance, "
                        "otherwise running variance",
                        "T2")
                .set_output(0, "input_delta",
                        "the gradient w.r.t the output of the batch "
                        "normalization",
                        "T1")
                .set_output(1, "gamma_delta",
                        "the gradient w.r.t the gamma of the batch "
                        "normalization",
                        "T2")
                .set_output(2, "beta_delta",
                        "the gradient w.r.t the beta of the batch "
                        "normalization",
                        "T2")
                .set_attr(op_attr::epsilon,
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_bn_bwd_output_shape)
                .set_type_constraint_function(check_bn_bwd_data_type))

DNNL_GRAPH_OP_SCHEMA(BiasAdd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "data tensor", "T")
                .set_input(1, "bias", "1D tensor", "T")
                .set_output(0, "output", "sum of input and bias", "T")
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_bias_add_output_shape))

DNNL_GRAPH_OP_SCHEMA(BiasAddBackprop, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_output(0, "bias_delta", "gradient tensor w.r.t. bias", "T")
                .set_attr(op_attr::data_format,
                        "the data format of input, the options are NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_bias_backprop_output_shape))

DNNL_GRAPH_OP_SCHEMA(Clamp, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::min, "lower bound of values in the output",
                        true, attribute_kind::f)
                .set_attr(op_attr::max, "upper bound of values in the output",
                        true, attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(ClampBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data",
                        "if use_dst is true, data is result of forward. Else, "
                        "data is src of forward.",
                        "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of Clamp.", "T")
                .set_attr(op_attr::min, "lower bound of values in the output",
                        true, attribute_kind::f)
                .set_attr(op_attr::max, "upper bound of values in the output",
                        true, attribute_kind::f)
                .set_attr(op_attr::use_dst,
                        "if true, use dst to calculate gradient; else use src",
                        false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Concat, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 64}))
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::axis,
                        "specifies which dimension to concatenate along", true,
                        attribute_kind::i)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_concat_output_shape))

DNNL_GRAPH_OP_SCHEMA(Convolution, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_input(1, "filter", "filter tensor", "T")
                .set_input(2, "bias", "bias tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvolutionBackpropData, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "weight", "weight tensor", "T1")
                .set_input(2, "output_shape",
                        "tensor, that specifies shape of "
                        "the output",
                        "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_attr(op_attr::output_padding,
                        "additional amount of paddings to be added to each "
                        "spatial axis in the output tensor",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .set_attr(op_attr::output_shape, "describing output shape",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .set_shape_inference_function(
                        infer_conv_bprop_data_output_shape)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvolutionBackpropFilters, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "output_delta",
                        "gradients tensor with respect to the output of the "
                        "convolution",
                        "T1")
                .set_input(2, "filter_shape",
                        "tensor, that specifies shape of filter", "T2")
                .set_output(0, "weight_delta",
                        "gradient tensor with respect to the weight of the "
                        "convolution",
                        "T1")
                .set_attr(op_attr::filter_shape, "describing filter shape",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .set_shape_inference_function(
                        infer_conv_bprop_filters_output_shape)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvTranspose, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_input(1, "weight", "weight tensor", "T")
                .set_input(2, "bias", "bias tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::output_padding,
                        "additional amount of paddings to be added to each "
                        "spatial axis in the output tensor",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_convtranspose_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvTransposeBackpropData, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "output_delta",
                        "gradients tensor with respect to the output of the "
                        "ConvTranspose",
                        "T")
                .set_input(1, "filter", "filter tensor", "T")
                .set_output(0, "input_delta", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_convtranspose_bprop_data_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvTransposeBackpropFilters, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "output_delta",
                        "gradients tensor with respect to the output of the "
                        "ConvTranspose",
                        "T1")
                .set_input(2, "filter_shape",
                        "tensor, that specifies shape of filter", "T2")
                .set_output(0, "filter_delta",
                        "gradient tensor with respect to the weight of the "
                        "ConvTranspose",
                        "T1")
                .set_attr(op_attr::filter_shape, "describing filter shape",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .set_shape_inference_function(
                        infer_convtranspose_bprop_filters_output_shape)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Divide, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T1")
                .set_input(1, "b", "second input tensor", "T2")
                .set_output(0, "output", "output tensor", "T3")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T3", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Elu, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::alpha, "scale for the negative factor", true,
                        attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(EluBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data",
                        "if use_dst is true, data is result of forward. Else, "
                        "data is src of forward.",
                        "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of Elu", "T")
                .set_attr(op_attr::alpha, "scale for the negative factor", true,
                        attribute_kind::f)
                .set_attr(op_attr::use_dst,
                        "if true, use dst to calculate gradient; else use src",
                        false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(End, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(0)
                .set_input(0, "input", "input tensor", "T")
                .set_type_constraints("T",
                        {data_type::f32, data_type::f16, data_type::bf16,
                                data_type::s8, data_type::u8, data_type::s32,
                                data_type::undef}))

DNNL_GRAPH_OP_SCHEMA(Equal, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T1")
                .set_input(1, "b", "second input tensor", "T1")
                .set_output(0, "output", "output tensor", "T2")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Erf, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Exp, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GELU, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GELUBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input_forward", "input of forward", "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of GELU", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Greater, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T1")
                .set_input(1, "b", "second input tensor", "T1")
                .set_output(0, "output", "output tensor", "T2")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(GreaterEqual, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T1")
                .set_input(1, "b", "second input tensor", "T1")
                .set_output(0, "output", "output tensor", "T2")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardSwish, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardSwishBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "forward input tensor of HSwish", "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of HSwish", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Index, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "indices", "indices tensor", "T2")
                .set_output(0, "output",
                        "a tensor with selected data from input tensor", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::s8, data_type::u8, data_type::s32})
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(Interpolate, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "data",
                        "Input tensor with data for interpolation", "T1")
                .set_input(1, "sizes",
                        "optional non-differentiable tensor, describing output"
                        " shape for spatial axes",
                        "T2")
                .set_output(0, "output",
                        "a tensor with selected data from input tensor", "T1")
                .set_attr(op_attr::mode, "specifies type of interpolation",
                        true, attribute_kind::s)
                .set_attr(op_attr::sizes,
                        "describing output shape for spatial axes", false,
                        attribute_kind::is)
                .set_attr(op_attr::scales, "describing scales for spatial axes",
                        false, attribute_kind::fs)
                .set_attr(op_attr::coordinate_transformation_mode,
                        "specifies how to transform the coordinate in the "
                        "resized tensor to the coordinate in the original "
                        "tensor",
                        false, attribute_kind::s, "half_pixel")
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_interpolate_output_shape))

DNNL_GRAPH_OP_SCHEMA(InterpolateBackprop, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "data",
                        "Input tensor with data for interpolation", "T1")
                .set_input(1, "output_delta",
                        "the gradient with respect to the output", "T1")
                .set_input(2, "sizes",
                        "(optional) tensor describing output shape for spatial "
                        "axes",
                        "T2")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to the input of "
                        "interpolate",
                        "T1")
                .set_attr(op_attr::mode, "specifies type of interpolation",
                        true, attribute_kind::s)
                .set_attr(op_attr::coordinate_transformation_mode,
                        "specifies how to transform the coordinate in the "
                        "resized tensor to the coordinate in the original "
                        "tensor",
                        false, attribute_kind::s, "half_pixel")
                .set_attr(op_attr::sizes,
                        "describing output shape for spatial axes", false,
                        attribute_kind::is)
                .set_attr(op_attr::scales, "describing scales for spatial axes",
                        false, attribute_kind::fs)
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LayerNorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 3}))
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "gamma",
                        "(optional) gamma scaling for normalized value", "T2")
                .set_input(2, "beta",
                        "(optional) bias added to the scaled normalized value",
                        "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_output(1, "mean",
                        "(optional) the mean calculated along the given axis",
                        "T2")
                .set_output(2, "variance",
                        "(optional) the std calculated along the given axis",
                        "T2")
                .set_attr(op_attr::keep_stats,
                        "used to indicate whether to output mean and variance",
                        false, attribute_kind::b, true)
                .set_attr(op_attr::begin_norm_axis,
                        "used to indicate which axis to perform layer "
                        "normalization",
                        false, attribute_kind::i, int64_t(-1))
                .set_attr(op_attr::use_affine,
                        "when set to True, this module has learnable "
                        "per-element affine parameters",
                        false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon,
                        "constant to improve numerical stability", false,
                        attribute_kind::f, 1e-5f)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_norm_output_shape)
                .set_type_constraint_function(check_ln_data_type))

DNNL_GRAPH_OP_SCHEMA(LayerNormBackprop, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5, 6}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 3}))
                .set_input(0, "input_tensor", "input tensor", "T1")
                .set_input(1, "output_delta", "the gradient w.r.t. the output",
                        "T1")
                .set_input(2, "mean", "mean of input", "T2")
                .set_input(3, "variance", "variance of input", "T2")
                .set_input(4, "gamma",
                        "(optional) gamma scaling for normalized value", "T2")
                .set_input(5, "beta",
                        "(optional) bias added to the scaled normalized value",
                        "T2")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to the output of the "
                        "layer normalization",
                        "T1")
                .set_output(1, "gamma_delta",
                        "(optional) the gradient tensor with respect to the "
                        "gamma of the layer normalization",
                        "T2")
                .set_output(2, "beta_delta",
                        "(optional) the gradient tensor with respect to the "
                        "beta of the layer normalization",
                        "T2")
                .set_attr(op_attr::begin_norm_axis,
                        "used to indicate which axis to perform layer "
                        "normalization",
                        false, attribute_kind::i, int64_t(-1))
                .set_attr(op_attr::use_affine,
                        "when set to True, this module has learnable "
                        "per-element affine parameters",
                        false, attribute_kind::b, true)
                .set_attr(op_attr::epsilon,
                        "constant to improve numerical stability", false,
                        attribute_kind::f, 1e-5f)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::f32, data_type::bf16})
                .set_shape_inference_function(infer_norm_bprop_output_shape)
                .set_type_constraint_function(check_ln_data_type))

DNNL_GRAPH_OP_SCHEMA(LeakyReLU, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::alpha, "coefficient of the leakage", true,
                        attribute_kind::f)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Less, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T1")
                .set_input(1, "b", "second input tensor", "T1")
                .set_output(0, "output", "output tensor", "T2")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(LessEqual, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T1")
                .set_input(1, "b", "second input tensor", "T1")
                .set_output(0, "output", "output tensor", "T2")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Log, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogSoftmax, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::axis,
                        "the axis of which the LogSoftmax is calculated", false,
                        attribute_kind::i, int64_t(-1))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogSoftmaxBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "output_delta",
                        "gradients tensor w.r.t. the output", "T")
                .set_input(1, "forward_result", "result of forward", "T")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of LogSoftmax",
                        "T")
                .set_attr(op_attr::axis,
                        "the axis of which the LogSoftmax is calculated", false,
                        attribute_kind::i, int64_t(-1))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogicalAnd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints("T", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogicalNot, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "a", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints("T", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogicalOr, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints("T", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogicalXor, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints("T", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(MatMul, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_input(2, "bias", "bias tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Maximum, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(MaxPool, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::strides, "the distance to slide the filter",
                        true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, "top and left padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::pads_end, "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::kernel, "size of each filter", true,
                        attribute_kind::is)
                .set_attr(op_attr::dilations,
                        "the distance in width and height between elements "
                        "in the filter",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr(op_attr::rounding_type,
                        "a type of rounding to be applied", false,
                        attribute_kind::s, "floor")
                .set_attr(op_attr::auto_pad, "how the padding is calculated",
                        false, attribute_kind::s, "None")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(MaxPoolBackprop, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "output_delta",
                        "the gradient tensor with respect to output", "T1")
                .set_input(2, "output_forward_indices",
                        "(optional) indices of max values in output tensor",
                        "T2")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to input", "T1")
                .set_attr(op_attr::strides, "the distance to slide the filter",
                        true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, "top and left padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::pads_end, "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::kernel, "size of each filter", true,
                        attribute_kind::is)
                .set_attr(op_attr::auto_pad, "how the padding is calculated",
                        false, attribute_kind::s, "None")
                .set_attr(op_attr::dilations,
                        "the distance in width and height between elements "
                        "in the filter",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 1))
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::u8, data_type::s32})
                .set_shape_inference_function(infer_pool_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(Minimum, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Mish, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(MishBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data",
                        "if use_dst is true, data is result of forward. Else, "
                        "data is src of forward.",
                        "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of Mish", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

// TODO(Yixin): for Multiply. input and output needs to have the same dtypes
// But in current pytorch bridge's type promotion system, there's no
// such constraints. So this feature is postponed.
DNNL_GRAPH_OP_SCHEMA(Multiply, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_commutative_inputs()
                .set_input(0, "a", "first input tensor", "T1")
                .set_input(1, "b", "second input tensor", "T2")
                .set_output(0, "output", "output tensor", "T3")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T3", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(NotEqual, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T1")
                .set_input(1, "b", "second input tensor", "T1")
                .set_output(0, "output", "output tensor", "T2")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::boolean})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Pow, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(PowBackprop, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input_forward", "input of forward", "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_input(2, "exponent", "exponent of input", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of pow", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(PowBackpropExponent, 1,
        op_schema_t()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input_forward", "input of forward", "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_input(2, "result_forward", "original output of pow", "T")
                .set_input(3, "exponent", "exponent of input", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of pow", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_exponent_output_shape))

DNNL_GRAPH_OP_SCHEMA(PReLU, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data", "input tensor", "T")
                .set_input(1, "slope", "slope tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr(op_attr::per_channel_broadcast,
                        "whether to apply per channel broadcast when slope is "
                        "1D tensor",
                        false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(PReLUBackprop, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(2)
                .set_input(0, "input_forward", "input of forward", "T")
                .set_input(1, "slope", "slope tensor", "T")
                .set_input(2, "output_delta",
                        "the gradient tensor with respect to the output of "
                        "prelu",
                        "T")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to the input of "
                        "prelu",
                        "T")
                .set_output(1, "slope_delta",
                        "the gradient tensor with respect to the slope", "T")
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_prelu_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(ReduceL1, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "axes",
                        "(optional) 1D tensor, specifies indices of input "
                        "data, along which the reduction is performed.",
                        "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceL2, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "axes",
                        "(optional) 1D tensor, specifies indices of input "
                        "data, along which the reduction is performed.",
                        "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceMax, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "axes",
                        "(optional) 1D tensor, specifies indices of input "
                        "data, along which the reduction is performed.",
                        "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceMean, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "axes",
                        "(optional) 1D tensor, specifies indices of input "
                        "data, along which the reduction is performed.",
                        "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceMin, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "axes",
                        "(optional) 1D tensor, specifies indices of input "
                        "data, along which the reduction is performed.",
                        "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceProd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "axes",
                        "(optional) 1D tensor, specifies indices of input "
                        "data, along which the reduction is performed.",
                        "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReduceSum, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_input(1, "axes",
                        "(optional) 1D tensor, specifies indices of input "
                        "data, along which the reduction is performed.",
                        "T2")
                .set_output(0, "output", "output tensor", "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints("T2", {data_type::s32})
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ReLU, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(ReLUBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data",
                        "if use_dst is true, data is result of forward. Else, "
                        "data is src of forward.",
                        "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of ReLU", "T")
                .set_attr(op_attr::use_dst,
                        "if true, use dst to calculate gradient; else use src",
                        false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Round, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Rsqrt, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Sigmoid, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Select, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "cond", "cond tensor with selection mask", "T1")
                .set_input(1, "then", "then tensor", "T2")
                .set_input(2, "else", "else input tensor", "T2")
                .set_output(0, "output", "output tensor", "T2")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints("T1", {data_type::boolean})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_select_output_shape))

DNNL_GRAPH_OP_SCHEMA(SigmoidBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data",
                        "if use_dst is true, data is result of forward. Else, "
                        "data is src of forward.",
                        "T")
                .set_input(1, "output_delta",
                        "gradient tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of Sigmoid", "T")
                .set_attr(op_attr::use_dst,
                        "if true, use dst to calculate gradient, else, use src",
                        false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftMax, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::axis,
                        "the axis of which the SoftMax is calculated", false,
                        attribute_kind::i, (int64_t)1)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftMaxBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "output_delta",
                        "gradients tensor w.r.t. the output", "T")
                .set_input(1, "forward_result", "result of forward", "T")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of SoftMax", "T")
                .set_attr(op_attr::axis,
                        "the axis of which the SoftMax is calculated", false,
                        attribute_kind::i, (int64_t)1)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftPlus, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::beta, "value for the Softplus formulation",
                        false, attribute_kind::i, int64_t(1))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftPlusBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input_forward", "input of forward", "T")
                .set_input(1, "output_delta",
                        "gradients tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of SoftPlus", "T")
                .set_attr(op_attr::beta, "value for the SoftPlus formulation",
                        false, attribute_kind::i, int64_t(1))
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Sqrt, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SqrtBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data",
                        "if use_dst is true, data is result of forward. Else, "
                        "data is src of forward.",
                        "T")
                .set_input(1, "output_delta",
                        "gradients tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of Sqrt", "T")
                .set_attr(op_attr::use_dst,
                        "if true, use dst to calculate gradient; else use "
                        "src.",
                        false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Square, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SquaredDifference, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Subtract, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor", "T")
                .set_input(1, "b", "second input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Tanh, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(TanhBackprop, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data",
                        "if use_dst is true, data is result of forward. Else, "
                        "data is src of forward.",
                        "T")
                .set_input(1, "output_delta",
                        "gradients tensor w.r.t. the output", "T")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of Tanh", "T")
                .set_attr(op_attr::use_dst,
                        "if true, use dst to calculate gradient; else use src",
                        false, attribute_kind::b, true)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Wildcard, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>(
                        {0, std::numeric_limits<size_t>::max()}))
                .set_outputs_option(op_schema_t::param_num_option::variadic)
                .set_num_outputs(std::set<size_t>(
                        {0, std::numeric_limits<size_t>::max()}))
                .set_input(0, "input", "input tensor", "any")
                .set_output(0, "output", "output tensor", "any")
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(Quantize, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", " fp32 tensor to be quantized", "T1")
                .set_output(0, "output", "quantized tensor", "T2")
                .set_attr(op_attr::qtype,
                        "specifies which quantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis,
                        "specifies dimension on which apply per-channel "
                        "quantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::scales, "apply in quantization formula",
                        true, attribute_kind::fs)
                .set_attr(op_attr::zps, "offset value that maps to float zero",
                        true, attribute_kind::is)
                .set_type_constraints("T1", {data_type::f32})
                .set_type_constraints("T2", {data_type::u8, data_type::s8})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Dequantize, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(
                        0, "input", "quantized tensor to be dequantized", "T1")
                .set_output(0, "output", "dequantized tensor", "T2")
                .set_attr(op_attr::qtype,
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis,
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr(op_attr::scales, "apply in quantization formula",
                        true, attribute_kind::fs)
                .set_attr(op_attr::zps, "offset value that maps to float zero",
                        true, attribute_kind::is)
                .set_type_constraints("T1", {data_type::u8, data_type::s8})
                .set_type_constraints("T2", {data_type::f32})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Reorder, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(TypeCast, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T1")
                .set_output(0, "output", "output tensor", "T2")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape)
                .set_type_constraint_function(check_typecast_data_type))

DNNL_GRAPH_OP_SCHEMA(StaticReshape, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "data", "multidimensional input tensor", "T")
                .set_output(0, "output",
                        "Output tensor with the same content as a tensor at "
                        "input data but with shape defined by input shape",
                        "T")
                .set_attr(op_attr::shape, "describing output shape", true,
                        attribute_kind::is)
                .set_attr(op_attr::special_zero,
                        "controls how zero values in shape are interpreted "
                        "shape",
                        true, attribute_kind::b)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_static_reshape_output_shape))

DNNL_GRAPH_OP_SCHEMA(DynamicReshape, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data", "multidimensional input tensor", "T1")
                .set_input(
                        1, "shape", "1D tensor describing output shape", "T2")
                .set_output(0, "output",
                        "Output tensor with the same content as a tensor at "
                        "input data but with shape defined by input shape",
                        "T1")
                .set_attr(op_attr::special_zero,
                        " controls how zero values in shape are interpreted "
                        "shape",
                        true, attribute_kind::b)
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::s8, data_type::u8, data_type::s32})
                .set_shape_inference_function(
                        infer_dynamic_reshape_output_shape))

DNNL_GRAPH_OP_SCHEMA(StaticTranspose, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "data", "the tensor to be transposed", "T")
                .set_output(0, "output",
                        "A tensor with shape and type matching 1st tensor.",
                        "T")
                .set_attr(op_attr::order,
                        "the permutation to apply to the axes of the input "
                        "shape",
                        true, attribute_kind::is)
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(
                        infer_static_transpose_output_shape))

DNNL_GRAPH_OP_SCHEMA(DynamicTranspose, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data", "the tensor to be transposed", "T1")
                .set_input(1, "order",
                        "the permutation to apply to the axes of the input "
                        "shape",
                        "T2")
                .set_output(0, "output",
                        "A tensor with shape and type matching 1st tensor.",
                        "T1")
                .set_type_constraints(
                        "T1", {data_type::f32, data_type::bf16, data_type::f16})
                .set_type_constraints(
                        "T2", {data_type::s8, data_type::u8, data_type::s32})
                .set_shape_inference_function(
                        infer_dynamic_transpose_output_shape))

DNNL_GRAPH_OP_SCHEMA(DynamicQuantize, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "", "T1")
                .set_input(1, "scales", "", "T1")
                .set_input(2, "zps", "", "T2")
                .set_output(0, "output", "", "T3")
                .set_attr(op_attr::qtype, "", false, attribute_kind::s,
                        "per_tensor")
                .set_attr(
                        op_attr::axis, "", false, attribute_kind::i, int64_t(1))
                .set_type_constraints("T1", {data_type::f32})
                .set_type_constraints(
                        "T2", {data_type::u8, data_type::s8, data_type::s32})
                .set_type_constraints("T3", {data_type::u8, data_type::s8})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(DynamicDequantize, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "", "T1")
                .set_input(1, "scales", "", "T2")
                .set_input(2, "zps", "", "T3")
                .set_output(0, "output", "", "T2")
                .set_attr(op_attr::qtype, "", false, attribute_kind::s,
                        "per_tensor")
                .set_attr(
                        op_attr::axis, "", false, attribute_kind::i, int64_t(1))
                .set_type_constraints("T1", {data_type::u8, data_type::s8})
                .set_type_constraints("T2", {data_type::f32})
                .set_type_constraints(
                        "T3", {data_type::u8, data_type::s8, data_type::s32})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Sign, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Negative, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Reciprocal, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor", "T")
                .set_output(0, "output", "output tensor", "T")
                .set_type_constraints(
                        "T", {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
