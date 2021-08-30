/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#ifndef BACKEND_DNNL_DNNL_OP_DEF_HPP
#define BACKEND_DNNL_DNNL_OP_DEF_HPP

#include <limits>
#include <set>
#include <vector>

#include "dnnl_shape_infer.hpp"
#include "interface/op_schema.hpp"
#include "interface/shape_infer.hpp"
#include "internal_ops.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

template <typename T>
op_schema get_op_schema();

DNNL_GRAPH_OP_SCHEMA(add_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(add_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(add_multiply, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_input(2, "other", "the second input tensor of multiply")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(maximum_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(maximum_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(maximum_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(avgpool_add, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_input(1, "other", "the second input tensor of add",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_attr("strides", "the distance to slide the filter", true,
                        attribute_kind::is)
                .set_attr("pads_begin", "top and left padding", true,
                        attribute_kind::is)
                .set_attr("pads_end", "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr("exclude_pad", "a type of pooling strategy", true,
                        attribute_kind::b)
                .set_attr("kernel", "size of each filter", true,
                        attribute_kind::is)
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("rounding_type", "a type of rounding to be applied",
                        false, attribute_kind::s, "floor")
                .set_attr("auto_pad", "how the padding is calculated", false,
                        attribute_kind::s, "None")
                .set_shape_inference_function(infer_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(maxpool_add, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_input(1, "other", "the second input tensor of add",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_attr("strides", "the distance to slide the filter", true,
                        attribute_kind::is)
                .set_attr("pads_begin", "top and left padding", true,
                        attribute_kind::is)
                .set_attr("pads_end", "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr("kernel", "size of each filter", true,
                        attribute_kind::is)
                .set_attr("dilations",
                        "the distance in width and height between elements "
                        "in the filter",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 1))
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("rounding_type", "a type of rounding to be applied",
                        false, attribute_kind::s, "floor")
                .set_attr("auto_pad", "how the padding is calculated", false,
                        attribute_kind::s, "None")
                .set_shape_inference_function(infer_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(int8_maxpool, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("strides", "the distance to slide the filter", true,
                        attribute_kind::is)
                .set_attr("pads_begin", "top and left padding", true,
                        attribute_kind::is)
                .set_attr("pads_end", "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr("kernel", "size of each filter", true,
                        attribute_kind::is)
                .set_attr("dilations",
                        "the distance in width and height between elements "
                        "in the filter",
                        false, attribute_kind::is,
                        std::vector<int64_t>(1, DNNL_GRAPH_MAX_NDIMS))
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("rounding_type", "a type of rounding to be applied",
                        false, attribute_kind::s, "floor")
                .set_attr("auto_pad", "how the padding is calculated", false,
                        attribute_kind::s, "None")
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(int8_avgpool, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("strides", "the distance to slide the filter", true,
                        attribute_kind::is)
                .set_attr("pads_begin", "top and left padding", true,
                        attribute_kind::is)
                .set_attr("pads_end", "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr("exclude_pad", "a type of pooling strategy", true,
                        attribute_kind::b)
                .set_attr("kernel", "size of each filter", true,
                        attribute_kind::is)
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("rounding_type", "a type of rounding to be applied",
                        false, attribute_kind::s, "floor")
                .set_attr("auto_pad", "how the padding is calculated", false,
                        attribute_kind::s, "None")
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(minimum_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(minimum_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(minimum_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(multiply_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(multiply_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(multiply_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "lhs", "first input tensor")
                .set_input(1, "rhs", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(relu_add, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_input(1, "other", "the second input tensor of add",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(conv_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_input(1, "weight", "weight tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_input(2, "bias", "bias tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_abs, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_add, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_input(1, "weight", "weight tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_input(2, "bias", "bias tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_input(3, "other", "the second input tensor of add",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_add_elu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("alpha", "scale for the negative factor", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_add_relu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_add_relu6, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("min",
                        "lower bound of values in the output, should be 0",
                        true, attribute_kind::f)
                .set_attr("max",
                        "upper bound of values in the output, should be 6",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_add_elu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("alpha", "scale for the negative factor", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_add_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_add_relu6, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("min",
                        "lower bound of values in the output, should be 0",
                        true, attribute_kind::f)
                .set_attr("max",
                        "upper bound of values in the output, should be 6",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_elu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .set_attr("alpha", "scale for the negative factor", true,
                        attribute_kind::f)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_hardtanh, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("min", "lower bound of values in the output", true,
                        attribute_kind::f)
                .set_attr("max", "upper bound of values in the output", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_relu6, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("min",
                        "lower bound of values in the output, should be 0",
                        true, attribute_kind::f)
                .set_attr("max",
                        "upper bound of values in the output, should be 6",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_sqrt, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_square, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_tanh, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bn, 1,
        op_schema()
                .set_num_inputs(6)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "gamma", "gamma scaling for normalized value")
                .set_input(
                        3, "beta", "bias added to the scaled normalized value")
                .set_input(4, "mean", "value for mean normalization")
                .set_input(5, "variance", "value for variance normalization")
                .set_output(0, "output", "output tensor")
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_bn, 1,
        op_schema()
                .set_num_inputs(7)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "gamma", "gamma scaling for normalized value")
                .set_input(
                        4, "beta", "bias added to the scaled normalized value")
                .set_input(5, "mean", "value for mean normalization")
                .set_input(6, "variance", "value for variance normalization")
                .set_output(0, "output", "output tensor")
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bn_add, 1,
        op_schema()
                .set_num_inputs(7)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "gamma", "gamma scaling for normalized value")
                .set_input(
                        3, "beta", "bias added to the scaled normalized value")
                .set_input(4, "mean", "value for mean normalization")
                .set_input(5, "variance", "value for variance normalization")
                .set_input(6, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bn_relu, 1,
        op_schema()
                .set_num_inputs(6)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "gamma", "gamma scaling for normalized value")
                .set_input(
                        3, "beta", "bias added to the scaled normalized value")
                .set_input(4, "mean", "value for mean normalization")
                .set_input(5, "variance", "value for variance normalization")
                .set_output(0, "output", "output tensor")
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_bn_relu, 1,
        op_schema()
                .set_num_inputs(7)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "gamma", "gamma scaling for normalized value")
                .set_input(
                        4, "beta", "bias added to the scaled normalized value")
                .set_input(5, "mean", "value for mean normalization")
                .set_input(6, "variance", "value for variance normalization")
                .set_output(0, "output", "output tensor")
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bn_add_relu, 1,
        op_schema()
                .set_num_inputs(7)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "gamma", "gamma scaling for normalized value")
                .set_input(
                        3, "beta", "bias added to the scaled normalized value")
                .set_input(4, "mean", "value for mean normalization")
                .set_input(5, "variance", "value for variance normalization")
                .set_input(6, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_bias_bn_add_relu, 1,
        op_schema()
                .set_num_inputs(8)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "gamma", "gamma scaling for normalized value")
                .set_input(
                        4, "beta", "bias added to the scaled normalized value")
                .set_input(5, "mean", "value for mean normalization")
                .set_input(6, "variance", "value for variance normalization")
                .set_input(7, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_input(1, "weight", "weight tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(bn_relu, 1,
        op_schema()
                .set_num_inputs(5)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "gamma", "gamma scaling for normalized value")
                .set_input(
                        2, "beta", "beta added to the scaled normalized value")
                .set_input(3, "mean", "value for mean normalization")
                .set_input(4, "variance", "value for variance normalization")
                .set_output(0, "output", "output tensor")
                .set_attr("epsilon",
                        "the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(matmul_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_add_gelu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_add_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_add, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_add_relu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_bn, 1,
        op_schema()
                .set_num_inputs(7)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "gamma", "gamma scaling for normalized value")
                .set_input(
                        4, "beta", "bias added to the scaled normalized value")
                .set_input(5, "mean", "value for mean normalization")
                .set_input(6, "variance", "value for variance normalization")
                .set_output(0, "output", "output tensor")
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_elu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("alpha", "scale for the negative factor", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_hardtanh, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("min", "lower bound of values in the output", true,
                        attribute_kind::f)
                .set_attr("max", "upper bound of values in the output", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_relu6, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("min",
                        "lower bound of values in the output, should be 0",
                        true, attribute_kind::f)
                .set_attr("max",
                        "upper bound of values in the output, should be 6",
                        true, attribute_kind::f)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_gelu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_add_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_elu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_hardtanh, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_gelu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_conv, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_conv, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_conv_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_conv_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_conv_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_conv_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_conv_bias_add_relu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_conv_add_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_conv_add_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_input(2, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_conv_bias_add_relu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_conv_bias_relu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_conv_bias, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_conv, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_conv_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_conv_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_conv_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_conv_bias_add_relu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_conv_add_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_conv, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_conv_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_conv_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_conv_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_conv_bias_add_relu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_conv_add_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul_bias_relu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_bias_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul_bias_sigmoid, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_gelu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul_gelu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_bias_gelu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul_bias_gelu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul_bias_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul_gelu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul_bias_gelu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::f32)
                .set_input(2, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_bias_add, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_quant_wei_matmul_bias_add, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_matmul_bias_add, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul_bias_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul_gelu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul_bias_gelu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_output(0, "output", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(x8s8f32_quant_wei_matmul_bias_add, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_input(1, "filter", "filter tensor", impl::data_type::s8)
                .set_input(2, "bias", "bias tensor", impl::data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_output(0, "output", "output tensor", impl::data_type::f32)
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "dequantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(mul_scales, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8,
                                impl::data_type::f32})
                .set_output(0, "y", "output tensor", impl::data_type::f32)
                .set_attr("qtype", "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis", "quantization type", false, attribute_kind::i,
                        int64_t(1))
                .set_attr("scales", "input scale", true, attribute_kind::fs)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(add_zps, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8,
                                impl::data_type::f32})
                .set_output(0, "y", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8})
                .set_attr("qtype", "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis", "quantization type", false, attribute_kind::i,
                        int64_t(1))
                .set_attr("zps", "input zero_point", true, attribute_kind::is)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(permute, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8,
                                impl::data_type::f32})
                .set_output(0, "y", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8,
                                impl::data_type::f32})
                .set_attr("from_format",
                        "the format of input, the options are NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("to_format",
                        "the format of output, the options are NCX and NXC",
                        false, attribute_kind::s, "NCX")
                .set_attr("permute_kind",
                        "if set to transpose then [from/to]_format will be "
                        "ignored",
                        false, attribute_kind::s, "none")
                .set_shape_inference_function(infer_permute_output_shape))

DNNL_GRAPH_OP_SCHEMA(to_group, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8,
                                impl::data_type::f32})
                .set_output(0, "y", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8,
                                impl::data_type::f32})
                .set_attr("groups", "the groups", false, attribute_kind::i,
                        (int64_t)1)
                .set_shape_inference_function(infer_to_group_output_shape))

DNNL_GRAPH_OP_SCHEMA(expand, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {impl::data_type::s8, impl::data_type::u8,
                                impl::data_type::f32})
                .set_output(0, "y", "output tensor",
                        {impl::data_type::s8, impl::data_type::u8,
                                impl::data_type::f32})
                .set_attr("insert_1dim", "where to insert 1 dim", false,
                        attribute_kind::s, "none")
                .set_attr("expand_to", "target ndims to expand", false,
                        attribute_kind::i, (int64_t)(-1))
                .set_shape_inference_function(infer_expand_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_convolution, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16, impl::data_type::u8,
                                impl::data_type::s8})
                .set_input(1, "filter", "filter tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16, impl::data_type::s8})
                .set_input(2, "bias", "bias tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16, impl::data_type::s32})
                .set_output(0, "output", "output tensor",
                        {impl::data_type::f32, impl::data_type::bf16,
                                impl::data_type::f16, impl::data_type::u8,
                                impl::data_type::s8})
                .set_shape_inference_function(infer_dnnl_conv_output_shape)
                .set_attr("output_format",
                        "the data format of output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(dnnl_pool, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("strides", "the distance to slide the filter", true,
                        attribute_kind::is)
                .set_attr("pads_begin", "top and left padding", true,
                        attribute_kind::is)
                .set_attr("pads_end", "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr("exclude_pad", "a type of pooling strategy", false,
                        attribute_kind::b)
                .set_attr("kernel", "size of each filter", true,
                        attribute_kind::is)
                .set_attr("dilations",
                        "the distance in width and height between elements "
                        "in the filter",
                        false, attribute_kind::is,
                        std::vector<int64_t>(1, DNNL_GRAPH_MAX_NDIMS))
                .set_attr("output_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("rounding_type", "a type of rounding to be applied",
                        false, attribute_kind::s, "floor")
                .set_attr("auto_pad", "how the padding is calculated", false,
                        attribute_kind::s, "None")
                .set_attr("kind", "pooling kind, maxpool or avgpool", true,
                        attribute_kind::s)
                .set_shape_inference_function(infer_dnnl_pool_output_shape))

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
