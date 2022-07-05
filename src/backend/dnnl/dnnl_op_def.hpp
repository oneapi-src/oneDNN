/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
op_schema_t get_op_schema();

DNNL_GRAPH_OP_SCHEMA(binary_post_ops_fusion, 1,
        op_schema_t()
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

DNNL_GRAPH_OP_SCHEMA(pool_binary, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "other", "the second input tensor of add")
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
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("dilations",
                        "the distance in width and height between elements "
                        "in the filter",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 1))
                .set_attr("rounding_type", "a type of rounding to be applied",
                        false, attribute_kind::s, "floor")
                .set_attr("auto_pad", "how the padding is calculated", false,
                        attribute_kind::s, "None")
                .set_shape_inference_function(infer_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(int8_maxpool, 1,
        op_schema_t()
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

DNNL_GRAPH_OP_SCHEMA(int8_maxpool_add, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
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
        op_schema_t()
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

DNNL_GRAPH_OP_SCHEMA(int8_avgpool_add, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
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

DNNL_GRAPH_OP_SCHEMA(eltwise_binary, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

// TODO(xxx) Merge this op into dnnl_convolution
DNNL_GRAPH_OP_SCHEMA(dnnl_conv_depthwise, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4, 5}))
                .set_num_outputs(std::set<size_t>({2, 3}))
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(
                        2, "other", "weight tensor for depthwise convolution")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                .set_output(2, "intermediate output",
                        "the intermediate output of base conv, which is needed "
                        "to create primitives")
                // Attributes inherited from Convolution
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("with_bias", "specifying if the op has a bias input",
                        false, attribute_kind::b, false)
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("dw_groups",
                        "the number of groups input / output channels are "
                        "divided into (for depthwise post-op)",
                        false, attribute_kind::i, (int64_t)1)
                .set_attr("dw_filter_format",
                        "the format of post depthwise weight, the options are "
                        "OIX, XIO",
                        false, attribute_kind::s, "XIO")
                .set_attr("dw_type",
                        "the type of post depthwise operation, the options are "
                        "k3s1p1 and k3s2p1",
                        true, attribute_kind::s)
                .set_attr("with_dw_bias",
                        "specifying if the fused dw conv has a bias input",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_conv_depthwise_output_shape))

DNNL_GRAPH_OP_SCHEMA(bn_relu, 1,
        op_schema_t()
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

// This op schema represents part of convolution related fusions.
// The fusion patterns should follow the below general rule:
//
//      Convolution + [Add|Mul] x [0, 32] + [ReLU|Abs|Elu|GELU] x [0, 32] +
//                                                          [Add|Mul] x [0, 32]
//
// In above rule, currently the supported binary ops are Add and Multiply. The
// supported eltwise ops are ReLU, Abs, Elu and GELU.
//
// [Add|Mul] x [0, 32] means the repetition times of block
// [Add|Mul] can be from 0 to 32. So do [ReLU|Abs|Elu|GELU] block and the second
// [Add|mul] block. Hence it will cover but not limited to the below patterns:
//
//  1. Convolution + [Add|Mul] + ... + [Add|Mul] + ReLU + ... + Abs + [Add|Mul]
//                                                      + ... + [Add|Mul]
//  2. Convolution + ReLU + ... + Elu + [Add|Mul] + ... + [Add|Mul]
//  3. Convolution + ReLU + ... + GELU
//  4. Convolution + [Add|Mul] + ... + [Add|Mul]
//  ......
DNNL_GRAPH_OP_SCHEMA(conv_bias_post_ops_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({3, 35}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(conv_post_ops_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 34}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_conv_post_ops_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 35}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

// This op schema represents all convtranspose related fusions.
// At the moment available are:
// convtranspose + bias,
// convtranspose + w/wo bias + binary add,
// convtranspose + w/wo bias + sum,
// convtranspose + w/wo bias + relu.
// Thanks to the unification of the op schema for these patterns,
// we can reduce the size of the binary.
DNNL_GRAPH_OP_SCHEMA(convtranspose_fusion, 1,
        op_schema_t()
                .set_num_inputs(std::set<size_t>({2, 3, 4}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_attr("output_padding",
                        "additional amount of paddings to be added to each "
                        "spatial axis in the output tensor",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .set_shape_inference_function(infer_convtranspose_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(quantized_convtranspose_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3, 4}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_input(3, "other", "other tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("output_padding",
                        "additional amount of paddings to be added to each "
                        "spatial axis in the output tensor",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .set_shape_inference_function(infer_convtranspose_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_post_ops_chain_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 34}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "filter", "filter tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(matmul_bias_post_ops_chain_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({3, 35}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "filter", "filter tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(int8_matmul_post_ops_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 35}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "filter", "filter tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(dnnl_mul_scales, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_input(1, "scales", "scales tensor")
                .set_output(0, "y", "output tensor")
                .set_attr("qtype", "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis", "quantization type", false, attribute_kind::i,
                        int64_t(1))
                .set_attr("scales", "input scale", false, attribute_kind::fs,
                        std::vector<float>())
                .set_attr("with_runtime_scales",
                        "indicate whether the op has runtime scales input",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_constant_scales, 1,
        op_schema_t()
                .set_num_inputs(0)
                .set_num_outputs(1)
                .set_output(0, "output", "output tensor")
                .set_attr("scales", "scales to store in constant storage", true,
                        attribute_kind::fs)
                .set_attr("shape", "describing output shape", true,
                        attribute_kind::is)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                .set_shape_inference_function(infer_dnnl_constant_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_add_zps, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_input(1, "zps", "zps tensor")
                .set_output(0, "y", "output tensor")
                .set_attr("qtype", "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis", "quantization type", false, attribute_kind::i,
                        int64_t(1))
                .set_attr("zps", "input zero_point", false, attribute_kind::is,
                        std::vector<int64_t>())
                .set_attr("with_runtime_zps",
                        "indicate whether the op has runtime zps input", false,
                        attribute_kind::b, false)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_sub_zps, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_input(1, "zps", "zps tensor")
                .set_output(0, "y", "output tensor")
                .set_attr("qtype", "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis", "quantization type", false, attribute_kind::i,
                        int64_t(1))
                .set_attr("zps", "input zero_point", false, attribute_kind::is,
                        std::vector<int64_t>())
                .set_attr("with_runtime_zps",
                        "indicate whether the op has runtime zps input", false,
                        attribute_kind::b, false)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_constant_zps, 1,
        op_schema_t()
                .set_num_inputs(0)
                .set_num_outputs(1)
                .set_output(0, "output", "output tensor")
                .set_attr("zps", "zero points to store in constant storage",
                        true, attribute_kind::is)
                .set_attr("shape", "describing output shape", true,
                        attribute_kind::is)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                .set_shape_inference_function(infer_dnnl_constant_output_shape))

DNNL_GRAPH_OP_SCHEMA(permute, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
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
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
                .set_attr("groups", "the groups", false, attribute_kind::i,
                        (int64_t)1)
                .set_attr("is_convtranspose",
                        "indicate whether this is for convtranspose", false,
                        attribute_kind::b, false)
                .set_shape_inference_function(infer_to_group_output_shape))

DNNL_GRAPH_OP_SCHEMA(from_group, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
                .set_attr("groups", "the groups", false, attribute_kind::i,
                        (int64_t)1)
                .set_attr("is_convtranspose",
                        "indicate whether this is for convtranspose", false,
                        attribute_kind::b, false)
                .set_shape_inference_function(infer_from_group_output_shape))

DNNL_GRAPH_OP_SCHEMA(expand, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
                .set_attr("axes",
                        "indices at which to insert the singleton dimension, "
                        "negative value means counting dimensions from the "
                        "back",
                        false, attribute_kind::is)
                .set_attr("insert_1dim", "where to insert 1 dim", false,
                        attribute_kind::s, "none")
                .set_attr("expand_to", "target ndims to expand", false,
                        attribute_kind::i, (int64_t)(-1))
                .set_shape_inference_function(infer_expand_output_shape))

DNNL_GRAPH_OP_SCHEMA(squeeze, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
                .set_attr("axes",
                        "which dims to be squeezed, negative "
                        "value means counting dimensions from the back",
                        false, attribute_kind::is)
                .set_shape_inference_function(infer_squeeze_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_convolution, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_input(1, "filter", "filter tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from Convolution.
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("with_bias", "specifying if the op has a bias input",
                        false, attribute_kind::b, false)
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_conv_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_convtranspose, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from ConvTranspose.
                .set_attr("output_padding",
                        "additional amount of paddings to be added to each "
                        "spatial axis in the output tensor",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("with_bias", "specifying if the op has a bias input",
                        false, attribute_kind::b, false)
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_convtranspose_bwd_data, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "output_delta",
                        "gradients tensor with respect to the output of the "
                        "ConvTranspose")
                .set_input(1, "filter", "filter tensor")
                .set_output(0, "input_delta", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from ConvTransposeBackpropData.
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_bwd_data_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_convtranspose_bwd_weights, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_input(1, "output_delta",
                        "gradients tensor with respect to the output of the "
                        "ConvTranspose")
                .set_input(2, "filter_shape",
                        "tensor, that specifies shape of filter")
                .set_output(0, "filter_delta",
                        "gradient tensor with respect to the weight of the "
                        "ConvTranspose")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from ConvTransposeBackpropFilters.
                .set_attr("filter_shape", "describing filter shape", false,
                        attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_bwd_weight_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_pool, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3}))
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                .set_output(2, "workspace", "workspace tensor")
                // Attributes inherited from MaxPool and AvgPool.
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
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 1))
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("rounding_type", "a type of rounding to be applied",
                        false, attribute_kind::s, "floor")
                .set_attr("auto_pad", "how the padding is calculated", false,
                        attribute_kind::s, "None")
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("kind", "pooling kind, maxpool or avgpool", true,
                        attribute_kind::s)
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                .set_attr("is_training", "whether this is for training", false,
                        attribute_kind::b)
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_pool_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3}))
                .set_num_outputs(2)
                .set_input(0, "output_delta",
                        "the gradient tensor with respect to output")
                .set_input(1, "output_forward_indices",
                        "(optional) indices of max values in output tensor")
                .set_input(2, "forward_src",
                        "(optional) source of forward operator")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to input")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
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
                .set_attr("auto_pad", "how the padding is calculated", false,
                        attribute_kind::s, "None")
                .set_attr("dilations",
                        "the distance in width and height between elements "
                        "in the filter",
                        false, attribute_kind::is,
                        std::vector<int64_t>(1, DNNL_GRAPH_MAX_NDIMS))
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("input_shape", "describing input shape", true,
                        attribute_kind::is)
                // New added attributes
                .set_attr("kind", "pooling kind, maxpool or avgpool", true,
                        attribute_kind::s)
                .set_shape_inference_function(infer_dnnl_pool_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_prelu, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "data", "input tensor")
                .set_input(1, "slope", "slope tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from PReLU
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("per_channel_broadcast",
                        "whether to apply per channel broadcast when slope is "
                        "1D tensor",
                        false, attribute_kind::b, true)
                // New added attributes
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_prelu_bwd, 1,
        op_schema_t()
                .set_num_inputs(3)
                .set_num_outputs(3)
                .set_input(0, "input_forward", "input of forward")
                .set_input(1, "slope", "slope tensor")
                .set_input(2, "output_delta",
                        "the gradient tensor with respect to the output of "
                        "prelu")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to the input of "
                        "prelu")
                .set_output(1, "slope_delta",
                        "the gradient tensor with respect to the slope")
                .set_output(2, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from PReLUBackprop
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                // New added attributes
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_prelu_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_bn_folding, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({5, 6}))
                .set_num_outputs(3)
                .set_input(0, "weight", "weight tensor")
                .set_input(1, "bias", "bias tensor")
                .set_input(2, "gamma", "gamma scaling for normalized value")
                .set_input(
                        3, "beta", "beta added to the scaled normalized value")
                .set_input(4, "mean", "value for mean normalization")
                .set_input(5, "variance", "value for variance normalization")
                .set_output(0, "updated_weight", "updated weight tensor")
                .set_output(1, "updated_bias", "updated bias tensor")
                .set_output(2, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // No corresponding frontend op
                // Attributes
                .set_attr("epsilon",
                        "the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr("with_bias", "specifying if the op has a bias input",
                        false, attribute_kind::b, false)
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("filter_format",
                        "the format of weight, the options are OIX, XIO", false,
                        attribute_kind::s, "XIO")
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                .set_shape_inference_function(infer_bn_folding_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_conv_bwd_data, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from ConvolutionBackpropData.
                .set_attr("output_padding",
                        "additional amount of paddings to be added to each "
                        "spatial axis in the output tensor",
                        false, attribute_kind::is,
                        std::vector<int64_t>(0, DNNL_GRAPH_MAX_NDIMS))
                .set_attr("output_shape", "describing output shape", false,
                        attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_conv_bwd_data_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_conv_bwd_weights, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_input(1, "output_delta",
                        "gradients tensor with respect to the output of the "
                        "convolution")
                .set_output(0, "weight_delta",
                        "gradient tensor with respect to the weight of the "
                        "convolution")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                .set_attr("filter_shape", "describing filter shape", false,
                        attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_conv_bwd_weight_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_batchnorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 6}))
                .set_input(0, "input", "input tensor")
                .set_input(1, "gamma", "gamma scaling for normalized value")
                .set_input(
                        2, "beta", "beta added to the scaled normalized value")
                .set_input(3, "mean", "value for mean normalization")
                .set_input(4, "variance", "value for variance normalization")
                .set_output(0, "output", "output tensor")
                .set_output(1, "running mean", "the computed running mean")
                .set_output(
                        2, "running variance", "the computed running variance")
                .set_output(3, "batch mean", "the computed batch mean")
                .set_output(4, "batch variance", "the computed batch variance")
                .set_output(5, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops. when output number is "
                        "2, this tensor will be the second one")
                // Attributes inherited from BatchNormInference and
                // BatchNormForwardTraining op
                .set_attr("epsilon",
                        "the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr("momentum",
                        "used for the computation of running_mean and "
                        "running_var",
                        false, attribute_kind::f)
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("is_training", "whether this is for training", false,
                        attribute_kind::b)
                .set_attr("fuse_relu", "whether to fuse relu (training only)",
                        false, attribute_kind::b)
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_batchnorm_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_batchnorm_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3, 4}))
                .set_input(0, "input", "input tensor")
                .set_input(1, "output_delta", "the gradient w.r.t. the output")
                .set_input(2, "gamma", "gamma scaling for normalized value")
                .set_input(3, "mean",
                        "if is_training is true, pass batch mean, otherwise "
                        "running mean")
                .set_input(4, "variance",
                        "if is_training is true, pass batch variance, "
                        "otherwise running variance")
                .set_output(0, "input_delta",
                        "the gradient w.r.t the output of the batch "
                        "normalization")
                .set_output(1, "gamma_delta",
                        "the gradient w.r.t the gamma of the batch "
                        "normalization")
                .set_output(2, "beta_delta",
                        "the gradient w.r.t the beta of the batch "
                        "normalization")
                .set_output(3, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops. when output number is "
                        "2, this tensor will be the second one")
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_shape_inference_function(infer_bn_bwd_output_shape))

// This op schema represents all interpolate related fusions.
// At the moment available are:
// interpolate + binary add,
// interpolate + sum,
// interpolate + eltwise.
// Thanks to the unification of the op schema for these patterns,
// we can reduce the size of the binary.
DNNL_GRAPH_OP_SCHEMA(interpolate_post_ops_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(
                        0, "data", "Input tensor with data for interpolation")
                .set_input(
                        1, "sizes", "describing output shape for spatial axes")
                .set_output(0, "output",
                        "a tensor with selected data from input tensor")
                .set_attr("mode", "specifies type of interpolation", true,
                        attribute_kind::s)
                .set_attr("sizes", "describing output shape for spatial axes",
                        true, attribute_kind::is)
                .set_attr("scales", "describing scales for spatial axes", false,
                        attribute_kind::fs)
                .set_attr("coordinate_transformation_mode",
                        "specifies how to transform the coordinate in the "
                        "resized tensor to the coordinate in the original "
                        "tensor",
                        false, attribute_kind::s, "half_pixel")
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_shape_inference_function(infer_interpolate_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_resampling_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(2)
                .set_input(
                        0, "data", "Input tensor with data for interpolation")
                .set_input(1, "output_delta",
                        "the gradient with respect to the output")
                .set_input(2, "sizes",
                        "(optional) tensor describing output shape for spatial "
                        "axes")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to the input of "
                        "interpolate")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                .set_attr("mode", "specifies type of interpolation", true,
                        attribute_kind::s)
                .set_attr("coordinate_transformation_mode",
                        "specifies how to transform the coordinate in the "
                        "resized tensor to the coordinate in the original "
                        "tensor",
                        false, attribute_kind::s, "half_pixel")
                .set_attr("sizes", "describing output shape for spatial axes",
                        false, attribute_kind::is)
                .set_attr("scales", "describing scales for spatial axes", false,
                        attribute_kind::fs)
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_sum, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_binary, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from front binary ops (Add, Multiply,
                // ...).
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                // Attributes inherited from front BiasAdd ops, will only take
                // effect when is_bias_add attr is true
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                // New added attributes
                .set_attr("is_bias_add",
                        "additional flag to indicate whether the op is lowered "
                        "from a BiasAdd op",
                        false, attribute_kind::b, false)
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("alg_kind",
                        "specifies algorithm kind, can be one of "
                        "add/sub/mul/div/min/max",
                        true, attribute_kind::i)
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(reorder_sum, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "src0", "input tensor")
                .set_input(1, "src1", "input tensor")
                .set_output(0, "dst", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(int8_reorder, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "src", "input tensor")
                .set_output(0, "dst", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_eltwise, 1,
        op_schema_t()
                // dnnl_eltwise can fuse dnnl_binary, so its input number is
                // variadic
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from front eltwise ops
                .set_attr("alpha",
                        "alpha, whose meaning is depended on the alg_kind",
                        false, attribute_kind::f, 0.f)
                .set_attr("beta",
                        "beta, whose meaning is depended on the alg_kind",
                        false, attribute_kind::f, 0.f)
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("alg_kind",
                        "specifies algorithm kind, can be one of "
                        "relu/tanh/sigmoid/elu/gelu/...",
                        true, attribute_kind::i)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_eltwise_bwd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(0, "forward_data",
                        "either input or output of forward version of eltwise "
                        "op")
                .set_input(1, "output_delta",
                        "the gradient with respect to the output")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to the input of "
                        "eltwise op")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                .set_attr("alpha",
                        "alpha, whose meaning is depended on the alg_kind",
                        false, attribute_kind::f, 0.f)
                .set_attr("beta",
                        "beta, whose meaning is depended on the alg_kind",
                        false, attribute_kind::f, 0.f)
                .set_attr("use_dst",
                        "if true, use dst to calculate gradient; else use src",
                        false, attribute_kind::b, false)
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("alg_kind",
                        "specifies algorithm kind, can be one of "
                        "relu/tanh/sigmoid/elu/gelu/...; algorithm version "
                        "depends on use_dst value",
                        true, attribute_kind::i)
                .set_attr("fwd_alg_kind",
                        "specifies algorithm kind of fwd op (differs from "
                        "alg_kind if use_dst flag equals true), can be one of "
                        "relu/tanh/sigmoid/elu/gelu/...",
                        true, attribute_kind::i)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_shuffle, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // No corresponding frontend op
                // Attributes
                .set_attr("axis",
                        "specifies the index of a dimension along which "
                        "shuffle is done",
                        true, attribute_kind::i)
                .set_attr("group",
                        "specifies the number of groups to split shuffle "
                        "dimension into",
                        true, attribute_kind::i)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_reduction, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_input(1, "axes",
                        "(optional) 1D tensor, specifies indices of input "
                        "data, along which the reduction is performed.")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from front reduction ops
                .SET_REDUCE_COMMON_ATTRS
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("alg_kind",
                        "specifies algorithm kind, can be one of "
                        "l1/l2/max/mean/min/prod/sum",
                        true, attribute_kind::i)
                .set_attr("p", "the p arg for Lp reduction", false,
                        attribute_kind::f, 0.0f)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_reduce_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_softmax_bwd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(
                        0, "output_delta", "gradients tensor w.r.t. the output")
                .set_input(1, "forward_result", "result of forward")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of SoftMax")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from SoftMaxBackprop
                .set_attr("axis",
                        "the axis of which the SoftMaxBackprop is calculated",
                        false, attribute_kind::i, (int64_t)1)
                // New added attributes
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_logsoftmax_bwd, 1,
        op_schema_t()
                .set_num_inputs(2)
                .set_num_outputs(2)
                .set_input(
                        0, "output_delta", "gradients tensor w.r.t. the output")
                .set_input(1, "forward_result", "result of forward")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of LogSoftmax")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from LogSoftmaxBackprop
                .set_attr("axis",
                        "the axis of which the LogSoftmaxBackprop is "
                        "calculated",
                        false, attribute_kind::i, (int64_t)-1)
                // New added attributes
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

// Represents all currently available reduction related fusions.
// Base-OP possibilites:
// [ReduceL1|ReduceL2|ReduceMax|ReduceMean|ReduceMin|ReduceProd|ReduceSum]
// Post-OP possibilites:
// [Abs, Clamp, Elu, Exp, GELU, Hardswish, Log, Sigmoid, SoftPlus, Pow, ReLU,
// Round, Sqrt, Square, Sigmoid+Multiply, Tanh, Add, Multiply, Maximum, Minimum,
// Divide, Subtract]
DNNL_GRAPH_OP_SCHEMA(reduction_post_ops_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "axes",
                        "(optional) 1D tensor, specifies indices of input "
                        "data, along which the reduction is performed.")
                .set_input(2, "other", "(optional) src1 tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("alg_kind",
                        "specifies algorithm kind, can be one of "
                        "l1/l2/max/mean/min/prod/sum",
                        true, attribute_kind::i)
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_REDUCE_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(dnnl_resampling, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(2)
                .set_input(
                        0, "data", "Input tensor with data for interpolation")
                .set_input(1, "sizes",
                        "optional non-differentiable tensor, describing output"
                        " shape for spatial axes")
                .set_output(0, "output",
                        "a tensor with selected data from input tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from Interpolate.
                .set_attr("mode", "specifies type of interpolation", true,
                        attribute_kind::s)
                .set_attr("sizes", "describing output shape for spatial axes",
                        false, attribute_kind::is)
                .set_attr("scales", "describing scales for spatial axes", false,
                        attribute_kind::fs)
                .set_attr("coordinate_transformation_mode",
                        "specifies how to transform the coordinate in the "
                        "resized tensor to the coordinate in the original "
                        "tensor",
                        false, attribute_kind::s, "half_pixel")
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_interpolate_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_concat, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 64}))
                .set_num_outputs(2)
                .set_input(0, "a", "first input tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from Concat
                .set_attr("axis",
                        "specifies which dimension to concatenate along", true,
                        attribute_kind::i)
                // New added attributes
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_concat_output_shape))

DNNL_GRAPH_OP_SCHEMA(quantized_concat_fusion, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 64}))
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("axis",
                        "specifies which dimension to concatenate along", true,
                        attribute_kind::i)
                .set_shape_inference_function(infer_concat_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_layernorm_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 6}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 2, 3, 4}))
                .set_input(0, "input_forward", "input tensor")
                .set_input(1, "output_delta",
                        "the gradient with respect to the output")
                .set_input(2, "mean", "mean of input")
                .set_input(3, "variance", "variance of input")
                .set_input(4, "gamma",
                        "(optional) gamma scaling for normalized value")
                .set_input(5, "beta",
                        "(optional) bias added to the scaled normalized value")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to the output of the "
                        "layer normalization")
                .set_output(1, "gamma_delta",
                        "(optional) the gradient tensor with respect to the "
                        "gamma of the layer normalization")
                .set_output(2, "beta_delta",
                        "(optional) the gradient tensor with respect to the "
                        "beta of the layer normalization")
                .set_output(3, "scratchpad",
                        "(optional) scratchpad tensor, which is a temporary "
                        "output and not connected to any other ops")
                .set_attr("with_gamma",
                        "when set to True, this module has learnable weights",
                        false, attribute_kind::b, true)
                .set_attr("with_beta",
                        "when set to True, this module has learnable bias",
                        false, attribute_kind::b, true)
                .set_attr("epsilon", "constant to improve numerical stability",
                        false, attribute_kind::f, 1e-5f)
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                .set_shape_inference_function(infer_norm_bprop_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_matmul, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({2, 32}))
                .set_num_outputs(2)
                .set_input(0, "src0", "input tensor")
                .set_input(1, "src1", "filter tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from MatMul.
                .SET_MATMUL_COMMON_ATTRS
                // New added attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps,, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("with_bias", "specifying if the op has a bias input",
                        false, attribute_kind::b, false)
                .set_attr("canonicalized",
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_matmul_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_softmax, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from SoftMax
                .set_attr("axis", "the axis of which the SoftMax is calculated",
                        false, attribute_kind::i, (int64_t)1)
                // New added attributes
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_logsoftmax, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(2)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from LogSoftmax
                .set_attr("axis", "the axis of which the SoftMax is calculated",
                        false, attribute_kind::i, (int64_t)1)
                // New added attributes
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_layernorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 4}))
                .set_input(0, "input", "input tensor")
                .set_input(1, "gamma",
                        "(optional) gamma scaling for normalized value")
                .set_input(2, "beta",
                        "(optional) bias added to the scaled normalized value")
                .set_output(0, "output", "output tensor")
                .set_output(1, "mean",
                        "(optional) the mean calculated along the given axis")
                .set_output(2, "variance",
                        "(optional) the std calculated along the given axis")
                .set_output(3, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // Attributes inherited from LayerNorm
                .set_attr("keep_stats",
                        "used to indicate whether to output mean and variance",
                        false, attribute_kind::b, true)
                .set_attr("begin_norm_axis",
                        "used to indicate which axis to perform layer "
                        "normalization",
                        false, attribute_kind::i, int64_t(-1))
                .set_attr("use_affine",
                        "when set to True, this module has learnable "
                        "per-element affine parameters",
                        false, attribute_kind::b, true)
                .set_attr("epsilon", "constant to improve numerical stability",
                        false, attribute_kind::f, 1e-5f)
                // New added attributes
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_norm_output_shape))

DNNL_GRAPH_OP_SCHEMA(dnnl_reorder, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(1) // No scratchpad
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                // TODO(xxx) Multiple ops will be mapped to dnnl_reorder
                // finally, how to deal with the attrs?
                .set_attr("qtype",
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                // Attributes
                .set_attr("primitive_attr_key", // TODO(qun) use fusion_info
                        "fusion information (such as zps,, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr("change_layout",
                        "if the attr is true, we can't do layout prop. so only "
                        "those ops inserted during layout prop change layout",
                        false, attribute_kind::b, false)
                .set_attr("scales", "the output scales", false,
                        attribute_kind::fs)
                .set_attr("src_zps", "the src zero points", false,
                        attribute_kind::is)
                .set_attr("dst_zps", "the src zero points", false,
                        attribute_kind::is)
                .set_attr("with_runtime_scales",
                        "indicate whether the op has runtime scales input",
                        false, attribute_kind::b, false)
                .set_attr("with_runtime_src_zps",
                        "indicate whether the op has runtime src zps input",
                        false, attribute_kind::b, false)
                .set_attr("with_runtime_dst_zps",
                        "indicate whether the op has runtime dst zps input",
                        false, attribute_kind::b, false)
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "scaling",
                        false, attribute_kind::i, int64_t(-1))
                .set_attr("is_constant",
                        "used in constant propagation to identify if the "
                        "output of this op is constant",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape))

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
