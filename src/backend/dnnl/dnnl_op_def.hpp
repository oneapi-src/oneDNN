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

#include "interface/op_schema.hpp"
#include "interface/shape_infer.hpp"

#include "backend/dnnl/dnnl_shape_infer.hpp"
#include "backend/dnnl/internal_attrs.hpp"
#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/layout_propagator.hpp"
#include "backend/dnnl/op_executable.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

#define SET_ATTR_IS_CONSTANT \
    set_attr(op_attr::is_constant, \
            "used in constant propagation to identify if the output of this " \
            "op is constant", \
            false, attribute_kind::b, false)

#define SET_EXECUTABLE_CREATOR(func) \
    set_additional_item<executable_creator_func>("executable_creator", {func})

#define SET_ARG_INDICES_GETTER(executable_class) \
    set_additional_item<arg_indices_getter_func>( \
            "arg_indices_getter", {executable_class::get_arg_indices})

#define SET_LAYOUT_PROPAGATOR(func) \
    set_additional_item<layout_propagator_func>("layout_propagator", {func})

#define SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS \
    set_attr(op_attr::strides, "the distance to slide the filter", true, \
            attribute_kind::is) \
            .set_attr(op_attr::pads_begin, "top and left padding", true, \
                    attribute_kind::is) \
            .set_attr(op_attr::pads_end, "bottom and right padding", true, \
                    attribute_kind::is) \
            .set_attr(op_attr::dilations, \
                    "the distance in width and height between elements " \
                    "in the filter", \
                    true, attribute_kind::is) \
            .set_attr(op_attr::auto_pad, "how the padding is calculated", \
                    false, attribute_kind::s, "None", \
                    {"None", "SAME_UPPER", "SAME_LOWER", "VALID"}) \
            .set_attr(op_attr::groups, \
                    "the number of groups input / output channels are " \
                    "divided into", \
                    false, attribute_kind::i, (int64_t)1) \
            .set_attr(op_attr::data_format, \
                    "the data format of input / output, the options are " \
                    "NCX and NXC", \
                    false, attribute_kind::s, "NXC", {"NXC", "NCX"}) \
            .set_attr(op_attr::filter_format, \
                    "the format of weight, the options are IOX, XOI and OIX", \
                    false, attribute_kind::s, "XOI", {"XOI", "IOX", "OIX"})

template <typename T>
op_schema_t get_op_schema();

DNNL_GRAPH_OP_SCHEMA(dnnl_mul_scales, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(0, "x", "input tensor")
                .set_input(1, "scales", "scales tensor")
                .set_output(0, "y", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                .set_attr(op_attr::qtype, "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, "quantization type", false,
                        attribute_kind::i, int64_t(1))
                .set_attr(op_attr::scales, "input scale", false,
                        attribute_kind::fs, std::vector<float>())
                .set_attr(op_attr::with_runtime_scales,
                        "indicate whether the op has runtime scales input",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_mul_scales)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<reorder_executable_t>)
                .SET_ARG_INDICES_GETTER(reorder_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_constant_scales, 1,
        op_schema_t()
                .set_num_inputs(0)
                .set_num_outputs(1)
                .set_output(0, "output", "output tensor")
                .set_attr(op_attr::scales,
                        "scales to store in constant storage", true,
                        attribute_kind::fs)
                .set_attr(op_attr::shape, "describing output shape", true,
                        attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_dnnl_constant_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_constant_filler)
                .SET_EXECUTABLE_CREATOR(executable_creator<const_scales_filler>)
                .SET_ARG_INDICES_GETTER(const_scales_filler))

DNNL_GRAPH_OP_SCHEMA(dnnl_add_zps, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_input(1, "zps", "zps tensor")
                .set_output(0, "y", "output tensor")
                .set_attr(op_attr::qtype, "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, "quantization type", false,
                        attribute_kind::i, int64_t(1))
                .set_attr(op_attr::zps, "input zero_point", false,
                        attribute_kind::is, std::vector<int64_t>())
                .set_attr(op_attr::with_runtime_zps,
                        "indicate whether the op has runtime zps input", false,
                        attribute_kind::b, false)
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_add_zps)
                .SET_EXECUTABLE_CREATOR(dummy_executable_creator)
                .set_additional_item<arg_indices_getter_func>(
                        "arg_indices_getter", {dummy_arg_indices_getter}))

DNNL_GRAPH_OP_SCHEMA(dnnl_sub_zps, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 2}))
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_input(1, "zps", "zps tensor")
                .set_output(0, "y", "output tensor")
                .set_attr(op_attr::qtype, "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr(op_attr::axis, "quantization type", false,
                        attribute_kind::i, int64_t(1))
                .set_attr(op_attr::zps, "input zero_point", false,
                        attribute_kind::is, std::vector<int64_t>())
                .set_attr(op_attr::with_runtime_zps,
                        "indicate whether the op has runtime zps input", false,
                        attribute_kind::b, false)
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_sub_zps)
                .SET_EXECUTABLE_CREATOR(dummy_executable_creator)
                .set_additional_item<arg_indices_getter_func>(
                        "arg_indices_getter", {dummy_arg_indices_getter}))

DNNL_GRAPH_OP_SCHEMA(dnnl_constant_zps, 1,
        op_schema_t()
                .set_num_inputs(0)
                .set_num_outputs(1)
                .set_output(0, "output", "output tensor")
                .set_attr(op_attr::zps,
                        "zero points to store in constant storage", true,
                        attribute_kind::is)
                .set_attr(op_attr::shape, "describing output shape", true,
                        attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_dnnl_constant_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_constant_filler)
                .SET_EXECUTABLE_CREATOR(executable_creator<const_zps_filler>)
                .SET_ARG_INDICES_GETTER(const_zps_filler))

// The logical axes will be permuted in the following manner:
// for (i = 0; i < ndims(); i++)
//     new_desc.dims()[permutation[i]] = dims()[i];
//
// Note: the permutation attr in dnnl_permute is quite different from the order
// attr in dnnl_transpose. The later one is inherited from StaticTranspose op
// and are used in the following manner:
// for (i = 0; i < ndims(); i++)
//     new_desc.dims()[i] = dims()[order[i]];
DNNL_GRAPH_OP_SCHEMA(dnnl_permute, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
                .set_attr(op_attr::permutation,
                        "the permutation to apply to the axes of the input "
                        "shape",
                        false, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_permute_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_permute)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_to_group, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
                .set_attr(op_attr::groups, "the groups", false,
                        attribute_kind::i, (int64_t)1)
                .set_attr(op_attr::is_convtranspose,
                        "indicate whether this is for convtranspose", false,
                        attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_to_group_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_to_group)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

// This op is used for grouped conv/deconv backward weight to convert a [g,
// oc/g, ic, kh, kw] shaped weight tensor to a [oc, ic, kh, kw] weight tensor.
// The former shaped weight tensor is required by oneDNN primitive, but the
// later one is required by oneDNN Graph users
DNNL_GRAPH_OP_SCHEMA(dnnl_from_group, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
                .set_attr(op_attr::groups, "the groups", false,
                        attribute_kind::i, (int64_t)1)
                .set_attr(op_attr::is_convtranspose,
                        "indicate whether this is for convtranspose", false,
                        attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_from_group_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_from_group)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_unsqueeze, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
                .set_attr(op_attr::axes,
                        "indices at which to insert the singleton dimension, "
                        "negative value means counting dimensions from the "
                        "back",
                        false, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache

                .set_shape_inference_function(infer_unsqueeze_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_unsqueeze)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_squeeze, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor")
                .set_output(0, "y", "output tensor")
                .set_attr(op_attr::axes,
                        "which dims to be squeezed, negative "
                        "value means counting dimensions from the back",
                        false, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_squeeze_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_squeeze)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_reshape, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "data", "multidimensional input tensor")
                .set_output(0, "output",
                        "Output tensor with the same content as a tensor at "
                        "input data but with shape defined by input shape")
                .set_attr(op_attr::shape, "describing output shape", true,
                        attribute_kind::is)
                .set_attr(op_attr::special_zero,
                        "controls how zero values in shape are interpreted "
                        "shape",
                        true, attribute_kind::b)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_static_reshape_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_reshape)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_transpose, 1,
        op_schema_t()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "data", "the tensor to be transposed")
                .set_output(0, "output",
                        "A tensor with shape and type matching 1st tensor.")
                .set_attr(op_attr::order,
                        "the permutation to apply to the axes of the input "
                        "shape",
                        true, attribute_kind::is)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(
                        infer_static_transpose_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_transpose)
                .SET_EXECUTABLE_CREATOR(executable_creator<memory_reparser_t>)
                .SET_ARG_INDICES_GETTER(memory_reparser_t))

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
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::with_bias,
                        "specifying if the op has a bias input", false,
                        attribute_kind::b, false)
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_conv_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_conv)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<conv_fwd_executable_t>)
                .SET_ARG_INDICES_GETTER(conv_fwd_executable_t))

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
                .set_attr(op_attr::output_padding,
                        "additional amount of paddings to be added to each "
                        "spatial axis in the output tensor",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::with_bias,
                        "specifying if the op has a bias input", false,
                        attribute_kind::b, false)
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_deconv)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<deconv_fwd_executable_t>)
                .SET_ARG_INDICES_GETTER(deconv_fwd_executable_t))

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
                .SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_bwd_data_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_deconv_bwd_data)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<deconv_bwd_data_executable_t>)
                .SET_ARG_INDICES_GETTER(deconv_bwd_data_executable_t))

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
                .set_attr(op_attr::filter_shape, "describing filter shape",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .SET_DNNL_CONVTRANSPOSE_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_convtranspose_bwd_weight_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_deconv_bwd_weights)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<deconv_bwd_weights_executable_t>)
                .SET_ARG_INDICES_GETTER(deconv_bwd_weights_executable_t))

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
                .set_attr(op_attr::strides, "the distance to slide the filter",
                        true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, "top and left padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::pads_end, "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::exclude_pad, "a type of pooling strategy",
                        false, attribute_kind::b)
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
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                .set_attr(op_attr::rounding_type,
                        "a type of rounding to be applied", false,
                        attribute_kind::s, "floor")
                .set_attr(op_attr::auto_pad, "how the padding is calculated",
                        false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                // New added attributes
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::kind, "pooling kind, maxpool or avgpool",
                        true, attribute_kind::s)
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_attr(op_attr::is_training, "whether this is for training",
                        false, attribute_kind::b)
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_pool_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_pool)
                .SET_EXECUTABLE_CREATOR(executable_creator<pool_executable_t>)
                .SET_ARG_INDICES_GETTER(pool_executable_t))

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
                .set_attr(op_attr::strides, "the distance to slide the filter",
                        true, attribute_kind::is)
                .set_attr(op_attr::pads_begin, "top and left padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::pads_end, "bottom and right padding", true,
                        attribute_kind::is)
                .set_attr(op_attr::exclude_pad, "a type of pooling strategy",
                        false, attribute_kind::b)
                .set_attr(op_attr::kernel, "size of each filter", true,
                        attribute_kind::is)
                .set_attr(op_attr::auto_pad, "how the padding is calculated",
                        false, attribute_kind::s, "None",
                        {"None", "SAME_UPPER", "SAME_LOWER", "VALID"})
                .set_attr(op_attr::dilations,
                        "the distance in width and height between elements "
                        "in the filter",
                        false, attribute_kind::is,
                        std::vector<int64_t>(1, DNNL_GRAPH_MAX_NDIMS))
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                .set_attr(op_attr::input_shape, "describing input shape", true,
                        attribute_kind::is)
                // New added attributes
                .set_attr(op_attr::kind, "pooling kind, maxpool or avgpool",
                        true, attribute_kind::s)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_dnnl_pool_bwd_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_pool_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<pool_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(pool_bwd_executable_t))

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
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                .set_attr(op_attr::per_channel_broadcast,
                        "whether to apply per channel broadcast when slope is "
                        "1D tensor",
                        false, attribute_kind::b, true)
                // New added attributes
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_prelu)
                .SET_EXECUTABLE_CREATOR(executable_creator<prelu_executable_t>)
                .SET_ARG_INDICES_GETTER(prelu_executable_t))

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
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_prelu_bwd_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_prelu_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<prelu_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(prelu_bwd_executable_t))

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
                .set_attr(op_attr::epsilon,
                        "the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr(op_attr::with_bias,
                        "specifying if the op has a bias input", false,
                        attribute_kind::b, false)
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                .set_attr(op_attr::filter_format,
                        "the format of weight, the options are OIX, XIO", false,
                        attribute_kind::s, "XIO", {"XIO", "OIX"})
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_bn_folding_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_bn_folding)
                .SET_EXECUTABLE_CREATOR(executable_creator<bn_folding_t>)
                .SET_ARG_INDICES_GETTER(bn_folding_t))

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
                .set_attr(op_attr::output_padding,
                        "additional amount of paddings to be added to each "
                        "spatial axis in the output tensor",
                        false, attribute_kind::is,
                        std::vector<int64_t>(0, DNNL_GRAPH_MAX_NDIMS))
                .set_attr(op_attr::output_shape, "describing output shape",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_conv_bwd_data_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_conv_bwd_data)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<conv_bwd_data_executable_t>)
                .SET_ARG_INDICES_GETTER(conv_bwd_data_executable_t))

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
                .set_attr(op_attr::filter_shape, "describing filter shape",
                        false, attribute_kind::is,
                        std::vector<int64_t>(DNNL_GRAPH_MAX_NDIMS, 0))
                .SET_CONV_COMMON_ATTRS
                // New added attributes
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(
                        infer_dnnl_conv_bwd_weight_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_conv_bwd_weights)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<conv_bwd_weights_executable_t>)
                .SET_ARG_INDICES_GETTER(conv_bwd_weights_executable_t))

// Note: if `is_training` is False, the `gamma` and `beta` are the second and
// third input (required), while `is_training` is True, the `gamma` and `beta`
// are the last two inputs (optional).
DNNL_GRAPH_OP_SCHEMA(dnnl_batchnorm, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3, 6, 7}))
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
                .set_output(6, "workspace", "(Optional) workspace tensor")
                // Attributes inherited from BatchNormInference and
                // BatchNormForwardTraining op
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
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::is_training, "whether this is for training",
                        false, attribute_kind::b)
                .set_attr(op_attr::fuse_relu,
                        "whether to fuse relu (training only)", false,
                        attribute_kind::b)
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_batchnorm_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_batchnorm)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<batchnorm_executable_t>)
                .SET_ARG_INDICES_GETTER(batchnorm_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_batchnorm_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 3, 4}))
                .set_input(0, "input", "input tensor")
                .set_input(1, "output_delta", "the gradient w.r.t. the output")
                .set_input(2, "mean",
                        "if is_training is true, pass batch mean, otherwise "
                        "running mean")
                .set_input(3, "variance",
                        "if is_training is true, pass batch variance, "
                        "otherwise running variance")
                .set_input(4, "gamma", "gamma scaling for normalized value")
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
                .set_attr(op_attr::epsilon,
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(
                        infer_dnnl_batchnorm_bwd_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_batchnorm_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<batchnorm_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(batchnorm_bwd_executable_t))

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
                        "(optional) tensor describing output shape for "
                        "spatial axes")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to the input of "
                        "interpolate")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                .set_attr(op_attr::mode, "specifies type of interpolation",
                        true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::coordinate_transformation_mode,
                        "specifies how to transform the coordinate in the "
                        "resized tensor to the coordinate in the original "
                        "tensor",
                        false, attribute_kind::s, "half_pixel", {"half_pixel"})
                .set_attr(op_attr::sizes,
                        "describing output shape for spatial axes", false,
                        attribute_kind::is)
                .set_attr(op_attr::scales, "describing scales for spatial axes",
                        false, attribute_kind::fs)
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_resampling_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<resampling_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(resampling_bwd_executable_t))

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
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_sum)
                .SET_EXECUTABLE_CREATOR(executable_creator<sum_executable_t>)
                .SET_ARG_INDICES_GETTER(sum_executable_t))

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
                .set_attr(op_attr::auto_broadcast,
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy", {"none", "numpy"})
                // Attributes inherited from front BiasAdd ops, will only take
                // effect when is_bias_add attr is true
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::is_bias_add,
                        "additional flag to indicate whether the op is lowered "
                        "from a BiasAdd op",
                        false, attribute_kind::b, false)
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::alg_kind,
                        "specifies algorithm kind, can be one of "
                        "add/sub/mul/div/min/max",
                        true, attribute_kind::i)
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_dnnl_binary_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_binary)
                .SET_EXECUTABLE_CREATOR(executable_creator<binary_executable_t>)
                .SET_ARG_INDICES_GETTER(binary_executable_t))

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
                .set_attr(op_attr::alpha,
                        "alpha, whose meaning is depended on the alg_kind",
                        false, attribute_kind::f, 0.f)
                .set_attr(op_attr::beta,
                        "beta, whose meaning is depended on the alg_kind",
                        false, attribute_kind::f, 0.f)
                // New added attributes
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::alg_kind,
                        "specifies algorithm kind, can be one of "
                        "relu/tanh/sigmoid/elu/gelu/...",
                        true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_eltwise)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<eltwise_executable_t>)
                .SET_ARG_INDICES_GETTER(eltwise_executable_t))

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
                .set_attr(op_attr::alpha,
                        "alpha, whose meaning is depended on the alg_kind",
                        false, attribute_kind::f, 0.f)
                .set_attr(op_attr::beta,
                        "beta, whose meaning is depended on the alg_kind",
                        false, attribute_kind::f, 0.f)
                .set_attr(op_attr::use_dst,
                        "if true, use dst to calculate gradient; else use src",
                        false, attribute_kind::b, false)
                // New added attributes
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::alg_kind,
                        "specifies algorithm kind, can be one of "
                        "relu/tanh/sigmoid/elu/gelu/...; algorithm version "
                        "depends on use_dst value",
                        true, attribute_kind::i)
                .set_attr(op_attr::fwd_alg_kind,
                        "specifies algorithm kind of fwd op (differs from "
                        "alg_kind if use_dst flag equals true), can be one of "
                        "relu/tanh/sigmoid/elu/gelu/...",
                        true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_eltwise_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<eltwise_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(eltwise_bwd_executable_t))

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
                .set_attr(op_attr::axis,
                        "specifies the index of a dimension along which "
                        "shuffle is done",
                        true, attribute_kind::i)
                .set_attr(op_attr::groups,
                        "specifies the number of groups to split shuffle "
                        "dimension into",
                        true, attribute_kind::i)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_shuffle)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<shuffle_executable_t>)
                .SET_ARG_INDICES_GETTER(shuffle_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_reduction, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
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
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::alg_kind,
                        "specifies algorithm kind, can be one of "
                        "l1/l2/max/mean/min/prod/sum",
                        true, attribute_kind::i)
                .set_attr(op_attr::p, "the p arg for Lp reduction", false,
                        attribute_kind::f, 0.0f)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_reduce_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_reduction)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<reduction_executable_t>)
                .SET_ARG_INDICES_GETTER(reduction_executable_t))

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
                .set_attr(op_attr::axis,
                        "the axis of which the SoftMaxBackprop is calculated",
                        false, attribute_kind::i, (int64_t)1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_softmax_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<softmax_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(softmax_bwd_executable_t))

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
                .set_attr(op_attr::axis,
                        "the axis of which the LogSoftmaxBackprop is "
                        "calculated",
                        false, attribute_kind::i, (int64_t)-1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_softmax_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<softmax_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(softmax_bwd_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_resampling, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>({1, 32}))
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
                .set_attr(op_attr::mode, "specifies type of interpolation",
                        true, attribute_kind::s,
                        {"nearest", "linear", "bilinear", "trilinear"})
                .set_attr(op_attr::sizes,
                        "describing output shape for spatial axes", false,
                        attribute_kind::is)
                .set_attr(op_attr::scales, "describing scales for spatial axes",
                        false, attribute_kind::fs)
                .set_attr(op_attr::coordinate_transformation_mode,
                        "specifies how to transform the coordinate in the "
                        "resized tensor to the coordinate in the original "
                        "tensor",
                        false, attribute_kind::s, "half_pixel", {"half_pixel"})
                .set_attr(op_attr::data_format,
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC", {"NXC", "NCX"})
                // New added attributes
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_interpolate_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_resampling)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<resampling_executable_t>)
                .SET_ARG_INDICES_GETTER(resampling_executable_t))

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
                .set_attr(op_attr::axis,
                        "specifies which dimension to concatenate along", true,
                        attribute_kind::i)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_concat_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_concat)
                .SET_EXECUTABLE_CREATOR(executable_creator<concat_executable_t>)
                .SET_ARG_INDICES_GETTER(concat_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_layernorm_bwd, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5, 6}))
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({2, 4}))
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
                        "scratchpad tensor, which is a temporary "
                        "output and not connected to any other ops")
                .set_attr(op_attr::use_affine,
                        "when set to True, this module has learnable weights",
                        false, attribute_kind::b, true)
                .set_attr(op_attr::begin_norm_axis,
                        "used to indicate which axis to perform layer "
                        "normalization",
                        false, attribute_kind::i, int64_t(-1))
                .set_attr(op_attr::epsilon,
                        "constant to improve numerical stability", false,
                        attribute_kind::f, 1e-5f)
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_shape_inference_function(infer_norm_bprop_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_layernorm_bwd)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<layernorm_bwd_executable_t>)
                .SET_ARG_INDICES_GETTER(layernorm_bwd_executable_t))

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
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps,, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::with_bias,
                        "specifying if the op has a bias input", false,
                        attribute_kind::b, false)
                .set_attr(op_attr::canonicalized,
                        "additional flag to indicate whether the op can be "
                        "directly mapped to DNNL primitive",
                        false, attribute_kind::b, false)
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_attr(op_attr::keep_dst_layout,
                        "if true, defined dst layout will be used to create "
                        "primitive instead of any",
                        false, attribute_kind::b, false)
                // Analysis rules
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_matmul)
                .SET_EXECUTABLE_CREATOR(executable_creator<matmul_executable_t>)
                .SET_ARG_INDICES_GETTER(matmul_executable_t))

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
                .set_attr(op_attr::axis,
                        "the axis of which the SoftMax is calculated", false,
                        attribute_kind::i, (int64_t)1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_softmax)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<softmax_executable_t>)
                .SET_ARG_INDICES_GETTER(softmax_executable_t))

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
                .set_attr(op_attr::axis,
                        "the axis of which the SoftMax is calculated", false,
                        attribute_kind::i, (int64_t)1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_softmax)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<softmax_executable_t>)
                .SET_ARG_INDICES_GETTER(softmax_executable_t))

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
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                // New added attributes
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_norm_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_layernorm)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<layernorm_executable_t>)
                .SET_ARG_INDICES_GETTER(layernorm_executable_t))

DNNL_GRAPH_OP_SCHEMA(dnnl_reorder, 1,
        op_schema_t()
                .set_inputs_option(op_schema_t::param_num_option::variadic)
                .set_outputs_option(op_schema_t::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 32}))
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_output(1, "scratchpad",
                        "scratchpad tensor, which is a temporary output and "
                        "not connected to any other ops")
                // TODO(xxx) Multiple ops will be mapped to dnnl_reorder
                // finally, how to deal with the attrs?
                .set_attr(op_attr::qtype,
                        "specifies which dequantization type is used", false,
                        attribute_kind::s, "per_tensor")
                // Attributes
                .set_attr(op_attr::fusion_info_key,
                        "fusion information (such as zps,, post-ops, ...) "
                        "generated by fusion passes.",
                        false, attribute_kind::i, (int64_t)-1)
                .set_attr(op_attr::change_layout,
                        "if the attr is true, we can't do layout prop. so only "
                        "those ops inserted during layout prop change layout",
                        false, attribute_kind::b, false)
                .set_attr(op_attr::scales, "the output scales", false,
                        attribute_kind::fs)
                .set_attr(op_attr::src_zps, "the src zero points", false,
                        attribute_kind::is)
                .set_attr(op_attr::dst_zps, "the src zero points", false,
                        attribute_kind::is)
                .set_attr(op_attr::with_runtime_scales,
                        "indicate whether the op has runtime scales input",
                        false, attribute_kind::b, false)
                .set_attr(op_attr::with_runtime_src_zps,
                        "indicate whether the op has runtime src zps input",
                        false, attribute_kind::b, false)
                .set_attr(op_attr::with_runtime_dst_zps,
                        "indicate whether the op has runtime dst zps input",
                        false, attribute_kind::b, false)
                .set_attr(op_attr::axis,
                        "specifies dimension on which apply per-channel "
                        "scaling",
                        false, attribute_kind::i, int64_t(-1))
                .SET_ATTR_IS_CONSTANT // used for constant prop and cache
                // Analysis rules
                .set_shape_inference_function(infer_identity_output_shape)
                .SET_LAYOUT_PROPAGATOR(layout_propagator_for_reorder)
                .SET_EXECUTABLE_CREATOR(
                        executable_creator<reorder_executable_t>)
                .SET_ARG_INDICES_GETTER(reorder_executable_t))

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
