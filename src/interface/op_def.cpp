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

#include "op_schema.hpp"
#include "shape_infer.hpp"

namespace dnnl {
namespace graph {
namespace impl {

DNNL_GRAPH_OP_SCHEMA(Abs, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Add, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Add_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Add_sigmoid, 1,
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

DNNL_GRAPH_OP_SCHEMA(AvgPool, 1,
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

DNNL_GRAPH_OP_SCHEMA(AvgPoolBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input_shape", "the dimensions of original input")
                .set_input(1, "output_delta",
                        "the gradient tensor with respect to output of avg "
                        "pool")
                .set_output(0, "input_delta",
                        "the the gradient tensor w.r.t. the input of avg pool")
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
                .set_attr("auto_pad", "how the padding is calculated", false,
                        attribute_kind::s, "None")
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(BatchNormInference, 1,
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

DNNL_GRAPH_OP_SCHEMA(BatchNormForwardTraining, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4, 5}))
                .set_num_outputs(5)
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
                .set_shape_inference_function(infer_bn_fwd_train_output_shape))

DNNL_GRAPH_OP_SCHEMA(BatchNormTrainingBackprop, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5, 6}))
                .set_outputs_option(op_schema::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 2, 3}))
                .set_input(0, "input", "input tensor")
                .set_input(1, "output_delta", "the gradient w.r.t. the output")
                .set_input(2, "gamma", "gamma scaling for normalized value")
                .set_input(
                        3, "beta", "beta added to the scaled normalized value")
                .set_input(4, "mean",
                        "if is_training is true, pass batch mean, otherwise "
                        "running mean")
                .set_input(5, "variance",
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
                .set_attr("epsilon",
                        " the number to be added to the variance to avoid "
                        "division by zero",
                        true, attribute_kind::f)
                .set_attr("is_training",
                        "used to indicate whether the operation is for "
                        "training",
                        false, attribute_kind::b, true)
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_shape_inference_function(infer_bn_bwd_output_shape))

DNNL_GRAPH_OP_SCHEMA(BiasAdd, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "data tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(1, "bias", "1D tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_output(0, "output", "sum of input and bias",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_attr("data_format",
                        "the data format of input / output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_shape_inference_function(infer_bias_add_output_shape))

DNNL_GRAPH_OP_SCHEMA(BiasAddBackprop, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(
                        0, "output_delta", "gradient tensor w.r.t. the output")
                .set_output(0, "bias_delta", "gradient tensor w.r.t. bias")
                .set_attr("data_format",
                        "the data format of input, the options are NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .set_shape_inference_function(infer_bias_backprop_output_shape))

DNNL_GRAPH_OP_SCHEMA(Clamp, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("min", "lower bound of values in the output", true,
                        attribute_kind::f)
                .set_attr("max", "upper bound of values in the output", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(ClampBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(
                        0, "output_delta", "gradients tensor w.r.t. the output")
                .set_input(1, "input_forward", "input of forward")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of Clamp.")
                .set_attr("min", "lower bound of values in the output", true,
                        attribute_kind::f)
                .set_attr("max", "upper bound of values in the output", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Concat, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>(
                        {1, std::numeric_limits<size_t>::max()}))
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("axis",
                        "specifies which dimension to concatenate along", true,
                        attribute_kind::i)
                .set_shape_inference_function(infer_concat_output_shape))

DNNL_GRAPH_OP_SCHEMA(Convolution, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(1, "filter", "filter tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(2, "bias", "bias tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_output(0, "output", "output tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvolutionBackpropData, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "output_shape",
                        "tensor, that specifies spatial shape of "
                        "the output")
                .set_output(0, "output", "output tensor")
                .set_attr("output_padding",
                        "additional amount of paddings to be added to each "
                        "spatial axis in the output tensor",
                        false, attribute_kind::is,
                        std::vector<int64_t>(0, DNNL_GRAPH_MAX_NDIMS))
                .set_shape_inference_function(
                        infer_conv_bprop_data_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(ConvolutionBackpropFilters, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "output_delta",
                        "gradients tensor with respect to the output of the "
                        "convolution")
                .set_output(0, "weight_delta",
                        "gradient tensor with respect to the weight of the "
                        "convolution")
                .set_shape_inference_function(
                        infer_conv_bprop_filters_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Divide, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Elu, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("alpha", "scale for the negative factor", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(EluBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "result_forward", "result of forward")
                .set_input(
                        1, "output_delta", "gradient tensor w.r.t. the output")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of Elu")
                .set_attr("alpha", "scale for the negative factor", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(End, 1,
        op_schema().set_num_inputs(1).set_num_outputs(0).set_input(
                0, "input", "input tensor"))

DNNL_GRAPH_OP_SCHEMA(Erf, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Exp, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GELU, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(GELUBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input_forward", "input of forward")
                .set_input(
                        1, "output_delta", "gradient tensor w.r.t. the output")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of GELU")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardTanh, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("min", "lower bound of values in the output", true,
                        attribute_kind::f)
                .set_attr("max", "upper bound of values in the output", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(HardTanhBackprop, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(
                        0, "output_delta", "gradient tensor w.r.t. the output")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of HardTanh")
                .set_attr("min", "lower bound of values in the output", true,
                        attribute_kind::f)
                .set_attr("max", "upper bound of values in the output", true,
                        attribute_kind::f)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Index, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "indices", "indices tensor")
                .set_output(0, "output",
                        "a tensor with selected data from input tensor")
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(Interpolate, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({3, 4}))
                .set_num_outputs(1)
                .set_input(
                        0, "data", "Input tensor with data for interpolation")
                .set_input(1, "sizes",
                        "1D tensor describing output shape for spatial axes")
                .set_input(2, "scales",
                        "1D tensor describing scales for spatial axes")
                .set_input(3, "axes",
                        "1D tensor specifying dimension indices where "
                        "interpolation is applied")
                .set_output(0, "output",
                        "a tensor with selected data from input tensor")
                .set_attr("mode", "specifies type of interpolation", true,
                        attribute_kind::s)
                .set_attr("shape_calculation_mode",
                        "specifies which input, sizes or scales, is used to "
                        "calculate an output shape",
                        true, attribute_kind::s)
                .set_attr("coordinate_transformation_mode",
                        "specifies how to transform the coordinate in the "
                        "resized tensor to the coordinate in the original "
                        "tensor",
                        false, attribute_kind::s, "half_pixel")
                .set_attr("nearest_mode",
                        "specifies round mode when mode == nearest and is used "
                        "only when mode == nearest.",
                        false, attribute_kind::s, "round_prefer_floor")
                .set_attr("antialias",
                        "antialias is a flag that specifies whether to perform "
                        "anti-aliasing.",
                        false, attribute_kind::b, false)
                .set_attr("pads_begin", "top and left padding", false,
                        attribute_kind::is,
                        std::vector<int64_t>(0, DNNL_GRAPH_MAX_NDIMS))
                .set_attr("pads_end", "bottom and right padding", false,
                        attribute_kind::is,
                        std::vector<int64_t>(0, DNNL_GRAPH_MAX_NDIMS))
                .set_attr("cube_coeff",
                        "specifies the parameter a for cubic interpolation",
                        false, attribute_kind::f, -0.75f)
                //todo(jihui):need to set real infer function
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(InterpolateBackprop, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({4, 5}))
                .set_num_outputs(std::set<size_t>({1, 2}))
                .set_input(
                        0, "data", "Input tensor with data for interpolation")
                .set_input(1, "output_delta",
                        "the gradient with respect to the output")
                .set_input(2, "sizes",
                        "1D tensor describing output shape for spatial axes")
                .set_input(3, "scales",
                        "1D tensor describing scales for spatial axes")
                .set_input(4, "axes",
                        "1D tensor specifying dimension indices where "
                        "interpolation is applied")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of Interpolate")
                .set_output(1, "scales_delta",
                        "the gradient tensor w.r.t. the input scales of "
                        "Interpolate. Required only when "
                        "shape_calculation_mode is scales.")
                .set_attr("mode", "specifies type of interpolation", true,
                        attribute_kind::s)
                .set_attr("shape_calculation_mode",
                        "specifies which input, sizes or scales, is used to "
                        "calculate an output shape",
                        true, attribute_kind::s)
                .set_attr("coordinate_transformation_mode",
                        "specifies how to transform the coordinate in the "
                        "resized tensor to the coordinate in the original "
                        "tensor",
                        false, attribute_kind::s, "half_pixel")
                .set_attr("nearest_mode",
                        "specifies round mode when mode == nearest and is used "
                        "only when mode == nearest.",
                        false, attribute_kind::s, "round_prefer_floor")
                .set_attr("antialias",
                        "antialias is a flag that specifies whether to perform "
                        "anti-aliasing.",
                        false, attribute_kind::b, false)
                .set_attr("pads_begin", "top and left padding", false,
                        attribute_kind::is,
                        std::vector<int64_t>(0, DNNL_GRAPH_MAX_NDIMS))
                .set_attr("pads_end", "bottom and right padding", false,
                        attribute_kind::is,
                        std::vector<int64_t>(0, DNNL_GRAPH_MAX_NDIMS))
                .set_attr("cube_coeff",
                        "specifies the parameter a for cubic interpolation",
                        false, attribute_kind::f, -0.75f)
                //todo(jihui):need to set real infer function
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(LayerNorm, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3}))
                .set_outputs_option(op_schema::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 3}))
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
                .set_shape_inference_function(infer_norm_output_shape))

DNNL_GRAPH_OP_SCHEMA(LayerNormBackprop, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({1, 3, 5}))
                .set_outputs_option(op_schema::param_num_option::optional)
                .set_num_outputs(std::set<size_t>({1, 3}))
                .set_input(0, "input", "input tensor")
                .set_input(1, "gamma",
                        "(optional) gamma scaling for normalized value")
                .set_input(2, "beta",
                        "(optional) bias added to the scaled normalized value")
                .set_input(3, "mean", "(optional) mean of input")
                .set_input(4, "variance", "(optional) variance input")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to the output of the "
                        "layer "
                        "normalization")
                .set_output(1, "gamma_delta",
                        "(optional) the gradient tensor with respect to the "
                        "gamma of "
                        "the layer normalization")
                .set_output(2, "beta_delta",
                        "(optional) the gradient tensor with respect to the "
                        "beta of the "
                        "layer normalization")
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
                .set_attr("use_stats",
                        "indicate whether to use input mean and variance",
                        false, attribute_kind::b, true)
                .set_shape_inference_function(infer_norm_bprop_output_shape))

DNNL_GRAPH_OP_SCHEMA(Log, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogSoftmax, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("axis",
                        "the axis of which the LogSoftmax is calculated", false,
                        attribute_kind::i, int64_t(-1))
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(LogSoftmaxBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(
                        0, "output_delta", "gradients tensor w.r.t. the output")
                .set_input(1, "forward_result", "input of forward")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of LogSoftmax")
                .set_attr("axis",
                        "the axis of which the LogSoftmax is calculated", false,
                        attribute_kind::i, int64_t(-1))
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(MatMul, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Maximum, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Maximum_add, 1,
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

DNNL_GRAPH_OP_SCHEMA(Maximum_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Maximum_sigmoid, 1,
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

DNNL_GRAPH_OP_SCHEMA(MaxPool, 1,
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
                .set_shape_inference_function(infer_pool_output_shape))

DNNL_GRAPH_OP_SCHEMA(MaxPoolBackprop, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "output_forward_indices",
                        "(optional) indices of max values in output tensor")
                .set_input(2, "output_delta",
                        "the gradient tensor with respect to output")
                .set_output(0, "input_delta",
                        "the gradient tensor with respect to input")
                .set_attr("strides", "the distance to slide the filter", true,
                        attribute_kind::is)
                .set_attr("pads_begin", "top and left padding", true,
                        attribute_kind::is)
                .set_attr("pads_end", "bottom and right padding", true,
                        attribute_kind::is)
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
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Minimum, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting "
                        "of input tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Minimum_add, 1,
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

DNNL_GRAPH_OP_SCHEMA(Minimum_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Minimum_sigmoid, 1,
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

DNNL_GRAPH_OP_SCHEMA(Multiply, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(Multiply_add, 1,
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

DNNL_GRAPH_OP_SCHEMA(Multiply_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Multiply_sigmoid, 1,
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

DNNL_GRAPH_OP_SCHEMA(Pow, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("auto_broadcast",
                        "specifies rules used for auto-broadcasting of input "
                        "tensors",
                        false, attribute_kind::s, "numpy")
                .set_shape_inference_function(
                        infer_elemwise_arithmetic_output_shape))

DNNL_GRAPH_OP_SCHEMA(PowBackprop, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input_forward", "input of forward")
                .set_input(
                        1, "output_delta", "gradient tensor w.r.t. the output")
                .set_input(2, "exponent", "exponent of input")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of pow")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(PowBackpropExponent, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input_forward", "input of forward")
                .set_input(
                        1, "output_delta", "gradient tensor w.r.t. the output")
                .set_input(2, "result_forward", "original output of pow")
                .set_input(3, "exponent", "exponent of input")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of pow")
                .set_shape_inference_function(infer_exponent_output_shape))

DNNL_GRAPH_OP_SCHEMA(ReduceSum, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "axis_indices",
                        "scalar or 1D tensor with axis indices for the 1st "
                        "input, along which reduction is performed")
                .set_output(0, "output", "output tensor")
                .set_attr("keep_dims",
                        "if true, holds axes that are used for reduction.",
                        false, attribute_kind::b, false)
                .set_shape_inference_function(infer_reduce_sum_output_shape))

DNNL_GRAPH_OP_SCHEMA(ReLU, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_output(0, "output", "output tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Relu_add, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(1, "other", "the second input tensor of add",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_output(0, "output", "output tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(ReLUBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(
                        0, "output_delta", "gradient tensor w.r.t. the output")
                .set_input(1, "arg",
                        "either forward input or output tensor of ReLU")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of ReLU")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Reshape, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data", "multidimensional input tensor")
                .set_input(1, "shape", "1D tensor describing output shape")
                .set_output(0, "output",
                        "Output tensor with the same content as a tensor at "
                        "input data but with shape defined by input shape")
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(Round, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Sigmoid, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SigmoidBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input",
                        "either forward input or output tensor of Sigmoid")
                .set_input(
                        1, "output_delta", "gradient tensor w.r.t. the output")
                .set_output(0, "input_delta",
                        "gradient tensor w.r.t. the input of Sigmoid")
                .set_attr("use_dst",
                        "if true, use dst to calculate gradient, else, use src",
                        false, attribute_kind::b, true)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftMax, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("axis", "the axis of which the SoftMax is calculated",
                        false, attribute_kind::i, (int64_t)1)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftMaxBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(
                        0, "output_delta", "gradients tensor w.r.t. the output")
                .set_input(1, "forward_result", "input of forward")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of SoftMax")
                .set_attr("axis", "the axis of which the SoftMax is calculated",
                        false, attribute_kind::i, (int64_t)1)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftPlus, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_attr("beta", "value for the Softplus formulation", false,
                        attribute_kind::i, int64_t(1))
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SoftPlusBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input_forward", "input of forward")
                .set_input(
                        1, "output_delta", "gradients tensor w.r.t. the output")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of SoftPlus")
                .set_attr("beta", "value for the SoftPlus formulation", false,
                        attribute_kind::i, int64_t(1))
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Sqrt, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(SqrtBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(
                        0, "output_delta", "gradients tensor w.r.t. the output")
                .set_input(1, "input",
                        "if use_dst is true, input is result of forward. Else, "
                        "input is src of forward.")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of Sqrt")
                .set_attr("use_dst",
                        "if true, use dst to calculate gradient; else use "
                        "src.",
                        false, attribute_kind::b, true)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Square, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Tanh, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(TanhBackprop, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input of forward")
                .set_input(
                        1, "output_delta", "gradients tensor w.r.t. the output")
                .set_output(0, "input_delta",
                        "the gradient tensor w.r.t. the input of Tanh")
                .set_attr("use_dst",
                        "if true, use dst to calculate gradient; else use src",
                        false, attribute_kind::b, true)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Transpose, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "data", "the tensor to be transposed")
                .set_input(1, "shape",
                        "the permutation to apply to the axes of the input "
                        "shape")
                .set_output(0, "output",
                        "A tensor with shape and type matching 1st tensor.")
                .set_shape_inference_function(infer_unsupported_output_shape))

DNNL_GRAPH_OP_SCHEMA(Wildcard, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::variadic)
                .set_num_inputs(std::set<size_t>(
                        {0, std::numeric_limits<size_t>::max()}))
                .set_outputs_option(op_schema::param_num_option::variadic)
                .set_num_outputs(std::set<size_t>(
                        {0, std::numeric_limits<size_t>::max()}))
                .set_input(0, "input", "input tensor",
                        {data_type::f32, data_type::f16, data_type::bf16,
                                data_type::s8, data_type::u8, data_type::s32,
                                data_type::undef})
                .set_output(0, "output", "output tensor",
                        {data_type::f32, data_type::f16, data_type::bf16,
                                data_type::s8, data_type::u8, data_type::s32,
                                data_type::undef})
                .set_shape_inference_function(infer_unsupported_output_shape))

// fusion ops
DNNL_GRAPH_OP_SCHEMA(Conv_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(1, "weight", "weight tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(2, "bias", "bias tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_output(0, "output", "output tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_bias_abs, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_bias_add, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(1, "weight", "weight tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(2, "bias", "bias tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(3, "other", "the second input tensor of add",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_output(0, "output", "output tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_bias_add_elu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bias_add_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bias_add_relu6, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_add_elu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_add_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_add_relu6, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bias_elu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bias_hardtanh, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_bias_relu6, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bias_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_bias_sqrt, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_bias_square, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_bias_tanh, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor")
                .set_input(1, "weight", "weight tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Conv_bn, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bias_bn, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bn_add, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bn_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bias_bn_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bn_add_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_bias_bn_add_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(Conv_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_input(1, "weight", "weight tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_output(0, "output", "output tensor",
                        {data_type::f32, data_type::bf16, data_type::f16})
                .set_shape_inference_function(infer_conv_output_shape)
                .SET_CONV_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(BatchNorm_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(MatMul_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_add, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_add_gelu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_add_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_bias_add, 1,
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

DNNL_GRAPH_OP_SCHEMA(MatMul_bias_add_relu, 1,
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

DNNL_GRAPH_OP_SCHEMA(MatMul_bias_bn, 1,
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

DNNL_GRAPH_OP_SCHEMA(MatMul_bias_elu, 1,
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

DNNL_GRAPH_OP_SCHEMA(MatMul_bias_hardtanh, 1,
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

DNNL_GRAPH_OP_SCHEMA(MatMul_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_bias_relu6, 1,
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

DNNL_GRAPH_OP_SCHEMA(MatMul_bias_gelu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_bias_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "bias", "bias tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_add_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_input(2, "other", "the second input tensor of add")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_elu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_hardtanh, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(MatMul_gelu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "a", "first input tensor")
                .set_input(1, "b", "second input tensor")
                .set_output(0, "output", "output tensor")
                .set_shape_inference_function(infer_matmul_output_shape)
                .SET_MATMUL_COMMON_ATTRS)

DNNL_GRAPH_OP_SCHEMA(Quantize, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", " fp32 tensor to be quantized",
                        data_type::f32)
                .set_output(0, "output", "quantized tensor",
                        {data_type::s8, data_type::u8})
                .set_attr("qtype", "specifies which quantization type is used",
                        false, attribute_kind::s, "per_tensor")
                .set_attr("axis",
                        "specifies dimension on which apply per-channel "
                        "quantization",
                        false, attribute_kind::i, int64_t(1))
                .set_attr("scales", "apply in quantization formula", true,
                        attribute_kind::fs)
                .set_attr("zps", "offset value that maps to float zero", true,
                        attribute_kind::is)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Dequantize, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "quantized tensor to be dequantized",
                        {data_type::s8, data_type::u8})
                .set_output(0, "output", "dequantized tensor", data_type::f32)
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
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Reorder, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::u8, data_type::s8})
                .set_output(0, "output", "output tensor",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::u8, data_type::s8})
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(INT8_Conv, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_Conv_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_input(2, "bias", "bias tensor", data_type::f32)
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_Conv_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_Conv_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_input(2, "bias", "bias tensor", data_type::f32)
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_Conv_bias_add_relu, 1,
        op_schema()
                .set_num_inputs(4)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_input(2, "bias", "bias tensor", data_type::f32)
                .set_input(3, "other", "add src1 tensor",
                        {data_type::s8, data_type::u8})
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_Conv_add_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_input(2, "other", "add src1 tensor",
                        {data_type::s8, data_type::u8})
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_MatMul, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_MatMul_bias, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_input(2, "bias", "bias tensor", data_type::f32)
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_MatMul_relu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_MatMul_bias_relu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_input(2, "bias", "bias tensor", data_type::f32)
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_MatMul_sigmoid, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_MatMul_bias_sigmoid, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_input(2, "bias", "bias tensor", data_type::f32)
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_MatMul_gelu, 1,
        op_schema()
                .set_num_inputs(2)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(INT8_MatMul_bias_gelu, 1,
        op_schema()
                .set_num_inputs(3)
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::s8, data_type::u8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::s8, data_type::u8})
                .set_input(2, "bias", "bias tensor", data_type::f32)
                .set_output(0, "output", "output tensor",
                        {data_type::s8, data_type::u8})
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

DNNL_GRAPH_OP_SCHEMA(Mul_scales, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {data_type::s8, data_type::u8, data_type::f32})
                .set_output(0, "y", "output tensor", data_type::f32)
                .set_attr("qtype", "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis", "quantization type", false, attribute_kind::i,
                        int64_t(1))
                .set_attr("scales", "input scale", true, attribute_kind::fs)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Add_zps, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {data_type::s8, data_type::u8, data_type::f32})
                .set_output(
                        0, "y", "output tensor", {data_type::s8, data_type::u8})
                .set_attr("qtype", "quantization type", false,
                        attribute_kind::s, "per_tensor")
                .set_attr("axis", "quantization type", false, attribute_kind::i,
                        int64_t(1))
                .set_attr("zps", "input zero_point", true, attribute_kind::is)
                .set_shape_inference_function(infer_identity_output_shape))

DNNL_GRAPH_OP_SCHEMA(Permute, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {data_type::s8, data_type::u8, data_type::f32})
                .set_output(0, "y", "output tensor",
                        {data_type::s8, data_type::u8, data_type::f32})
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

DNNL_GRAPH_OP_SCHEMA(To_group, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {data_type::s8, data_type::u8, data_type::f32})
                .set_output(0, "y", "output tensor",
                        {data_type::s8, data_type::u8, data_type::f32})
                .set_attr("groups", "the groups", false, attribute_kind::i,
                        (int64_t)1)
                .set_shape_inference_function(infer_to_group_output_shape))

DNNL_GRAPH_OP_SCHEMA(Broadcast, 1,
        op_schema()
                .set_num_inputs(1)
                .set_num_outputs(1)
                .set_input(0, "x", "input tensor",
                        {data_type::s8, data_type::u8, data_type::f32})
                .set_output(0, "y", "output tensor",
                        {data_type::s8, data_type::u8, data_type::f32})
                .set_attr("insert_1dim", "where to insert 1 dim", false,
                        attribute_kind::s, "none")
                .set_attr("broadcast_to", "target ndims to broadcast", false,
                        attribute_kind::i, (int64_t)(-1))
                .set_shape_inference_function(infer_broadcast_output_shape))

DNNL_GRAPH_OP_SCHEMA(Dnnl_convolution, 1,
        op_schema()
                .set_inputs_option(op_schema::param_num_option::optional)
                .set_num_inputs(std::set<size_t>({2, 3}))
                .set_num_outputs(1)
                .set_input(0, "input", "input tensor",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::u8, data_type::s8})
                .set_input(1, "filter", "filter tensor",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::s8})
                .set_input(2, "bias", "bias tensor",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::s32})
                .set_output(0, "output", "output tensor",
                        {data_type::f32, data_type::bf16, data_type::f16,
                                data_type::u8, data_type::s8})
                .set_shape_inference_function(infer_dnnl_conv_output_shape)
                .set_attr("output_format",
                        "the data format of output, the options are "
                        "NCX and NXC",
                        false, attribute_kind::s, "NXC")
                .SET_CONV_COMMON_ATTRS)

} // namespace impl
} // namespace graph
} // namespace dnnl
