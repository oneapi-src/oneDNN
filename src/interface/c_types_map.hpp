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

#ifndef INTERFACE_C_TYPES_MAP_HPP
#define INTERFACE_C_TYPES_MAP_HPP

#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_graph_types.h"

namespace dnnl {
namespace graph {
namespace impl {

using dim_t = dnnl_graph_dim_t;
using dims_t = dnnl_graph_dims_t;
using dims = std::vector<dim_t>;

using status_t = dnnl_graph_status_t;
namespace status {
const status_t success = dnnl_graph_success;
const status_t out_of_memory = dnnl_graph_out_of_memory;
const status_t invalid_arguments = dnnl_graph_invalid_arguments;
const status_t unimplemented = dnnl_graph_unimplemented;
const status_t interator_ends = dnnl_graph_iterator_ends;
const status_t runtime_error = dnnl_graph_runtime_error;
const status_t not_required = dnnl_graph_not_required;
const status_t invalid_graph = dnnl_graph_invalid_graph;
const status_t invalid_op = dnnl_graph_invalid_graph_op;
const status_t invalid_shape = dnnl_graph_invalid_shape;
const status_t invalid_data_type = dnnl_graph_invalid_data_type;
} // namespace status

using data_type_t = dnnl_graph_data_type_t;
namespace data_type {
const data_type_t undef = dnnl_graph_data_type_undef;
const data_type_t f16 = dnnl_graph_f16;
const data_type_t bf16 = dnnl_graph_bf16;
const data_type_t f32 = dnnl_graph_f32;
const data_type_t s32 = dnnl_graph_s32;
const data_type_t s8 = dnnl_graph_s8;
const data_type_t u8 = dnnl_graph_u8;
} // namespace data_type

using partition_policy_t = dnnl_graph_partition_policy_t;
namespace partition_policy {
const partition_policy_t max = dnnl_graph_partition_policy_max;
const partition_policy_t fusion = dnnl_graph_partition_policy_fusion;
const partition_policy_t debug = dnnl_graph_partition_policy_debug;
} // namespace partition_policy

using engine_kind_t = dnnl_graph_engine_kind_t;
namespace engine_kind {
const engine_kind_t any_engine = dnnl_graph_any_engine;
const engine_kind_t cpu = dnnl_graph_cpu;
const engine_kind_t gpu = dnnl_graph_gpu;
} // namespace engine_kind

using op_kind_t = dnnl_graph_op_kind_t;
namespace op_kind {
const op_kind_t Abs = dnnl_graph_op_abs;
const op_kind_t Add = dnnl_graph_op_add;
const op_kind_t AvgPool = dnnl_graph_op_avg_pool;
const op_kind_t AvgPoolBackprop = dnnl_graph_op_avg_pool_backprop;
const op_kind_t BatchNormForwardTraining
        = dnnl_graph_op_batch_norm_forward_training;
const op_kind_t BatchNormInference = dnnl_graph_op_batch_norm_inference;
const op_kind_t BatchNormTrainingBackprop = dnnl_graph_op_batch_norm_backprop;
const op_kind_t BiasAdd = dnnl_graph_op_bias_add;
const op_kind_t BiasAddBackprop = dnnl_graph_op_bias_add_backprop;
const op_kind_t Clamp = dnnl_graph_op_clamp;
const op_kind_t ClampBackprop = dnnl_graph_op_clamp_backprop;
const op_kind_t Concat = dnnl_graph_op_concat;
const op_kind_t Convolution = dnnl_graph_op_convolution;
const op_kind_t ConvolutionBackpropData
        = dnnl_graph_op_convolution_backprop_data;
const op_kind_t ConvolutionBackpropFilters
        = dnnl_graph_op_convolution_backprop_filters;
const op_kind_t ConvTranspose = dnnl_graph_op_conv_transpose;
const op_kind_t ConvTransposeBackpropData
        = dnnl_graph_op_conv_transpose_backprop_data;
const op_kind_t ConvTransposeBackpropFilters
        = dnnl_graph_op_conv_transpose_backprop_filters;
const op_kind_t Dequantize = dnnl_graph_op_dequantize;
const op_kind_t Divide = dnnl_graph_op_divide;
const op_kind_t DynamicDequantize = dnnl_graph_op_dynamic_dequantize;
const op_kind_t DynamicQuantize = dnnl_graph_op_dynamic_quantize;
const op_kind_t DynamicReshape = dnnl_graph_op_dynamic_reshape;
const op_kind_t DynamicTranspose = dnnl_graph_op_dynamic_transpose;
const op_kind_t Elu = dnnl_graph_op_elu;
const op_kind_t EluBackprop = dnnl_graph_op_elu_backprop;
const op_kind_t End = dnnl_graph_op_end;
const op_kind_t Erf = dnnl_graph_op_erf;
const op_kind_t Exp = dnnl_graph_op_exp;
const op_kind_t GELU = dnnl_graph_op_gelu;
const op_kind_t GELUBackprop = dnnl_graph_op_gelu_backprop;
const op_kind_t HardSwish = dnnl_graph_op_hard_swish;
const op_kind_t HardSwishBackprop = dnnl_graph_op_hard_swish_backprop;
const op_kind_t HardTanh = dnnl_graph_op_hard_tanh;
const op_kind_t HardTanhBackprop = dnnl_graph_op_hard_tanh_backprop;
const op_kind_t Index = dnnl_graph_op_index;
const op_kind_t Interpolate = dnnl_graph_op_interpolate;
const op_kind_t InterpolateBackprop = dnnl_graph_op_interpolate_backprop;
const op_kind_t LayerNorm = dnnl_graph_op_layer_norm;
const op_kind_t LayerNormBackprop = dnnl_graph_op_layer_norm_backprop;
const op_kind_t Log = dnnl_graph_op_log;
const op_kind_t LogSoftmax = dnnl_graph_op_log_softmax;
const op_kind_t LogSoftmaxBackprop = dnnl_graph_op_log_softmax_backprop;
const op_kind_t MatMul = dnnl_graph_op_matmul;
const op_kind_t Maximum = dnnl_graph_op_maximum;
const op_kind_t MaxPool = dnnl_graph_op_max_pool;
const op_kind_t MaxPoolBackprop = dnnl_graph_op_max_pool_backprop;
const op_kind_t Minimum = dnnl_graph_op_minimum;
const op_kind_t Multiply = dnnl_graph_op_multiply;
const op_kind_t Negative = dnnl_graph_op_negative;
const op_kind_t Pow = dnnl_graph_op_pow;
const op_kind_t PowBackprop = dnnl_graph_op_pow_backprop;
const op_kind_t PowBackpropExponent = dnnl_graph_op_pow_backprop_exponent;
const op_kind_t PReLU = dnnl_graph_op_prelu;
const op_kind_t PReLUBackprop = dnnl_graph_op_prelu_backprop;
const op_kind_t Quantize = dnnl_graph_op_quantize;
const op_kind_t Reciprocal = dnnl_graph_op_reciprocal;
const op_kind_t ReduceL1 = dnnl_graph_op_reduce_l1;
const op_kind_t ReduceL2 = dnnl_graph_op_reduce_l2;
const op_kind_t ReduceMax = dnnl_graph_op_reduce_max;
const op_kind_t ReduceMean = dnnl_graph_op_reduce_mean;
const op_kind_t ReduceMin = dnnl_graph_op_reduce_min;
const op_kind_t ReduceProd = dnnl_graph_op_reduce_prod;
const op_kind_t ReduceSum = dnnl_graph_op_reduce_sum;
const op_kind_t ReLU = dnnl_graph_op_relu;
const op_kind_t ReLUBackprop = dnnl_graph_op_relu_backprop;
const op_kind_t Reorder = dnnl_graph_op_reorder;
const op_kind_t Round = dnnl_graph_op_round;
const op_kind_t Sigmoid = dnnl_graph_op_sigmoid;
const op_kind_t SigmoidBackprop = dnnl_graph_op_sigmoid_backprop;
const op_kind_t Sign = dnnl_graph_op_sign;
const op_kind_t SoftMax = dnnl_graph_op_softmax;
const op_kind_t SoftMaxBackprop = dnnl_graph_op_softmax_backprop;
const op_kind_t SoftPlus = dnnl_graph_op_softplus;
const op_kind_t SoftPlusBackprop = dnnl_graph_op_softplus_backprop;
const op_kind_t Sqrt = dnnl_graph_op_sqrt;
const op_kind_t SqrtBackprop = dnnl_graph_op_sqrt_backprop;
const op_kind_t Square = dnnl_graph_op_square;
const op_kind_t SquaredDifference = dnnl_graph_op_squared_difference;
const op_kind_t StaticReshape = dnnl_graph_op_static_reshape;
const op_kind_t StaticTranspose = dnnl_graph_op_static_transpose;
const op_kind_t Subtract = dnnl_graph_op_subtract;
const op_kind_t Tanh = dnnl_graph_op_tanh;
const op_kind_t TanhBackprop = dnnl_graph_op_tanh_backprop;
const op_kind_t TypeCast = dnnl_graph_op_type_cast;
const op_kind_t Wildcard = dnnl_graph_op_wildcard;
const op_kind_t LastSymbol = dnnl_graph_op_last_symbol;
} // namespace op_kind

using logical_tensor_t = dnnl_graph_logical_tensor_t;

using layout_type_t = dnnl_graph_layout_type_t;
namespace layout_type {
const layout_type_t undef = dnnl_graph_layout_type_undef;
const layout_type_t any = dnnl_graph_layout_type_any;
const layout_type_t strided = dnnl_graph_layout_type_strided;
const layout_type_t opaque = dnnl_graph_layout_type_opaque;
} // namespace layout_type

using property_type_t = dnnl_graph_tensor_property_t;
namespace property_type {
const property_type_t undef = dnnl_graph_tensor_property_undef;
const property_type_t variable = dnnl_graph_tensor_property_variable;
const property_type_t constant = dnnl_graph_tensor_property_constant;
} // namespace property_type

using allocator_lifetime_t = dnnl_graph_allocator_lifetime_t;
namespace allocator_lifetime {
const allocator_lifetime_t persistent = dnnl_graph_allocator_persistent;
const allocator_lifetime_t output = dnnl_graph_allocator_output;
const allocator_lifetime_t temp = dnnl_graph_allocator_temp;
} // namespace allocator_lifetime

using attribute_kind_t = dnnl_graph_attribute_kind_t;
namespace attribute_kind {
const attribute_kind_t f = dnnl_graph_attribute_kind_f;
const attribute_kind_t fs = dnnl_graph_attribute_kind_fs;
const attribute_kind_t i = dnnl_graph_attribute_kind_i;
const attribute_kind_t is = dnnl_graph_attribute_kind_is;
const attribute_kind_t s = dnnl_graph_attribute_kind_s;
const attribute_kind_t b = dnnl_graph_attribute_kind_b;
} // namespace attribute_kind

using allocator_attr_t = dnnl_graph_allocator_attr_t;
using allocator_t = dnnl_graph_allocator;
using cpu_allocate_f = dnnl_graph_cpu_allocate_f;
using cpu_deallocate_f = dnnl_graph_cpu_deallocate_f;
using sycl_allocate_f = dnnl_graph_sycl_allocate_f;
using sycl_deallocate_f = dnnl_graph_sycl_deallocate_f;
using inplace_pair_t = dnnl_graph_inplace_pair_t;

using engine_t = dnnl_graph_engine;
using stream_t = dnnl_graph_stream;
using graph_t = dnnl_graph_graph;
using op_t = dnnl_graph_op;
using partition_t = dnnl_graph_partition;
using compiled_partition_t = dnnl_graph_compiled_partition;
using stream_t = dnnl_graph_stream;
using tensor_t = dnnl_graph_tensor;

} // namespace impl
} // namespace graph
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::graph::impl::op_kind_t> {
    using argument_type = dnnl::graph::impl::op_kind_t;
    using result_type = size_t;

    result_type operator()(const argument_type &x) const {
        using type = typename std::underlying_type<argument_type>::type;
        return std::hash<type>()(static_cast<type>(x));
    }
};
} // namespace std

#endif
