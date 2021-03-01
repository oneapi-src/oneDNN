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

using status_t = dnnl_graph_result_t;
namespace status {
const status_t success = dnnl_graph_result_success;
const status_t not_ready = dnnl_graph_result_not_ready;
const status_t device_not_found = dnnl_graph_result_error_device_not_found;
const status_t unsupported = dnnl_graph_result_error_unsupported;
const status_t invalid_argument = dnnl_graph_result_error_invalid_argument;
const status_t invalid_graph = dnnl_graph_result_error_invalid_graph;
const status_t invalid_shape = dnnl_graph_result_error_invalid_shape;
const status_t invalid_type = dnnl_graph_result_error_invalid_type;
const status_t invalid_op = dnnl_graph_result_error_invalid_op;
const status_t compile_fail = dnnl_graph_result_error_compile_fail;
const status_t miss_ins_outs = dnnl_graph_result_error_miss_ins_outs;
const status_t unknown = dnnl_graph_result_error_unknown;
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
const op_kind_t Abs = kAbs;
const op_kind_t Add = kAdd;
const op_kind_t AvgPool = kAvgPool;
const op_kind_t AvgPoolBackprop = kAvgPoolBackprop;
const op_kind_t BatchNormInference = kBatchNormInference;
const op_kind_t BatchNormForwardTraining = kBatchNormForwardTraining;
const op_kind_t BatchNormTrainingBackprop = kBatchNormTrainingBackprop;
const op_kind_t BiasAddBackprop = kBiasAddBackprop;
const op_kind_t Clamp = kClamp;
const op_kind_t ClampBackprop = kClampBackprop;
const op_kind_t Concat = kConcat;
const op_kind_t Convolution = kConvolution;
const op_kind_t ConvolutionBackpropData = kConvolutionBackpropData;
const op_kind_t ConvolutionBackpropFilters = kConvolutionBackpropFilters;
const op_kind_t Divide = kDivide;
const op_kind_t Elu = kElu;
const op_kind_t EluBackprop = kEluBackprop;
const op_kind_t Erf = kErf;
const op_kind_t Exp = kExp;
const op_kind_t GELU = kGELU;
const op_kind_t GELUBackprop = kGELUBackprop;
const op_kind_t HardTanh = kHardTanh;
const op_kind_t HardTanhBackprop = kHardTanhBackprop;
const op_kind_t LayerNorm = kLayerNorm;
const op_kind_t LayerNormBackprop = kLayerNormBackprop;
const op_kind_t Log = kLog;
const op_kind_t LogSoftmax = kLogSoftmax;
const op_kind_t LogSoftmaxBackprop = kLogSoftmaxBackprop;
const op_kind_t MatMul = kMatMul;
const op_kind_t Maximum = kMaximum;
const op_kind_t MaxPool = kMaxPool;
const op_kind_t MaxPoolBackprop = kMaxPoolBackprop;
const op_kind_t Minimum = kMinimum;
const op_kind_t Multiply = kMultiply;
const op_kind_t Pow = kPow;
const op_kind_t PowBackprop = kPowBackprop;
const op_kind_t ReduceSum = kReduceSum;
const op_kind_t ReLU = kReLU;
const op_kind_t ReLUBackprop = kReLUBackprop;
const op_kind_t Reshape = kReshape;
const op_kind_t Round = kRound;
const op_kind_t Sigmoid = kSigmoid;
const op_kind_t SigmoidBackprop = kSigmoidBackprop;
const op_kind_t SoftMax = kSoftMax;
const op_kind_t SoftMaxBackprop = kSoftMaxBackprop;
const op_kind_t SoftPlus = kSoftPlus;
const op_kind_t SoftPlusBackprop = kSoftPlusBackprop;
const op_kind_t Sqrt = kSqrt;
const op_kind_t SqrtBackprop = kSqrtBackprop;
const op_kind_t Square = kSquare;
const op_kind_t Tanh = kTanh;
const op_kind_t TanhBackprop = kTanhBackprop;
const op_kind_t Wildcard = kWildcard;
const op_kind_t BiasAdd = kBiasAdd;
const op_kind_t Interpolate = kInterpolate;
const op_kind_t Transpose = kTranspose;
const op_kind_t Index = kIndex;
const op_kind_t InterpolateBackprop = kInterpolateBackprop;
const op_kind_t PowBackpropExponent = kPowBackpropExponent;
const op_kind_t LastSymbol = kLastSymbol;

#define REGISTER_SYMBOL(s) #s,
const std::vector<std::string> op_kind_strings
        = {DNNL_GRAPH_FORALL_BUILDIN_OPS(REGISTER_SYMBOL) "LastSymbol"};
#undef REGISTER_SYMBOL
} // namespace op_kind

using logical_tensor_t = dnnl_graph_logical_tensor_t;

using layout_type_t = dnnl_graph_layout_type_t;
namespace layout_type {
const layout_type_t undef = dnnl_graph_layout_type_undef;
const layout_type_t any = dnnl_graph_layout_type_any;
const layout_type_t strided = dnnl_graph_layout_type_strided;
const layout_type_t opaque = dnnl_graph_layout_type_opaque;
} // namespace layout_type

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
using thread_pool_t = dnnl_graph_thread_pool;
using stream_attr_t = dnnl_graph_stream_attr;
using stream_t = dnnl_graph_stream;
using tensor_t = dnnl_graph_tensor;

// will be removed once the merge work is finished
using op_v2_t = dnnl_graph_op_v2;

} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
