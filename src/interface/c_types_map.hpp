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

#ifndef LLGA_INTERFACE_C_TYPES_MAP_HPP
#define LLGA_INTERFACE_C_TYPES_MAP_HPP

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

enum {
    kAny = 0x1234,
    kBatchNorm_relu,
    kBatchNormBwd_reluBwd,
    kBatchNormFwdTrain_relu,
    kConv_add,
    kConv_add_elu,
    kConv_add_relu,
    kConv_add_relu6,
    kConv_bias,
    kConv_bias_abs,
    kConv_bias_add,
    kConv_bias_add_elu,
    kConv_bias_add_relu,
    kConv_bias_add_relu6,
    kConv_bias_bn,
    kConv_bias_bn_add,
    kConv_bias_bn_add_relu,
    kConv_bias_bn_relu,
    kConv_bias_elu,
    kConv_bias_hardtanh,
    kConv_bias_relu,
    kConv_bias_relu6,
    kConv_bias_sigmoid,
    kConv_bias_sqrt,
    kConv_bias_square,
    kConv_bias_swish,
    kConv_bias_tanh,
    kConv_bn,
    kConv_bn_add,
    kConv_bn_add_relu,
    kConv_bn_relu,
    kConv_relu,
    kConvBwdF_biasAddBwd,
    kMatMul_bias,
    kMatMul_bias_add,
    kMatMul_bias_add_relu,
    kMatMul_bias_bn,
    kMatMul_bias_elu,
    kMatMul_bias_hardtanh,
    kMatMul_bias_relu,
    kMatMul_bias_relu6,
    kMatMul_bias_sigmoid,
    kMatMul_bias_swish,
    kMatMul_relu,
    kMatMul_elu,
    kMatMul_sigmoid,
    kMatMul_hardtanh,
    kMatMul_gelu,
    kMatMul_add,
    kMatMul_add_gelu,
    kMatMul_add_relu,
    kMatMul_add_sigmoid,
    kConvert,
};

// any means no need to identify when traversing graph
const op_kind_t any = static_cast<op_kind_t>(kAny);
// fused operators
const op_kind_t bn_relu = static_cast<op_kind_t>(kBatchNorm_relu);
const op_kind_t bn_bwd_relu_bwd = static_cast<op_kind_t>(kBatchNormBwd_reluBwd);
const op_kind_t bn_fwd_train_relu
        = static_cast<op_kind_t>(kBatchNormFwdTrain_relu);
const op_kind_t conv_add = static_cast<op_kind_t>(kConv_add);
const op_kind_t conv_add_elu = static_cast<op_kind_t>(kConv_add_elu);
const op_kind_t conv_add_relu = static_cast<op_kind_t>(kConv_add_relu);
const op_kind_t conv_add_relu6 = static_cast<op_kind_t>(kConv_add_relu6);
const op_kind_t conv_bias = static_cast<op_kind_t>(kConv_bias);
const op_kind_t conv_bias_abs = static_cast<op_kind_t>(kConv_bias_abs);
const op_kind_t conv_bias_add = static_cast<op_kind_t>(kConv_bias_add);
const op_kind_t conv_bias_add_elu = static_cast<op_kind_t>(kConv_bias_add_elu);
const op_kind_t conv_bias_add_relu
        = static_cast<op_kind_t>(kConv_bias_add_relu);
const op_kind_t conv_bias_add_relu6
        = static_cast<op_kind_t>(kConv_bias_add_relu6);
const op_kind_t conv_bias_bn = static_cast<op_kind_t>(kConv_bias_bn);
const op_kind_t conv_bias_bn_add = static_cast<op_kind_t>(kConv_bias_bn_add);
const op_kind_t conv_bias_bn_add_relu
        = static_cast<op_kind_t>(kConv_bias_bn_add_relu);
const op_kind_t conv_bias_bn_relu = static_cast<op_kind_t>(kConv_bias_bn_relu);
const op_kind_t conv_bias_elu = static_cast<op_kind_t>(kConv_bias_elu);
const op_kind_t conv_bias_hardtanh
        = static_cast<op_kind_t>(kConv_bias_hardtanh);
const op_kind_t conv_bias_relu = static_cast<op_kind_t>(kConv_bias_relu);
const op_kind_t conv_bias_relu6 = static_cast<op_kind_t>(kConv_bias_relu6);
const op_kind_t conv_bias_sigmoid = static_cast<op_kind_t>(kConv_bias_sigmoid);
const op_kind_t conv_bias_sqrt = static_cast<op_kind_t>(kConv_bias_sqrt);
const op_kind_t conv_bias_square = static_cast<op_kind_t>(kConv_bias_square);
const op_kind_t conv_bias_swish = static_cast<op_kind_t>(kConv_bias_swish);
const op_kind_t conv_bias_tanh = static_cast<op_kind_t>(kConv_bias_tanh);
const op_kind_t conv_bn = static_cast<op_kind_t>(kConv_bn);
const op_kind_t conv_bn_add = static_cast<op_kind_t>(kConv_bn_add);
const op_kind_t conv_bn_add_relu = static_cast<op_kind_t>(kConv_bn_add_relu);
const op_kind_t conv_bn_relu = static_cast<op_kind_t>(kConv_bn_relu);
const op_kind_t conv_relu = static_cast<op_kind_t>(kConv_relu);
const op_kind_t conv_bwd_f_biasadd_bwd
        = static_cast<op_kind_t>(kConvBwdF_biasAddBwd);
const op_kind_t matmul_bias = static_cast<op_kind_t>(kMatMul_bias);
const op_kind_t matmul_bias_add = static_cast<op_kind_t>(kMatMul_bias_add);
const op_kind_t matmul_bias_add_relu
        = static_cast<op_kind_t>(kMatMul_bias_add_relu);
const op_kind_t matmul_bias_bn = static_cast<op_kind_t>(kMatMul_bias_bn);
const op_kind_t matmul_bias_elu = static_cast<op_kind_t>(kMatMul_bias_elu);
const op_kind_t matmul_bias_hardtanh
        = static_cast<op_kind_t>(kMatMul_bias_hardtanh);
const op_kind_t matmul_bias_relu = static_cast<op_kind_t>(kMatMul_bias_relu);
const op_kind_t matmul_bias_relu6 = static_cast<op_kind_t>(kMatMul_bias_relu6);
const op_kind_t matmul_bias_sigmoid
        = static_cast<op_kind_t>(kMatMul_bias_sigmoid);
const op_kind_t matmul_bias_swish = static_cast<op_kind_t>(kMatMul_bias_swish);
const op_kind_t matmul_relu = static_cast<op_kind_t>(kMatMul_relu);
const op_kind_t matmul_elu = static_cast<op_kind_t>(kMatMul_elu);
const op_kind_t matmul_sigmoid = static_cast<op_kind_t>(kMatMul_sigmoid);
const op_kind_t matmul_hardtanh = static_cast<op_kind_t>(kMatMul_hardtanh);
const op_kind_t matmul_gelu = static_cast<op_kind_t>(kMatMul_gelu);
const op_kind_t matmul_add = static_cast<op_kind_t>(kMatMul_add);
const op_kind_t matmul_add_gelu = static_cast<op_kind_t>(kMatMul_add_gelu);
const op_kind_t matmul_add_relu = static_cast<op_kind_t>(kMatMul_add_relu);
const op_kind_t matmul_add_sigmoid
        = static_cast<op_kind_t>(kMatMul_add_sigmoid);
// for data conversion
const op_kind_t convert = static_cast<op_kind_t>(kConvert);

#define REGISTER_SYMBOL(s) #s,
const std::vector<std::string> op_kind_strings
        = {DNNL_GRAPH_FORALL_BUILDIN_OPS(REGISTER_SYMBOL) "LastSymbol"};
#undef REGISTER_SYMBOL

const std::vector<std::string> internal_op_strings = {"Any", "BatchNorm_relu",
        "BatchNormBwd_reluBwd", "BatchNormFwdTrain_relu", "Conv_add",
        "Conv_add_elu", "Conv_add_relu", "Conv_add_relu6", "Conv_bias",
        "Conv_bias_abs", "Conv_bias_add", "Conv_bias_add_elu",
        "Conv_bias_add_relu", "Conv_bias_add_relu6", "Conv_bias_bn",
        "Conv_bias_bn_add", "Conv_bias_bn_add_relu", "Conv_bias_bn_relu",
        "Conv_bias_elu", "Conv_bias_hardtanh", "Conv_bias_relu",
        "Conv_bias_relu6", "Conv_bias_sigmoid", "Conv_bias_sqrt",
        "Conv_bias_square", "Conv_bias_swish", "Conv_bias_tanh", "Conv_bn",
        "Conv_bn_add", "Conv_bn_add_relu", "Conv_bn_relu", "Conv_relu",
        "ConvBwdF_biasAddBwd", "MatMul_bias", "MatMul_bias_add",
        "MatMul_bias_add_relu", "MatMul_bias_bn", "MatMul_bias_elu",
        "MatMul_bias_hardtanh", "MatMul_bias_relu", "MatMul_bias_relu6",
        "MatMul_bias_sigmoid", "MatMul_bias_swish", "MatMul_relu", "MatMul_elu",
        "MatMul_sigmoid", "MatMul_hardtanh", "MatMul_gelu", "MatMul_add",
        "MatMul_add_gelu", "MatMul_add_relu", "MatMul_add_sigmoid", "Convert"};

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
using graph_t = dnnl_graph_graph;
using op_t = dnnl_graph_op;
using partition_t = dnnl_graph_partition;
using compiled_partition_t = dnnl_graph_compiled_partition;

} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
