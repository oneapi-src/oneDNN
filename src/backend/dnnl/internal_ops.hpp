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

// We define those internal used operators in this file. For those operators
// defined on API can be found at src/interface/c_types_map.hpp.

#ifndef BACKEND_DNNL_INTERNAL_OPS_HPP
#define BACKEND_DNNL_INTERNAL_OPS_HPP

#include <string>
#include <vector>

#include "interface/c_types_map.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace op_kind {

// X(s, v):
// s will be the internal op kind value, can be accessed via impl::op_kind::s.
// v will be used to define the name string of each op kind.
#define INTERNAL_OPS \
    X(bn_relu, BatchNorm_relu) \
    X(bn_bwd_relu_bwd, BatchNormBwd_reluBwd) \
    X(bn_fwd_train_relu, BatchNormFwdTrain_relu) \
    X(binary_post_ops_fusion, Binary_post_ops_fusion) \
    X(dnnl_conv_depthwise, Dnnl_conv_depthwise) \
    X(conv_bias_post_ops_fusion, conv_bias_post_ops_fusion) \
    X(convtranspose_fusion, ConvTranspose_fusion) \
    X(pool_binary, Pool_binary) \
    X(int8_conv_post_ops_fusion, INT8_conv_post_ops_chain) \
    X(int8_matmul_post_ops_fusion, INT8_matmul_post_ops_fusion) \
    X(int8_matmul, INT8_MatMul) \
    X(int8_quant_wei_matmul, INT8_Quant_wei_MatMul) \
    X(int8_matmul_bias, INT8_MatMul_bias) \
    X(int8_quant_wei_matmul_bias, INT8_Quant_wei_MatMul_bias) \
    X(int8_matmul_relu, INT8_MatMul_relu) \
    X(int8_quant_wei_matmul_relu, INT8_Quant_wei_MatMul_relu) \
    X(int8_matmul_bias_relu, INT8_MatMul_bias_relu) \
    X(int8_quant_wei_matmul_bias_relu, INT8_Quant_wei_MatMul_bias_relu) \
    X(int8_matmul_sigmoid, INT8_MatMul_sigmoid) \
    X(int8_quant_wei_matmul_sigmoid, INT8_Quant_wei_MatMul_sigmoid) \
    X(int8_matmul_bias_sigmoid, INT8_MatMul_bias_sigmoid) \
    X(int8_quant_wei_matmul_bias_sigmoid, INT8_Quant_wei_MatMul_bias_sigmoid) \
    X(int8_matmul_gelu, INT8_MatMul_gelu) \
    X(int8_quant_wei_matmul_gelu, INT8_Quant_wei_MatMul_gelu) \
    X(int8_matmul_bias_gelu, INT8_MatMul_bias_gelu) \
    X(int8_quant_wei_matmul_bias_gelu, INT8_Quant_wei_MatMul_bias_gelu) \
    X(int8_matmul_add, INT8_MatMul_add) \
    X(int8_quant_wei_matmul_add, INT8_Quant_wei_MatMul_add) \
    X(int8_matmul_bias_add, INT8_MatMul_bias_add) \
    X(int8_quant_wei_matmul_bias_add, INT8_Quant_wei_MatMul_bias_add) \
    X(x8s8float_matmul_add, X8S8FLOAT_MatMul_add) \
    X(x8s8float_matmul_bias_add, X8S8FLOAT_MatMul_bias_add) \
    X(dnnl_mul_scales, Dnnl_mul_scales) \
    X(dnnl_constant_scales, Dnnl_constant_scales) \
    X(dnnl_add_zps, Dnnl_add_zps) \
    X(dnnl_sub_zps, Dnnl_sub_zps) \
    X(dnnl_constant_zps, Dnnl_constant_zps) \
    X(permute, Permute) \
    X(to_group, To_group) \
    X(from_group, From_group) \
    X(expand, Expand) \
    X(squeeze, Squeeze) \
    X(dnnl_convolution, Dnnl_convolution) \
    X(dnnl_convtranspose, Dnnl_convtranspose) \
    X(int8_maxpool, INT8_MaxPool) \
    X(int8_maxpool_add, INT8_MaxPool_add) \
    X(eltwise_binary, Eltwise_binary) \
    X(x8x8float_matmul, X8X8FLOAT_MatMul) \
    X(x8s8float_matmul_bias, X8S8FLOAT_MatMul_bias) \
    X(x8s8f32_matmul_relu, X8S8F32_MatMul_relu) \
    X(x8s8f32_matmul_bias_relu, X8S8F32_MatMul_bias_relu) \
    X(x8s8f32_matmul_sigmoid, X8S8F32_MatMul_sigmoid) \
    X(x8s8f32_matmul_bias_sigmoid, X8S8F32_MatMul_bias_sigmoid) \
    X(x8s8f32_matmul_gelu, X8S8F32_MatMul_gelu) \
    X(x8s8f32_matmul_bias_gelu, X8S8F32_MatMul_bias_gelu) \
    X(x8s8f32_quant_wei_matmul_add, X8S8F32_Quant_wei_MatMul_add) \
    X(x8s8f32_quant_wei_matmul_bias_add, X8S8F32_Quant_wei_MatMul_bias_add) \
    X(x8s8f32_quant_wei_matmul, X8S8F32_Quant_wei_MatMul) \
    X(x8s8f32_quant_wei_matmul_bias, X8S8F32_Quant_wei_MatMul_bias) \
    X(x8s8f32_quant_wei_matmul_relu, X8S8F32_Quant_wei_MatMul_relu) \
    X(x8s8f32_quant_wei_matmul_bias_relu, X8S8F32_Quant_wei_MatMul_bias_relu) \
    X(x8s8f32_quant_wei_matmul_sigmoid, X8S8F32_Quant_wei_MatMul_sigmoid) \
    X(x8s8f32_quant_wei_matmul_bias_sigmoid, \
            X8S8F32_Quant_wei_MatMul_bias_sigmoid) \
    X(x8s8f32_quant_wei_matmul_gelu, X8S8F32_Quant_wei_MatMul_gelu) \
    X(x8s8f32_quant_wei_matmul_bias_gelu, X8S8F32_Quant_wei_MatMul_bias_gelu) \
    X(int8_avgpool, INT8_AvgPool) \
    X(int8_avgpool_add, INT8_AvgPool_add) \
    X(dnnl_pool, Dnnl_pool) \
    X(dnnl_bn_folding, Dnnl_bn_folding) \
    X(dnnl_conv_bwd_data, Dnnl_conv_bwd_data) \
    X(x8x8float_matmul_div, X8X8FLOAT_MatMul_div) \
    X(x8x8float_matmul_div_add, X8X8FLOAT_MatMul_div_add) \
    X(dnnl_batchnorm, Dnnl_batchnorm) \
    X(dnnl_binary, Dnnl_binary) \
    X(dnnl_eltwise, Dnnl_eltwise) \
    X(dnnl_eltwise_bwd, Dnnl_eltwise_bwd) \
    X(dnnl_shuffle, Dnnl_shuffle) \
    X(interpolate_post_ops_fusion, Interpolate_post_ops_fusion) \
    X(dnnl_sum, Dnnl_sum) \
    X(dnnl_reduction, Dnnl_reduction) \
    X(reduction_post_ops_fusion, Reduction_post_ops_fusion) \
    X(conv_simple_resblock, Conv_simple_resblock) \
    X(int8_MHA, INT8_MHA) \
    X(f32_MHA, F32_MHA) \
    X(chained_relu, Chained_relu) \
    X(dnnl_prelu, Dnnl_prelu) \
    X(dnnl_prelu_bwd, Dnnl_prelu_bwd) \
    X(reorder_sum, Reorder_sum) \
    X(int8_reorder, INT8_reorder) \
    X(dnnl_batchnorm_bwd, Dnnl_batchnorm_bwd) \
    X(dnnl_softmax_bwd, Dnnl_softmax_bwd) \
    X(dnnl_logsoftmax_bwd, Dnnl_logsoftmax_bwd) \
    X(dnnl_resampling, Dnnl_resampling) \
    X(dnnl_resampling_bwd, Dnnl_resampling_bwd) \
    X(dnnl_concat, Dnnl_concat) \
    X(dnnl_layernorm_bwd, Dnnl_layernorm_bwd) \
    X(dnnl_conv_bwd_weights, Dnnl_conv_bwd_weights) \
    X(dnnl_pool_bwd, Dnnl_pool_bwd) \
    X(dnnl_matmul, Dnnl_matmul) \
    X(dnnl_softmax, Dnnl_softmax) \
    X(dnnl_logsoftmax, Dnnl_logsoftmax) \
    X(dnnl_layernorm, Dnnl_layernorm) \
    X(dnnl_reorder, Dnnl_reorder) \
    X(float_conv_fusion, Float_conv_fusion) \
    X(conv_post_ops_fusion, conv_post_ops_fusion) \
    X(quantized_convtranspose_fusion, Quantized_convtranspose_fusion) \
    X(quantized_concat_fusion, Quantized_concat_fusion) \
    X(matmul_post_ops_chain_fusion, MatMul_post_ops_chain_fusion) \
    X(matmul_bias_post_ops_chain_fusion, MatMul_bias_post_ops_chain_fusion) \
    X(dnnl_convtranspose_bwd_data, Dnnl_convtranspose_bwd_data) \
    X(dnnl_convtranspose_bwd_weights, Dnnl_convtranspose_bwd_weights)

enum {
    kDNNL_INTERNAL_OP_STARTER = 0x1234,
#define X(s, v) k##v,
    INTERNAL_OPS
#undef X
};

#define X(s, v) const op_kind_t s = static_cast<op_kind_t>(k##v);
INTERNAL_OPS
#undef X

#define X(s, v) #v,
const std::vector<std::string> internal_op_strings = {INTERNAL_OPS};
#undef X

#undef INTERNAL_OPS

} // namespace op_kind
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
