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

// We define those internal used operators in this file. For those operators
// defined on API can be found at src/interface/c_types_map.hpp.

#ifndef INTERFACE_INTERNAL_OPS_HPP
#define INTERFACE_INTERNAL_OPS_HPP

#include <string>
#include <vector>

namespace dnnl {
namespace graph {
namespace impl {
namespace op_kind {

// X(s, v):
// s will be the internal op kind value, can be accessed via impl::op_kind::s.
// v will be used to define the name string of each op kind.
#define INTERNAL_OPS \
    X(bn_relu, BatchNorm_relu) \
    X(bn_bwd_relu_bwd, BatchNormBwd_reluBwd) \
    X(bn_fwd_train_relu, BatchNormFwdTrain_relu) \
    X(conv_add, Conv_add) \
    X(conv_add_elu, Conv_add_elu) \
    X(conv_add_relu, Conv_add_relu) \
    X(conv_add_relu6, Conv_add_relu6) \
    X(conv_bias, Conv_bias) \
    X(conv_bias_abs, Conv_bias_abs) \
    X(conv_bias_add, Conv_bias_add) \
    X(conv_bias_add_elu, Conv_bias_add_elu) \
    X(conv_bias_add_relu, Conv_bias_add_relu) \
    X(conv_bias_add_relu6, Conv_bias_add_relu6) \
    X(conv_bias_bn, Conv_bias_bn) \
    X(conv_bias_bn_add, Conv_bias_bn_add) \
    X(conv_bias_bn_add_relu, Conv_bias_bn_add_relu) \
    X(conv_bias_bn_relu, Conv_bias_bn_relu) \
    X(conv_bias_elu, Conv_bias_elu) \
    X(conv_bias_hardtanh, Conv_bias_hardtanh) \
    X(conv_bias_relu, Conv_bias_relu) \
    X(conv_bias_relu6, Conv_bias_relu6) \
    X(conv_bias_sigmoid, Conv_bias_sigmoid) \
    X(conv_bias_sqrt, Conv_bias_sqrt) \
    X(conv_bias_square, Conv_bias_square) \
    X(conv_bias_swish, Conv_bias_swish) \
    X(conv_bias_tanh, Conv_bias_tanh) \
    X(conv_bn, Conv_bn) \
    X(conv_bn_add, Conv_bn_add) \
    X(conv_bn_add_relu, Conv_bn_add_relu) \
    X(conv_bn_relu, Conv_bn_relu) \
    X(conv_relu, Conv_relu) \
    X(conv_bwd_f_biasadd_bwd, ConvBwdF_biasAddBwd) \
    X(matmul_bias, MatMul_bias) \
    X(matmul_bias_add, MatMul_bias_add) \
    X(matmul_bias_add_relu, MatMul_bias_add_relu) \
    X(matmul_bias_bn, MatMul_bias_bn) \
    X(matmul_bias_elu, MatMul_bias_elu) \
    X(matmul_bias_hardtanh, MatMul_bias_hardtanh) \
    X(matmul_bias_relu, MatMul_bias_relu) \
    X(matmul_bias_relu6, MatMul_bias_relu6) \
    X(matmul_bias_gelu, MatMul_bias_gelu) \
    X(matmul_bias_sigmoid, MatMul_bias_sigmoid) \
    X(matmul_bias_swish, MatMul_bias_swish) \
    X(matmul_relu, MatMul_relu) \
    X(matmul_elu, MatMul_elu) \
    X(matmul_sigmoid, MatMul_sigmoid) \
    X(matmul_hardtanh, MatMul_hardtanh) \
    X(matmul_gelu, MatMul_gelu) \
    X(matmul_add, MatMul_add) \
    X(matmul_add_gelu, MatMul_add_gelu) \
    X(matmul_add_relu, MatMul_add_relu) \
    X(matmul_add_sigmoid, MatMul_add_sigmoid) \
    X(int8_conv, INT8_Conv) \
    X(int8_conv_bias, INT8_Conv_bias) \
    X(int8_conv_relu, INT8_Conv_relu) \
    X(int8_conv_bias_relu, INT8_Conv_bias_relu) \
    X(int8_conv_add_relu, INT8_Conv_add_relu) \
    X(int8_conv_bias_add_relu, INT8_Conv_bias_add_relu) \
    X(int8_matmul, INT8_MatMul) \
    X(int8_matmul_bias, INT8_MatMul_bias) \
    X(int8_matmul_relu, INT8_MatMul_relu) \
    X(int8_matmul_bias_relu, INT8_MatMul_bias_relu) \
    X(int8_matmul_sigmoid, INT8_MatMul_sigmoid) \
    X(int8_matmul_bias_sigmoid, INT8_MatMul_bias_sigmoid) \
    X(int8_matmul_gelu, INT8_MatMul_gelu) \
    X(int8_matmul_bias_gelu, INT8_MatMul_bias_gelu) \
    X(mul_scales, Mul_scales) \
    X(add_zps, Add_zps) \
    X(permute, Permute) \
    X(to_group, To_group) \
    X(broadcast, Broadcast) \
    X(dnnl_convolution, Dnnl_convolution) \
    X(relu_add, Relu_add) \
    X(add_relu, Add_relu) \
    X(add_sigmoid, Add_sigmoid) \
    X(multiply_relu, Multiply_relu) \
    X(multiply_sigmoid, Multiply_sigmoid) \
    X(maximum_relu, Maximum_relu) \
    X(maximum_sigmoid, Maximum_sigmoid) \
    X(minimum_relu, Minimum_relu) \
    X(minimum_sigmoid, Minimum_sigmoid) \
    X(multiply_add, Multiply_add) \
    X(maximum_add, Maximum_add) \
    X(minimum_add, Minimum_add)

enum {
    kAny = 0x1234,
#define X(s, v) k##v,
    INTERNAL_OPS
#undef X
};

const op_kind_t any = static_cast<op_kind_t>(kAny);
#define X(s, v) const op_kind_t s = static_cast<op_kind_t>(k##v);
INTERNAL_OPS
#undef X

#define X(s, v) #v,
const std::vector<std::string> internal_op_strings = {"Any", INTERNAL_OPS};
#undef X

#undef INTERNAL_OPS

} // namespace op_kind
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
