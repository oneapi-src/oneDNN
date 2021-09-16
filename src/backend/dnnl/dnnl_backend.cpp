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

#include <utility>

#include "utils/compatible.hpp"

#include "dnnl_backend.hpp"
#include "dnnl_opset.hpp"
#include "kernels/kernels.hpp"
#include "patterns/binary_fusion.hpp"
#include "patterns/bn_fusion.hpp"
#include "patterns/conv_fusion.hpp"
#include "patterns/eltwise_fusion.hpp"
#include "patterns/gelu_fusion.hpp"
#include "patterns/matmul_fusion.hpp"
#include "patterns/pool_fusion.hpp"
#include "patterns/single_op_pattern.hpp"
#include "tensor.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

bool dnnl_layout_id_manager::is_mem_desc_equal(
        const impl::utils::any &mem_desc1,
        const impl::utils::any &mem_desc2) const {
    auto &md1 = impl::utils::any_cast<const memory::desc &>(mem_desc1);
    auto &md2 = impl::utils::any_cast<const memory::desc &>(mem_desc2);
    return md1 == md2;
}

dnnl_backend::dnnl_backend(const std::string &name, float priority)
    : backend(std::move(name), priority) {
    bool ret = register_op_schemas() && register_passes() && register_kernels();
    if (!ret) { throw std::runtime_error(name + " initialize failed"); }
}

bool dnnl_backend::register_op_schemas() {
    register_dnnl_opset_schema();
    return true;
}

bool dnnl_backend::register_passes() {
    DNNL_BACKEND_REGISTER_PASSES_CALL(binary_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(bn_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(conv_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(gelu_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(matmul_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(single_op_pass, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(pool_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(eltwise_fusion, pass_registry_);

    return true;
}

bool dnnl_backend::register_kernels() {
    // Register DNNL kernel
#define DECLARE_KERNEL_EX(kernel_class_, counter) \
    static auto _registered_dnnl_kernel_##kernel_class_##_##counter##_

#define DECLARE_KERNEL(kernel_class_, counter) \
    DECLARE_KERNEL_EX(kernel_class_, counter)

#define DNNL_REGISTER_KERNEL(op_kind_, kernel_class_) \
    DECLARE_KERNEL(kernel_class_, __COUNTER__) \
            = kernel_registry_.register_kernel( \
                    op_kind_, &kernel_registry::create_kernel<kernel_class_>);

    // concat
    DNNL_REGISTER_KERNEL(impl::op_kind::Concat, concat);

    // conv related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::Convolution, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_add, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_add_elu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_add_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_add_relu6, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_add, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_add_elu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_add_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_add_relu6, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_bn, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_bn_add, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_bn_add_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_bn_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_elu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_sigmoid, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_swish, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_relu6, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_hardtanh, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_square, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_tanh, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_abs, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bias_sqrt, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bn, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bn_add, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bn_add_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_bn_relu, float_conv_fwd)
    DNNL_REGISTER_KERNEL(op_kind::conv_relu, float_conv_fwd)

    DNNL_REGISTER_KERNEL(impl::op_kind::ConvolutionBackpropData, conv_bwd_data)

    DNNL_REGISTER_KERNEL(impl::op_kind::ConvolutionBackpropFilters,
            convolution_backward_weights)
    DNNL_REGISTER_KERNEL(
            op_kind::conv_bwd_f_biasadd_bwd, convolution_backward_weights)

    // convtranspose related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::ConvTranspose, convtranspose_forward)

    // bn related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::BatchNormInference,
            batch_normalization_forward_inference)
    DNNL_REGISTER_KERNEL(
            op_kind::bn_relu, batch_normalization_forward_inference)
    DNNL_REGISTER_KERNEL(impl::op_kind::BatchNormForwardTraining,
            batch_normalization_forward_training)
    DNNL_REGISTER_KERNEL(impl::op_kind::BatchNormTrainingBackprop,
            batch_normalization_backward)

    // binary operators
    DNNL_REGISTER_KERNEL(impl::op_kind::Add, binary)
    DNNL_REGISTER_KERNEL(op_kind::add_relu, binary)
    DNNL_REGISTER_KERNEL(op_kind::add_sigmoid, binary)
    DNNL_REGISTER_KERNEL(op_kind::add_multiply, binary)
    DNNL_REGISTER_KERNEL(impl::op_kind::Multiply, binary)
    DNNL_REGISTER_KERNEL(op_kind::multiply_add, binary)
    DNNL_REGISTER_KERNEL(op_kind::multiply_relu, binary)
    DNNL_REGISTER_KERNEL(op_kind::multiply_sigmoid, binary)
    DNNL_REGISTER_KERNEL(impl::op_kind::Maximum, binary)
    DNNL_REGISTER_KERNEL(op_kind::maximum_add, binary)
    DNNL_REGISTER_KERNEL(op_kind::maximum_relu, binary)
    DNNL_REGISTER_KERNEL(op_kind::maximum_sigmoid, binary)
    DNNL_REGISTER_KERNEL(impl::op_kind::Minimum, binary)
    DNNL_REGISTER_KERNEL(op_kind::minimum_add, binary)
    DNNL_REGISTER_KERNEL(op_kind::minimum_relu, binary)
    DNNL_REGISTER_KERNEL(op_kind::minimum_sigmoid, binary)

    // bias_add
    DNNL_REGISTER_KERNEL(impl::op_kind::BiasAdd, bias_add)

    // elementwise related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::Abs, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::Elu, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::Exp, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::GELU, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::HardTanh, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReLU, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::Sqrt, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::Square, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::Tanh, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::Pow, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::Log, eltwise_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::ReLUBackprop, eltwise_backward)
    DNNL_REGISTER_KERNEL(impl::op_kind::GELUBackprop, eltwise_backward)

    // matmul related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::MatMul, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_relu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_elu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_sigmoid, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_hardtanh, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_gelu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_relu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_gelu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_relu6, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_elu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_sigmoid, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_swish, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_hardtanh, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_add, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_add_relu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_bias_bn, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_add, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_add_gelu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_add_relu, float_matmul)
    DNNL_REGISTER_KERNEL(op_kind::matmul_add_sigmoid, float_matmul)

    // pooling related operators
    DNNL_REGISTER_KERNEL(impl::op_kind::AvgPool, pooling_forward)
    DNNL_REGISTER_KERNEL(op_kind::avgpool_add, pooling_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::MaxPool, pooling_forward)
    DNNL_REGISTER_KERNEL(op_kind::maxpool_add, pooling_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::AvgPoolBackprop, pooling_backward)
    DNNL_REGISTER_KERNEL(impl::op_kind::MaxPoolBackprop, pooling_backward)

    // softmax operators
    DNNL_REGISTER_KERNEL(impl::op_kind::SoftMax, softmax_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::SoftMaxBackprop, softmax_backward)

    // logsoftmax operators
    DNNL_REGISTER_KERNEL(impl::op_kind::LogSoftmax, logsoftmax_forward)
    DNNL_REGISTER_KERNEL(impl::op_kind::LogSoftmaxBackprop, logsoftmax_backward)

    // layernorm kernel
    DNNL_REGISTER_KERNEL(impl::op_kind::LayerNorm, layer_normalization_forward)

    //interpolate kernel
    DNNL_REGISTER_KERNEL(impl::op_kind::Interpolate, resampling_forward)
    DNNL_REGISTER_KERNEL(
            impl::op_kind::InterpolateBackprop, resampling_backward)

    // reorder kernel
    DNNL_REGISTER_KERNEL(impl::op_kind::Reorder, reorder)
    DNNL_REGISTER_KERNEL(impl::op_kind::TypeCast, reorder)

    // quantize and dequantize kernel
    DNNL_REGISTER_KERNEL(impl::op_kind::Quantize, quantize_dequantize)
    DNNL_REGISTER_KERNEL(impl::op_kind::Dequantize, quantize_dequantize)

    // quantized conv
    DNNL_REGISTER_KERNEL(op_kind::int8_conv, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_bias, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_bias_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_conv_bias_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_bias, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_bias_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_conv_bias_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv, quantized_conv)
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_conv_bias_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv_bias_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_conv_bias, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_quant_wei_conv, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_quant_wei_conv_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_quant_wei_conv_bias, quantized_conv)
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_conv_bias_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_conv_add_relu, quantized_conv)
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_conv_bias_add_relu, quantized_conv)

    // quantized matmul
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_matmul_bias_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8float_matmul_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8float_matmul_bias_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8x8f32_matmul, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8float_matmul_bias, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_bias_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_matmul_bias_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_matmul_bias_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul_bias, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_bias_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_bias_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_bias_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::int8_quant_wei_matmul_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::int8_quant_wei_matmul_bias_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias_add, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8s8f32_quant_wei_matmul, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias_relu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias_sigmoid, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(
            op_kind::x8s8f32_quant_wei_matmul_bias_gelu, quantized_matmul);
    DNNL_REGISTER_KERNEL(op_kind::x8x8float_matmul_div, quantized_matmul);

    //eltwise+binary ops
    DNNL_REGISTER_KERNEL(op_kind::relu_add, eltwise_forward);

    // quantized pooling
    DNNL_REGISTER_KERNEL(op_kind::int8_maxpool, quantized_pooling);
    DNNL_REGISTER_KERNEL(op_kind::int8_avgpool, quantized_pooling);

#undef DNNL_REGISTER_KERNEL

    return true;
}

size_t dnnl_backend::get_mem_size(const impl::logical_tensor_t &lt) const {
    auto md = make_dnnl_memory_desc(lt);
    return md.get_size();
}

bool dnnl_backend::compare_logical_tensor(const impl::logical_tensor_t &lhs,
        const impl::logical_tensor_t &rhs) const {
    auto md1 = make_dnnl_memory_desc(lhs);
    auto md2 = make_dnnl_memory_desc(rhs);
    return md1 == md2;
}

impl::utils::optional<size_t> dnnl_backend::set_mem_desc(
        const impl::utils::any &mem_desc) {
    return layout_id_manager_.set_mem_desc(mem_desc);
}

impl::utils::optional<impl::utils::any> dnnl_backend::get_mem_desc(
        const size_t &layout_id) const {
    return layout_id_manager_.get_mem_desc(layout_id);
}

DNNL_GRAPH_REGISTER_BACKEND(dnnl_backend::get_singleton())

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
