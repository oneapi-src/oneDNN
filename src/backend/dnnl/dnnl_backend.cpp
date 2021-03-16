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
#include "operators.hpp"
#include "passes.hpp"
#include "tensor.hpp"
#include "transformation_pass.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

bool dnnl_layout_id_manager::is_mem_desc_equal(
        const impl::utils::any &mem_desc1,
        const impl::utils::any &mem_desc2) const {
    auto &md1 = impl::utils::any_cast<const tensor::desc &>(mem_desc1);
    auto &md2 = impl::utils::any_cast<const tensor::desc &>(mem_desc2);
    return md1 == md2;
}

dnnl_backend::dnnl_backend(const std::string &name, float priority)
    : backend(std::move(name), priority) {
    bool ret = register_passes() && register_kernels();
    if (!ret) { throw std::runtime_error(name + " initialize failed"); }
}

bool dnnl_backend::register_passes() {
    DNNL_BACKEND_REGISTER_PASSES_CALL(bn_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(conv_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(gelu_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(matmul_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PASSES_CALL(single_node_pass, pass_registry_);

    return true;
}

bool dnnl_backend::register_kernels() {
    // Register DNNL kernel
#define DNNL_REGISTER_KERNEL(op_kind_, kernel_class_) \
    static auto _flag_##op_kind_##_ \
            = kernel_registry_.register_kernel(op_kind::op_kind_, \
                    &kernel_registry::create_kernel<kernel_class_>); \
    (void)_flag_##op_kind_##_;

    // conv related operators
    DNNL_REGISTER_KERNEL(Convolution, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_add, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_add_elu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_add_relu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_add_relu6, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_add, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_add_elu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_add_relu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_add_relu6, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_bn, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_bn_add, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_bn_add_relu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_bn_relu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_elu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_relu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_sigmoid, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_relu6, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_hardtanh, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_square, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_tanh, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_abs, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bias_sqrt, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bn, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bn_add, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bn_add_relu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_bn_relu, convolution_forward)
    DNNL_REGISTER_KERNEL(conv_relu, convolution_forward)
    DNNL_REGISTER_KERNEL(ConvolutionBackpropData, convolution_backward_data)
    DNNL_REGISTER_KERNEL(
            ConvolutionBackpropFilters, convolution_backward_weights)
    DNNL_REGISTER_KERNEL(conv_bwd_f_biasadd_bwd, convolution_backward_weights)

    // bn related operators
    DNNL_REGISTER_KERNEL(
            BatchNormInference, batch_normalization_forward_inference)
    DNNL_REGISTER_KERNEL(bn_relu, batch_normalization_forward_inference)
    DNNL_REGISTER_KERNEL(
            BatchNormForwardTraining, batch_normalization_forward_training)
    DNNL_REGISTER_KERNEL(
            BatchNormTrainingBackprop, batch_normalization_backward)

    // binary operators
    DNNL_REGISTER_KERNEL(Add, binary)
    DNNL_REGISTER_KERNEL(Multiply, binary)
    DNNL_REGISTER_KERNEL(Maximum, binary)
    DNNL_REGISTER_KERNEL(Minimum, binary)

    // bias_add
    DNNL_REGISTER_KERNEL(BiasAdd, bias_add)

    // elementwise related operators
    DNNL_REGISTER_KERNEL(Abs, eltwise_forward)
    DNNL_REGISTER_KERNEL(Elu, eltwise_forward)
    DNNL_REGISTER_KERNEL(Exp, eltwise_forward)
    DNNL_REGISTER_KERNEL(GELU, eltwise_forward)
    DNNL_REGISTER_KERNEL(HardTanh, eltwise_forward)
    DNNL_REGISTER_KERNEL(ReLU, eltwise_forward)
    DNNL_REGISTER_KERNEL(Sqrt, eltwise_forward)
    DNNL_REGISTER_KERNEL(Square, eltwise_forward)
    DNNL_REGISTER_KERNEL(Tanh, eltwise_forward)
    DNNL_REGISTER_KERNEL(Pow, eltwise_forward)
    DNNL_REGISTER_KERNEL(Log, eltwise_forward)
    DNNL_REGISTER_KERNEL(ReLUBackprop, eltwise_backward)
    DNNL_REGISTER_KERNEL(GELUBackprop, eltwise_backward)

    // matmul related operators
    DNNL_REGISTER_KERNEL(MatMul, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_relu, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_elu, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_sigmoid, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_hardtanh, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_gelu, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_bias, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_bias_relu, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_bias_relu6, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_bias_elu, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_bias_sigmoid, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_bias_hardtanh, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_bias_add, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_bias_add_relu, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_bias_bn, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_add, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_add_gelu, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_add_relu, matmul_forward)
    DNNL_REGISTER_KERNEL(matmul_add_sigmoid, matmul_forward)

    // pooling related operators
    DNNL_REGISTER_KERNEL(MaxPool, pooling_forward)
    DNNL_REGISTER_KERNEL(AvgPool, pooling_forward)
    DNNL_REGISTER_KERNEL(AvgPoolBackprop, pooling_backward)
    DNNL_REGISTER_KERNEL(MaxPoolBackprop, pooling_backward)

    // softmax operators
    DNNL_REGISTER_KERNEL(SoftMax, softmax_forward)
    DNNL_REGISTER_KERNEL(SoftMaxBackprop, softmax_backward)

    // layernorm kernel
    DNNL_REGISTER_KERNEL(LayerNorm, layer_normalization_forward)

    // reorder kernel
    DNNL_REGISTER_KERNEL(convert, reorder)

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

std::shared_ptr<partition_impl_t> dnnl_backend::create_conversion(
        const impl::engine_kind_t engine_kind,
        const impl::logical_tensor_t &input,
        const impl::logical_tensor_t &output) {
    logical_tensor_wrapper ltw {&output};
    assert(ltw.is_opaque());

    auto pimpl = std::make_shared<dnnl_partition_impl_t>(engine_kind);
    pimpl->fused_op_ = impl::utils::make_unique<op_t>(op_kind::convert);
    pimpl->inputs_.push_back(input);
    pimpl->outputs_.push_back(output);

    return pimpl;
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
