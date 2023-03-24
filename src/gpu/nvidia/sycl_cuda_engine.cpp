/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "common/impl_list_item.hpp"
#include "common/utils.hpp"

#include "sycl/sycl_utils.hpp"

#include "gpu/nvidia/cudnn_batch_normalization.hpp"
#include "gpu/nvidia/cudnn_binary.hpp"
#include "gpu/nvidia/cudnn_conv_inner_product.hpp"
#include "gpu/nvidia/cudnn_convolution.hpp"
#include "gpu/nvidia/cudnn_deconvolution.hpp"
#include "gpu/nvidia/cudnn_eltwise.hpp"
#include "gpu/nvidia/cudnn_gemm_inner_product.hpp"
#include "gpu/nvidia/cudnn_lrn.hpp"
#include "gpu/nvidia/cudnn_matmul.hpp"
#include "gpu/nvidia/cudnn_pooling.hpp"
#include "gpu/nvidia/cudnn_reduction.hpp"
#include "gpu/nvidia/cudnn_resampling.hpp"
#include "gpu/nvidia/cudnn_softmax.hpp"
#include "gpu/nvidia/sycl_cuda_compat.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"

#include "gpu/sycl/ref_batch_normalization.hpp"
#include "gpu/sycl/ref_binary.hpp"
#include "gpu/sycl/ref_eltwise.hpp"
#include "gpu/sycl/ref_layer_normalizations.hpp"
#include "gpu/sycl/ref_lrn.hpp"
#include "gpu/sycl/ref_pooling.hpp"
#include "gpu/sycl/ref_prelu.hpp"
#include "gpu/sycl/ref_resampling.hpp"
#include "gpu/sycl/ref_shuffle.hpp"
#include "gpu/sycl/ref_softmax.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

bool is_nvidia_gpu(const ::sycl::device &dev) {
    constexpr int nvidia_vendor_id = 0x10DE;
    return dev.is_gpu()
            && dev.get_info<::sycl::info::device::vendor_id>()
            == nvidia_vendor_id;
}

status_t cuda_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index) {
    CHECK(nvidia::check_device(engine_kind));
    std::unique_ptr<nvidia::sycl_cuda_engine_t, engine_deleter_t> cuda_engine(
            (new nvidia::sycl_cuda_engine_t(dev, ctx, index)));
    if (!cuda_engine) return status::out_of_memory;

    CHECK(cuda_engine->init());
    *engine = cuda_engine.release();

    return status::success;
}

sycl_cuda_engine_t::sycl_cuda_engine_t(engine_kind_t kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : base_t(kind, dev, ctx, index) {
    underlying_context_type();
    set_cudnn_handle();
    set_cublas_handle();
}

sycl_cuda_engine_t::sycl_cuda_engine_t(
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : sycl_cuda_engine_t(engine_kind::gpu, dev, ctx, index) {
    assert(is_nvidia_gpu(dev));
}

status_t sycl_cuda_engine_t::set_cublas_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the cublas handle.
    cuda_sycl_scoped_context_handler_t sc(*this);
    cublasHandle_t handle;
    CHECK(CUBLAS_EXECUTE_FUNC_S(cublasCreate, &handle));
    cublas_handle_.set(
            std::unique_ptr<cublasHandle_t, void (*)(cublasHandle_t *)>(
                    new cublasHandle_t(handle), [](cublasHandle_t *h) {
                        if (h != nullptr)
                            CUBLAS_EXECUTE_FUNC_V(cublasDestroy, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}

status_t sycl_cuda_engine_t::set_cudnn_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the cudnn handle.
    cuda_sycl_scoped_context_handler_t sc(*this);
    cudnnHandle_t handle;
    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreate, &handle));
    cudnn_handle_.set(std::unique_ptr<cudnnHandle_t, void (*)(cudnnHandle_t *)>(
            new cudnnHandle_t(handle), [](cudnnHandle_t *h) {
                if (h != nullptr) CUDNN_EXECUTE_FUNC_V(cudnnDestroy, *h);
                delete h;
            }));
    handle = nullptr;
    return status::success;
}

CUcontext sycl_cuda_engine_t::get_underlying_context() const {
    return compat::get_native<CUcontext>(context());
}

CUdevice sycl_cuda_engine_t::get_underlying_device() const {
    return compat::get_native<CUdevice>(device());
}

status_t sycl_cuda_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return sycl_cuda_stream_t::create_stream(stream, this, flags);
}

status_t sycl_cuda_engine_t::create_stream(
        stream_t **stream, ::sycl::queue &queue) {
    return sycl_cuda_stream_t::create_stream(stream, this, queue);
}

status_t sycl_cuda_engine_t::underlying_context_type() {
    // this is a costly function which take avarage up to 75ms
    // on titanrx. So we must run it once and store the variable
    // in primary_context_;
    CUcontext primary, current;
    CUcontext desired = compat::get_native<CUcontext>(context());
    CUdevice cuda_device = compat::get_native<CUdevice>(device());
    CHECK(CUDA_EXECUTE_FUNC_S(cuCtxGetCurrent, &current));

    unsigned int flags;
    int is_primary_active;
    CHECK(CUDA_EXECUTE_FUNC_S(cuDevicePrimaryCtxGetState, cuda_device, &flags,
            &is_primary_active));

    // If primary context is active, current context will be the primary context
    // So we can do the comparison without the expensive calls to CtxRetain and CtxRelease
    if (current == desired || is_primary_active) {
        primary_context_
                = (current == desired) ? (is_primary_active == 1) : false;
    } else {
        CHECK(CUDA_EXECUTE_FUNC_S(
                cuDevicePrimaryCtxRetain, &primary, cuda_device));
        CHECK(CUDA_EXECUTE_FUNC_S(cuDevicePrimaryCtxRelease, cuda_device));
        primary_context_ = (primary == desired);
    }
    return status::success;
}

cudnnHandle_t *sycl_cuda_engine_t::get_cudnn_handle() {
    if (!cudnn_handle_.is_set()) set_cudnn_handle();
    return cudnn_handle_.get().get();
}

cublasHandle_t *sycl_cuda_engine_t::get_cublas_handle() {
    if (!cublas_handle_.is_set()) set_cublas_handle();
    return cublas_handle_.get().get();
}

device_id_t sycl_cuda_engine_t::device_id() const {
    return device_id_t(static_cast<int>(impl::sycl::backend_t::nvidia),
            static_cast<uint64_t>(compat::get_native<CUdevice>(device())),
            static_cast<uint64_t>(0));
}

void sycl_cuda_engine_t::activate_stream_cublas(CUstream cuda_stream) {
    cuda_sycl_scoped_context_handler_t sc(*this);
    cudaStream_t current_stream_id = nullptr;
    auto cublas_handle = get_cublas_handle();
    CUBLAS_EXECUTE_FUNC(cublasGetStream, *cublas_handle, &current_stream_id);
    if (current_stream_id != cuda_stream) {
        CUBLAS_EXECUTE_FUNC(cublasSetStream, *cublas_handle, cuda_stream);
    }
}

void sycl_cuda_engine_t::activate_stream_cudnn(CUstream cuda_stream) {
    cuda_sycl_scoped_context_handler_t sc(*this);
    cudaStream_t current_stream_id = nullptr;
    auto cudnn_handle = get_cudnn_handle();
    CUDNN_EXECUTE_FUNC(cudnnGetStream, *cudnn_handle, &current_stream_id);
    if (current_stream_id != cuda_stream) {
        CUDNN_EXECUTE_FUNC(cudnnSetStream, *cudnn_handle, cuda_stream);
    }
}

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
constexpr dnnl::impl::impl_list_item_t sycl_cuda_impl_list[] = {
        // Elementwise
        INSTANCE(cudnn_eltwise_fwd_t)
        INSTANCE(cudnn_eltwise_bwd_t)
        INSTANCE(sycl::ref_sycl_eltwise_fwd_t)
        INSTANCE(sycl::ref_sycl_eltwise_bwd_t)

        // Deconvolution
        INSTANCE(cudnn_deconvolution_fwd_t)
        INSTANCE(cudnn_deconvolution_bwd_data_t)
        INSTANCE(cudnn_deconvolution_bwd_weights_t)

        // Convolution
        INSTANCE(cudnn_convolution_fwd_t)
        INSTANCE(cudnn_convolution_bwd_data_t)
        INSTANCE(cudnn_convolution_bwd_weights_t)

        // Batch Normalization
        INSTANCE(cudnn_batch_normalization_fwd_t)
        INSTANCE(cudnn_batch_normalization_bwd_t)
        INSTANCE(sycl::ref_batch_normalization_fwd_t)
        INSTANCE(sycl::ref_batch_normalization_bwd_t)

        // Layer Normalization
        INSTANCE(sycl::ref_layer_normalization_fwd_t)
        INSTANCE(sycl::ref_layer_normalization_bwd_t)

        // PReLU
        INSTANCE(sycl::ref_prelu_fwd_t)
        INSTANCE(sycl::ref_prelu_bwd_t)

        // Pooling
        INSTANCE(cudnn_pooling_fwd_t)
        INSTANCE(cudnn_pooling_bwd_t)
        INSTANCE(sycl::ref_pooling_fwd_t)
        INSTANCE(sycl::ref_pooling_bwd_t)

        // LRN
        INSTANCE(cudnn_lrn_fwd_t)
        INSTANCE(cudnn_lrn_bwd_t)
        INSTANCE(sycl::ref_sycl_lrn_fwd_t)
        INSTANCE(sycl::ref_sycl_lrn_bwd_t)

        // Inner Product
        INSTANCE(cudnn_gemm_inner_product_fwd_t)
        INSTANCE(cudnn_conv_inner_product_fwd_t)
        INSTANCE(cudnn_gemm_inner_product_bwd_data_t)
        INSTANCE(cudnn_conv_inner_product_bwd_data_t)
        INSTANCE(cudnn_gemm_inner_product_bwd_weights_t)
        INSTANCE(cudnn_conv_inner_product_bwd_weights_t)

        // Softmax
        INSTANCE(cudnn_softmax_fwd_t)
        INSTANCE(cudnn_softmax_bwd_t)
        INSTANCE(sycl::ref_sycl_softmax_fwd_t)
        INSTANCE(sycl::ref_sycl_softmax_bwd_t)

        // Binary
        INSTANCE(cudnn_binary_t)
        INSTANCE(sycl::ref_binary_t)

        // MatMul
        INSTANCE(cudnn_matmul_t)

        // Resampling
        INSTANCE(cudnn_resampling_fwd_t)
        INSTANCE(cudnn_resampling_bwd_t)
        INSTANCE(sycl::ref_resampling_fwd_t)
        INSTANCE(sycl::ref_resampling_bwd_t)

        // Reduction
        INSTANCE(cudnn_reduction_t)

        // Shuffle
        INSTANCE(sycl::ref_shuffle_t)
        nullptr,
};
// clang-format on
} // namespace
const dnnl::impl::impl_list_item_t *sycl_cuda_engine_t::get_implementation_list(
        const op_desc_t *) const {
    return sycl_cuda_impl_list;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
