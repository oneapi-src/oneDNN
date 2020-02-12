/***************************************************************************
 *  Copyright 2020 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 **************************************************************************/

#include "nvidia/sycl_cuda_engine.hpp"
#include "nvidia/cudnn_batch_normalization.hpp"
#include "nvidia/cudnn_binary.hpp"
#include "nvidia/cudnn_conv_inner_product.hpp"
#include "nvidia/cudnn_convolution.hpp"
#include "nvidia/cudnn_deconvolution.hpp"
#include "nvidia/cudnn_eltwise.hpp"
#include "nvidia/cudnn_gemm_inner_product.hpp"
#include "nvidia/cudnn_lrn.hpp"
#include "nvidia/cudnn_matmul.hpp"
#include "nvidia/cudnn_pooling.hpp"
#include "nvidia/cudnn_resampling.hpp"
#include "nvidia/cudnn_softmax.hpp"
#include "nvidia/sycl_cuda_scoped_context.hpp"
#include "nvidia/sycl_cuda_stream.hpp"
#include "sycl/sycl_utils.hpp"
#include <CL/sycl/backend/cuda.hpp>

namespace dnnl {
namespace impl {
namespace cuda {

status_t sycl_cuda_engine_t::set_cublas_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the cublas handle.
    cublasHandle_t handle;
    cuda_sycl_scoped_context_handler_t sc(*this);
    CHECK(CUBLAS_EXECUTE_FUNC_S(cublasCreate, &handle));
    cublas_handle_.reset(new cublasHandle_t(handle));
    handle = nullptr;
    return status::success;
}

status_t sycl_cuda_engine_t::set_cudnn_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the cublas handle.
    cudnnHandle_t handle;
    cuda_sycl_scoped_context_handler_t sc(*this);
    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreate, &handle));
    cudnn_handle_.reset(new cudnnHandle_t(handle));
    handle = nullptr;
    return status::success;
}

CUcontext sycl_cuda_engine_t::get_underlying_context() const {
    return cl::sycl::get_native<cl::sycl::backend::cuda>(context());
}

status_t sycl_cuda_engine_t::create_stream(
        stream_t **stream, unsigned flags, const stream_attr_t *) {
    return sycl_cuda_stream_t::create_stream(stream, this, flags);
}

status_t sycl_cuda_engine_t::create_stream(
        stream_t **stream, cl::sycl::queue &queue) {
    return sycl_cuda_stream_t::create_stream(stream, this, queue);
}

status_t sycl_cuda_engine_t::underlying_context_type() {
    // this is a costly function which take avarage up to 75ms
    // on titanrx. So we must run it once and store the variable
    // in  is_primary_context_;
    CUcontext primary;
    CUcontext desired
            = cl::sycl::get_native<cl::sycl::backend::cuda>(context());
    CUdevice cuda_device
            = cl::sycl::get_native<cl::sycl::backend::cuda>(device());
    CHECK(CUDA_EXECUTE_FUNC_S(cuDevicePrimaryCtxRetain, &primary, cuda_device));
    CHECK(CUDA_EXECUTE_FUNC_S(cuDevicePrimaryCtxRelease, cuda_device));
    primary_context_ = (primary == desired);
    return status::success;
}

device_id_t sycl_cuda_engine_t::device_id() const {
    return device_id_t(static_cast<int>(sycl::backend_t::nvidia),
            static_cast<uint64_t>(
                    cl::sycl::get_native<cl::sycl::backend::cuda>(device())),
            static_cast<uint64_t>(0));
}

namespace {
using namespace dnnl::impl::data_type;
#define INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
// clang-format off
static const dnnl::impl::engine_t::primitive_desc_create_f
sycl_cuda_impl_list[] =
        {   /* binary */
            INSTANCE(cudnn_binary_t),
            /* softmax */
            INSTANCE(cudnn_softmax_fwd_t),
            INSTANCE(cudnn_softmax_bwd_t),
            /*eltwise*/
            INSTANCE(cudnn_eltwise_fwd_t),
            INSTANCE(cudnn_eltwise_bwd_t),
            /* batch normalization */
            INSTANCE(cudnn_batch_normalization_fwd_t),
            INSTANCE(cudnn_batch_normalization_bwd_t),
            /* inner product GEMM*/
            INSTANCE(cudnn_gemm_inner_product_fwd_t),
            INSTANCE(cudnn_gemm_inner_product_bwd_data_t),
            INSTANCE(cudnn_gemm_inner_product_bwd_weights_t),
            /* inner product CONV*/
            INSTANCE(cudnn_conv_inner_product_fwd_t),
            INSTANCE(cudnn_conv_inner_product_bwd_data_t),
            INSTANCE(cudnn_conv_inner_product_bwd_weights_t),
            /* Convolution */
            INSTANCE(cudnn_convolution_fwd_t),
            INSTANCE(cudnn_convolution_bwd_data_t),
            INSTANCE(cudnn_convolution_bwd_weights_t),
            /* pooling */
            INSTANCE(cudnn_pooling_fwd_t),
            INSTANCE(cudnn_pooling_bwd_t),
            /* lrn */
            INSTANCE(cudnn_lrn_fwd_t),
            INSTANCE(cudnn_lrn_bwd_t),
            /* resampling */
            INSTANCE(cudnn_resampling_fwd_t),
            INSTANCE(cudnn_resampling_bwd_t),
            /* Deconvolution */
            INSTANCE(cudnn_deconvolution_fwd_t),
            INSTANCE(cudnn_deconvolution_bwd_data_t),
            INSTANCE(cudnn_deconvolution_bwd_weights_t),
            /* Matmul */
            INSTANCE(cudnn_matmul_t),
            nullptr
        };
// clang-format on
#undef INSTANCE
} // namespace
const dnnl::impl::engine_t::primitive_desc_create_f *
sycl_cuda_engine_t::get_implementation_list(const op_desc_t *) const {
    return sycl_cuda_impl_list;
}

} // namespace cuda
} // namespace impl
} // namespace dnnl
