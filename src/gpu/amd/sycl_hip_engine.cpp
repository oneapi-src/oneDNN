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
#include "hip/hip_runtime.h"

#include "miopen/miopen.h"
#include "sycl/sycl_utils.hpp"

#include "gpu/amd/miopen_batch_normalization.hpp"
#include "gpu/amd/miopen_binary.hpp"
#include "gpu/amd/miopen_convolution.hpp"
#include "gpu/amd/miopen_deconvolution.hpp"
#include "gpu/amd/miopen_eltwise.hpp"
#include "gpu/amd/miopen_gemm_inner_product.hpp"
#include "gpu/amd/miopen_lrn.hpp"
#include "gpu/amd/miopen_matmul.hpp"
#include "gpu/amd/miopen_pooling.hpp"
#include "gpu/amd/miopen_reduction.hpp"
#include "gpu/amd/miopen_softmax.hpp"
#include "gpu/amd/sycl_hip_compat.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

bool is_amd_gpu(const ::sycl::device &dev) {
    constexpr int amd_vendor_id = 0x1002;
    return dev.is_gpu()
            && dev.get_info<::sycl::info::device::vendor_id>() == amd_vendor_id;
}

status_t hip_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index) {
    CHECK(amd::check_device(engine_kind));
    std::unique_ptr<amd::sycl_hip_engine_t, engine_deleter_t> hip_engine(
            (new amd::sycl_hip_engine_t(dev, ctx, index)));
    if (!hip_engine) return status::out_of_memory;

    CHECK(hip_engine->init());
    *engine = hip_engine.release();

    return status::success;
}

sycl_hip_engine_t::sycl_hip_engine_t(engine_kind_t kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : base_t(kind, dev, ctx, index) {
    set_miopen_handle();
    set_rocblas_handle();
}

sycl_hip_engine_t::sycl_hip_engine_t(
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
    : sycl_hip_engine_t(engine_kind::gpu, dev, ctx, index) {
    assert(is_amd_gpu(dev));
}

status_t sycl_hip_engine_t::set_rocblas_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the rocblas handle.
    hip_sycl_scoped_context_handler_t sc(*this);
    rocblas_handle handle;
    CHECK(ROCBLAS_EXECUTE_FUNC_S(rocblas_create_handle, &handle));
    rocblas_handle_.set(
            std::unique_ptr<rocblas_handle, void (*)(rocblas_handle *)>(
                    new rocblas_handle(handle), [](rocblas_handle *h) {
                        if (h != nullptr)
                            ROCBLAS_EXECUTE_FUNC_V(rocblas_destroy_handle, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}

status_t sycl_hip_engine_t::set_miopen_handle() {
    // scoped context will make sure the top of the stack context is
    // the engine context while creating the miopen handle.
    hip_sycl_scoped_context_handler_t sc(*this);
    miopenHandle_t handle;
    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreate, &handle));
    miopen_handle_.set(
            std::unique_ptr<miopenHandle_t, void (*)(miopenHandle_t *)>(
                    new miopenHandle_t(handle), [](miopenHandle_t *h) {
                        if (h != nullptr)
                            MIOPEN_EXECUTE_FUNC_V(miopenDestroy, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}
hipCtx_t sycl_hip_engine_t::get_underlying_context() const {
    return compat::get_native<hipCtx_t>(context());
}

hipDevice_t sycl_hip_engine_t::get_underlying_device() const {
    return compat::get_native<hipDevice_t>(device());
}

status_t sycl_hip_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return sycl_hip_stream_t::create_stream(stream, this, flags);
}

status_t sycl_hip_engine_t::create_stream(
        stream_t **stream, ::sycl::queue &queue) {
    return sycl_hip_stream_t::create_stream(stream, this, queue);
}

miopenHandle_t *sycl_hip_engine_t::get_miopen_handle() {
    if (!miopen_handle_.is_set()) set_miopen_handle();
    return miopen_handle_.get().get();
}

rocblas_handle *sycl_hip_engine_t::get_rocblas_handle() {
    if (!rocblas_handle_.is_set()) set_rocblas_handle();
    return rocblas_handle_.get().get();
}

device_id_t sycl_hip_engine_t::device_id() const {
    return device_id_t(static_cast<int>(impl::sycl::backend_t::amd),
            static_cast<uint64_t>(compat::get_native<hipDevice_t>(device())),
            static_cast<uint64_t>(0));
}

void sycl_hip_engine_t::activate_stream_rocblas(HIPstream hip_stream) {
    hip_sycl_scoped_context_handler_t sc(*this);
    hipStream_t current_stream_id = nullptr;
    auto rocblas_handle = get_rocblas_handle();
    ROCBLAS_EXECUTE_FUNC(
            rocblas_get_stream, *rocblas_handle, &current_stream_id);
    if (current_stream_id != hip_stream) {
        ROCBLAS_EXECUTE_FUNC(rocblas_set_stream, *rocblas_handle, hip_stream);
    }
}

void sycl_hip_engine_t::activate_stream_miopen(HIPstream hip_stream) {
    hip_sycl_scoped_context_handler_t sc(*this);
    hipStream_t current_stream_id = nullptr;
    auto miopen_handle = get_miopen_handle();
    MIOPEN_EXECUTE_FUNC_S(miopenGetStream, *miopen_handle, &current_stream_id);
    if (current_stream_id != hip_stream) {
        MIOPEN_EXECUTE_FUNC_S(miopenSetStream, *miopen_handle, hip_stream);
    }
}

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
constexpr dnnl::impl::impl_list_item_t sycl_hip_impl_list[] = {
        // Binary
        INSTANCE(miopen_binary_t)
        // Elementwise
        INSTANCE(miopen_eltwise_fwd_t)
        INSTANCE(miopen_eltwise_bwd_t)
        // Softmax
        INSTANCE(miopen_softmax_fwd_t)
        INSTANCE(miopen_softmax_bwd_t)
        // LRN
        INSTANCE(miopen_lrn_fwd_t)
        INSTANCE(miopen_lrn_bwd_t)
        // Pooling
        INSTANCE(miopen_pooling_fwd_t)
        INSTANCE(miopen_pooling_bwd_t)
        // Reduction
        INSTANCE(miopen_reduction_t)
        // MatMul
        INSTANCE(miopen_matmul_t)
        // Inner Product
        INSTANCE(miopen_gemm_inner_product_fwd_t)
        INSTANCE(miopen_gemm_inner_product_bwd_data_t)
        INSTANCE(miopen_gemm_inner_product_bwd_weights_t)
        // Convolution
        INSTANCE(miopen_convolution_fwd_t)
        INSTANCE(miopen_convolution_bwd_data_t)
        INSTANCE(miopen_convolution_bwd_weights_t)
        // Batch Normalization
        INSTANCE(miopen_batch_normalization_fwd_t)
        INSTANCE(miopen_batch_normalization_bwd_t)
        // Deconvolution
        INSTANCE(miopen_deconvolution_fwd_t)
        INSTANCE(miopen_deconvolution_bwd_data_t)
        INSTANCE(miopen_deconvolution_bwd_weights_t)

        nullptr,
};
// clang-format on
} // namespace

const dnnl::impl::impl_list_item_t *sycl_hip_engine_t::get_implementation_list(
        const op_desc_t *) const {
    return sycl_hip_impl_list;
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
