/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_NVIDIA_SYCL_CUDA_ENGINE_HPP
#define GPU_NVIDIA_SYCL_CUDA_ENGINE_HPP

#include <cudnn.h>
#include <cublas_v2.h>

#include "common/stream.hpp"
#include "common/thread_local_storage.hpp"
#include "gpu/gpu_impl_list.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"
#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_engine_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

class sycl_cuda_engine_t : public dnnl::impl::sycl::sycl_engine_base_t {
public:
    using base_t = dnnl::impl::sycl::sycl_engine_base_t;

    sycl_cuda_engine_t(engine_kind_t kind, const ::sycl::device &dev,
            const ::sycl::context &ctx, size_t index);
    sycl_cuda_engine_t(const ::sycl::device &dev, const ::sycl::context &ctx,
            size_t index);

    status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, ::sycl::queue &queue);

    void activate_stream_cudnn(CUstream cuda_stream);
    void activate_stream_cublas(CUstream cuda_stream);

    const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return gpu::gpu_impl_list_t::get_reorder_implementation_list(
                src_md, dst_md);
    }

    const impl_list_item_t *get_concat_implementation_list() const override {
        return gpu::gpu_impl_list_t::get_concat_implementation_list();
    }

    const impl_list_item_t *get_sum_implementation_list() const override {
        return gpu::gpu_impl_list_t::get_sum_implementation_list();
    }

    const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc) const override {
        return gpu::gpu_impl_list_t::get_implementation_list(desc);
    }

    CUcontext get_underlying_context() const;
    CUdevice get_underlying_device() const;
    cudnnHandle_t *get_cudnn_handle();
    cublasHandle_t *get_cublas_handle();
    device_id_t device_id() const override;

private:
    status_t set_cudnn_handle();
    status_t set_cublas_handle();
    // To avoid performance penalty cudnn/cublas required to have one handle per
    // thread per context therefor the handles will be the properties of the
    // engine. an engine can be assigned to multiple streams: lets say engine
    // eng(kind, 0); stream str1(eng,...); stream str2(eng,...); stream
    // str3(eng,...); In multi-threading environment both engin and stream
    // should be created in a different thread in order to allow safe
    // multi-threading programming If all the streams belongs to one thread, the
    // same handle will be used for all. Creation of handle is expensive and
    // must be avoided when it is not necessary.
    utils::thread_local_storage_t<
            std::unique_ptr<cudnnHandle_t, void (*)(cudnnHandle_t *)>>
            cudnn_handle_;
    utils::thread_local_storage_t<
            std::unique_ptr<cublasHandle_t, void (*)(cublasHandle_t *)>>
            cublas_handle_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
