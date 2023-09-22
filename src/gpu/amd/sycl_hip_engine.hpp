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

#ifndef GPU_AMD_SYCL_HIP_ENGINE_HPP
#define GPU_AMD_SYCL_HIP_ENGINE_HPP

#include <stdexcept>
#include "common/stream.hpp"
#include "common/thread_local_storage.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
#include "miopen/miopen.h"
#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_engine_base.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

class hip_gpu_engine_impl_list_t {
public:
    static const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md, const memory_desc_t *dst_md);
    static const dnnl::impl::impl_list_item_t *
    get_concat_implementation_list() {
        static impl_list_item_t hip_concat_impl_list[] = {
                nullptr,
        };
        return hip_concat_impl_list;
    }
    static const dnnl::impl::impl_list_item_t *get_sum_implementation_list() {
        static impl_list_item_t hip_sum_impl_list[] = {
                nullptr,
        };
        return hip_sum_impl_list;
    }
};

class sycl_hip_engine_t : public dnnl::impl::sycl::sycl_engine_base_t {
public:
    using base_t = dnnl::impl::sycl::sycl_engine_base_t;

    sycl_hip_engine_t(engine_kind_t kind, const ::sycl::device &dev,
            const ::sycl::context &ctx, size_t index);
    sycl_hip_engine_t(const ::sycl::device &dev, const ::sycl::context &ctx,
            size_t index);

    status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, ::sycl::queue &queue);

    const dnnl::impl::impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return hip_gpu_engine_impl_list_t::get_reorder_implementation_list(
                src_md, dst_md);
    }

    const dnnl::impl::impl_list_item_t *
    get_concat_implementation_list() const override {
        return hip_gpu_engine_impl_list_t::get_concat_implementation_list();
    }

    const dnnl::impl::impl_list_item_t *
    get_sum_implementation_list() const override {
        return hip_gpu_engine_impl_list_t::get_sum_implementation_list();
    }

    void activate_stream_miopen(HIPstream hip_stream);
    void activate_stream_rocblas(HIPstream hip_stream);

    const impl_list_item_t *get_implementation_list(
            const op_desc_t *) const override;
    hipCtx_t get_underlying_context() const;
    hipDevice_t get_underlying_device() const;
    miopenHandle_t *get_miopen_handle();
    rocblas_handle *get_rocblas_handle();
    const bool has_primary_context() const { return primary_context_; }
    device_id_t device_id() const override;

protected:
    ~sycl_hip_engine_t() override = default;

private:
    status_t set_miopen_handle();
    status_t set_rocblas_handle();
    utils::thread_local_storage_t<
            std::unique_ptr<miopenHandle_t, void (*)(miopenHandle_t *)>>
            miopen_handle_;
    utils::thread_local_storage_t<
            std::unique_ptr<rocblas_handle, void (*)(rocblas_handle *)>>
            rocblas_handle_;
    bool primary_context_;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
