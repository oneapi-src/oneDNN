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

#ifndef SYCL_CUDA_ENGINE_HPP
#define SYCL_CUDA_ENGINE_HPP

#include <cudnn.h>
#include <cublas_v2.h>

#include <CL/sycl.hpp>

#include "common/stream.hpp"
#include "nvidia/sycl_cuda_utils.hpp"
#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_engine_base.hpp"

namespace dnnl {
namespace impl {
namespace cuda {

class cuda_gpu_engine_impl_list_t {
public:
    static const dnnl::impl::engine_t::reorder_primitive_desc_create_f *
    get_reorder_implementation_list(
            const memory_desc_t *src_md, const memory_desc_t *dst_md);
    static const dnnl::impl::engine_t::concat_primitive_desc_create_f *
    get_concat_implementation_list();
    static const dnnl::impl::engine_t::sum_primitive_desc_create_f *
    get_sum_implementation_list();
}; // namespace cuda

class sycl_cuda_engine_t : public dnnl::impl::sycl::sycl_engine_base_t {
public:
    using base_t = dnnl::impl::sycl::sycl_engine_base_t;
    sycl_cuda_engine_t(engine_kind_t kind, const cl::sycl::device &dev,
            const cl::sycl::context &ctx)
        : base_t(kind, dev, ctx) {
        underlying_context_type();
        set_cudnn_handle();
        set_cublas_handle();
    }

    sycl_cuda_engine_t(
            const cl::sycl::device &dev, const cl::sycl::context &ctx)
        : sycl_cuda_engine_t(engine_kind::gpu, dev, ctx) {
        assert(dev.is_gpu());
        constexpr int nvidia_vendor_id = 0x10DE;
        assert(dev.get_info<cl::sycl::info::device::vendor_id>()
                == nvidia_vendor_id);
    }

    virtual status_t create_stream(
            stream_t **stream, unsigned flags, const stream_attr_t *) override;
    status_t create_stream(stream_t **stream, cl::sycl::queue &queue);

    virtual const dnnl::impl::engine_t::reorder_primitive_desc_create_f *
    get_reorder_implementation_list(const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return cuda_gpu_engine_impl_list_t::get_reorder_implementation_list(
                src_md, dst_md);
    }

    virtual const dnnl::impl::engine_t::concat_primitive_desc_create_f *
    get_concat_implementation_list() const override {
        return cuda_gpu_engine_impl_list_t::get_concat_implementation_list();
    }

    virtual const dnnl::impl::engine_t::sum_primitive_desc_create_f *
    get_sum_implementation_list() const override {
        return cuda_gpu_engine_impl_list_t::get_sum_implementation_list();
    }

    virtual const primitive_desc_create_f *get_implementation_list(
            const op_desc_t *) const override;
    CUcontext get_underlying_context() const;
    cudnnHandle_t *get_cudnn_handle() const { return cudnn_handle_.get(); }
    cublasHandle_t *get_cublas_handle() const { return cublas_handle_.get(); }
    const bool has_primary_context() const { return primary_context_; }
    virtual device_id_t device_id() const override;

private:
    // This functions sets the context type. Since cuda requires different
    // approach in retaining/releasing primary/non-primary context.
    status_t underlying_context_type();
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
    std::unique_ptr<cudnnHandle_t, std::function<void(cudnnHandle_t *)>>
            cudnn_handle_ {nullptr, [](cudnnHandle_t *h) {
                               if (h != nullptr) {
                                   CUDNN_EXECUTE_FUNC_V(cudnnDestroy, *h);
                                   h = nullptr;
                               }
                           }};
    std::unique_ptr<cublasHandle_t, std::function<void(cublasHandle_t *)>>
            cublas_handle_ {nullptr, [](cublasHandle_t *h) {
                                if (h != nullptr) {
                                    CUBLAS_EXECUTE_FUNC_V(cublasDestroy, *h);
                                    h = nullptr;
                                }
                            }};
    bool primary_context_;
};

} // namespace cuda
} // namespace impl
} // namespace dnnl

#endif // SYCL_CUDA_ENGINE_HPP
