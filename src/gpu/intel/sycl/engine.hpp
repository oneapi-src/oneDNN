/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_SYCL_ENGINE_HPP
#define GPU_INTEL_SYCL_ENGINE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory_storage.hpp"

#include "xpu/ocl/utils.hpp"
#include "xpu/sycl/engine_impl.hpp"

#include "gpu/intel/compute/compute_engine.hpp"

#include "gpu/intel/ocl/ocl_gpu_engine.hpp"
#include "gpu/intel/ocl/ocl_gpu_kernel.hpp"

#include "gpu/intel/sycl/compat.hpp"
#include "gpu/intel/sycl/utils.hpp"

#include "gpu/intel/sycl/sycl_interop_gpu_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

status_t engine_create(impl::engine_t **engine, engine_kind_t engine_kind,
        const ::sycl::device &dev, const ::sycl::context &ctx, size_t index);

class engine_t : public gpu::intel::compute::compute_engine_t {
public:
    engine_t(
            const ::sycl::device &dev, const ::sycl::context &ctx, size_t index)
        : gpu::intel::compute::compute_engine_t(new xpu::sycl::engine_impl_t(
                engine_kind::gpu, dev, ctx, index)) {}

    status_t init() override {
        CHECK(init_impl());
        CHECK(gpu::intel::compute::compute_engine_t::init());

        return status::success;
    }

    status_t create_memory_storage(memory_storage_t **storage, unsigned flags,
            size_t size, void *handle) override;

    status_t create_stream(
            impl::stream_t **stream, impl::stream_impl_t *stream_impl) override;

    status_t convert_to_sycl(
            std::vector<gpu::intel::compute::kernel_t> &kernels,
            const std::vector<gpu::intel::compute::kernel_t> &ocl_kernels,
            const std::vector<const char *> &kernel_names,
            gpu::intel::ocl::ocl_gpu_engine_t *ocl_engine) const {
        kernels = std::vector<gpu::intel::compute::kernel_t>(
                kernel_names.size());
        for (size_t i = 0; i < ocl_kernels.size(); ++i) {
            if (!ocl_kernels[i]) continue;
            auto *k = utils::downcast<gpu::intel::ocl::ocl_gpu_kernel_t *>(
                    ocl_kernels[i].impl());
            xpu::binary_t binary;
            CHECK(k->get_binary(ocl_engine, binary));
            CHECK(create_kernel_from_binary(
                    kernels[i], binary, kernel_names[i]));
        }
        return status::success;
    }

    status_t create_kernel_from_binary(gpu::intel::compute::kernel_t &kernel,
            const xpu::binary_t &binary,
            const char *kernel_name) const override {
        std::vector<gpu::intel::compute::scalar_type_t> arg_types;

        std::unique_ptr<::sycl::kernel> sycl_kernel;
        CHECK(gpu::intel::sycl::compat::make_kernel(
                sycl_kernel, this, binary, kernel_name));

        std::shared_ptr<gpu::intel::compute::kernel_impl_t> kernel_impl
                = std::make_shared<
                        gpu::generic::sycl::sycl_interop_gpu_kernel_t>(
                        *sycl_kernel, arg_types);
        kernel = std::move(kernel_impl);
        return status::success;
    }

    status_t create_kernels_from_cache_blob(const cache_blob_t &cache_blob,
            std::vector<gpu::intel::compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const override {
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::intel::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        auto status = gpu::intel::sycl::create_ocl_engine(&ocl_engine, this);
        if (status != status::success) return status;

        std::vector<gpu::intel::compute::kernel_t> ocl_kernels;
        CHECK(ocl_engine->create_kernels_from_cache_blob(
                cache_blob, ocl_kernels, kernel_names));
        CHECK(convert_to_sycl(
                kernels, ocl_kernels, kernel_names, ocl_engine.get()));
        return status::success;
    }

    status_t create_kernel(gpu::intel::compute::kernel_t *kernel,
            gpu::intel::jit::jit_generator_base *jitter,
            const cache_blob_t &cache_blob) const override {

        UNUSED(cache_blob);
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::intel::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        CHECK(gpu::intel::sycl::create_ocl_engine(&ocl_engine, this));

        auto kernel_name = jitter->kernel_name();

        xpu::binary_t binary = jitter->get_binary(
                ocl_engine->context(), ocl_engine->device());
        return create_kernel_from_binary(*kernel, binary, kernel_name);
    }

    status_t create_kernels(std::vector<gpu::intel::compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const gpu::intel::compute::kernel_ctx_t &kernel_ctx,
            const cache_blob_t &cache_blob) const override {
        UNUSED(cache_blob);
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::intel::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        CHECK(gpu::intel::sycl::create_ocl_engine(&ocl_engine, this));

        std::vector<gpu::intel::compute::kernel_t> ocl_kernels;
        CHECK(ocl_engine->create_kernels(
                &ocl_kernels, kernel_names, kernel_ctx, cache_blob));
        CHECK(convert_to_sycl(
                *kernels, ocl_kernels, kernel_names, ocl_engine.get()));
        return status::success;
    }

    const ::sycl::device &device() const { return impl()->device(); }
    const ::sycl::context &context() const { return impl()->context(); }

    xpu::sycl::backend_t backend() const { return impl()->backend(); }

    cl_device_id ocl_device() const {
        if (backend() != xpu::sycl::backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device().is_cpu() || device().is_gpu());
        return xpu::ocl::make_wrapper(
                xpu::sycl::compat::get_native<cl_device_id>(device()));
    }

    cl_context ocl_context() const {
        if (backend() != xpu::sycl::backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device().is_cpu() || device().is_gpu());
        return xpu::ocl::make_wrapper(
                xpu::sycl::compat::get_native<cl_context>(context()));
    }

    gpu::intel::gpu_utils::device_id_t device_id() const override {
        return gpu::intel::sycl::device_id(device());
    }

protected:
    const xpu::sycl::engine_impl_t *impl() const {
        return (const xpu::sycl::engine_impl_t *)impl::engine_t::impl();
    }

    ~engine_t() override = default;
    status_t init_device_info() override;
};

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
