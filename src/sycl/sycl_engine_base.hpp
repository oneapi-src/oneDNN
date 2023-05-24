/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef SYCL_ENGINE_BASE_HPP
#define SYCL_ENGINE_BASE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory_storage.hpp"
#include "common/stream.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/sycl/sycl_interop_gpu_kernel.hpp"
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_engine_id.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_engine_base_t : public gpu::compute::compute_engine_t {
public:
    sycl_engine_base_t(engine_kind_t kind, const ::sycl::device &dev,
            const ::sycl::context &ctx, size_t index)
        : gpu::compute::compute_engine_t(kind, runtime_kind::sycl, index)
        , device_(dev)
        , context_(ctx)
        , backend_(backend_t::unknown) {}

    status_t init() override {
        backend_ = get_sycl_backend(device_);
        if (!utils::one_of(backend_, backend_t::host, backend_t::opencl,
                    backend_t::level0, backend_t::nvidia, backend_t::amd))
            return status::invalid_arguments;

        CHECK(check_device(kind(), device_, context_));
        CHECK(gpu::compute::compute_engine_t::init());

        return status::success;
    }

    status_t create_memory_storage(memory_storage_t **storage, unsigned flags,
            size_t size, void *handle) override;

    status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, ::sycl::queue &queue);

    status_t create_compiled_bundle(gpu::compute::compiled_bundle_t &generator,
            const std::vector<const char *> &kernel_names,
            const gpu::compute::kernel_ctx_t &kernel_ctx) const override {
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        auto status = create_ocl_engine(&ocl_engine);
        if (status != status::success) return status;
        return ocl_engine->create_compiled_bundle(
                generator, kernel_names, kernel_ctx);
    }

    status_t create_compiled_kernel(gpu::compute::compiled_kernel_t &generator,
            gpu::jit::jit_generator_base &jitter) const override {
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        auto status = create_ocl_engine(&ocl_engine);
        if (status != status::success) return status;
        return ocl_engine->create_compiled_kernel(generator, jitter);
    }

    status_t convert_to_sycl(std::vector<gpu::compute::kernel_t> &kernels,
            const std::vector<gpu::compute::kernel_t> &ocl_kernels,
            const std::vector<const char *> &kernel_names,
            gpu::ocl::ocl_gpu_engine_t *ocl_engine) const {
        kernels = std::vector<gpu::compute::kernel_t>(kernel_names.size());
        for (size_t i = 0; i < ocl_kernels.size(); ++i) {
            if (!ocl_kernels[i]) continue;
            auto *k = utils::downcast<gpu::ocl::ocl_gpu_kernel_t *>(
                    ocl_kernels[i].impl());
            gpu::compute::binary_t binary;
            CHECK(k->get_binary(ocl_engine, binary));
            std::unique_ptr<::sycl::kernel> sycl_kernel;
            CHECK(compat::make_kernel(
                    sycl_kernel, this, binary, kernel_names[i]));

            kernels[i] = gpu::compute::kernel_t(
                    new gpu::sycl::sycl_interop_gpu_kernel_t(
                            *sycl_kernel, k->arg_types()));
        }
        return status::success;
    }

    status_t create_kernels_from_bundle(
            std::vector<gpu::compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names,
            const gpu::compute::compiled_bundle_t &generator) const override {
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        auto status = create_ocl_engine(&ocl_engine);
        if (status != status::success) return status;

        std::vector<gpu::compute::kernel_t> ocl_kernels;
        ocl_engine->create_kernels_from_bundle(
                ocl_kernels, kernel_names, generator);

        CHECK(convert_to_sycl(
                kernels, ocl_kernels, kernel_names, ocl_engine.get()));
        return status::success;
    }

    status_t create_kernel_from_binary(gpu::compute::kernel_t &kernel,
            const gpu::compute::binary_t &binary,
            const char *kernel_name) const override {
        gpu::ocl::dump_kernel_binary(binary, kernel_name);

        std::vector<gpu::compute::scalar_type_t> arg_types;

        std::unique_ptr<::sycl::kernel> sycl_kernel;
        CHECK(compat::make_kernel(sycl_kernel, this, binary, kernel_name));

        kernel = gpu::compute::kernel_t(
                new gpu::sycl::sycl_interop_gpu_kernel_t(
                        *sycl_kernel, arg_types));
        return status::success;
    }

    status_t create_kernels_from_cache_blob(cache_blob_t cache_blob,
            std::vector<gpu::compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names) const override {
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        auto status = create_ocl_engine(&ocl_engine);
        if (status != status::success) return status;

        std::vector<gpu::compute::kernel_t> ocl_kernels;
        CHECK(ocl_engine->create_kernels_from_cache_blob(
                cache_blob, ocl_kernels, kernel_names));
        CHECK(convert_to_sycl(
                kernels, ocl_kernels, kernel_names, ocl_engine.get()));
        return status::success;
    }

    status_t create_kernel(gpu::compute::kernel_t *kernel,
            gpu::jit::jit_generator_base *jitter,
            cache_blob_t cache_blob) const override {

        UNUSED(cache_blob);
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        auto status = create_ocl_engine(&ocl_engine);
        if (status != status::success) return status;

        auto kernel_name = jitter->kernel_name();

        gpu::compute::binary_t binary = jitter->get_binary(
                ocl_engine->context(), ocl_engine->device());
        return create_kernel_from_binary(*kernel, binary, kernel_name);
    }

    status_t create_kernels(std::vector<gpu::compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const gpu::compute::kernel_ctx_t &kernel_ctx,
            cache_blob_t cache_blob) const override {
        UNUSED(cache_blob);
        if (kind() != engine_kind::gpu) {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        auto status = create_ocl_engine(&ocl_engine);
        if (status != status::success) return status;

        std::vector<gpu::compute::kernel_t> ocl_kernels;
        CHECK(ocl_engine->create_kernels(
                &ocl_kernels, kernel_names, kernel_ctx, cache_blob));
        CHECK(convert_to_sycl(
                *kernels, ocl_kernels, kernel_names, ocl_engine.get()));
        return status::success;
    }

    const ::sycl::device &device() const { return device_; }
    const ::sycl::context &context() const { return context_; }

    backend_t backend() const { return backend_; }

    cl_device_id ocl_device() const {
        if (backend() != backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device_.is_cpu() || device_.is_gpu());
        return gpu::ocl::make_ocl_wrapper(
                compat::get_native<cl_device_id>(device()));
    }
    cl_context ocl_context() const {
        if (backend() != backend_t::opencl) {
            assert(!"not expected");
            return nullptr;
        }
        assert(device_.is_cpu() || device_.is_gpu());
        return gpu::ocl::make_ocl_wrapper(
                compat::get_native<cl_context>(context()));
    }

    device_id_t device_id() const override { return sycl_device_id(device_); }

    engine_id_t engine_id() const override {
        return engine_id_t(new sycl_engine_id_impl_t(
                device(), context(), kind(), runtime_kind(), index()));
    }

protected:
    ~sycl_engine_base_t() override = default;
    status_t init_device_info() override;

private:
    ::sycl::device device_;
    ::sycl::context context_;

    backend_t backend_;

    status_t create_ocl_engine(
            std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                    *ocl_engine) const {
        gpu::ocl::ocl_engine_factory_t f(engine_kind::gpu);

        if (backend_ == backend_t::opencl) {
            engine_t *ocl_engine_ptr;
            size_t index;
            CHECK(gpu::ocl::get_ocl_device_index(&index, ocl_device()));
            CHECK(f.engine_create(
                    &ocl_engine_ptr, ocl_device(), ocl_context(), index));
            ocl_engine->reset(utils::downcast<gpu::ocl::ocl_gpu_engine_t *>(
                    ocl_engine_ptr));
        } else if (backend_ == backend_t::level0) {
            engine_t *ocl_engine_ptr;
            // FIXME: This does not work for multi-GPU systems. OpenCL engine
            // should be created based on the Level0 device to ensure that a
            // program is compiled for the same physical device. However,
            // OpenCL does not provide any API to match its devices with
            // Level0.
            CHECK(f.engine_create(&ocl_engine_ptr, 0));
            ocl_engine->reset(utils::downcast<gpu::ocl::ocl_gpu_engine_t *>(
                    ocl_engine_ptr));
        } else {
            assert(!"not expected");
            return status::invalid_arguments;
        }

        return status::success;
    }
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
