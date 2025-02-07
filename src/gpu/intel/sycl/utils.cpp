/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "gpu/intel/sycl/utils.hpp"
#include "gpu/intel/sycl/engine.hpp"

#include "gpu/intel/sycl/l0/utils.hpp"

#include "xpu/ocl/engine_factory.hpp"
#include "xpu/ocl/utils.hpp"

#include "xpu/sycl/compat.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

::sycl::nd_range<3> to_sycl_nd_range(
        const gpu::intel::compute::nd_range_t &range) {
    const auto &local_range = range.local_range();
    const auto &global_range = range.global_range();

    assert(range.ndims() <= 3);
    auto sycl_global_range = ::sycl::range<3>(
            global_range.ndims() >= 3 ? global_range[2] : 1,
            global_range.ndims() >= 2 ? global_range[1] : 1, global_range[0]);

    if (!local_range) {
        assert(!"not expected");
        return ::sycl::nd_range<3>(
                sycl_global_range, ::sycl::range<3>(1, 1, 1));
    }

    auto sycl_local_range = ::sycl::range<3>(
            local_range.ndims() >= 3 ? local_range[2] : 1,
            local_range.ndims() >= 2 ? local_range[1] : 1, local_range[0]);
    return ::sycl::nd_range<3>(sycl_global_range, sycl_local_range);
}

struct uuid2ocl_dev_t {
    uuid2ocl_dev_t() = default;

    status_t add(xpu::device_uuid_t uuid,
            const xpu::ocl::wrapper_t<cl_device_id> &d) {
        auto it = mapper_.insert(std::make_pair(uuid, d));
        if (!it.second) return status::runtime_error;
        return status::success;
    }

    cl_device_id get(xpu::device_uuid_t uuid) const {
        auto it = mapper_.find(uuid);
        if (it == mapper_.end()) return nullptr;
        return it->second;
    }

    bool empty() const { return mapper_.empty(); }

    ~uuid2ocl_dev_t() {
        if (!is_destroying_cache_safe()) {
            release();
            return;
        }
    }

private:
    using mapper_t = std::unordered_map<xpu::device_uuid_t,
            xpu::ocl::wrapper_t<cl_device_id>, xpu::device_uuid_hasher_t>;

    void release() {
        auto t = utils::make_unique<mapper_t>();
        std::swap(*t, mapper_);
        // This explicitly leaks memory so that the cache doesn't get destroyed
        t.release();
    }
    mapper_t mapper_;
};

status_t sycl_dev2ocl_dev(cl_device_id *ocl_dev, const ::sycl::device &dev) {
#if !defined(cl_khr_device_uuid)
#error "cl_khr_device_uuid is required"
#endif
    using namespace gpu::intel::compute;
    assert(xpu::sycl::get_backend(dev) == xpu::sycl::backend_t::level0);
    if (xpu::sycl::get_backend(dev) != xpu::sycl::backend_t::level0)
        return status::runtime_error;

    static const uuid2ocl_dev_t uuid2ocl_dev = []() {
        auto uuid2ocl_dev_tmp = uuid2ocl_dev_t();

        std::vector<cl_device_id> ocl_devices;
        std::vector<xpu::ocl::wrapper_t<cl_device_id>> ocl_sub_devices;
        auto status = xpu::ocl::get_devices(
                &ocl_devices, &ocl_sub_devices, CL_DEVICE_TYPE_GPU);
        assert(status == status::success);
        MAYBE_UNUSED(status);

        const auto register_ocl_dev
                = [&uuid2ocl_dev_tmp](
                          const xpu::ocl::wrapper_t<cl_device_id> &d) {
                      xpu::device_uuid_t ocl_dev_uuid;
                      auto status = xpu::ocl::get_device_uuid(ocl_dev_uuid, d);
                      assert(status == status::success);
                      status = uuid2ocl_dev_tmp.add(std::move(ocl_dev_uuid), d);
                      assert(status == status::success);
                      MAYBE_UNUSED(status);
                  };

        for (cl_device_id d : ocl_devices) {
            register_ocl_dev(xpu::ocl::make_wrapper(d));
        }
        for (const auto &sd_wrapper : ocl_sub_devices) {
            register_ocl_dev(sd_wrapper);
        }

        return uuid2ocl_dev_tmp;
    }();

    if (uuid2ocl_dev.empty()) return status::runtime_error;

    const xpu::device_uuid_t l0_dev_uuid
            = gpu::intel::sycl::get_device_uuid(dev);
    auto d = uuid2ocl_dev.get(l0_dev_uuid);

    if (!d) return status::runtime_error;

    *ocl_dev = d;

    return status::success;
}

static status_t create_ocl_engine(
        std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t>
                *ocl_engine,
        const ::sycl::device &sycl_dev,
        const ::sycl::context *sycl_ctx = nullptr) {
    xpu::ocl::engine_factory_t f(engine_kind::gpu);
    const auto backend = xpu::sycl::get_backend(sycl_dev);

    // The SYCL context is always provided for OpenCL backend.
    if (backend == xpu::sycl::backend_t::opencl && !sycl_ctx)
        return status::runtime_error;
    xpu::ocl::wrapper_t<cl_device_id> ocl_dev;
    xpu::ocl::wrapper_t<cl_context> ocl_ctx;

    switch (backend) {
        case xpu::sycl::backend_t::opencl:
            ocl_dev = xpu::ocl::make_wrapper(
                    xpu::sycl::compat::get_native<cl_device_id>(sycl_dev));
            ocl_ctx = xpu::ocl::make_wrapper(
                    xpu::sycl::compat::get_native<cl_context>(*sycl_ctx));
            break;
        case xpu::sycl::backend_t::level0: {
            cl_device_id d {nullptr};
            CHECK(sycl_dev2ocl_dev(&d, sycl_dev));
            ocl_dev = xpu::ocl::make_wrapper(d, true);

            cl_int err;
            ocl_ctx = xpu::ocl::make_wrapper(
                    clCreateContext(nullptr, 1, &d, nullptr, nullptr, &err));
            OCL_CHECK(err);
            break;
        }
        default: assert(!"not expected"); return status::invalid_arguments;
    }
    impl::engine_t *ocl_engine_ptr;
    size_t index;
    CHECK(xpu::ocl::get_device_index(&index, ocl_dev));
    CHECK(f.engine_create(&ocl_engine_ptr, ocl_dev, ocl_ctx, index));
    ocl_engine->reset(
            utils::downcast<gpu::intel::ocl::engine_t *>(ocl_engine_ptr));
    return status::success;
}

status_t create_ocl_engine(
        std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t>
                *ocl_engine,
        const gpu::intel::sycl::engine_t *engine) {
    const auto sycl_ctx = engine->context();
    return create_ocl_engine(ocl_engine, engine->device(), &sycl_ctx);
}

status_t get_kernel_binary(
        const ::sycl::kernel &kernel, xpu::binary_t &binary) {
    auto devs = kernel.get_context().get_devices();
    assert(!devs.empty());
    switch (xpu::sycl::get_backend(devs[0])) {
        case xpu::sycl::backend_t::level0: {
            auto bundle = kernel.get_kernel_bundle();
            auto module_vec = ::sycl::get_native<
                    ::sycl::backend::ext_oneapi_level_zero>(bundle);
            auto module = module_vec[0];
            size_t module_binary_size;
            xpu::binary_t module_binary;
            CHECK(gpu::intel::sycl::func_zeModuleGetNativeBinary(
                    module, &module_binary_size, nullptr));
            module_binary.resize(module_binary_size);
            CHECK(gpu::intel::sycl::func_zeModuleGetNativeBinary(
                    module, &module_binary_size, module_binary.data()));
            {
                std::unique_ptr<gpu::intel::ocl::engine_t, engine_deleter_t>
                        ocl_engine;
                CHECK(create_ocl_engine(&ocl_engine, devs[0]));
                xpu::ocl::wrapper_t<cl_program> ocl_program;
                CHECK(xpu::ocl::create_program(ocl_program,
                        ocl_engine->device(), ocl_engine->context(),
                        module_binary));

                cl_int err;
                auto name = kernel.get_info<
                        ::sycl::info::kernel::function_name>();
                auto ocl_kernel = xpu::ocl::make_wrapper(
                        clCreateKernel(ocl_program, name.c_str(), &err));
                OCL_CHECK(err);
                CHECK(gpu::intel::ocl::get_ocl_kernel_binary(
                        ocl_kernel, binary));
            }
            return status::success;
        }
        case xpu::sycl::backend_t::opencl: {
            auto ocl_kernel
                    = ::sycl::get_native<::sycl::backend::opencl>(kernel);
            CHECK(gpu::intel::ocl::get_ocl_kernel_binary(ocl_kernel, binary));
            return status::success;
        }
        default: return status::runtime_error;
    }
}

gpu_utils::device_id_t device_id(const ::sycl::device &dev) {
    if (xpu::sycl::is_host(dev))
        return std::make_tuple(
                static_cast<int>(xpu::sycl::backend_t::host), 0, 0);

    gpu_utils::device_id_t device_id = gpu_utils::device_id_t {
            static_cast<int>(xpu::sycl::backend_t::unknown), 0, 0};
    switch (xpu::sycl::get_backend(dev)) {
        case xpu::sycl::backend_t::opencl: {
            auto ocl_device = xpu::ocl::make_wrapper(
                    xpu::sycl::compat::get_native<cl_device_id>(dev));
            device_id = std::make_tuple(
                    static_cast<int>(xpu::sycl::backend_t::opencl),
                    reinterpret_cast<uint64_t>(ocl_device.get()), 0);
            break;
        }
        case xpu::sycl::backend_t::level0: {
            device_id = std::tuple_cat(std::make_tuple(static_cast<int>(
                                               xpu::sycl::backend_t::level0)),
                    gpu::intel::sycl::get_device_uuid(dev));
            break;
        }
        case xpu::sycl::backend_t::unknown: assert(!"unknown backend"); break;
        default: assert(!"unreachable");
    }
    assert(std::get<0>(device_id)
            != static_cast<int>(xpu::sycl::backend_t::unknown));
    return device_id;
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
