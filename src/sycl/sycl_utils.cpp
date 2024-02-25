/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#include "sycl/sycl_utils.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_engine_base.hpp"

#include "sycl/level_zero_utils.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>

#ifdef DNNL_SYCL_CUDA
// Do not include sycl_cuda_utils.hpp because it's intended for use in
// gpu/nvidia directory only.

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {
bool compare_cuda_devices(const ::sycl::device &lhs, const ::sycl::device &rhs);
}
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

#ifdef DNNL_SYCL_HIP
// Do not include sycl_cuda_utils.hpp because it's intended for use in
// gpu/amd directory only.
namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {
bool compare_hip_devices(const ::sycl::device &lhs, const ::sycl::device &rhs);
}
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
namespace dnnl {
namespace impl {
namespace sycl {

backend_t get_sycl_gpu_backend() {
    // Create default GPU device and query its backend (assumed as default)
    static backend_t default_backend = []() {
        const backend_t fallback = backend_t::opencl;

        const auto gpu_type = ::sycl::info::device_type::gpu;
        if (::sycl::device::get_devices(gpu_type).empty()) return fallback;

        ::sycl::device dev {compat::gpu_selector_v};
        backend_t backend = get_sycl_backend(dev);

        return backend;
    }();

    return default_backend;
}

bool are_equal(const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_be = get_sycl_backend(lhs);
    auto rhs_be = get_sycl_backend(rhs);
    if (lhs_be != rhs_be) return false;

    // Only one host device exists.
    if (lhs_be == backend_t::host) return true;

    if (lhs_be == backend_t::opencl) {
        // Use wrapper objects to avoid memory leak.
        auto lhs_ocl_handle = compat::get_native<cl_device_id>(lhs);
        auto rhs_ocl_handle = compat::get_native<cl_device_id>(rhs);
        return lhs_ocl_handle == rhs_ocl_handle;
    }

    if (lhs_be == backend_t::level0) { return compare_ze_devices(lhs, rhs); }

#ifdef DNNL_SYCL_CUDA
    if (lhs_be == backend_t::nvidia) {
        return gpu::nvidia::compare_cuda_devices(lhs, rhs);
    }
#endif

#ifdef DNNL_SYCL_HIP
    if (lhs_be == backend_t::amd) {
        return gpu::amd::compare_hip_devices(lhs, rhs);
    }
#endif
    assert(!"not expected");
    return false;
}

device_id_t sycl_device_id(const ::sycl::device &dev) {
    if (is_host(dev))
        return std::make_tuple(static_cast<int>(backend_t::host), 0, 0);

    device_id_t device_id
            = device_id_t {static_cast<int>(backend_t::unknown), 0, 0};
    switch (get_sycl_backend(dev)) {
        case backend_t::opencl: {
            auto ocl_device = gpu::ocl::make_ocl_wrapper(
                    compat::get_native<cl_device_id>(dev));
            device_id = std::make_tuple(static_cast<int>(backend_t::opencl),
                    reinterpret_cast<uint64_t>(ocl_device.get()), 0);
            break;
        }
        case backend_t::level0: {
            device_id = std::tuple_cat(
                    std::make_tuple(static_cast<int>(backend_t::level0)),
                    get_device_uuid(dev));
            break;
        }
        case backend_t::unknown: assert(!"unknown backend"); break;
        default: assert(!"unreachable");
    }
    assert(std::get<0>(device_id) != static_cast<int>(backend_t::unknown));
    return device_id;
}

bool dev_ctx_consistency_check(
        const ::sycl::device &dev, const ::sycl::context &ctx) {
    auto ctx_devs = ctx.get_devices();

    // Try to find the given device in the given context.
    auto it = std::find_if(ctx_devs.begin(), ctx_devs.end(),
            [&](const ::sycl::device &ctx_dev) {
                return are_equal(ctx_dev, dev);
            });
    // If found.
    if (it != ctx_devs.end()) return true;

    // If not found and the given device is not a sub-device.
    if (!is_subdevice(dev)) return false;

    // Try to find a parent device of the given sub-device in the given
    // context.
    while (is_subdevice(dev)) {
        auto parent_dev = get_parent_device(dev);
        it = std::find_if(ctx_devs.begin(), ctx_devs.end(),
                [&](const ::sycl::device &ctx_dev) {
                    return are_equal(ctx_dev, parent_dev);
                });
        // If found.
        if (it != ctx_devs.end()) return true;
    }

    return false;
}

status_t check_device(engine_kind_t eng_kind, const ::sycl::device &dev,
        const ::sycl::context &ctx) {
    // Check device and context consistency.
    if (!dev_ctx_consistency_check(dev, ctx)) return status::invalid_arguments;

    // Check engine kind and device consistency.
    if (eng_kind == engine_kind::cpu && !dev.is_cpu() && !is_host(dev))
        return status::invalid_arguments;
    if (eng_kind == engine_kind::gpu && !dev.is_gpu())
        return status::invalid_arguments;

#if !defined(DNNL_SYCL_CUDA) && !defined(DNNL_SYCL_HIP)
    // Check that platform is an Intel platform.
    if (!is_host(dev) && !is_intel_platform(dev.get_platform()))
        return status::invalid_arguments;
#endif

    return status::success;
}

struct uuid2ocl_dev_t {
    uuid2ocl_dev_t() = default;

    status_t add(gpu::compute::device_uuid_t uuid,
            const gpu::ocl::ocl_wrapper_t<cl_device_id> &d) {
        auto it = mapper_.insert(std::make_pair(uuid, d));
        if (!it.second) return status::runtime_error;
        return status::success;
    }

    cl_device_id get(gpu::compute::device_uuid_t uuid) const {
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
    using mapper_t = std::unordered_map<gpu::compute::device_uuid_t,
            gpu::ocl::ocl_wrapper_t<cl_device_id>,
            gpu::compute::device_uuid_hasher_t>;

    void release() {
        auto t = utils::make_unique<mapper_t>();
        std::swap(*t, mapper_);
        t.release();
    }
    mapper_t mapper_;
};

status_t sycl_dev2ocl_dev(cl_device_id *ocl_dev, const ::sycl::device &dev) {
#if !defined(cl_khr_device_uuid)
#error "cl_khr_device_uuid is required"
#endif
    using namespace gpu::compute;
    assert(get_sycl_backend(dev) == backend_t::level0);
    if (get_sycl_backend(dev) != backend_t::level0)
        return status::runtime_error;

    static const uuid2ocl_dev_t uuid2ocl_dev = []() {
        auto uuid2ocl_dev_tmp = uuid2ocl_dev_t();

        std::vector<cl_device_id> ocl_devices;
        std::vector<gpu::ocl::ocl_wrapper_t<cl_device_id>> ocl_sub_devices;
        auto st = gpu::ocl::get_ocl_devices(
                &ocl_devices, &ocl_sub_devices, CL_DEVICE_TYPE_GPU);
        assert(st == status::success);
        MAYBE_UNUSED(st);

        const auto register_ocl_dev
                = [&uuid2ocl_dev_tmp](
                          const gpu::ocl::ocl_wrapper_t<cl_device_id> &d) {
                      device_uuid_t ocl_dev_uuid;
                      auto st = gpu::ocl::get_device_uuid(ocl_dev_uuid, d);
                      assert(st == status::success);
                      st = uuid2ocl_dev_tmp.add(ocl_dev_uuid, d);
                      assert(st == status::success);
                      MAYBE_UNUSED(st);
                  };

        for (cl_device_id d : ocl_devices) {
            register_ocl_dev(gpu::ocl::make_ocl_wrapper(d));
        }
        for (const auto &sd_wrapper : ocl_sub_devices) {
            register_ocl_dev(sd_wrapper);
        }

        return uuid2ocl_dev_tmp;
    }();

    if (uuid2ocl_dev.empty()) return status::runtime_error;

    const device_uuid_t l0_dev_uuid = get_device_uuid(dev);
    auto d = uuid2ocl_dev.get(l0_dev_uuid);

    if (!d) return status::runtime_error;

    *ocl_dev = d;

    return status::success;
}

static status_t create_ocl_engine(
        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                *ocl_engine,
        const ::sycl::device &sycl_dev,
        const ::sycl::context *sycl_ctx = nullptr) {
    gpu::ocl::ocl_engine_factory_t f(engine_kind::gpu);
    const auto backend = get_sycl_backend(sycl_dev);

    // The SYCL context is always provided for OpenCL backend.
    if (backend == backend_t::opencl && !sycl_ctx) return status::runtime_error;
    gpu::ocl::ocl_wrapper_t<cl_device_id> ocl_dev;
    gpu::ocl::ocl_wrapper_t<cl_context> ocl_ctx;

    switch (backend) {
        case backend_t::opencl:
            ocl_dev = gpu::ocl::make_ocl_wrapper(
                    compat::get_native<cl_device_id>(sycl_dev));
            ocl_ctx = gpu::ocl::make_ocl_wrapper(
                    compat::get_native<cl_context>(*sycl_ctx));
            break;
        case backend_t::level0: {
            cl_device_id d {nullptr};
            CHECK(sycl_dev2ocl_dev(&d, sycl_dev));
            ocl_dev = gpu::ocl::make_ocl_wrapper(d, true);

            cl_int err;
            ocl_ctx = gpu::ocl::make_ocl_wrapper(
                    clCreateContext(nullptr, 1, &d, nullptr, nullptr, &err));
            OCL_CHECK(err);
            break;
        }
        default: assert(!"not expected"); return status::invalid_arguments;
    }
    engine_t *ocl_engine_ptr;
    size_t index;
    CHECK(gpu::ocl::get_ocl_device_index(&index, ocl_dev));
    CHECK(f.engine_create(&ocl_engine_ptr, ocl_dev, ocl_ctx, index));
    ocl_engine->reset(
            utils::downcast<gpu::ocl::ocl_gpu_engine_t *>(ocl_engine_ptr));
    return status::success;
}

status_t create_ocl_engine(
        std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                *ocl_engine,
        const sycl_engine_base_t *engine) {
    const auto sycl_ctx = engine->context();
    return create_ocl_engine(ocl_engine, engine->device(), &sycl_ctx);
}

status_t get_kernel_binary(
        const ::sycl::kernel &kernel, gpu::compute::binary_t &binary) {
    auto devs = kernel.get_context().get_devices();
    assert(!devs.empty());
    switch (get_sycl_backend(devs[0])) {
        case backend_t::level0: {
            auto bundle = kernel.get_kernel_bundle();
            auto module_vec = ::sycl::get_native<
                    ::sycl::backend::ext_oneapi_level_zero>(bundle);
            auto module = module_vec[0];
            size_t module_binary_size;
            gpu::compute::binary_t module_binary;
            CHECK(func_zeModuleGetNativeBinary(
                    module, &module_binary_size, nullptr));
            module_binary.resize(module_binary_size);
            CHECK(func_zeModuleGetNativeBinary(
                    module, &module_binary_size, module_binary.data()));
            {
                std::unique_ptr<gpu::ocl::ocl_gpu_engine_t, engine_deleter_t>
                        ocl_engine;
                CHECK(create_ocl_engine(&ocl_engine, devs[0]));
                gpu::ocl::ocl_wrapper_t<cl_program> ocl_program;
                CHECK(gpu::ocl::create_ocl_program(ocl_program,
                        ocl_engine->device(), ocl_engine->context(),
                        module_binary));

                cl_int err;
                auto name = kernel.get_info<
                        ::sycl::info::kernel::function_name>();
                auto ocl_kernel = gpu::ocl::make_ocl_wrapper(
                        clCreateKernel(ocl_program, name.c_str(), &err));
                OCL_CHECK(err);
                CHECK(gpu::ocl::get_ocl_kernel_binary(ocl_kernel, binary));
            }
            return status::success;
        }
        case backend_t::opencl: {
            auto ocl_kernel
                    = ::sycl::get_native<::sycl::backend::opencl>(kernel);
            CHECK(gpu::ocl::get_ocl_kernel_binary(ocl_kernel, binary));
            return status::success;
        }
        default: return status::runtime_error;
    }
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
