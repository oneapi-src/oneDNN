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

#include "sycl/sycl_device_info.hpp"
#include "gpu/intel/sycl/utils.hpp"

#include "gpu/intel/ocl/ocl_engine.hpp"
#include "gpu/intel/ocl/ocl_gpu_hw_info.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "gpu/intel/sycl/compat.hpp"
#include "gpu/sycl/sycl_gpu_engine.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_device_info_t::init_arch(impl::engine_t *engine) {
    auto *sycl_engine = utils::downcast<const sycl_engine_base_t *>(engine);
    auto &device = sycl_engine->device();

    // skip cpu engines
    if (!device.is_gpu()) return status::success;

    // skip other vendors
    if (!xpu::sycl::is_intel_device(device)) return status::success;

    auto be = xpu::sycl::get_backend(device);
    if (be == xpu::sycl::backend_t::opencl) {
        cl_int err = CL_SUCCESS;

        auto ocl_dev = xpu::sycl::compat::get_native<cl_device_id>(device);
        auto ocl_dev_wrapper = xpu::ocl::make_wrapper(ocl_dev);

        auto ocl_ctx_wrapper = xpu::ocl::make_wrapper(
                clCreateContext(nullptr, 1, &ocl_dev, nullptr, nullptr, &err));
        OCL_CHECK(err);

        gpu::intel::ocl::init_gpu_hw_info(engine, ocl_dev_wrapper,
                ocl_ctx_wrapper, ip_version_, gpu_arch_, stepping_id_,
                native_extensions_, mayiuse_systolic_, mayiuse_ngen_kernels_);
    } else if (be == xpu::sycl::backend_t::level0) {
        // TODO: add support for L0 binary ngen check
        // XXX: query from ocl_engine for now
        std::unique_ptr<gpu::intel::ocl::ocl_gpu_engine_t, engine_deleter_t>
                ocl_engine;
        CHECK(gpu::intel::sycl::create_ocl_engine(&ocl_engine, sycl_engine));

        auto *dev_info = ocl_engine->device_info();
        ip_version_ = dev_info->ip_version();
        gpu_arch_ = dev_info->gpu_arch();
        stepping_id_ = dev_info->stepping_id();
        mayiuse_systolic_ = dev_info->mayiuse_systolic();
        mayiuse_ngen_kernels_ = dev_info->mayiuse_ngen_kernels();
    } else {
        assert(!"not_expected");
    }

    return status::success;
}

status_t sycl_device_info_t::init_device_name(engine_t *engine) {
    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    name_ = device.get_info<::sycl::info::device::name>();
    return status::success;
}

status_t sycl_device_info_t::init_runtime_version(engine_t *engine) {
    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    auto driver_version
            = device.get_info<::sycl::info::device::driver_version>();

    if (runtime_version_.set_from_string(driver_version.c_str())
            != status::success) {
        runtime_version_.major = 0;
        runtime_version_.minor = 0;
        runtime_version_.build = 0;
    }

    return status::success;
}

status_t sycl_device_info_t::init_extensions(engine_t *engine) {
    using namespace gpu::intel::compute;

    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    extensions_ = gpu::intel::sycl::compat::init_extensions(device);

    // Handle future extensions, not yet supported by the DPC++ API
    extensions_
            |= (uint64_t)get_future_extensions(gpu_arch(), mayiuse_systolic());

    return status::success;
}

status_t sycl_device_info_t::init_attributes(engine_t *engine) {
    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    if (device.is_gpu() && xpu::sycl::is_intel_device(device)) {
        xpu::sycl::backend_t be = xpu::sycl::get_backend(device);
        if (be == xpu::sycl::backend_t::opencl) {
            // XXX: OpenCL backend get_info() queries below are not yet
            // supported so query OpenCL directly.
            cl_device_id ocl_dev
                    = xpu::sycl::compat::get_native<cl_device_id>(device);
            CHECK(gpu::intel::ocl::get_ocl_device_eu_count(
                    ocl_dev, gpu_arch_, &eu_count_));
        } else {
            auto slices = device.get_info<
                    xpu::sycl::compat::ext_intel_gpu_slices>();
            auto sub_slices = device.get_info<
                    xpu::sycl::compat::ext_intel_gpu_subslices_per_slice>();
            auto eus_per_subslice = device.get_info<::sycl::info::device::
                            ext_intel_gpu_eu_count_per_subslice>();
            if (gpu_arch_ == gpu::intel::compute::gpu_arch_t::xe2)
                eus_per_subslice
                        = 8; /* override incorrect driver information */
            eu_count_ = slices * sub_slices * eus_per_subslice;
        }
    } else {
        eu_count_ = device.get_info<::sycl::info::device::max_compute_units>();
    }
    max_wg_size_ = device.get_info<::sycl::info::device::max_work_group_size>();
    l3_cache_size_
            = device.get_info<::sycl::info::device::global_mem_cache_size>();
    mayiuse_system_memory_allocators_
            = device.has(::sycl::aspect::usm_system_allocations);
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
