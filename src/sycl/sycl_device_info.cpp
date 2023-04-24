/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_utils.hpp"

#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_gpu_hw_info.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/sycl/sycl_gpu_engine.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_device_info_t::init_arch(engine_t *engine) {
    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();

    // skip cpu engines
    if (!device.is_gpu()) return status::success;

    // skip other vendors
    if (!is_intel_device(device)) return status::success;

    backend_t be = get_sycl_backend(device);
    if (be == backend_t::opencl) {
        cl_int err = CL_SUCCESS;

        auto ocl_dev = compat::get_native<cl_device_id>(device);
        auto ocl_dev_wrapper = gpu::ocl::make_ocl_wrapper(ocl_dev);

        auto ocl_ctx_wrapper = gpu::ocl::make_ocl_wrapper(
                clCreateContext(nullptr, 1, &ocl_dev, nullptr, nullptr, &err));
        OCL_CHECK(err);

        gpu::ocl::init_gpu_hw_info(engine, ocl_dev_wrapper, ocl_ctx_wrapper,
                gpu_arch_, stepping_id_, mayiuse_systolic_,
                mayiuse_ngen_kernels_);
    } else if (be == backend_t::level0) {
        // TODO: add support for L0 binary ngen check
        // XXX: query from ocl_engine for now
        gpu::ocl::ocl_engine_factory_t f(engine_kind::gpu);

        engine_t *engine;
        CHECK(f.engine_create(&engine, 0));

        std::unique_ptr<gpu::compute::compute_engine_t, engine_deleter_t>
                compute_engine(
                        utils::downcast<gpu::compute::compute_engine_t *>(
                                engine));

        auto *dev_info = compute_engine->device_info();
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
    using namespace gpu::compute;

    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    extensions_ = compat::init_extensions(device);

    // Handle future extensions, not yet supported by the DPC++ API
    extensions_
            |= (uint64_t)get_future_extensions(gpu_arch(), mayiuse_systolic());

    return status::success;
}

status_t sycl_device_info_t::init_attributes(engine_t *engine) {
    auto &device
            = utils::downcast<const sycl_engine_base_t *>(engine)->device();
    if (device.is_gpu() && is_intel_device(device)) {
        backend_t be = get_sycl_backend(device);
        if (be == backend_t::opencl) {
            // XXX: OpenCL backend get_info() queries below are not yet
            // supported so query OpenCL directly.
            cl_device_id ocl_dev = compat::get_native<cl_device_id>(device);
            CHECK(gpu::ocl::get_ocl_device_eu_count(ocl_dev, &eu_count_));
        } else {
            auto slices = device.get_info<compat::ext_intel_gpu_slices>();
            auto sub_slices = device.get_info<
                    compat::ext_intel_gpu_subslices_per_slice>();
            auto eus_per_subslice = device.get_info<::sycl::info::device::
                            ext_intel_gpu_eu_count_per_subslice>();
            eu_count_ = slices * sub_slices * eus_per_subslice;
        }
    } else {
        eu_count_ = device.get_info<::sycl::info::device::max_compute_units>();
    }
    max_wg_size_ = device.get_info<::sycl::info::device::max_work_group_size>();
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
