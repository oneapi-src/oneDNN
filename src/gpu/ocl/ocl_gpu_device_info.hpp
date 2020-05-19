/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_OCL_OCL_GPU_DEVICE_INFO_HPP
#define GPU_OCL_OCL_GPU_DEVICE_INFO_HPP

#include <string>
#include <vector>
#include <CL/cl.h>

#include "common/z_magic.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static compute::device_ext_t get_extensions(compute::gpu_arch_t gpu_arch) {
    uint64_t extensions = 0;
    switch (gpu_arch) {
        case compute::gpu_arch_t::gen12lp:
            extensions |= (uint64_t)
                    compute::device_ext_t::intel_subgroup_local_block_io;
        case compute::gpu_arch_t::gen9:
            extensions |= (uint64_t)compute::device_ext_t::khr_fp16;
            extensions |= (uint64_t)compute::device_ext_t::intel_subgroups;
            extensions
                    |= (uint64_t)compute::device_ext_t::intel_subgroups_short;
            break;
        case compute::gpu_arch_t::unknown: break;
    }
    return (compute::device_ext_t)extensions;
}

class ocl_gpu_device_info_t : public compute::device_info_t {
public:
    ocl_gpu_device_info_t(cl_device_id device) : device_(device) {}

    status_t init() override {
        CHECK(init_arch());
        CHECK(init_extensions());
        CHECK(init_attributes());

        // Device name
        size_t size_name {0};
        cl_int err = clGetDeviceInfo(
                device_, CL_DEVICE_NAME, 0, nullptr, &size_name);
        OCL_CHECK(err);

        std::string dev_name;
        dev_name.resize(size_name);
        err = clGetDeviceInfo(
                device_, CL_DEVICE_NAME, size_name, &dev_name[0], &size_name);
        OCL_CHECK(err);
        set_name(dev_name);

        // OpenCL runtime version
        size_t size_driver_version {0};
        err = clGetDeviceInfo(
                device_, CL_DRIVER_VERSION, 0, nullptr, &size_driver_version);
        OCL_CHECK(err);
        std::string driver_version;
        driver_version.resize(size_driver_version);
        err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, size_driver_version,
                &driver_version[0], nullptr);
        OCL_CHECK(err);

        driver_version[size_driver_version - 1] = '\0';
        compute::runtime_version_t runtime_version;
        if (runtime_version.set_from_string(&driver_version[0])
                != status::success) {
            runtime_version.major = 0;
            runtime_version.minor = 0;
            runtime_version.build = 0;
        }
        set_runtime_version(runtime_version);

        CHECK(init_arch());
        CHECK(init_extensions());
        CHECK(init_attributes());

        return status::success;
    }

    bool has(compute::device_ext_t ext) const override {
        return has(extensions_, ext);
    }

    compute::gpu_arch_t gpu_arch() const override { return gpu_arch_; }

    int eu_count() const override { return eu_count_; }
    int hw_threads() const override { return hw_threads_; }
    size_t llc_cache_size() const override { return llc_cache_size_; }

private:
    status_t init_arch() {
        if (name().find("Gen9") != std::string::npos)
            gpu_arch_ = compute::gpu_arch_t::gen9;
        else if (name().find("Gen12LP") != std::string::npos)
            gpu_arch_ = compute::gpu_arch_t::gen12lp;
        else
            gpu_arch_ = compute::gpu_arch_t::unknown;

        return status::success;
    }

    status_t init_extensions() {
        // Handle extensions provided by the OpenCL runtime
        size_t size_ext {0};
        cl_int err = clGetDeviceInfo(
                device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &size_ext);
        OCL_CHECK(err);

        std::string dev_ext;
        dev_ext.resize(size_ext);

        err = clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, size_ext,
                &dev_ext[0], &size_ext);
        OCL_CHECK(err);

        for (uint64_t i_ext = 1; i_ext < (uint64_t)compute::device_ext_t::last;
                i_ext <<= 1) {
            const char *s_ext = ext2cl_str((compute::device_ext_t)i_ext);
            if (s_ext != nullptr && dev_ext.find(s_ext) != std::string::npos) {
                extensions_ |= i_ext;
            }
        }

        // Handle future extensions, not yet supported by the OpenCL API
        extensions_ |= (uint64_t)get_extensions(gpu_arch_);

        return status::success;
    }

    status_t init_attributes() {
        cl_uint eu_count;
        cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(cl_uint), &eu_count, nullptr);
        eu_count_ = (err == CL_SUCCESS) ? eu_count : 0;

        // Assume 7 threads by default
        int32_t threads_per_eu = 7;

        switch (gpu_arch_) {
            case compute::gpu_arch_t::gen9: threads_per_eu = 7; break;
            case compute::gpu_arch_t::gen12lp: threads_per_eu = 7; break;
            default: break;
        }

        hw_threads_ = eu_count_ * threads_per_eu;
        llc_cache_size_ = get_llc_cache_size();
        return status::success;
    }

    bool has(uint64_t extensions, compute::device_ext_t ext) const {
        return extensions & (uint64_t)ext;
    }

    size_t get_llc_cache_size() const;

    cl_device_id device_ = nullptr;

    int32_t eu_count_ = 0;
    int32_t hw_threads_ = 0;
    size_t llc_cache_size_ = 0;

    // extensions_ and gpu_arch_ describe effective extensions and GPU architecutre.
    uint64_t extensions_ = 0;

    compute::gpu_arch_t gpu_arch_ = compute::gpu_arch_t::unknown;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_OCL_GPU_DEVICE_INFO_HPP
