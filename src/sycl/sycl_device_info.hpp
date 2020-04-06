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

#ifndef SYCL_DEVICE_INFO_HPP
#define SYCL_DEVICE_INFO_HPP

#include <vector>
#include <CL/sycl.hpp>

#include "cpu/cpu_isa_traits.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_device_info_t : public gpu::compute::device_info_t {
public:
    sycl_device_info_t(const cl::sycl::device &device)
        : device_(device), ext_(0), eu_count_(0), hw_threads_(0) {}

    virtual status_t init() override {
        // Extensions
        for (uint64_t i_ext = 1;
                i_ext < (uint64_t)gpu::compute::device_ext_t::last;
                i_ext <<= 1) {
            const char *s_ext = ext2cl_str((gpu::compute::device_ext_t)i_ext);
            if (s_ext != nullptr && device_.has_extension(s_ext)) {
                ext_ |= i_ext;
            }
        }

        // Device name
        auto dev_name = device_.get_info<cl::sycl::info::device::name>();
        set_name(dev_name);

        // EU count
        eu_count_
                = device_.get_info<cl::sycl::info::device::max_compute_units>();

        // Gen9 value, for GPU, for now
        int threads_per_eu = (device_.is_gpu() ? 7 : 1);
        hw_threads_ = eu_count_ * threads_per_eu;

        // Integrated GPUs share LLC with CPU which is L3 cache on CPU.
        size_t cache_size
                = cpu::get_per_core_cache_size(3) * cpu::get_num_cores();
        llc_cache_size_ = (size_t)cache_size;

        // Runtime version
        auto driver_version
                = device_.get_info<cl::sycl::info::device::driver_version>();
        gpu::compute::runtime_version_t runtime_version;
        if (runtime_version.set_from_string(driver_version.c_str())
                != status::success) {
            runtime_version.major = 0;
            runtime_version.minor = 0;
            runtime_version.build = 0;
        }
        set_runtime_version(runtime_version);

        return status::success;
    }

    virtual bool has(gpu::compute::device_ext_t ext) const override {
        return ext_ & (uint64_t)ext;
    }

    virtual int eu_count() const override { return eu_count_; }
    virtual int hw_threads() const override { return hw_threads_; }
    virtual size_t llc_cache_size() const override { return llc_cache_size_; }

private:
    cl::sycl::device device_;
    uint64_t ext_;
    int32_t eu_count_, hw_threads_;
    size_t llc_cache_size_ = 0;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_DEVICE_INFO_HPP
