/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef OCL_DEVICE_INFO_HPP
#define OCL_DEVICE_INFO_HPP

#include <vector>
#include <CL/cl.h>

#include "compute/device_info.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

class ocl_device_info_t : public compute::device_info_t {
public:
    ocl_device_info_t(cl_device_id device)
        : device_(device)
        , ext_(0)
        , eu_count_(0)
        , hw_threads_(0)
        , runtime_version_ {0, 0, 0} {}

    virtual status_t init() override {
        // Extensions
        size_t size_ext {0};
        cl_int err = clGetDeviceInfo(
                device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &size_ext);
        OCL_CHECK(err);

        std::vector<char> c_ext(size_ext / sizeof(char));
        err = clGetDeviceInfo(
                device_, CL_DEVICE_EXTENSIONS, size_ext, &c_ext[0], &size_ext);
        OCL_CHECK(err);

        for (uint64_t i_ext = 1; i_ext < (uint64_t)compute::device_ext_t::last;
                i_ext <<= 1) {
            const char *s_ext = ext2cl_str((compute::device_ext_t)i_ext);
            if (s_ext != nullptr && strstr(&c_ext[0], s_ext) != nullptr) {
                ext_ |= i_ext;
            }
        }

        // Device name
        size_t size_name {0};
        err = clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, nullptr, &size_name);
        OCL_CHECK(err);

        std::vector<char> c_name(size_name / sizeof(char));
        err = clGetDeviceInfo(
                device_, CL_DEVICE_NAME, size_name, &c_name[0], &size_name);
        OCL_CHECK(err);

        // EU count
        cl_uint eu_count;
        err = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(cl_uint), &eu_count, nullptr);
        eu_count_ = (err == CL_SUCCESS) ? eu_count : 0;

        // Gen9 value, for now
        static constexpr auto threads_per_eu = 7;
        hw_threads_ = eu_count_ * threads_per_eu;

        // OpenCL runtime version
        size_t size_driver_version {0};
        err = clGetDeviceInfo(
                device_, CL_DRIVER_VERSION, 0, nullptr, &size_driver_version);
        OCL_CHECK(err);

        std::vector<char> c_driver_version(size_driver_version / sizeof(char));
        err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, size_driver_version,
                &c_driver_version[0], nullptr);
        OCL_CHECK(err);

        c_driver_version[size_driver_version - 1] = '\0';
        if (runtime_version_.set_from_string(&c_driver_version[0])
                != status::success) {
            runtime_version_.major = 0;
            runtime_version_.minor = 0;
            runtime_version_.build = 0;
        }

        return status::success;
    }

    virtual bool has(compute::device_ext_t ext) const override {
        return ext_ & (uint64_t)ext;
    }

    virtual int eu_count() const override { return eu_count_; }
    virtual int hw_threads() const override { return hw_threads_; }

    virtual const compute::runtime_version_t &runtime_version() const override {
        return runtime_version_;
    }

private:
    cl_device_id device_;
    uint64_t ext_;
    int32_t eu_count_, hw_threads_;
    compute::runtime_version_t runtime_version_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif // OCL_DEVICE_INFO_HPP
