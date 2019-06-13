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

#ifndef CL_DEVICE_INFO_HPP
#define CL_DEVICE_INFO_HPP

#include <CL/cl.h>
#include <stdint.h>
#include <string.h>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/z_magic.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

enum class cl_device_ext_t {
    khr_fp16 = 1 << 0,
    intel_subgroups = 1 << 1,
    intel_subgroups_short = 1 << 2,
    last
};

static const char *ext2str(cl_device_ext_t ext) {
#define CASE(x) \
    case cl_device_ext_t::x: return STRINGIFY(CONCAT2(cl_, x));
    switch (ext) {
        CASE(khr_fp16);
        CASE(intel_subgroups);
        CASE(intel_subgroups_short);
    default: return nullptr;
    }
#undef CASE
}

struct runtime_version_t {
    int major;
    int minor;
    int build;

    friend bool operator==(
            const runtime_version_t &v1, const runtime_version_t &v2) {
        return (v1.major == v2.major) && (v1.minor == v2.minor)
                && (v1.build == v2.build);
    }

    friend bool operator!=(
            const runtime_version_t &v1, const runtime_version_t &v2) {
        return !(v1 == v2);
    }

    friend bool operator<(
            const runtime_version_t &v1, const runtime_version_t &v2) {
        if (v1.major < v2.major)
            return true;
        if (v1.major > v2.major)
            return false;
        if (v1.minor < v2.minor)
            return true;
        if (v1.minor > v2.minor)
            return false;
        return (v1.build < v2.build);
    }

    friend bool operator>(
            const runtime_version_t &v1, const runtime_version_t &v2) {
        return (v2 < v1);
    }

    friend bool operator<=(
            const runtime_version_t &v1, const runtime_version_t &v2) {
        return !(v1 > v2);
    }

    friend bool operator>=(
            const runtime_version_t &v1, const runtime_version_t &v2) {
        return !(v1 < v2);
    }

    status_t set_from_string(const char *s) {
        int i_major = 0, i = 0;

        for (; s[i] != '.'; i++)
            if (!s[i])
                return status::invalid_arguments;

        auto i_minor = ++i;

        for (; s[i] != '.'; i++)
            if (!s[i])
                return status::invalid_arguments;

        auto i_build = ++i;

        major = atoi(&s[i_major]);
        minor = atoi(&s[i_minor]);
        build = atoi(&s[i_build]);

        return status::success;
    }
};

struct cl_device_info_t {
public:
    cl_device_info_t(cl_device_id device)
        : device_(device)
        , ext_(0)
        , eu_count_(0)
        , hw_threads_(0)
        , runtime_version_{ 0, 0, 0 } {}

    cl_int init() {
        // Extensions.
        size_t size_ext{ 0 };
        cl_int err = clGetDeviceInfo(
                device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &size_ext);
        if (err != CL_SUCCESS)
            return err;

        std::vector<char> c_ext(size_ext / sizeof(char));
        err = clGetDeviceInfo(
                device_, CL_DEVICE_EXTENSIONS, size_ext, &c_ext[0], &size_ext);
        if (err != CL_SUCCESS) {
            return err;
        }

        for (uint64_t i_ext = 1; i_ext < (uint64_t)cl_device_ext_t::last;
                i_ext <<= 1) {
            const char *s_ext = ext2str((cl_device_ext_t)i_ext);
            if (s_ext != nullptr && strstr(&c_ext[0], s_ext) != nullptr) {
                ext_ |= i_ext;
            }
        }

        // Device name.
        size_t size_name{ 0 };
        err = clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, nullptr, &size_name);
        if (err != CL_SUCCESS)
            return err;

        std::vector<char> c_name(size_name / sizeof(char));
        err = clGetDeviceInfo(
                device_, CL_DEVICE_NAME, size_name, &c_name[0], &size_name);
        if (err != CL_SUCCESS)
            return err;

        // EU count.
        cl_uint eu_count;
        err = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(cl_uint), &eu_count, NULL);
        eu_count_ = (err == CL_SUCCESS) ? eu_count : 0;

        static constexpr auto threads_per_eu = 7; /* Gen9 value, for now */
        hw_threads_ = eu_count_ * threads_per_eu;

        // OpenCL runtime version
        size_t size_driver_version{ 0 };
        err = clGetDeviceInfo(
                device_, CL_DRIVER_VERSION, 0, nullptr, &size_driver_version);
        if (err != CL_SUCCESS)
            return err;

        std::vector<char> c_driver_version(size_driver_version / sizeof(char));
        err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, size_driver_version,
                &c_driver_version[0], NULL);
        if (err != CL_SUCCESS)
            return err;

        c_driver_version[size_driver_version - 1] = '\0';
        if (runtime_version_.set_from_string(&c_driver_version[0])
                != status::success) {
            runtime_version_.major = 0;
            runtime_version_.minor = 0;
            runtime_version_.build = 0;
        }

        return CL_SUCCESS;
    }

    bool has(cl_device_ext_t ext) const { return ext_ & (uint64_t)ext; }

    int eu_count() const { return eu_count_; }
    int hw_threads() const { return hw_threads_; }

    const runtime_version_t &runtime_version() const {
        return runtime_version_;
    }

private:
    cl_device_id device_;
    uint64_t ext_;
    int32_t eu_count_, hw_threads_;
    runtime_version_t runtime_version_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
