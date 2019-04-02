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
struct cl_device_info_t {
public:
    cl_device_info_t(cl_device_id device) : device_(device), ext_(0) {}

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
        return err;
    }

    bool has(cl_device_ext_t ext) const { return ext_ & (uint64_t)ext; }

private:
    cl_device_id device_;
    uint64_t ext_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
