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

#ifndef DNNL_TEST_COMMON_OCL_HPP
#define DNNL_TEST_COMMON_OCL_HPP

#include "dnnl.hpp"
#include "gtest/gtest.h"

#include <CL/cl.h>

#define OCL_CHECK(x) \
    do { \
        int s = int(x); \
        EXPECT_EQ(s, CL_SUCCESS) << "OpenCL error: " << s; \
    } while (0)

static inline cl_device_id find_ocl_device(cl_device_type dev_type) {
    cl_int err;
    const size_t max_platforms = 16;

    cl_uint nplatforms;
    cl_platform_id ocl_platforms[max_platforms];
    err = clGetPlatformIDs(max_platforms, ocl_platforms, &nplatforms);
    if (err != CL_SUCCESS) {
        // OpenCL has no support on the platform.
        return nullptr;
    }

    for (cl_uint i = 0; i < nplatforms; ++i) {
        cl_platform_id ocl_platform = ocl_platforms[i];

        const size_t max_platform_vendor_size = 256;
        std::string platform_vendor(max_platform_vendor_size + 1, 0);
        OCL_CHECK(clGetPlatformInfo(ocl_platform, CL_PLATFORM_VENDOR,
                max_platform_vendor_size * sizeof(char), &platform_vendor[0],
                nullptr));
        cl_uint ndevices;
        cl_device_id ocl_dev;
        err = clGetDeviceIDs(ocl_platform, dev_type, 1, &ocl_dev, &ndevices);
        if (err == CL_SUCCESS) { return ocl_dev; }
    }
    return nullptr;
}

#endif
