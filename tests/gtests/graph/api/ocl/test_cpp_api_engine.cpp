/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <vector>
#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_ocl.hpp"

#include "api/test_api_common.hpp"
#include "test_allocator.hpp"

using namespace dnnl::graph;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#define GRAPH_TEST_OCL_CHECK(x) \
    do { \
        cl_int s = (x); \
        if (s != CL_SUCCESS) { \
            std::cout << "[" << __FILE__ << ":" << __LINE__ << "] '" << #x \
                      << "' failed (status code: " << s << ")." << std::endl; \
            exit(1); \
        } \
    } while (0)

static void *ocl_malloc_shared(
        size_t size, size_t alignment, cl_device_id dev, cl_context ctx) {
    using F = void *(*)(cl_context, cl_device_id, cl_ulong *, size_t, cl_uint,
            cl_int *);
    if (size == 0) return nullptr;

    cl_platform_id platform;
    GRAPH_TEST_OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const char *f_name = "clSharedMemAllocINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    cl_int err;
    void *p = f(ctx, dev, nullptr, size, static_cast<cl_uint>(alignment), &err);
    GRAPH_TEST_OCL_CHECK(err);
    return p;
}

static void ocl_free(
        void *ptr, cl_device_id dev, const cl_context ctx, cl_event event) {
    if (event != nullptr) clWaitForEvents(1, &event);
    if (nullptr == ptr) return;
    using F = cl_int (*)(cl_context, void *);
    cl_platform_id platform;
    GRAPH_TEST_OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const char *f_name = "clMemFreeINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    GRAPH_TEST_OCL_CHECK(f(ctx, ptr));
}

TEST(OCLApi, Engine) {
    dnnl::engine::kind ekind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
    SKIP_IF(ekind != dnnl::engine::kind::gpu,
            "skip ocl api test for non-gpu engine.");
    cl_uint num_platforms = 0;
    GRAPH_TEST_OCL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    if (num_platforms > 0) {
        GRAPH_TEST_OCL_CHECK(
                clGetPlatformIDs(num_platforms, platforms.data(), NULL));
    } else {
        throw "Cannot find openCL platform!";
    }

    std::vector<cl_device_id> gpu_device_ids;
    for (cl_platform_id &platform_id : platforms) {
        cl_uint num_devices;
        if (!clGetDeviceIDs(
                    platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices)) {
            std::vector<cl_device_id> device_ids(num_devices);
            GRAPH_TEST_OCL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU,
                    num_devices, device_ids.data(), NULL));
            gpu_device_ids.insert(
                    gpu_device_ids.end(), device_ids.begin(), device_ids.end());
        }
    }
    if (gpu_device_ids.empty()) { throw "Cannot find OpenCL device!"; }

    cl_device_id device_id = gpu_device_ids[0]; // select a device
    cl_int err = 0;
    auto ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    GRAPH_TEST_OCL_CHECK(err);

    EXPECT_NO_THROW({
        auto alloc = ocl_interop::make_allocator(ocl_malloc_shared, ocl_free);
        auto eng = ocl_interop::make_engine_with_allocator(
                device_id, ctx, alloc);
    });

    auto alloc = ocl_interop::make_allocator(ocl_malloc_shared, ocl_free);
    {
        auto cache_blob
                = dnnl::ocl_interop::get_engine_cache_blob_id(device_id);
        EXPECT_NO_THROW({
            ocl_interop::make_engine_with_allocator(
                    device_id, ctx, alloc, cache_blob);
        });
    }

    {
        auto eng = dnnl::ocl_interop::make_engine(device_id, ctx);
        auto cache_blob = dnnl::ocl_interop::get_engine_cache_blob(eng);
        EXPECT_NO_THROW({
            ocl_interop::make_engine_with_allocator(
                    device_id, ctx, alloc, cache_blob);
        });
    }
}
#endif
