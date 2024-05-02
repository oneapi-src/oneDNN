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

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL

#include "graph/utils/ocl_usm_utils.hpp"

#include "common/cpp_compat.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
namespace ocl {
namespace {
template <typename F>
struct ext_func_t {
    ext_func_t(const char *name) {
        for (size_t i = 0; i < intel_platforms().size(); ++i) {
            auto p = intel_platforms()[i];
            auto f = reinterpret_cast<F>(
                    clGetExtensionFunctionAddressForPlatform(p, name));
            auto it = ext_func_ptrs_.insert({p, f});
            assert(it.second);
            MAYBE_UNUSED(it);
        }
    }

    template <typename... Args>
    typename cpp_compat::invoke_result<F, Args...>::type operator()(
            cl_platform_id platform, Args... args) const {
        if (!ext_func_ptrs_.count(platform)) {
            throw std::runtime_error("Don't support the platform");
        }
        auto f = ext_func_ptrs_.at(platform);
        return f(args...);
    }

private:
    static const std::vector<cl_platform_id> &intel_platforms() {
        static auto intel_platforms = get_intel_platforms();
        return intel_platforms;
    }

    static std::vector<cl_platform_id> get_intel_platforms() {
        cl_uint num_platforms = 0;
        cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> intel_platforms;
        char vendor_name[128] = {};
        for (cl_platform_id p : platforms) {
            err = clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor_name),
                    vendor_name, nullptr);
            if (err != CL_SUCCESS) continue;
            if (std::string(vendor_name).find("Intel") != std::string::npos)
                intel_platforms.push_back(p);
        }

        // OpenCL can return a list of platforms that contains duplicates.
        std::sort(intel_platforms.begin(), intel_platforms.end());
        intel_platforms.erase(
                std::unique(intel_platforms.begin(), intel_platforms.end()),
                intel_platforms.end());
        return intel_platforms;
    }

    std::unordered_map<cl_platform_id, F> ext_func_ptrs_;
};
} // namespace

void *malloc_shared(const cl_device_id dev, const cl_context ctx, size_t size,
        size_t alignment) {
    using clSharedMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);
    if (size == 0) return nullptr;
    static ext_func_t<clSharedMemAllocINTEL_func_t> ext_func(
            "clSharedMemAllocINTEL");

    cl_platform_id platform;
    UNUSED_OCL_RESULT(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));

    cl_int err;
    void *p = ext_func(platform, ctx, dev, nullptr, size,
            static_cast<cl_uint>(alignment), &err);
    assert(dnnl::impl::utils::one_of(err, CL_SUCCESS, CL_OUT_OF_RESOURCES,
            CL_OUT_OF_HOST_MEMORY, CL_INVALID_BUFFER_SIZE));
    return p;
}

void free(void *ptr, const cl_device_id dev, const cl_context ctx) {
    if (nullptr == ptr) return;
    using F = cl_int (*)(cl_context, void *);
    static ext_func_t<F> ext_func("clMemBlockingFreeINTEL");
    cl_platform_id platform;
    UNUSED_OCL_RESULT(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    cl_int err = ext_func(platform, ctx, ptr);
    UNUSED_OCL_RESULT(err);
}

} // namespace ocl
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
