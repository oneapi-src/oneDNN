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

#ifndef COMMON_XPU_OCL_UTILS_HPP
#define COMMON_XPU_OCL_UTILS_HPP

#include <CL/cl.h>

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#endif

#include "oneapi/dnnl/dnnl_config.h"

#include "common/c_types_map.hpp"
#include "common/cpp_compat.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {

// Dynamically loaded opencl functions
namespace {

void *find_cl_symbol(const char *symbol) {
#if defined(__linux__)
    void *handle = dlopen("libOpencl.so.1", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    // Use LOAD_LIBRARY_SEARCH_SYSTEM32 flag to avoid DLL hijacking issue.
    HMODULE handle = LoadLibraryExA(
            "OpenCL.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#endif
    if (!handle) {
        VERROR(common, opencl, "cannot find opencl library");
        assert(!"not expected");
        return nullptr;
    }

#if defined(__linux__)
    void *f = reinterpret_cast<void *>(dlsym(handle, symbol));
#elif defined(_WIN32)
    void *f = reinterpret_cast<void *>(GetProcAddress(handle, symbol));
#endif

    if (!f) {
        VERROR(common, opencl, "cannot find symbol: %s", symbol);
        assert(!"not expected");
    }
    return f;
}

template <typename F>
F find_cl_symbol(const char *symbol) {
    return (F)find_cl_symbol(symbol);
}

#define CL_INDIRECT_API(f) \
    template <typename... Args> \
    auto call_##f(Args &&...args) { \
        static auto f_ = find_cl_symbol<decltype(&f)>(#f); \
        return f_(std::forward<Args>(args)...); \
    }

CL_INDIRECT_API(clBuildProgram)
CL_INDIRECT_API(clCreateBuffer)
CL_INDIRECT_API(clCreateContext)
CL_INDIRECT_API(clCreateKernel)
CL_INDIRECT_API(clCloneKernel)
CL_INDIRECT_API(clCreateProgramWithBinary)
CL_INDIRECT_API(clCreateProgramWithSource)
CL_INDIRECT_API(clCreateSubBuffer)
CL_INDIRECT_API(clCreateSubDevices)
CL_INDIRECT_API(clEnqueueMapBuffer)
CL_INDIRECT_API(clEnqueueWriteBuffer)
CL_INDIRECT_API(clEnqueueReadBuffer)
CL_INDIRECT_API(clEnqueueCopyBuffer)
CL_INDIRECT_API(clGetEventProfilingInfo)
CL_INDIRECT_API(clEnqueueFillBuffer)
CL_INDIRECT_API(clGetCommandQueueInfo)
CL_INDIRECT_API(clEnqueueMarkerWithWaitList)
CL_INDIRECT_API(clEnqueueUnmapMemObject)
CL_INDIRECT_API(clFinish)
CL_INDIRECT_API(clGetContextInfo)
CL_INDIRECT_API(clGetDeviceIDs)
CL_INDIRECT_API(clGetDeviceInfo)
CL_INDIRECT_API(clGetExtensionFunctionAddressForPlatform)
CL_INDIRECT_API(clGetKernelArgInfo)
CL_INDIRECT_API(clGetKernelInfo)
CL_INDIRECT_API(clGetMemObjectInfo)
CL_INDIRECT_API(clGetPlatformIDs)
CL_INDIRECT_API(clGetPlatformInfo)
CL_INDIRECT_API(clGetProgramBuildInfo)
CL_INDIRECT_API(clGetProgramInfo)
CL_INDIRECT_API(clReleaseCommandQueue)
CL_INDIRECT_API(clReleaseContext)
CL_INDIRECT_API(clReleaseDevice)
CL_INDIRECT_API(clReleaseEvent)
CL_INDIRECT_API(clReleaseKernel)
CL_INDIRECT_API(clReleaseMemObject)
CL_INDIRECT_API(clReleaseProgram)
CL_INDIRECT_API(clReleaseSampler)
CL_INDIRECT_API(clRetainCommandQueue)
CL_INDIRECT_API(clRetainContext)
CL_INDIRECT_API(clRetainDevice)
CL_INDIRECT_API(clRetainEvent)
CL_INDIRECT_API(clRetainKernel)
CL_INDIRECT_API(clRetainMemObject)
CL_INDIRECT_API(clRetainProgram)
CL_INDIRECT_API(clRetainSampler)
CL_INDIRECT_API(clSetKernelArg)
CL_INDIRECT_API(clWaitForEvents)
CL_INDIRECT_API(clCreateCommandQueue)
CL_INDIRECT_API(clEnqueueNDRangeKernel)
#ifdef CL_VERSION_2_0
CL_INDIRECT_API(clSetKernelArgSVMPointer)
CL_INDIRECT_API(clCreateCommandQueueWithProperties)
#endif
#undef CL_INDIRECT_API
} // namespace

namespace xpu {
namespace ocl {

status_t convert_to_dnnl(cl_int cl_status);
const char *convert_cl_int_to_str(cl_int cl_status);

#define MAYBE_REPORT_ERROR(msg) \
    do { \
        VERROR(primitive, gpu, msg); \
    } while (0)

#define MAYBE_REPORT_OCL_ERROR(s) \
    do { \
        VERROR(primitive, ocl, "errcode %d,%s,%s:%d", int(s), \
                dnnl::impl::xpu::ocl::convert_cl_int_to_str(s), __FILENAME__, \
                __LINE__); \
    } while (0)

#define OCL_CHECK_V(x) \
    do { \
        cl_int s = x; \
        if (s != CL_SUCCESS) { \
            MAYBE_REPORT_OCL_ERROR(s); \
            return; \
        } \
    } while (0)

#define OCL_CHECK(x) \
    do { \
        cl_int s = x; \
        if (s != CL_SUCCESS) { \
            MAYBE_REPORT_OCL_ERROR(s); \
            return dnnl::impl::xpu::ocl::convert_to_dnnl(s); \
        } \
    } while (0)

#define UNUSED_OCL_RESULT(x) \
    do { \
        cl_int s = x; \
        if (s != CL_SUCCESS) { MAYBE_REPORT_OCL_ERROR(s); } \
        assert(s == CL_SUCCESS); \
        MAYBE_UNUSED(s); \
    } while (false)

// OpenCL objects reference counting traits
template <typename T>
struct ref_traits;
//{
//    static void retain(T t) {}
//    static void release(T t) {}
//};

template <>
struct ref_traits<cl_context> {
    static void retain(cl_context t) { UNUSED_OCL_RESULT(call_clRetainContext(t)); }
    static void release(cl_context t) {
        UNUSED_OCL_RESULT(call_clReleaseContext(t));
    }
};

template <>
struct ref_traits<cl_command_queue> {
    static void retain(cl_command_queue t) {
        UNUSED_OCL_RESULT(call_clRetainCommandQueue(t));
    }
    static void release(cl_command_queue t) {
        UNUSED_OCL_RESULT(call_clReleaseCommandQueue(t));
    }
};

template <>
struct ref_traits<cl_program> {
    static void retain(cl_program t) { UNUSED_OCL_RESULT(call_clRetainProgram(t)); }
    static void release(cl_program t) {
        UNUSED_OCL_RESULT(call_clReleaseProgram(t));
    }
};

template <>
struct ref_traits<cl_kernel> {
    static void retain(cl_kernel t) { UNUSED_OCL_RESULT(call_clRetainKernel(t)); }
    static void release(cl_kernel t) { UNUSED_OCL_RESULT(call_clReleaseKernel(t)); }
};

template <>
struct ref_traits<cl_mem> {
    static void retain(cl_mem t) { UNUSED_OCL_RESULT(call_clRetainMemObject(t)); }
    static void release(cl_mem t) { UNUSED_OCL_RESULT(call_clReleaseMemObject(t)); }
};

template <>
struct ref_traits<cl_sampler> {
    static void retain(cl_sampler t) { UNUSED_OCL_RESULT(call_clRetainSampler(t)); }
    static void release(cl_sampler t) {
        UNUSED_OCL_RESULT(call_clReleaseSampler(t));
    }
};

template <>
struct ref_traits<cl_event> {
    static void retain(cl_event t) { UNUSED_OCL_RESULT(call_clRetainEvent(t)); }
    static void release(cl_event t) { UNUSED_OCL_RESULT(call_clReleaseEvent(t)); }
};

template <>
struct ref_traits<cl_device_id> {
    static void retain(cl_device_id t) { UNUSED_OCL_RESULT(call_clRetainDevice(t)); }
    static void release(cl_device_id t) {
        UNUSED_OCL_RESULT(call_clReleaseDevice(t));
    }
};

// Generic class providing RAII support for OpenCL objects
template <typename T>
struct wrapper_t {
    wrapper_t(T t = nullptr, bool retain = false) : t_(t) {
        if (retain) { do_retain(); }
    }

    wrapper_t(const wrapper_t &other) : t_(other.t_) { do_retain(); }

    wrapper_t(wrapper_t &&other) noexcept : wrapper_t() { swap(*this, other); }

    wrapper_t &operator=(wrapper_t other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(wrapper_t &a, wrapper_t &b) noexcept {
        using std::swap;
        swap(a.t_, b.t_);
    }

    ~wrapper_t() { do_release(); }

    operator T() const { return t_; }
    T get() const { return t_; }
    T &unwrap() { return t_; }
    const T &unwrap() const { return t_; }

    T release() {
        T t = t_;
        t_ = nullptr;
        return t;
    }

private:
    T t_;

    void do_retain() {
        if (t_) { ref_traits<T>::retain(t_); }
    }

    void do_release() {
        if (t_) { ref_traits<T>::release(t_); }
    }
};

// Constructs an OpenCL wrapper object (providing RAII support)
template <typename T>
wrapper_t<T> make_wrapper(T t, bool retain = false) {
    return wrapper_t<T>(t, retain);
}

cl_platform_id get_platform(cl_device_id device);
cl_platform_id get_platform(engine_t *engine);

template <typename F>
struct ext_func_t {
    ext_func_t(const char *ext_func_name, const char *vendor_name = "Intel")
        : ext_func_ptrs_(vendor_platforms(vendor_name).size()) {
        for (size_t i = 0; i < vendor_platforms(vendor_name).size(); ++i) {
            auto p = vendor_platforms(vendor_name)[i];
            auto it = ext_func_ptrs_.insert(
                    {p, load_ext_func(p, ext_func_name)});
            assert(it.second);
            MAYBE_UNUSED(it);
        }
    }

    template <typename... Args>
    typename cpp_compat::invoke_result<F, Args...>::type operator()(
            engine_t *engine, Args... args) const {
        auto f = get_func(engine);
        return f(args...);
    }

    F get_func(engine_t *engine) const {
        return get_func(get_platform(engine));
    }

    F get_func(cl_platform_id platform) const {
        return ext_func_ptrs_.at(platform);
    }

private:
    std::unordered_map<cl_platform_id, F> ext_func_ptrs_;

    static F load_ext_func(cl_platform_id platform, const char *ext_func_name) {
        return reinterpret_cast<F>(call_clGetExtensionFunctionAddressForPlatform(
                platform, ext_func_name));
    }

    static const std::vector<cl_platform_id> &vendor_platforms(
            const char *vendor_name) {
        static auto vendor_platforms = get_vendor_platforms(vendor_name);
        return vendor_platforms;
    }

    static std::vector<cl_platform_id> get_vendor_platforms(
            const char *vendor_name) {
        cl_uint num_platforms = 0;
        cl_int err = call_clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> platforms(num_platforms);
        err = call_clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> vendor_platforms;
        char platform_vendor_name[128] = {};
        for (cl_platform_id p : platforms) {
            err = call_clGetPlatformInfo(p, CL_PLATFORM_VENDOR,
                    sizeof(platform_vendor_name), platform_vendor_name,
                    nullptr);
            if (err != CL_SUCCESS) continue;
            if (std::string(platform_vendor_name).find(vendor_name)
                    != std::string::npos)
                vendor_platforms.push_back(p);
        }

        // OpenCL can return a list of platforms that contains duplicates.
        std::sort(vendor_platforms.begin(), vendor_platforms.end());
        vendor_platforms.erase(
                std::unique(vendor_platforms.begin(), vendor_platforms.end()),
                vendor_platforms.end());
        return vendor_platforms;
    }
};

std::string get_kernel_name(cl_kernel kernel);

status_t get_devices(std::vector<cl_device_id> *devices,
        cl_device_type device_type, cl_uint vendor_id = 0x8086);

status_t get_devices(std::vector<cl_device_id> *devices,
        std::vector<wrapper_t<cl_device_id>> *sub_devices,
        cl_device_type device_type);

status_t get_device_index(size_t *index, cl_device_id device);

cl_platform_id get_platform(cl_device_id device);
cl_platform_id get_platform(engine_t *engine);

status_t create_program(ocl::wrapper_t<cl_program> &ocl_program,
        cl_device_id dev, cl_context ctx, const xpu::binary_t &binary);

status_t get_device_uuid(xpu::device_uuid_t &uuid, cl_device_id ocl_dev);

// Check for three conditions:
// 1. Device and context are compatible, i.e. the device belongs to
//    the context devices.
// 2. Device type matches the passed engine kind
// 3. Device/context platfrom is an Intel platform
status_t check_device(engine_kind_t eng_kind, cl_device_id dev, cl_context ctx);

status_t clone_kernel(cl_kernel kernel, cl_kernel *cloned_kernel);

#ifdef DNNL_ENABLE_MEM_DEBUG
cl_mem DNNL_WEAK clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret);
#else
cl_mem clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret);
#endif

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
