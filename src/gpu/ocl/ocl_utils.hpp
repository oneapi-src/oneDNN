/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_OCL_OCL_UTILS_HPP
#define GPU_OCL_OCL_UTILS_HPP

#include <cinttypes>
#include <memory>
#include <sstream>
#include <string.h>
#include <string>
#include <utility>
#include <vector>
#include <CL/cl.h>
#include <initializer_list>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "common/c_types_map.hpp"
#include "common/cpp_compat.hpp"
#include "common/internal_defs.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/compute/kernel_arg_list.hpp"
#include "gpu/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

namespace compute {
class kernel_t;
}

namespace ocl {

inline status_t convert_to_dnnl(cl_int cl_status) {
    switch (cl_status) {
        case CL_SUCCESS: return status::success;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_OUT_OF_RESOURCES:
        case CL_OUT_OF_HOST_MEMORY: return status::out_of_memory;
        case CL_DEVICE_NOT_FOUND:
        case CL_DEVICE_NOT_AVAILABLE:
        case CL_COMPILER_NOT_AVAILABLE:
        case CL_PROFILING_INFO_NOT_AVAILABLE:
        case CL_MEM_COPY_OVERLAP:
        case CL_IMAGE_FORMAT_MISMATCH:
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        case CL_BUILD_PROGRAM_FAILURE:
        case CL_MAP_FAILURE:
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        case CL_COMPILE_PROGRAM_FAILURE:
        case CL_LINKER_NOT_AVAILABLE:
        case CL_LINK_PROGRAM_FAILURE:
        case CL_DEVICE_PARTITION_FAILED:
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        case CL_INVALID_PLATFORM:
        case CL_INVALID_DEVICE: return status::runtime_error;
        case CL_INVALID_VALUE:
        case CL_INVALID_DEVICE_TYPE:
        case CL_INVALID_CONTEXT:
        case CL_INVALID_QUEUE_PROPERTIES:
        case CL_INVALID_COMMAND_QUEUE:
        case CL_INVALID_HOST_PTR:
        case CL_INVALID_MEM_OBJECT:
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        case CL_INVALID_IMAGE_SIZE:
        case CL_INVALID_SAMPLER:
        case CL_INVALID_BINARY:
        case CL_INVALID_BUILD_OPTIONS:
        case CL_INVALID_PROGRAM:
        case CL_INVALID_PROGRAM_EXECUTABLE:
        case CL_INVALID_KERNEL_NAME:
        case CL_INVALID_KERNEL_DEFINITION:
        case CL_INVALID_KERNEL:
        case CL_INVALID_ARG_INDEX:
        case CL_INVALID_ARG_VALUE:
        case CL_INVALID_ARG_SIZE:
        case CL_INVALID_KERNEL_ARGS:
        case CL_INVALID_WORK_DIMENSION:
        case CL_INVALID_WORK_GROUP_SIZE:
        case CL_INVALID_WORK_ITEM_SIZE:
        case CL_INVALID_GLOBAL_OFFSET:
        case CL_INVALID_EVENT_WAIT_LIST:
        case CL_INVALID_EVENT:
        case CL_INVALID_OPERATION:
        case CL_INVALID_GL_OBJECT:
        case CL_INVALID_BUFFER_SIZE:
        case CL_INVALID_MIP_LEVEL:
        case CL_INVALID_GLOBAL_WORK_SIZE: return status::invalid_arguments;

        default: return status::runtime_error;
    }
}

// Ordered by value as defined by opencl
inline const char *convert_cl_int_to_str(cl_int cl_status) {
#define CL_STATUS_CASE(status) \
    case status: return #status
    switch (cl_status) {
        CL_STATUS_CASE(CL_SUCCESS);
        CL_STATUS_CASE(CL_DEVICE_NOT_FOUND);
        CL_STATUS_CASE(CL_DEVICE_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_COMPILER_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        CL_STATUS_CASE(CL_OUT_OF_RESOURCES);
        CL_STATUS_CASE(CL_OUT_OF_HOST_MEMORY);
        CL_STATUS_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_MEM_COPY_OVERLAP);
        CL_STATUS_CASE(CL_IMAGE_FORMAT_MISMATCH);
        CL_STATUS_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        CL_STATUS_CASE(CL_BUILD_PROGRAM_FAILURE);
        CL_STATUS_CASE(CL_MAP_FAILURE);
        CL_STATUS_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        CL_STATUS_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        CL_STATUS_CASE(CL_COMPILE_PROGRAM_FAILURE);
        CL_STATUS_CASE(CL_LINKER_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_LINK_PROGRAM_FAILURE);
        CL_STATUS_CASE(CL_DEVICE_PARTITION_FAILED);
        CL_STATUS_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
        CL_STATUS_CASE(CL_INVALID_VALUE);
        CL_STATUS_CASE(CL_INVALID_DEVICE_TYPE);
        CL_STATUS_CASE(CL_INVALID_PLATFORM);
        CL_STATUS_CASE(CL_INVALID_DEVICE);
        CL_STATUS_CASE(CL_INVALID_CONTEXT);
        CL_STATUS_CASE(CL_INVALID_QUEUE_PROPERTIES);
        CL_STATUS_CASE(CL_INVALID_COMMAND_QUEUE);
        CL_STATUS_CASE(CL_INVALID_HOST_PTR);
        CL_STATUS_CASE(CL_INVALID_MEM_OBJECT);
        CL_STATUS_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        CL_STATUS_CASE(CL_INVALID_IMAGE_SIZE);
        CL_STATUS_CASE(CL_INVALID_SAMPLER);
        CL_STATUS_CASE(CL_INVALID_BINARY);
        CL_STATUS_CASE(CL_INVALID_BUILD_OPTIONS);
        CL_STATUS_CASE(CL_INVALID_PROGRAM);
        CL_STATUS_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        CL_STATUS_CASE(CL_INVALID_KERNEL_NAME);
        CL_STATUS_CASE(CL_INVALID_KERNEL_DEFINITION);
        CL_STATUS_CASE(CL_INVALID_KERNEL);
        CL_STATUS_CASE(CL_INVALID_ARG_INDEX);
        CL_STATUS_CASE(CL_INVALID_ARG_VALUE);
        CL_STATUS_CASE(CL_INVALID_ARG_SIZE);
        CL_STATUS_CASE(CL_INVALID_KERNEL_ARGS);
        CL_STATUS_CASE(CL_INVALID_WORK_DIMENSION);
        CL_STATUS_CASE(CL_INVALID_WORK_GROUP_SIZE);
        CL_STATUS_CASE(CL_INVALID_WORK_ITEM_SIZE);
        CL_STATUS_CASE(CL_INVALID_GLOBAL_OFFSET);
        CL_STATUS_CASE(CL_INVALID_EVENT_WAIT_LIST);
        CL_STATUS_CASE(CL_INVALID_EVENT);
        CL_STATUS_CASE(CL_INVALID_OPERATION);
        CL_STATUS_CASE(CL_INVALID_GL_OBJECT);
        CL_STATUS_CASE(CL_INVALID_BUFFER_SIZE);
        CL_STATUS_CASE(CL_INVALID_MIP_LEVEL);
        CL_STATUS_CASE(CL_INVALID_GLOBAL_WORK_SIZE);

        default: return "unknown macro name";
    }
}
enum { OCL_BUFFER_ALIGNMENT = 128 };

#define MAYBE_REPORT_ERROR(msg) \
    do { \
        VERROR(primitive, gpu, msg); \
    } while (0)

#define MAYBE_REPORT_OCL_ERROR(s) \
    do { \
        VERROR(primitive, ocl, "errcode %d,%s,%s:%d", int(s), \
                gpu::ocl::convert_cl_int_to_str(s), __FILENAME__, __LINE__); \
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
            return dnnl::impl::gpu::ocl::convert_to_dnnl(s); \
        } \
    } while (0)

#define UNUSED_OCL_RESULT(x) \
    do { \
        cl_int s = x; \
        if (s != CL_SUCCESS) { MAYBE_REPORT_OCL_ERROR(s); } \
        assert(s == CL_SUCCESS); \
        MAYBE_UNUSED(s); \
    } while (false)

// Check for three conditions:
// 1. Device and context are compatible, i.e. the device belongs to
//    the context devices.
// 2. Device type matches the passed engine kind
// 3. Device/context platfrom is an Intel platform
status_t check_device(engine_kind_t eng_kind, cl_device_id dev, cl_context ctx);

status_t get_ocl_devices(
        std::vector<cl_device_id> *devices, cl_device_type device_type);

status_t get_ocl_device_index(size_t *index, cl_device_id device);

cl_platform_id get_ocl_platform(cl_device_id device);
cl_platform_id get_ocl_platform(engine_t *engine);

namespace details {

// OpenCL objects reference counting traits
template <typename T>
struct ocl_ref_traits;
//{
//    static void retain(T t) {}
//    static void release(T t) {}
//};

template <>
struct ocl_ref_traits<cl_context> {
    static void retain(cl_context t) { UNUSED_OCL_RESULT(clRetainContext(t)); }
    static void release(cl_context t) {
        UNUSED_OCL_RESULT(clReleaseContext(t));
    }
};

template <>
struct ocl_ref_traits<cl_command_queue> {
    static void retain(cl_command_queue t) {
        UNUSED_OCL_RESULT(clRetainCommandQueue(t));
    }
    static void release(cl_command_queue t) {
        UNUSED_OCL_RESULT(clReleaseCommandQueue(t));
    }
};

template <>
struct ocl_ref_traits<cl_program> {
    static void retain(cl_program t) { UNUSED_OCL_RESULT(clRetainProgram(t)); }
    static void release(cl_program t) {
        UNUSED_OCL_RESULT(clReleaseProgram(t));
    }
};

template <>
struct ocl_ref_traits<cl_kernel> {
    static void retain(cl_kernel t) { UNUSED_OCL_RESULT(clRetainKernel(t)); }
    static void release(cl_kernel t) { UNUSED_OCL_RESULT(clReleaseKernel(t)); }
};

template <>
struct ocl_ref_traits<cl_mem> {
    static void retain(cl_mem t) { UNUSED_OCL_RESULT(clRetainMemObject(t)); }
    static void release(cl_mem t) { UNUSED_OCL_RESULT(clReleaseMemObject(t)); }
};

template <>
struct ocl_ref_traits<cl_sampler> {
    static void retain(cl_sampler t) { UNUSED_OCL_RESULT(clRetainSampler(t)); }
    static void release(cl_sampler t) {
        UNUSED_OCL_RESULT(clReleaseSampler(t));
    }
};

template <>
struct ocl_ref_traits<cl_event> {
    static void retain(cl_event t) { UNUSED_OCL_RESULT(clRetainEvent(t)); }
    static void release(cl_event t) { UNUSED_OCL_RESULT(clReleaseEvent(t)); }
};

template <>
struct ocl_ref_traits<cl_device_id> {
    static void retain(cl_device_id t) { UNUSED_OCL_RESULT(clRetainDevice(t)); }
    static void release(cl_device_id t) {
        UNUSED_OCL_RESULT(clReleaseDevice(t));
    }
};

} // namespace details

// Generic class providing RAII support for OpenCL objects
template <typename T>
struct ocl_wrapper_t {
    ocl_wrapper_t(T t = nullptr, bool retain = false) : t_(t) {
        if (retain) { do_retain(); }
    }

    ocl_wrapper_t(const ocl_wrapper_t &other) : t_(other.t_) { do_retain(); }

    ocl_wrapper_t(ocl_wrapper_t &&other) noexcept : ocl_wrapper_t() {
        swap(*this, other);
    }

    ocl_wrapper_t &operator=(ocl_wrapper_t other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(ocl_wrapper_t &a, ocl_wrapper_t &b) noexcept {
        using std::swap;
        swap(a.t_, b.t_);
    }

    ~ocl_wrapper_t() { do_release(); }

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
        if (t_) { details::ocl_ref_traits<T>::retain(t_); }
    }

    void do_release() {
        if (t_) { details::ocl_ref_traits<T>::release(t_); }
    }
};

// Constructs an OpenCL wrapper object (providing RAII support)
template <typename T>
ocl_wrapper_t<T> make_ocl_wrapper(T t, bool retain = false) {
    return ocl_wrapper_t<T>(t, retain);
}

template <typename F>
struct ext_func_t {
    ext_func_t(const char *name) : ext_func_ptrs_(intel_platforms().size()) {
        for (size_t i = 0; i < intel_platforms().size(); ++i) {
            auto p = intel_platforms()[i];
            auto it = ext_func_ptrs_.insert({p, load_ext_func(p, name)});
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
        return get_func(get_ocl_platform(engine));
    }

    F get_func(cl_platform_id platform) const {
        return ext_func_ptrs_.at(platform);
    }

private:
    std::unordered_map<cl_platform_id, F> ext_func_ptrs_;

    static F load_ext_func(cl_platform_id platform, const char *name) {
        return reinterpret_cast<F>(
                clGetExtensionFunctionAddressForPlatform(platform, name));
    }

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
};

status_t get_ocl_kernel_arg_type(compute::scalar_type_t *type,
        cl_kernel ocl_kernel, int idx, bool allow_undef = false);

#ifdef DNNL_ENABLE_MEM_DEBUG
cl_mem DNNL_WEAK clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret);
#else
cl_mem clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret);
#endif

status_t get_ocl_program_binary(
        cl_program program, cl_device_id device, compute::binary_t &binary);

status_t get_ocl_program_binary(
        cl_kernel kernel, cl_device_id device, compute::binary_t &binary);

status_t get_ocl_kernel_binary(cl_kernel ocl_kernel, compute::binary_t &binary);

status_t get_ocl_program_binary_size(
        cl_kernel kernel, cl_device_id device, size_t *size);

void debugdump_processed_source(const std::string &source,
        const std::string &options, const std::string &ocl_options);

status_t get_kernel_arg_types(cl_kernel ocl_kernel,
        std::vector<gpu::compute::scalar_type_t> *arg_types);

status_t get_ocl_device_eu_count(cl_device_id device, int32_t *eu_count);

status_t get_ocl_device_enabled_systolic_intel(
        cl_device_id device, bool &systolic_enabled);

status_t get_ocl_device_enabled_native_float_atomics(
        cl_device_id device, uint64_t &native_extensions, bool is_xelpg);

status_t clone_kernel(cl_kernel kernel, cl_kernel *cloned_kernel);

status_t create_ocl_program(gpu::ocl::ocl_wrapper_t<cl_program> &ocl_program,
        cl_device_id dev, cl_context ctx, const gpu::compute::binary_t &binary);

status_t get_device_uuid(
        gpu::compute::device_uuid_t &uuid, cl_device_id ocl_dev);

status_t get_ocl_devices(std::vector<cl_device_id> *devices,
        std::vector<ocl_wrapper_t<cl_device_id>> *sub_devices,
        cl_device_type device_type);

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
