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

#ifndef OCL_UTILS_HPP
#define OCL_UTILS_HPP

#include <cinttypes>
#include <memory>
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
#include "common/engine.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace ocl {
namespace ocl_utils {

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
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return status::runtime_error;
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
        case CL_INVALID_KERNEL_DEFINITION: // FI
        case CL_INVALID_KERNEL:
        case CL_INVALID_ARG_INDEX:
        case CL_INVALID_ARG_VALUE: return status::invalid_arguments;

        default: return status::runtime_error;
    }
}

#ifndef NDEBUG
#define MAYBE_REPORT_OCL_ERROR(s) \
    do { \
        if (get_verbose()) \
            printf("dnnl_verbose,gpu,ocl_error,%d\n", (int)(s)); \
    } while (0)
#define OCL_CHECK_V(x) \
    do { \
        cl_int s = x; \
        if (s != CL_SUCCESS) { \
            MAYBE_REPORT_OCL_ERROR(s); \
            return; \
        } \
    } while (0)
#else
#define MAYBE_REPORT_OCL_ERROR(s)
#define OCL_CHECK_V(x) x
#endif

#define OCL_CHECK(x) \
    do { \
        cl_int s = x; \
        if (s != CL_SUCCESS) { \
            MAYBE_REPORT_OCL_ERROR(s); \
            return dnnl::impl::ocl::ocl_utils::convert_to_dnnl(s); \
        } \
    } while (0)

// Check for two conditions:
// 1. Device and context are compatible, i.e. the device belongs to
//    the context devices.
// 2. Device type matches the passed engine kind (only GPU supported).
inline status_t check_device(
        engine_kind_t eng_kind, cl_device_id dev, cl_context ctx) {
    assert(dev && ctx);
    assert(eng_kind == engine_kind::gpu);

    size_t dev_bytes;
    OCL_CHECK(
            clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, nullptr, &dev_bytes));

    std::vector<cl_device_id> ctx_devices(dev_bytes / sizeof(cl_device_id));
    OCL_CHECK(clGetContextInfo(
            ctx, CL_CONTEXT_DEVICES, dev_bytes, &ctx_devices[0], nullptr));

    for (size_t i = 0; i < ctx_devices.size(); ++i) {
        if (ctx_devices[i] == dev) {
            cl_device_type dev_type;
            OCL_CHECK(clGetDeviceInfo(
                    dev, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL));
            if ((dev_type & CL_DEVICE_TYPE_GPU) == 0) {
                return status::invalid_arguments;
            }
            return status::success;
        }
    }
    return status::invalid_arguments;
}

inline void get_optimal_lws(const size_t *gws, size_t *lws, size_t n) {
    const size_t lws_max = 256;
    const size_t optimal_lws_values[]
            = {256, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1};
    size_t total_lws = 1;
    for (size_t i = 0; i < n; ++i) {
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = 0;
        while (rest_lws < optimal_lws_values[lws_idx])
            lws_idx++;

        while (gws[i] % optimal_lws_values[lws_idx])
            lws_idx++;

        lws[i] = optimal_lws_values[lws_idx];
        total_lws *= optimal_lws_values[lws_idx];
    }
}

status_t get_ocl_devices(
        std::vector<cl_device_id> *devices, cl_device_type device_type);

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
    static void retain(cl_context t) { clRetainContext(t); }
    static void release(cl_context t) { clReleaseContext(t); }
};

template <>
struct ocl_ref_traits<cl_command_queue> {
    static void retain(cl_command_queue t) { clRetainCommandQueue(t); }
    static void release(cl_command_queue t) { clReleaseCommandQueue(t); }
};

template <>
struct ocl_ref_traits<cl_program> {
    static void retain(cl_program t) { clRetainProgram(t); }
    static void release(cl_program t) { clReleaseProgram(t); }
};

template <>
struct ocl_ref_traits<cl_kernel> {
    static void retain(cl_kernel t) { clRetainKernel(t); }
    static void release(cl_kernel t) { clReleaseKernel(t); }
};

template <>
struct ocl_ref_traits<cl_mem> {
    static void retain(cl_mem t) { clRetainMemObject(t); }
    static void release(cl_mem t) { clReleaseMemObject(t); }
};

template <>
struct ocl_ref_traits<cl_sampler> {
    static void retain(cl_sampler t) { clRetainSampler(t); }
    static void release(cl_sampler t) { clReleaseSampler(t); }
};

template <>
struct ocl_ref_traits<cl_event> {
    static void retain(cl_event t) { clRetainEvent(t); }
    static void release(cl_event t) { clReleaseEvent(t); }
};

template <>
struct ocl_ref_traits<cl_device_id> {
    static void retain(cl_device_id t) { clRetainDevice(t); }
    static void release(cl_device_id t) { clReleaseDevice(t); }
};

} // namespace details

// Generic class providing RAII support for OpenCL objects
template <typename T>
struct ocl_wrapper_t {
    ocl_wrapper_t(T t = nullptr, bool retain = false) : t_(t) {
        if (retain) { do_retain(); }
    }

    ocl_wrapper_t(const ocl_wrapper_t &other) : t_(other.t_) { do_retain(); }

    ocl_wrapper_t(ocl_wrapper_t &&other) noexcept : t_(std::move(other.t_)) {}

    ocl_wrapper_t &operator=(ocl_wrapper_t other) {
        using std::swap;
        swap(t_, other.t_);
        return *this;
    }

    ~ocl_wrapper_t() { do_release(); }

    operator T() const { return t_; }
    T get() const { return t_; }

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
ocl_wrapper_t<T> make_ocl_wrapper(T t) {
    return ocl_wrapper_t<T>(t);
}

} // namespace ocl_utils

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif
