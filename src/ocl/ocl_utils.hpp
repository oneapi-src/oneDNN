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

#include <CL/cl.h>
#include <cinttypes>
#include <initializer_list>
#include <memory>
#include <string.h>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {
namespace ocl_utils {

inline status_t convert_to_mkldnn(cl_int cl_status) {
    switch (cl_status) {
    case CL_SUCCESS: return status::success;
    case CL_DEVICE_NOT_FOUND:
    case CL_DEVICE_NOT_AVAILABLE:
    case CL_COMPILER_NOT_AVAILABLE:
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    case CL_OUT_OF_RESOURCES:
    case CL_OUT_OF_HOST_MEMORY:
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

#define OCL_CHECK(x)                                                   \
    do {                                                               \
        cl_int s = x;                                                  \
        if (s != CL_SUCCESS) {                                         \
            if (mkldnn_verbose()->level >= 5) {                        \
                printf("Error from OpenCL: %d\n", s);                  \
            }                                                          \
            return mkldnn::impl::ocl::ocl_utils::convert_to_mkldnn(s); \
        }                                                              \
    } while (0)

#define OCL_CHECK_V(x)                            \
    do {                                          \
        cl_int s = x;                             \
        if (s != CL_SUCCESS) {                    \
            printf("Error from OpenCL: %d\n", s); \
            exit(1);                              \
            return;                               \
        }                                         \
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
    const size_t optimal_lws_values[] = { 256, 224, 192, 160, 128, 96, 64, 32,
        16, 8, 7, 6, 5, 4, 3, 2, 1 };
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
        if (retain) {
            do_retain();
        }
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
        if (t_) {
            details::ocl_ref_traits<T>::retain(t_);
        }
    }

    void do_release() {
        if (t_) {
            details::ocl_ref_traits<T>::release(t_);
        }
    }
};

// Constructs an OpenCL wrapper object (providing RAII support)
template <typename T>
ocl_wrapper_t<T> make_ocl_wrapper(T t) {
    return ocl_wrapper_t<T>(t);
}

} // namespace ocl_utils

// Stores an argument for an OpenCL kernel
class ocl_kernel_arg_t
{
public:
    static constexpr size_t max_size = 8;

    size_t size() const { return size_; }
    bool is_global() const { return kind_ == kind_t::global; }

    void set_value(const memory_storage_t &mem_storage) {
        kind_ = kind_t::global;
        size_ = 0;
        value_ = static_cast<const void *>(&mem_storage);
    }

    template <typename T,
            typename = typename std::enable_if<std::is_arithmetic<T>::value
                    || std::is_same<T, float16_t>::value>::type>
    void set_value(const T &value) {
        kind_ = kind_t::scalar;
        new (&scalar_storage_) T(value);
        size_ = sizeof(T);
        value_ = nullptr;
    }

    void set_value(size_t size, std::nullptr_t) {
        kind_ = kind_t::local;
        size_ = size;
        value_ = nullptr;
    }

    const void *value() const {
        assert(kind_ != kind_t::undef);
        if (kind_ == kind_t::scalar)
            return static_cast<const void *>(&scalar_storage_);
        return value_;
    }

private:
    enum class kind_t {
        undef,
        global,
        local,
        scalar,
    };

    kind_t kind_ = kind_t::undef;
    size_t size_ = 0;
    const void *value_ = nullptr;

    typename std::aligned_storage<max_size, max_size>::type scalar_storage_;
};

// RAII wrapper for an OpenCL kernel and its arguments
struct ocl_kernel_t {
    ocl_kernel_t(cl_kernel kernel = nullptr) : kernel_(kernel) {}

    ocl_kernel_t(const ocl_kernel_t &other)
        : kernel_(other.kernel_), nargs_(other.nargs_) {
        if (kernel_) {
            clRetainKernel(kernel_);
        }
        utils::array_copy(args_, other.args_, nargs_);
    }

    ocl_kernel_t(ocl_kernel_t &&other)
        : kernel_(other.kernel_), nargs_(other.nargs_) {
        other.kernel_ = nullptr;
        utils::array_copy(args_, other.args_, nargs_);
    }

    ocl_kernel_t &operator=(const ocl_kernel_t &other) {
        MKLDNN_SHORT_CIRCUIT_SELF_ASSIGN(other);

        if (kernel_) {
            clReleaseKernel(kernel_);
        }

        kernel_ = other.kernel_;
        if (kernel_) {
            clRetainKernel(kernel_);
        }
        nargs_ = other.nargs_;
        for (size_t i = 0; i < nargs_; i++) {
            args_[i] = other.args_[i];
        }
        return *this;
    }

    ~ocl_kernel_t() {
        if (kernel_) {
            clReleaseKernel(kernel_);
        }
    }

    explicit operator bool() const { return kernel(); }

    cl_kernel kernel() const { return kernel_; }

    size_t nargs() const { return nargs_; }

    const ocl_kernel_arg_t *args() const { return args_; }

    void set_arg(int index, const memory_storage_t &mem_storage) const {
        assert(index < max_args);
        ((size_t &)(nargs_)) = nstl::max(nargs_, size_t(index) + 1);
        ((ocl_kernel_arg_t *)args_)[index].set_value(mem_storage);
    }

    template <class T,
            typename = typename std::enable_if<std::is_arithmetic<T>::value
                    || std::is_same<T, float16_t>::value>::type>
    void set_arg(int index, const T &value) const {
        static_assert(sizeof(T) <= ocl_kernel_arg_t::max_size,
                "Type size is too large");

        assert(index < max_args);
        ((size_t &)(nargs_)) = nstl::max(nargs_, size_t(index) + 1);
        ((ocl_kernel_arg_t *)args_)[index].set_value(value);
    }

    void set_arg(int index, size_t sz, std::nullptr_t) const {
        assert(index < max_args);
        ((size_t &)(nargs_)) = nstl::max(nargs_, size_t(index) + 1);
        ((ocl_kernel_arg_t *)args_)[index].set_value(sz, nullptr);
    }

private:
    static constexpr int max_args = 20;

    cl_kernel kernel_ = nullptr;
    size_t nargs_ = 0;
    ocl_kernel_arg_t args_[max_args];
};

// Stores global/local ranges to use for kernel enqueueing
struct cl_nd_range_t {
    cl_nd_range_t(size_t n, const size_t *global_range,
            const size_t *local_range = nullptr) {
        with_local_range_ = bool(local_range);

        for (size_t i = 0; i < 3; ++i) {
            global_range_[i] = (i < n) ? global_range[i] : 1;
            if (with_local_range_) {
                local_range_[i] = (i < n) ? local_range[i] : 1;
            }
        }
    }

    cl_nd_range_t(
            const size_t *global_range, const size_t *local_range = nullptr)
        : cl_nd_range_t(3, global_range, local_range) {}

    template <typename int_type>
    cl_nd_range_t(std::initializer_list<int_type> global_range,
            std::initializer_list<int_type> local_range = {}) {
        with_local_range_ = (local_range.size() > 0);
        if (with_local_range_) {
            assert(global_range.size() == local_range.size());
        }
        size_t n = global_range.size();
        for (size_t i = 0; i < 3; i++) {
            global_range_[i] = (i < n) ? *(global_range.begin() + i) : 1;
            if (with_local_range_) {
                local_range_[i] = (i < n) ? *(local_range.begin() + i) : 1;
            }
        }
    }

    size_t ndims() const { return 3; }
    const size_t *global_range() const { return global_range_; }

    const size_t *local_range() const {
        return with_local_range_ ? local_range_ : nullptr;
    }

    bool is_zero() const {
        return global_range_[0] == 0 || global_range_[1] == 0
                || global_range_[2] == 0;
    }

private:
    size_t global_range_[3];
    size_t local_range_[3];
    bool with_local_range_;
};

struct ocl_jit_t {
    ocl_jit_t(const char *code) : code_(code), options_(), program_(nullptr) {
        code_size_ = strlen(code_);

        options_.reserve(256);
        options_.push_back('\0');

        set_default_options();
    }

    ~ocl_jit_t() {
        if (program_)
            clReleaseProgram(program_);
    }

    ocl_jit_t(ocl_jit_t &&other) {
        code_ = other.code_;
        code_size_ = other.code_size_;
        options_ = std::move(other.options_);
        program_ = other.program_;

        other.reset();
    }

    ocl_jit_t &operator=(ocl_jit_t &&other) {
        code_ = std::move(other.code_);
        code_size_ = other.code_size_;
        options_ = std::move(other.options_);
        program_ = other.program_;

        other.reset();

        return *this;
    }

    const char *get_code() const { return code_; }
    const char *get_options() const { return options_.data(); }
    size_t get_code_size() const { return code_size_; }

    void define_int(const char *variable, int64_t value) {
        int var_sz = snprintf(nullptr, 0, " -D%s=%" PRId64, variable, value);
        size_t old_sz = options_.size();
        size_t new_sz = options_.size() + var_sz;

        options_.resize(new_sz);
        char *tmp = options_.data() + old_sz - 1;
        snprintf(tmp, var_sz + 1, " -D%s=%" PRId64, variable, value);
    }

    void define_float(const char *variable, float value) {
        union {
            float f;
            uint32_t u;
        } f2u = { value };

        int var_sz
                = snprintf(nullptr, 0, " -D%s=as_float(0x%x)", variable, f2u.u);
        size_t old_sz = options_.size();
        size_t new_sz = options_.size() + var_sz;

        options_.resize(new_sz);
        char *tmp = options_.data() + old_sz - 1;
        snprintf(tmp, var_sz + 1, " -D%s=as_float(0x%x)", variable, f2u.u);
    }

    void add_option(const char *option) {
        int var_sz = snprintf(nullptr, 0, " %s", option);
        size_t old_sz = options_.size();
        size_t new_sz = options_.size() + var_sz;

        options_.resize(new_sz);
        char *tmp = options_.data() + old_sz - 1;
        snprintf(tmp, var_sz + 1, " %s", option);
    }

    void set_data_type(data_type_t dt) {
        switch (dt) {
        case data_type::f16: define_int("DT_F16", 1); break;
        case data_type::f32: define_int("DT_F32", 1); break;
        case data_type::s8: define_int("DT_S8", 1); break;
        case data_type::u8: define_int("DT_U8", 1); break;
        case data_type::s32: define_int("DT_S32", 1); break;
        default: assert(!"unknown data type"); break;
        }
    }

    status_t build(const engine_t *engine);

    ocl_kernel_t get_kernel(const char *kernel_name) {
        assert(program_);
        cl_int err;
        cl_kernel kernel = clCreateKernel(program_, kernel_name, &err);
        assert(err == CL_SUCCESS);
        return ocl_kernel_t(kernel);
    }

private:
    void set_default_options() {
        // By default fp32 division and sqrt is not IEEE-compliant
        add_option("-cl-fp32-correctly-rounded-divide-sqrt");
    }

    void reset() {
        code_ = nullptr;
        code_size_ = 0;
        program_ = nullptr;
    }

    const char *code_;
    size_t code_size_;
    std::vector<char> options_;
    cl_program program_;

    MKLDNN_DISALLOW_COPY_AND_ASSIGN(ocl_jit_t);
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
