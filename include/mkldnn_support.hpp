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

#ifndef MKLDNN_SUPPORT_HPP
#define MKLDNN_SUPPORT_HPP

#include "mkldnn_config.h"
#include "mkldnn_types.h"

#include <type_traits>

#if MKLDNN_WITH_SYCL
#include <CL/sycl.hpp>
#endif

namespace mkldnn {
namespace impl {

#if MKLDNN_WITH_SYCL

#define MKLDNN_HANDLE_ALL_TYPES        \
    MKLDNN_CASE(mkldnn_f16, uint16_t); \
    MKLDNN_CASE(mkldnn_f32, float);    \
    MKLDNN_CASE(mkldnn_s32, int32_t);  \
    MKLDNN_CASE(mkldnn_s8, int8_t);    \
    MKLDNN_CASE(mkldnn_u8, uint8_t);

namespace sycl {

/// Helper class implementing type erasure for SYCL buffer.
class untyped_sycl_buffer_t
{
public:
    template <typename T, int ndims>
    untyped_sycl_buffer_t(const cl::sycl::buffer<T, ndims> &buf) {
        static_assert(ndims == 1, "Only 1D buffers supported");
        buf_ptr_ = new cl::sycl::buffer<T, ndims>(buf);

#define MKLDNN_CASE(dt, t)           \
    if (std::is_same<T, t>::value) { \
        data_type_ = dt;             \
        return;                      \
    }

        MKLDNN_HANDLE_ALL_TYPES

#undef MKLDNN_CASE

        assert(!"not expected");
    }

    untyped_sycl_buffer_t(mkldnn_data_type_t data_type, size_t size)
        : data_type_(data_type) {
        if (size == 0)
            return;

#define MKLDNN_CASE(dt, t)                             \
    case dt:                                           \
        buf_ptr_ = new cl::sycl::buffer<t, 1>(         \
                cl::sycl::range<1>(size / sizeof(t))); \
        break

        switch (data_type_) {
            MKLDNN_HANDLE_ALL_TYPES
        default: assert(!"not expected");
        }
#undef MKLDNN_CASE
    }

    untyped_sycl_buffer_t(void *data, mkldnn_data_type_t data_type, size_t size)
        : data_type_(data_type) {
        if (size == 0)
            return;

#define MKLDNN_CASE(dt, t)                                                     \
    case dt:                                                                   \
        buf_ptr_ = new cl::sycl::buffer<t, 1>(                                 \
                static_cast<t *>(data), cl::sycl::range<1>(size / sizeof(t))); \
        break

        switch (data_type_) {
            MKLDNN_HANDLE_ALL_TYPES
        default: assert(!"not expected");
        }
#undef MKLDNN_CASE
    }

    untyped_sycl_buffer_t(untyped_sycl_buffer_t &&other)
        : data_type_(other.data_type_), buf_ptr_(other.buf_ptr_) {
        other.buf_ptr_ = nullptr;
    }

    ~untyped_sycl_buffer_t() {
        if (buf_ptr_) {
#define MKLDNN_CASE(dt, t) \
    case dt: delete static_cast<cl::sycl::buffer<t, 1> *>(buf_ptr_); break

            switch (data_type_) {
                MKLDNN_HANDLE_ALL_TYPES
            default: assert(!"not expected");
            }
#undef MKLDNN_CASE
        }
    }

    untyped_sycl_buffer_t(const untyped_sycl_buffer_t &other)
        : untyped_sycl_buffer_t(other.reinterpret<uint8_t, 1>()) {}

    untyped_sycl_buffer_t &operator=(untyped_sycl_buffer_t &other) = delete;

    mkldnn_data_type_t data_type() const { return data_type_; }

    size_t get_size() const {
#define MKLDNN_CASE(dt, t) \
    case dt: return static_cast<cl::sycl::buffer<t, 1> *>(buf_ptr_)->get_size()
        switch (data_type_) {
            MKLDNN_HANDLE_ALL_TYPES
        default: assert(!"not expected");
        }
#undef MKLDNN_CASE
        return 0;
    }

    template <typename T, int ndims = 1>
    cl::sycl::buffer<T, ndims> reinterpret() const {
#define MKLDNN_CASE(dt, t)                                            \
    case dt: {                                                        \
        auto &buf = *static_cast<cl::sycl::buffer<t, 1> *>(buf_ptr_); \
        return buf.template reinterpret<T, ndims>(                    \
                cl::sycl::range<1>(buf.get_size() / sizeof(T)));      \
    }

        switch (data_type_) {
            MKLDNN_HANDLE_ALL_TYPES
        default: assert(!"not expected");
        }
#undef MKLDNN_CASE
        return cl::sycl::buffer<T, 1>(
                static_cast<T *>(nullptr), cl::sycl::range<1>(0));
    }

    template <typename T, int ndims>
    bool is_compatible() const {
        if (ndims != 1)
            return false;

#define MKLDNN_CASE(dt, t) \
    case dt: return std::is_same<T, t>::value;

        switch (data_type_) {
            MKLDNN_HANDLE_ALL_TYPES
        default: return false;
        }
#undef MKLDNN_CASE
        return false;
    }

    template <class T, int ndims = 1>
    cl::sycl::buffer<T, ndims> &sycl_buffer() const {
        static_assert(ndims == 1, "Only 1D buffers supported");
        bool is_compat = is_compatible<T, ndims>();
        assert(is_compat);
        (void)is_compat;
        return *static_cast<cl::sycl::buffer<T, ndims> *>(buf_ptr_);
    }

    template <cl::sycl::access::mode mode>
    void set_as_arg(cl::sycl::handler &cgh, int index) const {
#define MKLDNN_CASE(dt, t)                                             \
    case dt:                                                           \
        cgh.set_arg(index, sycl_buffer<t, 1>().get_access<mode>(cgh)); \
        break

        switch (data_type_) {
            MKLDNN_HANDLE_ALL_TYPES
        default: assert(!"not expected");
        }
#undef MKLDNN_CASE
    }

    template <cl::sycl::access::mode access_mode, typename dtor_callback_t>
    void *map_data(const dtor_callback_t &dtor_callback) const {
#define MKLDNN_CASE(dt, t)                                                    \
    case dt: {                                                                \
        auto &sycl_buf = sycl_buffer<t, 1>();                                 \
        auto sycl_host_acc = sycl_buf.get_access<access_mode>();              \
        auto *sycl_host_acc_ptr = new decltype(sycl_host_acc)(sycl_host_acc); \
        dtor_callback([=]() { delete sycl_host_acc_ptr; });                   \
        return static_cast<void *>(sycl_host_acc_ptr->get_pointer());         \
    }

        switch (data_type_) {
            MKLDNN_HANDLE_ALL_TYPES
        default: assert(!"not expected"); break;
        }
#undef MKLDNN_CASE
        return nullptr;
    }

    void copy_from(const void *ptr, size_t size) {
        constexpr auto write_mode = cl::sycl::access::mode::write;
#define MKLDNN_CASE(dt, t)                                      \
    case dt: {                                                  \
        const auto *data = static_cast<const t *>(ptr);         \
        auto &sycl_buf = sycl_buffer<t, 1>();                   \
        auto sycl_host_acc = sycl_buf.get_access<write_mode>(); \
        for (size_t i = 0; i < size / sizeof(*data); i++) {     \
            sycl_host_acc[i] = data[i];                         \
        }                                                       \
        break;                                                  \
    }

        switch (data_type_) {
            MKLDNN_HANDLE_ALL_TYPES
        default: assert(!"not expected"); break;
        }
#undef MKLDNN_CASE
    }

    void copy_to(void *ptr, size_t size) const {
        constexpr auto read_mode = cl::sycl::access::mode::read;
#define MKLDNN_CASE(dt, t)                                     \
    case dt: {                                                 \
        auto *data = static_cast<t *>(ptr);                    \
        auto &sycl_buf = sycl_buffer<t, 1>();                  \
        auto sycl_host_acc = sycl_buf.get_access<read_mode>(); \
        for (size_t i = 0; i < size / sizeof(*data); i++) {    \
            data[i] = sycl_host_acc[i];                        \
        }                                                      \
        break;                                                 \
    }
        switch (data_type_) {
            MKLDNN_HANDLE_ALL_TYPES
        default: assert(!"not expected"); break;
        }
#undef MKLDNN_CASE
    }

private:
    mkldnn_data_type_t data_type_;
    void *buf_ptr_ = nullptr;
};

} // namespace sycl

#undef MKLDNN_HANDLE_ALL_TYPES

#endif

} // namespace impl
} // namespace mkldnn

#endif
