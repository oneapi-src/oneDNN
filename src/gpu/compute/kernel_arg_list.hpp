/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef GPU_COMPUTE_KERNEL_ARG_LIST_HPP
#define GPU_COMPUTE_KERNEL_ARG_LIST_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/memory_storage.hpp"
#include "common/nstl.hpp"
#include "common/verbose.hpp"
#include "gpu/ocl/types_interop.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

enum class kernel_arg_kind_t {
    undef,
    global,
    local,
    scalar,
    svm,
};

enum class scalar_type_t {
    undef,
    _char,
    _bfloat16,
    _float,
    _half,
    _int,
    _long,
    _short,
    _uchar,
    _uint,
    _ulong,
    _ushort,
    _zero_pad_mask_t,
    _int64x3_t,
};

template <typename T>
struct scalar_type_traits {};

template <>
struct scalar_type_traits<float16_t> {
    static const auto type = scalar_type_t::_half;
};
template <>
struct scalar_type_traits<bfloat16_t> {
    static const auto type = scalar_type_t::_bfloat16;
};
template <>
struct scalar_type_traits<float> {
    static const auto type = scalar_type_t::_float;
};

template <>
struct scalar_type_traits<uint8_t> {
    static const auto type = scalar_type_t::_uchar;
};
template <>
struct scalar_type_traits<uint16_t> {
    static const auto type = scalar_type_t::_ushort;
};
template <>
struct scalar_type_traits<uint32_t> {
    static const auto type = scalar_type_t::_uint;
};
template <>
struct scalar_type_traits<uint64_t> {
    static const auto type = scalar_type_t::_ulong;
};

template <>
struct scalar_type_traits<int8_t> {
    static const auto type = scalar_type_t::_char;
};
template <>
struct scalar_type_traits<int16_t> {
    static const auto type = scalar_type_t::_short;
};
template <>
struct scalar_type_traits<int32_t> {
    static const auto type = scalar_type_t::_int;
};
template <>
struct scalar_type_traits<int64_t> {
    static const auto type = scalar_type_t::_long;
};
template <>
struct scalar_type_traits<zero_pad_mask_t> {
    static const auto type = scalar_type_t::_zero_pad_mask_t;
};
template <>
struct scalar_type_traits<int64x3_t> {
    static const auto type = scalar_type_t::_int64x3_t;
};

class kernel_arg_t {
public:
    kernel_arg_kind_t kind() const { return kind_; }
    scalar_type_t scalar_type() const { return scalar_type_; }
    size_t size() const { return size_; }

    bool is_global() const { return kind() == kernel_arg_kind_t::global; }
    bool is_local() const { return kind() == kernel_arg_kind_t::local; }
    bool is_svm_pointer() const { return kind_ == kernel_arg_kind_t::svm; }

    kernel_arg_t &set_value(const memory_storage_t &storage) {
        kind_ = kernel_arg_kind_t::global;
        size_ = 0;
        value_ = static_cast<const void *>(&storage);
        return *this;
    }

    template <typename T>
    kernel_arg_t &set_value(const T &value, void *&data_pool) {
        assert(size_ <= sizeof(T));
        if (value_ == nullptr) {
            assert(data_pool != nullptr);
            size_ = sizeof(T);
            data_pool = utils::align_ptr(data_pool, alignof(T));
            value_ = data_pool;
            data_pool = static_cast<char *>(data_pool) + size_;
        }
        kind_ = kernel_arg_kind_t::scalar;
        scalar_type_ = scalar_type_traits<T>::type;
        new (const_cast<void *>(value_)) T(value);
        return *this;
    }

    kernel_arg_t &set_value(size_t size, std::nullptr_t) {
        kind_ = kernel_arg_kind_t::local;
        size_ = size;
        value_ = nullptr;
        return *this;
    }

    void set_value(void *svm_ptr, kernel_arg_kind_t kind) {
        assert(kind == kernel_arg_kind_t::svm);
        kind_ = kernel_arg_kind_t::svm;
        size_ = 0;
        value_ = svm_ptr;
    }

    const void *value() const {
        assert(kind() != kernel_arg_kind_t::undef);
        return value_;
    }

    template <typename T>
    T as() const {
        assert(kind() == kernel_arg_kind_t::scalar);
        assert(scalar_type() == scalar_type_traits<T>::type);
        return *(const T *)value();
    }

private:
    kernel_arg_kind_t kind_ = kernel_arg_kind_t::undef;
    scalar_type_t scalar_type_ = scalar_type_t::undef;
    size_t size_ = 0;
    const void *value_ = nullptr;
};

class kernel_arg_list_t {
public:
    kernel_arg_list_t() { nargs_ = 0; }

    void append(const memory_storage_t &storage) {
        assert(nargs_ + 1 < max_args);
        args_[nargs_++].set_value(storage);
    }

    void append(void *value, kernel_arg_kind_t kind) {
        assert(nargs_ + 1 < max_args);
        args_[nargs_++].set_value(value, kind);
    }

    template <class T>
    void append(const T &value) {
        assert(nargs_ + 1 < max_args);
        args_[nargs_++].set_value(value, unused_storage);

        assert(unused_storage
                <= reinterpret_cast<char *>(&scalar_storage_) + storage_size);
    }

    void append(size_t size, std::nullptr_t) {
        assert(nargs_ + 1 < max_args);
        args_[nargs_++].set_value(size, nullptr);
    }

    void set(int index, const memory_storage_t &storage) {
        assert(index < max_args);
        nargs_ = nstl::max(nargs_, index + 1);
        args_[index].set_value(storage);
    }

    void set(int index, void *value, kernel_arg_kind_t kind) {
        assert(index < max_args);
        nargs_ = nstl::max(nargs_, index + 1);
        args_[index].set_value(value, kind);
    }

    template <class T>
    void set(int index, const T &value) {
        assert(index < max_args);
        nargs_ = nstl::max(nargs_, index + 1);
        args_[index].set_value(value, unused_storage);

        assert(unused_storage
                <= reinterpret_cast<char *>(&scalar_storage_) + storage_size);
    }

    void set(int index, size_t size, std::nullptr_t) {
        assert(index < max_args);
        nargs_ = nstl::max(nargs_, index + 1);
        args_[index].set_value(size, nullptr);
    }

    int nargs() const { return nargs_; }

    const kernel_arg_t &get(int index) const {
        assert(index < nargs());
        return args_[index];
    }

    const memory_storage_t &get_memory_storage(int index) const {
        assert(args_[index].kind() == kernel_arg_kind_t::global);
        return *static_cast<const memory_storage_t *>(args_[index].value());
    }

private:
    static constexpr int max_args = 96;
    static constexpr int storage_size = 512;
    static constexpr int storage_alginment = 8;

    int nargs_ = 0;
    kernel_arg_t args_[max_args];
    typename std::aligned_storage<storage_size, storage_alginment>::type
            scalar_storage_;
    void *unused_storage = &scalar_storage_;

    kernel_arg_list_t(const kernel_arg_list_t &) = delete;
    kernel_arg_list_t(kernel_arg_list_t &&) = delete;
    kernel_arg_list_t &operator=(const kernel_arg_list_t &) = delete;
    kernel_arg_list_t &operator=(kernel_arg_list_t &&) = delete;
};

template <typename T>
void set_scalar_arg_cvt(kernel_arg_list_t &arg_list, int index, T scalar,
        scalar_type_t requested_type) {
    if (scalar_type_traits<T>::type == requested_type) {
        arg_list.set(index, scalar);
        return;
    }

    switch (requested_type) {
        case scalar_type_t::_half:
            arg_list.set(index, (float16_t)scalar);
            break;
        case scalar_type_t::_uchar: arg_list.set(index, (uint8_t)scalar); break;
        case scalar_type_t::_char: arg_list.set(index, (int8_t)scalar); break;
        default: assert(!"Cannot convert scalar to the requested type.");
    }
}

inline status_t check_scalar_arguments(const kernel_arg_list_t &arg_list,
        const std::vector<scalar_type_t> &arg_types) {
    // Some kernels may not support argument validation.
    if (arg_types.empty()) return status::success;

    for (int i = 0; i < arg_list.nargs(); i++) {
        auto &arg = arg_list.get(i);
        auto req_arg_type = arg_types[i];

        if (!arg.is_global() && !arg.is_local() && !arg.is_svm_pointer()) {
            if (req_arg_type == gpu::compute::scalar_type_t::undef) {
                // Types of kernel arguments may not be available when zebin
                // is used.
                continue;
            }

            if (req_arg_type != arg.scalar_type()) {
                VERROR(primitive, gpu,
                        "type of a scalar kernel argument #%d is "
                        "different from the type of the given scalar",
                        i);
                return status::invalid_arguments;
            }
        }
    }
    return status::success;
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_KERNEL_ARG_LIST_HPP
