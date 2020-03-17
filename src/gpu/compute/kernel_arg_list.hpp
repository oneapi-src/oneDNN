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

#ifndef GPU_COMPUTE_KERNEL_ARG_LIST_HPP
#define GPU_COMPUTE_KERNEL_ARG_LIST_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/memory_storage.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

enum class kernel_arg_kind_t {
    undef,
    global,
    local,
    scalar,
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

class kernel_arg_t {
public:
    static constexpr size_t max_size = 8;

    kernel_arg_kind_t kind() const { return kind_; }
    scalar_type_t scalar_type() const { return scalar_type_; }
    size_t size() const { return size_; }

    bool is_global() const { return kind() == kernel_arg_kind_t::global; }
    bool is_local() const { return kind() == kernel_arg_kind_t::local; }

    kernel_arg_t &set_value(const memory_storage_t &storage) {
        kind_ = kernel_arg_kind_t::global;
        size_ = 0;
        value_ = static_cast<const void *>(&storage);
        return *this;
    }

    template <typename T,
            typename = typename std::enable_if<std::is_arithmetic<T>::value
                    || std::is_same<T, float16_t>::value
                    || std::is_same<T, bfloat16_t>::value>::type>
    kernel_arg_t &set_value(const T &value) {
        kind_ = kernel_arg_kind_t::scalar;
        scalar_type_ = scalar_type_traits<T>::type;
        new (&scalar_storage_) T(value);
        size_ = sizeof(T);
        value_ = nullptr;
        return *this;
    }

    kernel_arg_t &set_value(size_t size, std::nullptr_t) {
        kind_ = kernel_arg_kind_t::local;
        size_ = size;
        value_ = nullptr;
        return *this;
    }

    const void *value() const {
        assert(kind() != kernel_arg_kind_t::undef);
        if (kind() == kernel_arg_kind_t::scalar)
            return static_cast<const void *>(&scalar_storage_);
        return value_;
    }

    template <typename T>
    T as() const {
        assert(kind() == kernel_arg_kind_t::scalar);
        assert(scalar_type() == scalar_type_traits<T>::type);
        return *(const T *)value();
    }

    static kernel_arg_t cast(
            scalar_type_t other_type, const kernel_arg_t &other);

private:
    kernel_arg_kind_t kind_ = kernel_arg_kind_t::undef;
    scalar_type_t scalar_type_ = scalar_type_t::undef;
    size_t size_ = 0;
    const void *value_ = nullptr;

    typename std::aligned_storage<max_size, max_size>::type scalar_storage_;
};

class kernel_arg_list_t {
public:
    void set(int index, const memory_storage_t &storage) {
        assert(index < max_args);
        nargs_ = nstl::max(nargs_, index + 1);
        args_[index].set_value(storage);
    }

    template <class T,
            typename = typename std::enable_if<std::is_arithmetic<T>::value
                    || std::is_same<T, float16_t>::value
                    || std::is_same<T, bfloat16_t>::value>::type>
    void set(int index, const T &value) {
        static_assert(
                sizeof(T) <= kernel_arg_t::max_size, "Type size is too large");

        assert(index < max_args);
        nargs_ = nstl::max(nargs_, index + 1);
        args_[index].set_value(value);
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
    static constexpr int max_args = 32;

    int nargs_ = 0;
    kernel_arg_t args_[max_args];
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_KERNEL_ARG_LIST_HPP
