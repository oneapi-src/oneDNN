/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_COMPUTE_KERNEL_ARG_LIST_HPP
#define GPU_INTEL_COMPUTE_KERNEL_ARG_LIST_HPP

#include <cassert>
#include <cstddef>
#include <string>
#include <type_traits>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/memory_storage.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
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
    _f4_e2m1,
    _f4_e3m0,
    _hfloat8,
    _bfloat8,
    _bfloat16,
    _float,
    _double,
    _half,
    _int,
    _int4,
    _long,
    _short,
    _uchar,
    _uint,
    _uint4,
    _ulong,
    _ushort,
    _zero_pad_mask_t,
    _int64x2_t,
    _int64x3_t,
    _int64x4_t,
    _int64x5_t,
    _int64x6_t,
    _dispatch_gws_rt_params_t,
};

inline std::string to_string(scalar_type_t type) {
#define CASE(x) \
    case scalar_type_t::x: return #x

    switch (type) {
        CASE(undef);
        CASE(_char);
        CASE(_f4_e2m1);
        CASE(_f4_e3m0);
        CASE(_hfloat8);
        CASE(_bfloat8);
        CASE(_bfloat16);
        CASE(_float);
        CASE(_double);
        CASE(_half);
        CASE(_int);
        CASE(_int4);
        CASE(_long);
        CASE(_short);
        CASE(_uchar);
        CASE(_uint);
        CASE(_uint4);
        CASE(_ulong);
        CASE(_ushort);
        CASE(_zero_pad_mask_t);
        CASE(_int64x2_t);
        CASE(_int64x3_t);
        CASE(_int64x4_t);
        CASE(_int64x5_t);
        CASE(_int64x6_t);
        CASE(_dispatch_gws_rt_params_t);
    }
    return "unexpected";
#undef CASE
}

template <typename T>
struct scalar_type_traits_t {};

template <>
struct scalar_type_traits_t<float16_t> {
    static const auto type = scalar_type_t::_half;
};
template <>
struct scalar_type_traits_t<bfloat16_t> {
    static const auto type = scalar_type_t::_bfloat16;
};
template <>
struct scalar_type_traits_t<float> {
    static const auto type = scalar_type_t::_float;
};
template <>
struct scalar_type_traits_t<double> {
    static const auto type = scalar_type_t::_double;
};

template <>
struct scalar_type_traits_t<uint8_t> {
    static const auto type = scalar_type_t::_uchar;
};
template <>
struct scalar_type_traits_t<uint16_t> {
    static const auto type = scalar_type_t::_ushort;
};
template <>
struct scalar_type_traits_t<uint32_t> {
    static const auto type = scalar_type_t::_uint;
};
template <>
struct scalar_type_traits_t<uint64_t> {
    static const auto type = scalar_type_t::_ulong;
};

template <>
struct scalar_type_traits_t<int8_t> {
    static const auto type = scalar_type_t::_char;
};
template <>
struct scalar_type_traits_t<int16_t> {
    static const auto type = scalar_type_t::_short;
};
template <>
struct scalar_type_traits_t<int32_t> {
    static const auto type = scalar_type_t::_int;
};
template <>
struct scalar_type_traits_t<int64_t> {
    static const auto type = scalar_type_t::_long;
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
        scalar_type_ = scalar_type_traits_t<T>::type;
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
        assert(scalar_type() == scalar_type_traits_t<T>::type);
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
    kernel_arg_list_t() { args_.reserve(512); }

    void append(const memory_storage_t &storage) {
        args_.emplace_back();
        args_.back().set_value(storage);
    }

    void append(void *value, kernel_arg_kind_t kind) {
        args_.emplace_back();
        args_.back().set_value(value, kind);
    }

    template <class T>
    void append(const T &value) {
        args_.emplace_back();
        args_.back().set_value(value, unused_storage);

        assert(unused_storage
                <= reinterpret_cast<char *>(&scalar_storage_) + storage_size);
    }

    void append(size_t size, std::nullptr_t) {
        args_.emplace_back();
        args_.back().set_value(size, nullptr);
    }

    void set(int index, const memory_storage_t &storage) {
        assert(index < storage_size);
        if ((index + 1) > nargs()) { args_.resize(index + 1); };
        args_[index].set_value(storage);
    }

    void set(int index, void *value, kernel_arg_kind_t kind) {
        assert(index < storage_size);
        if ((index + 1) > nargs()) { args_.resize(index + 1); };
        args_[index].set_value(value, kind);
    }

    template <class T>
    void set(int index, const T &value) {
        assert(index < storage_size);
        if ((index + 1) > nargs()) { args_.resize(index + 1); };
        args_[index].set_value(value, unused_storage);

        assert(unused_storage
                <= reinterpret_cast<char *>(&scalar_storage_) + storage_size);
    }

    void set(int index, size_t size, std::nullptr_t) {
        assert(index < storage_size);
        if ((index + 1) > nargs()) { args_.resize(index + 1); };
        args_[index].set_value(size, nullptr);
    }

    int nargs() const { return static_cast<int>(args_.size()); }

    const kernel_arg_t &get(int index) const {
        assert(index < nargs());
        return args_[index];
    }

    const memory_storage_t &get_memory_storage(int index) const {
        assert(args_[index].kind() == kernel_arg_kind_t::global);
        return *static_cast<const memory_storage_t *>(args_[index].value());
    }

private:
    static constexpr int storage_size = 2048;
    static constexpr int storage_alignment = 8;

    std::vector<kernel_arg_t> args_;
    typename std::aligned_storage<storage_size, storage_alignment>::type
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
    if (scalar_type_traits_t<T>::type == requested_type) {
        arg_list.set(index, scalar);
        return;
    }

    switch (requested_type) {
        case scalar_type_t::_half:
            arg_list.set(index, (float16_t)scalar);
            break;
        case scalar_type_t::_double: arg_list.set(index, (double)scalar); break;
        case scalar_type_t::_uchar: arg_list.set(index, (uint8_t)scalar); break;
        case scalar_type_t::_char: arg_list.set(index, (int8_t)scalar); break;
        default: assert(!"Cannot convert scalar to the requested type.");
    }
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_COMPUTE_KERNEL_ARG_LIST_HPP
