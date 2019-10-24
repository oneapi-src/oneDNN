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

#ifndef COMPUTE_KERNEL_ARG_LIST_HPP
#define COMPUTE_KERNEL_ARG_LIST_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/memory_storage.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {
namespace compute {

class kernel_arg_t {
public:
    enum class kind_t {
        undef,
        global,
        local,
        scalar,
    };

    static constexpr size_t max_size = 8;

    kind_t kind() const { return kind_; }
    size_t size() const { return size_; }
    bool is_global() const { return kind_ == kind_t::global; }

    void set_value(const memory_storage_t &storage) {
        kind_ = kind_t::global;
        size_ = 0;
        value_ = static_cast<const void *>(&storage);
    }

    template <typename T,
            typename = typename std::enable_if<std::is_arithmetic<T>::value
                    || std::is_same<T, float16_t>::value
                    || std::is_same<T, bfloat16_t>::value>::type>
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
    kind_t kind_ = kind_t::undef;
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
        assert(args_[index].kind() == kernel_arg_t::kind_t::global);
        return *static_cast<const memory_storage_t *>(args_[index].value());
    }

private:
    static constexpr int max_args = 32;

    int nargs_ = 0;
    kernel_arg_t args_[max_args];
};

} // namespace compute
} // namespace impl
} // namespace dnnl

#endif // COMPUTE_KERNEL_ARG_LIST_HPP
