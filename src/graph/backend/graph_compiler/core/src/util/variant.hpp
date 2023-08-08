/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_VARIANT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_VARIANT_HPP
#include <stdexcept>
#include <stdint.h>
#include <utility>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace variant_impl {
using std::size_t;
template <size_t arg1, size_t... args>
struct const_max;

template <size_t arg1, size_t arg2, size_t... args>
struct const_max<arg1, arg2, args...> {
    static constexpr size_t value = arg1 >= arg2
            ? const_max<arg1, args...>::value
            : const_max<arg2, args...>::value;
};

template <size_t v>
struct const_max<v> {
    static constexpr size_t value = v;
};

template <typename... Args>
struct helper;

template <typename T, typename... Args>
struct helper<T, Args...> {
    using inner = helper<Args...>;
    static constexpr size_t id = sizeof...(Args) + 1;

    template <typename T2, bool is_same>
    struct handle_get_id {
        static constexpr size_t call() { return id; }
        static constexpr bool has_type() { return true; }
    };

    template <typename T2>
    struct handle_get_id<T2, false> {
        static constexpr size_t call() { return inner::template get_id<T2>(); }
        static constexpr bool has_type() {
            return inner::template has_type<T2>();
        }
    };

    template <typename T2>
    static constexpr size_t get_id() {
        return handle_get_id<T2, std::is_same<T2, T>::value>::call();
    }

    template <typename T2>
    static constexpr bool has_type() {
        return handle_get_id<T2, std::is_same<T2, T>::value>::has_type();
    }

    static void move(size_t curid, void *src, void *dst) {
        if (id == curid) {
            new (dst) T(std::move(*reinterpret_cast<T *>(src)));
        } else {
            inner::move(curid, src, dst);
        }
    }

    // use the name copyit instead of copy to make linter happy
    template <typename T2 = T>
    static void copyit(size_t curid, const void *src, void *dst) {
        if (id == curid) {
            new (dst) T(*reinterpret_cast<const T *>(src));
        } else {
            inner::copyit(curid, src, dst);
        }
    }

    static void destroy(size_t curid, void *data) {
        if (id == curid) {
            reinterpret_cast<T *>(data)->~T();
        } else {
            inner::destroy(curid, data);
        }
    }

    template <typename T2, bool is_T2_parent_of_T>
    struct handle_as {
        // case when is_T2_parent_of_T=true
        static T2 *call(size_t curid, void *data) {
            if (id == curid) {
                // pointer of T and T2 may not be the same, if T2 is not the
                // only base class of T. Use static_cast to ensure correct
                // casting
                return static_cast<T2 *>(reinterpret_cast<T *>(data));
            }
            return inner::template as<T2>(curid, data);
        }
    };

    template <typename T2>
    struct handle_as<T2, false> {
        // case when is_T2_parent_of_T=false
        static T2 *call(size_t curid, void *data) {
            return inner::template as<T2>(curid, data);
        }
    };

    template <typename T2>
    static T2 *as(size_t curid, void *data) {
        constexpr bool T2_is_base_of_T
                = std::is_same<T2, T>::value || std::is_base_of<T2, T>::value;
        return handle_as<T2, T2_is_base_of_T>::call(curid, data);
    }

    template <typename T2, bool is_T_convertible_to_T2>
    struct handle_cast {
        // case when is_T2_parent_of_T=true
        static T2 call(size_t curid, const void *data) {
            if (id == curid) { return *reinterpret_cast<const T *>(data); }
            return inner::template cast<T2>(curid, data);
        }
    };

    template <typename T2>
    struct handle_cast<T2, false> {
        // case when is_T2_parent_of_T=false
        static T2 call(size_t curid, const void *data) {
            return inner::template cast<T2>(curid, data);
        }
    };

    template <typename T2>
    static T2 cast(size_t curid, const void *data) {
        return handle_cast<T2, std::is_convertible<T, T2>::value>::call(
                curid, data);
    }
};

template <>
struct helper<> {
    static void move(size_t curid, void *src, void *dst) {}
    static void copyit(size_t curid, const void *src, void *dst) {}
    static void destroy(size_t curid, void *data) {}

    template <typename T2>
    static constexpr bool has_type() {
        return false;
    }

    template <typename T2>
    static T2 *as(size_t curid, void *data) {
        return nullptr;
    }

    template <typename T2>
    static T2 cast(size_t curid, const void *data) {
        throw std::runtime_error("Bad variant cast");
    }
};

template <typename T, typename variantT>
struct copy_or_move_handler {
    static_assert(variantT::helper_t::template has_type<T>(),
            "The variant does not include this type");
    static void call(variantT &ths, T &&src) {
        auto id = ths.template get_id_of_type<T>();
        new (&ths.data_) T(std::move(src));
        ths.id_ = id;
    }

    // using template to avoid instantiating the use of copy ctor when it is
    // never used
    template <typename T2 = variantT>
    static void call(variantT &ths, const T &src) {
        auto id = ths.template get_id_of_type<T>();
        new (&ths.data_) T(src);
        ths.id_ = id;
    }
};

template <typename variantT>
struct copy_or_move_handler<variantT, variantT> {
    static void call(variantT &ths, variantT &&other) {
        variantT::helper_t::move(other.id_, &other.data_, &ths.data_);
        ths.id_ = other.id_;
        other.id_ = 0;
    }

    // using template to avoid instantiating the use of copy ctor when it is
    // never used
    template <typename T2 = variantT>
    static void call(variantT &ths, const variantT &other) {
        variantT::helper_t::copyit(other.id_, &other.data_, &ths.data_);
        ths.id_ = other.id_;
    }
};

} // namespace variant_impl

/**
 * @brief The type-safe tagged union. The candidates types are decided by the
 * template arguments Args. When getting the values from the variant, it will
 * check the real type stored in the variant at the run time. It supports
 * move/copy assignment and constructor. The variant is by default "empty". An
 * empty variant has not any valid data.
 * @note Each type in a variant type has a unique type id in the current
 * variant's scope. Type id 0 is for empty variant without any value. The last
 * type of the candidate types has type id 1. The type id is increased by 1 for
 * each type from right to left.
 *
 * @tparam Args the candidate types. The size and the alignment of variant will
 * be the max values of those in candidate types.
 */
template <typename... Args>
struct variant {
private:
    using size_t = std::size_t;
    template <typename T>
    using copy_or_move_handler
            = variant_impl::copy_or_move_handler<typename std::decay<T>::type,
                    variant>;
    static const size_t size = variant_impl::const_max<sizeof(Args)...>::value;
    static const size_t alignment
            = variant_impl::const_max<alignof(Args)...>::value;

    using buffer_t = typename std::aligned_storage<size, alignment>::type;
    size_t id_ = 0;
    buffer_t data_;

    template <typename T, typename variantT>
    friend struct variant_impl::copy_or_move_handler;

public:
    using helper_t = variant_impl::helper<Args...>;
    // the variant is by default "empty". An empty variant has not any valid
    // data
    variant() = default;

    // copy/move from variant/candidate types
    template <typename T>
    variant(T &&v) {
        copy_or_move_handler<T>::call(*this, std::forward<T>(v));
    }

    // copy/move assign from variant/candidate types
    template <typename T>
    variant &operator=(T &&other) {
        clear();
        copy_or_move_handler<T>::call(*this, std::forward<T>(other));
        return *this;
    }

    // check if the stored type is exactly the same as T
    template <typename T>
    bool isa() const {
        return id_ == get_id_of_type<T>();
    }

    // get the type index of the stored value. returns 0 if it is empty.
    size_t get_id() const { return id_; }

    // get the type index of a type in candidates
    template <typename T>
    static constexpr size_t get_id_of_type() {
        static_assert(helper_t::template has_type<T>(),
                "The variant does not include this type");
        return helper_t::template get_id<typename std::decay<T>::type>();
    }

    // return true of the type is in candidates
    template <typename T>
    static constexpr bool has_type() {
        return helper_t::template has_type<typename std::decay<T>::type>();
    }

    // return true of the variant is not empty
    bool defined() const { return id_ != 0; }

    // get the stored object. If the stored type is not exactly the same as T,
    // throw an error.
    template <typename T>
    T &get() {
        if (id_ == get_id_of_type<T>()) {
            return *reinterpret_cast<T *>(&data_);
        } else {
            throw std::runtime_error("Bad variant cast");
        }
    }

    // get the stored object. If the stored type is not exactly the same as T,
    // throw an error.
    template <typename T>
    const T &get() const {
        if (id_ == get_id_of_type<T>()) {
            return *reinterpret_cast<const T *>(&data_);
        } else {
            throw std::runtime_error("Bad variant cast");
        }
    }

    ~variant() { helper_t::destroy(id_, &data_); }

    // get the stored object's pointer, if T is same as or base type of the
    // stored type. Otherwise, return null. Note that if the stored type is X,
    // it returns a pointer to X
    template <typename T>
    const T *as_or_null() const {
        return helper_t::template as<typename std::decay<T>::type>(
                id_, const_cast<buffer_t *>(&data_));
    }

    // get the stored object's pointer, if T is same as or base type of the
    // stored type. Otherwise, return null. Note that if the stored type is X,
    // it returns a pointer to X
    template <typename T>
    T *as_or_null() {
        return helper_t::template as<typename std::decay<T>::type>(id_, &data_);
    }

    // get the stored object's reference, if T is same as or base type of the
    // stored type. Otherwise, throw an error
    template <typename T>
    T &as() {
        auto ret = as_or_null<T>();
        if (!ret) { throw std::runtime_error("Bad variant cast"); }
        return *ret;
    }

    // get the stored object's reference, if T is same as or base type of the
    // stored type. Otherwise, throw an error
    template <typename T>
    const T &as() const {
        auto ret = as_or_null<T>();
        if (!ret) { throw std::runtime_error("Bad variant cast"); }
        return *ret;
    }

    // Convert the stored object to type T and return the converted object, if
    // the stored type can be converted to T. Otherwise, throw an error
    template <typename T>
    T cast() const {
        return helper_t::template cast<T>(id_, &data_);
    }

    // Clear the variant. After calling this function, the variant object will
    // be empty.
    void clear() {
        helper_t::destroy(id_, &data_);
        id_ = 0;
    }
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
