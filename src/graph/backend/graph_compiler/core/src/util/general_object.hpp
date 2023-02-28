
/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_GENERAL_OBJECT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_GENERAL_OBJECT_HPP

#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <type_traits>
#include <unordered_map>
#include <util/def.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
struct any_t;
namespace reflection {
struct class_metadata;
struct vector_metadata;

enum class basic_type {
    t_int32_t,
    t_int64_t,
    t_uint32_t,
    t_uint64_t,
    t_float,
    t_double,
    t_bool,
    t_string,
    t_class,
};

struct type {
    basic_type base_;
    unsigned array_depth_;
    // only if base_ == t_class, meta_ is valid
    class_metadata *meta_;
    constexpr bool operator==(const type &other) const {
        return other.base_ == base_ && other.array_depth_ == array_depth_
                && other.meta_ == meta_;
    }
    // three-way compares with another type, this<other => -1, this==other => 0,
    // this>other => 1.
    SC_INTERNAL_API int cmp(const type &other) const;
    SC_INTERNAL_API size_t size() const;

    /**
     * Gets the unique string representation of a type.
     * int32 => i
     * int64 => i
     * uint32 => u
     * uint64 => U
     * float => f
     * double => d
     * bool => b
     * string => s
     * class => class name in class_metadata
     * */
    SC_INTERNAL_API std::string to_string() const;
};

// any class extending this class will be tagged "reflection enabled".
// json reader and writer will use reflection on this class
struct reflection_enabled_t {};

template <typename T, typename Dummy = int>
struct type_registry;

/**
 * The type-erased container for user-defined classes.
 * It holds the concrete values of an object
 * */
struct SC_INTERNAL_API general_object_t {
    // the type-erased pointer to the user-defined object
    std::unique_ptr<char[]> data_;
    // the function table of a user-defined class
    std::shared_ptr<class_metadata> vtable_;
    ~general_object_t() { release(); }
    general_object_t(general_object_t &&other);
    general_object_t(std::unique_ptr<char[]> &&data,
            const std::shared_ptr<class_metadata> &vtable);
    general_object_t() : vtable_(nullptr) {}

    general_object_t(const general_object_t &other) = delete;
    general_object_t &operator=(const general_object_t &other) = delete;

    void release();

    std::unique_ptr<void, void (*)(void *)> move_to_unique_ptr();

    template <typename T>
    static general_object_t make() {
        static_assert(std::is_class<T>::value,
                "general_object_t::make must be applied on a class");
        return type_registry<T>::metadata()->make_instance();
    }

    template <typename T>
    static general_object_t make(T &&v) {
        using decay_t = typename std::decay<T>::type;
        auto ret = make<decay_t>();
        *ret.template unchecked_get_as<decay_t>() = std::forward<T>(v);
        return ret;
    }

    template <typename T>
    T *get_as() const {
        assert(type_registry<T>::metadata() == vtable_.get());
        return reinterpret_cast<T *>(data_.get());
    }

    template <typename T>
    T *unchecked_get_as() const {
        return reinterpret_cast<T *>(data_.get());
    }

    general_object_t &operator=(general_object_t &&other) {
        release();
        data_ = std::move(other.data_);
        vtable_ = other.vtable_;
        other.vtable_ = nullptr;
        return *this;
    }

    void copy_from(const std::unordered_map<std::string, any_t> &m);
    void copy_to(std::unordered_map<std::string, any_t> &m);

    static void copy_from_any_map(
            const std::unordered_map<std::string, any_t> &m, void *object,
            class_metadata *vtable);
    static void copy_to_any_map(std::unordered_map<std::string, any_t> &m,
            void *object, class_metadata *vtable);

    void *get() const { return data_.get(); }
};

/**
 * The type-erased container for user-defined classes. It has shared ownership
 * of the pointer
 * */
struct SC_INTERNAL_API shared_general_object_t {
    // the type-erased pointer to the user-defined object
    std::shared_ptr<void> data_;
    // the function table of a user-defined class
    std::shared_ptr<class_metadata> vtable_;

    shared_general_object_t(std::nullptr_t) {};
    shared_general_object_t() = default;
    shared_general_object_t(const shared_general_object_t &) = default;
    shared_general_object_t(shared_general_object_t &&) = default;
    shared_general_object_t(general_object_t &&other) {
        *this = std::move(other);
    }

    shared_general_object_t &operator=(general_object_t &&other) {
        vtable_ = other.vtable_;
        data_ = other.move_to_unique_ptr();
        return *this;
    }

    shared_general_object_t &operator=(shared_general_object_t &&other) {
        vtable_ = std::move(other.vtable_);
        data_ = std::move(other.data_);
        return *this;
    }

    shared_general_object_t &operator=( // NOLINT
            const shared_general_object_t &other) { // NOLINT
        vtable_ = other.vtable_;
        data_ = other.data_;
        return *this;
    }

    template <typename T>
    T *get_as() const {
        assert(type_registry<T>::metadata() == vtable_.get());
        return reinterpret_cast<T *>(data_.get());
    }

    template <typename T>
    T *unchecked_get_as() const {
        return reinterpret_cast<T *>(data_.get());
    }

    operator bool() const { return bool(data_); }

    void *get() const { return data_.get(); };
};

// General reference. Similar to general_object_t, except that this is a borrow
// of the pointer and do not take the ownership
struct SC_INTERNAL_API general_ref_t {
    // the type-erased pointer to the user-defined object
    void *data_;
    // the type of the data
    type type_;

    general_ref_t() : data_(nullptr) {}
    general_ref_t(void *data, type tp) : data_(data), type_(tp) {}
    template <typename T>
    static general_ref_t from(T &obj) {
        type thetype = type_registry<typename std::decay<T>::type>::type_;
        return {(void *)&obj, thetype};
    }

    static general_ref_t from(general_object_t &obj);
    static general_ref_t from(const shared_general_object_t &obj);
    static general_ref_t from(shared_general_object_t &obj);
    static general_ref_t from(const general_object_t &obj);
    bool cmp_equal(general_ref_t ori_param) const;
    // three-way compares with another ref, this<other => -1, this==other => 0,
    // this>other => 1.
    int cmp(general_ref_t other) const;
    // computes the hash code
    size_t hash() const;
};

} // namespace reflection
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
