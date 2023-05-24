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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_ANY_MAP_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_ANY_MAP_HPP

#include <iostream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>
#include "assert.hpp"
#include "def.hpp"
#include <type_traits>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace reflection {
struct type;
}

namespace any_detail {

using unary_func = void (*)(void *);
using binary_func = void (*)(void *ths, void *other);
using binary_const_func = void (*)(void *ths, const void *other);
struct any_vtable_t {
    size_t size_;
    const std::type_info &typeinfo_;
    unary_func destructor_;
    // move assign: operator =(T&&)
    binary_func move_assigner_;
    // move constructor: T(T&&)
    binary_func move_ctor_;
    // copy assign: operator =(const T&)
    binary_const_func copy_assigner_;
    // copy constructor: T(const T&)
    binary_const_func copy_ctor_;
    any_vtable_t(size_t size, const std::type_info &typeinfo,
            unary_func destructor, binary_func move_assigner,
            binary_func move_ctor, binary_const_func copy_assigner,
            binary_const_func copy_ctor)
        : size_(size)
        , typeinfo_(typeinfo)
        , destructor_(destructor)
        , move_assigner_(move_assigner)
        , move_ctor_(move_ctor)
        , copy_assigner_(copy_assigner)
        , copy_ctor_(copy_ctor) {
        set_rtti_to_vtable_map(&typeinfo_, this);
    }

    SC_API static void set_rtti_to_vtable_map(
            const std::type_info *typeinfo, any_vtable_t *);
    SC_API static any_vtable_t *get_vtable_by_rtti(
            const std::type_info *typeinfo);
};

template <typename T>
struct registry;

// if is_move_assignable, use move
template <bool movable, typename T>
struct move_assign_impl_t {
    static inline void call(void *ths, void *other) {
        T *ptr = reinterpret_cast<T *>(ths);
        T *oth = reinterpret_cast<T *>(other);
        *ptr = std::move(*oth);
    }
};

// if not is_move_assignable
template <typename T>
struct move_assign_impl_t<false, T> {
    static constexpr void (*call)(void *ths, void *other) = nullptr;
};

// if is_move_constructible, use move
template <bool movable, typename T>
struct move_constru_impl_t {
    static inline void call(void *ths, void *other) {
        T *oth = reinterpret_cast<T *>(other);
        new (ths) T(std::move(*oth));
    }
};

// if not is_move_constructible
template <typename T>
struct move_constru_impl_t<false, T> {
    static constexpr void (*call)(void *ths, void *other) = nullptr;
};

// if is_copy_assignable, use copy
template <bool copyable, typename T>
struct copy_assign_impl_t {
    static inline void call(void *ths, const void *other) {
        T *ptr = reinterpret_cast<T *>(ths);
        const T *oth = reinterpret_cast<const T *>(other);
        *ptr = *oth;
    }
};

// if not is_copy_assignable
template <typename T>
struct copy_assign_impl_t<false, T> {
    static constexpr void (*call)(void *ths, const void *other) = nullptr;
};

// if is_copy_constructible, use copy
template <bool copyable, typename T>
struct copy_constru_impl_t {
    static inline void call(void *ths, const void *other) {
        const T *oth = reinterpret_cast<const T *>(other);
        new (ths) T(*oth);
    }
};

// if not is_copy_constructible
template <typename T>
struct copy_constru_impl_t<false, T> {
    static constexpr void (*call)(void *ths, const void *other) = nullptr;
};

template <typename T>
struct destructor_impl_t {
    static inline void destructor(void *p) {
        T *ptr = reinterpret_cast<T *>(p);
        ptr->~T();
    }
};

template <typename T, std::size_t N>
struct destructor_impl_t<T[N]> {
    using impl_t = std::array<T, N>;
    static inline void destructor(void *p) {
        auto *ptr = reinterpret_cast<impl_t *>(p);
        ptr->~impl_t();
    }
};

template <typename T>
struct registry {
    static SC_API any_vtable_t vtable;
    static constexpr any_vtable_t *get_vtable() { return &vtable; }
};

#if defined(__clang__) || !defined(SC_DLL) || defined(SC_DLL_EXPORTS)
template <typename T>
SC_API any_vtable_t registry<T>::vtable(sizeof(T), typeid(T),
        destructor_impl_t<T>::destructor,
        move_assign_impl_t<std::is_move_assignable<T>::value, T>::call,
        move_constru_impl_t<std::is_move_constructible<T>::value, T>::call,
        copy_assign_impl_t<std::is_copy_assignable<T>::value, T>::call,
        copy_constru_impl_t<std::is_copy_constructible<T>::value, T>::call);
#endif

template <typename T>
struct assign_impl;

// pattern matching impl

template <typename Ret, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (F::*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (F::*)(Arg, Rest...) const);

template <typename F>
decltype(first_argument_helper(&F::operator())) first_argument_helper(F);

template <typename T>
using first_argument = typename std::decay<decltype(
        first_argument_helper(std::declval<T>()))>::type;
} // namespace any_detail

/**
 * The type-erased container for any type. It will save the
 * destructor/copyer/mover to correctly destruct/move/copy the contained object.
 * assume we have `any_t` variables: any_val and any_val2. We also have a
 * variable of type `T`: var. Requirements on the type `T`:
 *  1. in the assignment `any_val = var`:
 *      a) if any_val.isa<T>(), then requires that type T is copy assignable
 *      b) else, requires that type T is copy constructible
 *  2. in the assignment `any_val = std::move(var)`:
 *      a) if any_val.isa<T>(), then requires that type T is move assignable
 *      b) else, requires that type T is move constructible
 *  3. in the assignment `any_val = any_val2`:
 *      a) if any_val has the same type of any_val2, then requires that type T
 *          is copy assignable
 *      b) else, requires that type T is copy constructible
 *  4. in the assignment `any_val = std::move(any_val2)`:
 *      requires nothing. It will move the data pointer from any_val2 to any_val
 *  5. in the construction `any_t new_val = any_val`, requires that T is copy
 *      constructible
 *  6. in the construction `any_t new_val = std::move(any_val2)`, it requires
 *     nothing. It will move the data pointer from any_val2 to any_val
 *
 *  We can roughly say: if you want to move an `any_t`, the object in it should
 * be movable. if you want to copy an `any_t`, the object in it should be
 * copyable
 *
 * If the requirement is not satisfied, will abort
 * @note simple types (e.g. int, float, bool) are allowed here
 * */
struct any_t {
    static constexpr size_t INLINE_BUFFER_SIZE = 64 - sizeof(void *);

private:
    // for large objects with size > INLINE_BUFFER_SIZE bytes, `any_t` will
    // allocate a buffer on the heap for it and internally store a pointer to
    // the object. For small objects <= INLINE_BUFFER_SIZE bytes, it will use
    // the `inlined_buffer_` to store it.
    union buffer_or_pointer {
        char *ptr_;
        char inlined_buffer_[INLINE_BUFFER_SIZE];
        buffer_or_pointer() : inlined_buffer_ {0} {}
    } data_;
    const any_detail::any_vtable_t *vtable_ = nullptr;
    template <class T>
    friend struct any_detail::assign_impl;

    SC_API void create_buffer(const any_detail::any_vtable_t *vt);

    // switches the buffer to a type. If we are already holding this type, do
    // nothing and return true. Else, we release the held object, create new
    // buffer, change the vtable and return false
    SC_API bool switch_buffer_to_type(const any_detail::any_vtable_t *vt);

    SC_API void copy_from(const void *data, const any_detail::any_vtable_t *vt);

    SC_API void move_from(void *data, const any_detail::any_vtable_t *vt);

    // if we move from any_t, we can simply "steal" the pointer and vtable
    SC_API void move_from_any(any_t &&v);

public:
    void *get_raw() {
        if (!vtable_) { return nullptr; }
        if (vtable_->size_ <= INLINE_BUFFER_SIZE) {
            return &data_.inlined_buffer_[0];
        } else {
            return data_.ptr_;
        }
    }
    const void *get_raw() const { return const_cast<any_t *>(this)->get_raw(); }
    /**
     * Clears the contained object. Calls the destructor and free the buffer.
     * */
    SC_API void clear();

    /**
     * Compares this with other. Requires that both this and other's type have
     * reflection::type registered. (Basic types like int and classes registered
     * in reflection are okay.). If the requirement is not satisfied or any_t of
     * this or other is empty, it will throw an exception
     * @return -1 if *this<other. 0 if *this==other. 1 if *this>other.
     * */
    SC_API int cmp(const any_t &other) const;

    /**
     * Computes the hash code , using reflection::general_ref_t::hash. Requires
     * that the type have reflection::type registered. If the requirement is not
     * satisfied or this is empty, it will throw an exception
     * */
    SC_API size_t hash() const;
    /**
     * Pattern matching on the type of the contained value of this any_t
     * The parameters starting from the second are the functions that matches
     * the contained type. They should have the signature of `void (T)`, where T
     * can be a constant reference or a base type. If the contained type in this
     * any_t matches T, the function will be called. The first parameter of
     * `match(...)` is a function, which will be called when no type is matched.
     * */
    template <typename T, typename T1, typename... Args>
    bool match(T defaults, T1 func1, Args &&...args) const {
        using MatchedT = any_detail::first_argument<T1>;
        if (isa<MatchedT>()) {
            func1(get<MatchedT>());
            return true;
        }
        return match(std::forward<T>(defaults), std::forward<Args>(args)...);
    }

    template <typename T, typename T1>
    bool match(T defaults, T1 func1) const {
        using MatchedT = any_detail::first_argument<T1>;
        if (isa<MatchedT>()) {
            func1(get<MatchedT>());
            return true;
        }
        defaults();
        return false;
    }

    any_t() = default;

    // keep this overload function to make compiler happy
    any_t(const any_t &v) { copy_from(v.get_raw(), v.vtable_); }
    // keep this overload function to make compiler happy
    any_t(any_t &&v) { move_from_any(std::move(v)); }

    template <typename T>
    any_t(T &&v) {
        *this = std::forward<T>(v);
    }

    // C++ templates cannot correctly override T&& and const T& (all goes to
    // T&&), so we need to use std::decay to choose the correct version
    template <typename T>
    any_t &operator=(T &&v) {
        any_detail::assign_impl<typename std::decay<T>::type>::call(
                *this, std::forward<T>(v));
        return *this;
    }

    // keep this overload function to make compiler happy
    any_t &operator=(any_t &&v) {
        if (this == &v) return *this;
        move_from_any(std::move(v));
        return *this;
    }
    // keep this overload function to make compiler happy
    any_t &operator=(const any_t &v) {
        if (this == &v) return *this;
        copy_from(v.get_raw(), v.vtable_);
        return *this;
    }

    ~any_t() { clear(); }

    template <typename T>
    void set(const T &v) {
        *this = v;
    }

    template <typename T>
    static any_detail::any_vtable_t *get_vtable() {
        return any_detail::registry<typename std::decay<T>::type>::get_vtable();
    }

    template <typename T>
    T *get_or_null() {
        if (!isa<T>()) { return nullptr; }
        T *ptr = reinterpret_cast<T *>(get_raw());
        return ptr;
    }

    template <typename T>
    const T *get_or_null() const {
        return const_cast<any_t *>(this)->get_or_null<T>();
    }

    template <typename T>
    T &get() {
        COMPILE_ASSERT(isa<T>(),
                "Incorrect type for any_t::get, this = "
                        << vtable_->typeinfo_.name() << ", expected "
                        << get_vtable<T>()->typeinfo_.name());
        T *ptr = reinterpret_cast<T *>(get_raw());
        return *ptr;
    }

    template <typename T>
    const T &get() const {
        return const_cast<any_t *>(this)->get<T>();
    }

    // returns if the `any_t` has an object of type T
    template <typename T>
    bool isa() const {
        return vtable_ == get_vtable<T>();
    }

    // returns the type_info of the contained object. If is empty, return null
    const std::type_info *type_code() const {
        return vtable_ ? &vtable_->typeinfo_ : nullptr;
    }

    const any_detail::any_vtable_t *vtable() const { return vtable_; }

    // returns true if there is any value in this `any_t`
    bool empty() const { return !vtable_; }

    // makes an any_t by the reflection type
    static SC_API any_t make_by_type(const reflection::type *type);
};

namespace any_detail {
template <typename T>
struct assign_impl {
    inline static void call(any_t &ths, const T &v) {
        ths.copy_from(&v, any_detail::registry<T>::get_vtable());
    }

    inline static void call(any_t &ths, T &&v) {
        ths.move_from(&v, any_detail::registry<T>::get_vtable());
    }
};

template <>
struct assign_impl<any_t> {
    inline static void call(any_t &ths, const any_t &v) {
        ths.copy_from(v.get_raw(), v.vtable_);
    }

    inline static void call(any_t &ths, any_t &&v) {
        ths.move_from_any(std::move(v));
    }
};

template <>
struct assign_impl<const char *> {
    inline static void call(any_t &ths, const char *v) {
        assign_impl<std::string>::call(ths, std::string(v));
    }
};

template <>
struct assign_impl<char *> {
    inline static void call(any_t &ths, char *v) {
        assign_impl<const char *>::call(ths, v);
    }
};
} // namespace any_detail

struct SC_API any_map_t {
private:
    std::unordered_map<std::string, any_t> impl_;

public:
    any_map_t() = default;
    any_map_t(const std::unordered_map<std::string, any_t> &impl)
        : impl_(impl) {}
    any_map_t(std::initializer_list<std::pair<const std::string, any_t>> init)
        : impl_(init) {}

    size_t size() const { return impl_.size(); };

    const std::unordered_map<std::string, any_t> &as_map() const {
        return impl_;
    }
    std::unordered_map<std::string, any_t> &as_map() { return impl_; }

    any_t &get_any(const std::string &v);

    const any_t &get_any(const std::string &v) const;

    template <typename T>
    T &get(const std::string &v) {
        return get_any(v).get<T>();
    };

    template <typename T>
    const T &get(const std::string &v) const {
        return get_any(v).get<T>();
    };

    // gets a value. if key is not found, returns default_value
    template <typename T>
    const T &get_or_else(const std::string &k, const T &default_value) const {
        if (!has_key(k)) { return default_value; }
        return get_any(k).get<T>();
    };

    // gets the pointer of a value. if key is not found, returns nullptr
    template <typename T>
    T *get_or_null(const std::string &k) {
        if (!has_key(k)) { return nullptr; }
        return &get_any(k).get<T>();
    };

    // gets the pointer of a value. if key is not found, returns nullptr
    template <typename T>
    const T *get_or_null(const std::string &k) const {
        if (!has_key(k)) { return nullptr; }
        return &get_any(k).get<T>();
    };

    template <typename T>
    void set(const std::string &k, const T &v) {
        if (has_key(k)) {
            get_any(k) = v;
        } else {
            any_t anyv = v;
            impl_.insert(std::make_pair(k, std::move(anyv)));
        }
    };

    bool has_key(const std::string &v) const {
        auto itr = impl_.find(v);
        return itr != impl_.end();
    }

    void remove(const std::string &v) { impl_.erase(v); }

    any_t &operator[](const std::string &v) {
        if (!has_key(v)) { impl_.insert(std::make_pair(v, any_t())); }
        return get_any(v);
    }
    /**
     * Compares the map with other using reflection
     * Requires that the types in it have reflection::type registered. If not,
     * it will throw an exception
     * */
    bool operator==(const any_map_t &other) const;

    bool operator!=(const any_map_t &other) const { return !(*this == other); }

    /**
     * Computes the hash code using reflection
     * Requires that the types in it have reflection::type registered. If not,
     * it will throw an exception
     * */
    size_t hash() const;

    // a static version for get_or_else. It also returns defaultv when v is null
    template <typename T>
    inline static const T fetch_or_else(
            const any_map_t *v, const std::string &k, const T &defaultv) {
        if (!v) { return defaultv; }
        return v->get_or_else(k, defaultv);
    }

    // a static version for get_or_null. It also returns null when v is null
    template <typename T>
    inline static const T *fetch_or_null(
            const any_map_t *v, const std::string &k) {
        if (!v) { return nullptr; }
        return v->get_or_null<T>(k);
    }
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
