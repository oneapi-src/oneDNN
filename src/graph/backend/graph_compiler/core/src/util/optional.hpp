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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_OPTIONAL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_OPTIONAL_HPP
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <utility>
#ifdef _GLIBCXX_DEBUG
#include <string.h>
#endif

// MSVC had a old bug that it can not optimize empty base (EBO) with
// multiple-inheritance. The bug was fixed but EBO is disabled by default. We
// need to enable it with this macro
#ifdef _MSC_VER
#define SC_EMPTY_BASE_OPTIMIZE __declspec(empty_bases)
#else
#define SC_EMPTY_BASE_OPTIMIZE
#endif

namespace std {
template <class T, class Deleter>
class unique_ptr;

template <class T>
class shared_ptr;
} // namespace std

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

template <typename T>
struct optional;

namespace optional_impl {
using std::size_t;

template <typename T>
struct optional_base {
    bool has_value_;

    void init_as_empty(void *v) {
        has_value_ = false;
        // make GCC _GLIBCXX_DEBUG mode happy, or it may complain that it is
        // uninitialized
#ifdef _GLIBCXX_DEBUG
        ::memset(v, 0, sizeof(T));
#endif
    }
    void set_has_value(void *v) { has_value_ = true; }
    bool has_value_impl(const void *v) const { return has_value_; }
};

template <typename T>
struct pointer_like_optional_impl {
    void init_as_empty(void *v) { new (v) T(nullptr); }
    void set_has_value(void *v) {}
    bool has_value_impl(const void *v) const {
        return bool(*reinterpret_cast<const T *>(v));
    }
};

template <typename T>
struct optional_base<T *> : public pointer_like_optional_impl<T *> {};

template <typename T, typename Deleter>
struct optional_base<std::unique_ptr<T, Deleter>>
    : public pointer_like_optional_impl<std::unique_ptr<T, Deleter>> {};

template <typename T>
struct optional_base<std::shared_ptr<T>>
    : public pointer_like_optional_impl<std::shared_ptr<T>> {};

template <typename T>
struct extract_nested_optional {};

template <typename T>
struct extract_nested_optional<optional<T>> {
    using internal_t = T;
};

template <typename T, typename Func>
struct extract_mapper_func_ret {
    using type = typename std::decay<decltype(
            std::declval<Func>()(std::declval<T>()))>::type;
};

template <typename T, typename Func>
using mapper_func_ret_t = typename extract_mapper_func_ret<T, Func>::type;

template <typename TOptional, bool copyable>
struct handle_copyable {
    handle_copyable() = default;
    handle_copyable(const handle_copyable &other) {
        static_cast<TOptional &>(*this).copy_from(
                static_cast<const TOptional &>(other));
    }
    using T = typename extract_nested_optional<TOptional>::internal_t;
    handle_copyable(const T &v) {
        auto &ths = static_cast<TOptional &>(*this);
        new (&ths.storage_) T(v);
        ths.set_has_value(&ths.storage_);
    }
};

template <typename TOptional>
struct handle_copyable<TOptional, false> {
    handle_copyable() = default;
    handle_copyable(const handle_copyable &other) = delete;
};

} // namespace optional_impl

struct none_opt {};

/**
 * @brief Optional container for type T. Can either be empty or contain a value
 * of T. It will also treat nullptr for pointer types as empty
 *
 * @tparam The contained type
 */
template <typename T>
struct SC_EMPTY_BASE_OPTIMIZE optional
    : protected optional_impl::optional_base<T>,
      protected optional_impl::handle_copyable<optional<T>,
              std::is_copy_constructible<T>::value> {
private:
    template <typename TOptional, bool copyable>
    friend struct optional_impl::handle_copyable;

    using size_t = std::size_t;
    using storage_type = typename std::aligned_storage<sizeof(T),
            std::alignment_of<T>::value>::type;
    storage_type storage_;
    using impl_t = typename optional_impl::optional_base<T>;

    static constexpr bool copyable = std::is_copy_constructible<T>::value;
    using copy_handler_t =
            typename optional_impl::handle_copyable<optional<T>, copyable>;

    const T *get_impl() const { return reinterpret_cast<const T *>(&storage_); }

    T *get_impl() { return reinterpret_cast<T *>(&storage_); }

    // called when storage_ is not initialized, or is cleared
    void move_from(optional &&other) {
        if (other.has_value()) {
            new (&storage_) T(std::move(*other.get_impl()));
            impl_t::set_has_value(&storage_);
            other.clear();
        } else {
            impl_t::init_as_empty(&storage_);
        }
    }

    template <typename optionalT,
            typename Dummy = typename std::enable_if<copyable
                            && std::is_same<optionalT, optional>::value,
                    int>::type>
    void copy_from(const optionalT &other) {
        if (other.has_value()) {
            new (&storage_) T(*other.get_impl());
            impl_t::set_has_value(&storage_);
        } else {
            impl_t::init_as_empty(&storage_);
        }
    }

public:
    optional(const none_opt &) { impl_t::init_as_empty(&storage_); }
    // the optional is by default "empty".
    optional() { impl_t::init_as_empty(&storage_); }

    optional(const optional &v) = default;
    using copy_handler_t::copy_handler_t;

    // move constructors
    optional(optional &&other) { move_from(std::move(other)); }
    optional(T &&v) {
        new (&storage_) T(std::move(v));
        impl_t::set_has_value(&storage_);
    }
    optional &operator=(optional &&other) {
        if (&other == this) return *this;
        clear();
        move_from(std::move(other));
        return *this;
    }

    // template <typename T2,
    //         typename Dummy = typename std::enable_if<
    //                 copyable && std::is_same<T2, T>::value, int>::type>
    // optional &operator=(const optional<T2> &other) {
    //     clear();
    //     copy_from(static_cast<const TOptional &>(other));
    //     return ths;
    // }

    bool has_value() const { return impl_t::has_value_impl(&storage_); }

    /**
     * @brief Transform the contained value by a function "f" and box the return
     * value in another optional, if this optioanl is not empty. Otherwise,
     * return an empty optional
     *
     * @param f the transform function. Takes an argument of T/const T&. Its
     * return value will be boxed into an optional
     * @return optional<Ret> returns a boxed value if current optional is not
     * empty. Otherwise, returns none_opt
     */
    template <typename Func,
            typename Ret = typename optional_impl::mapper_func_ret_t<T, Func>>
    optional<Ret> map(Func &&f) const {
        if (has_value()) { return optional<Ret>(f(*get_impl())); }
        return none_opt {};
    }

    /**
     * @brief Transform the contained value by a function "f" and return the
     * result of "f" function, if this optional is not empty. Otherwise, return
     * an empty optional
     *
     * @param f the transform function. Takes an argument of T/const T&. Its
     * return value should be an optional<Ret>
     * @return optional<Ret> returns "f"'s result if current optional is not
     * empty. Otherwise, returns none_opt
     */
    template <typename Func,
            typename Ret = typename optional_impl::extract_nested_optional<
                    typename optional_impl::mapper_func_ret_t<T,
                            Func>>::internal_t>
    optional<Ret> flat_map(Func &&f) const {
        if (has_value()) { return f(*get_impl()); }
        return none_opt {};
    }

    /**
     * @brief Return *this, if this optional is not empty. Otherwise, return the
     * result optional of function "f"
     *
     * @param f the function. Takes no arguments. Its return value
     * should be an optional<T>
     * @return optional<T>
     */
    template <typename Func>
    optional or_else(Func &&f) && {
        if (has_value()) { return std::move(*this); }
        return f();
    }

    /**
     * @brief Return *this, if this optional is not empty and the function "f"
     * returns true. Otherwise, return an empty optional
     *
     * @param f the filter function. Takes an argument of T/const T&. Its return
     * value should be a boolean
     * @return optional<T>
     */
    template <typename Func>
    optional filter(Func &&f) && {
        if (has_value() && f(*get_impl())) { return std::move(*this); }
        return none_opt {};
    }

    // Clear the optional. After calling this function, the optional object will
    // be empty.
    void clear() {
        if (has_value()) {
            get_impl()->~T();
            impl_t::init_as_empty(&storage_);
        }
    }

    ~optional() { clear(); }

    /**
     * @brief Gets the contained value. Will throw an exception if it is empty
     *
     * @return T& the contained value
     */
    T &get() {
        if (has_value()) { return *get_impl(); }
        throw std::runtime_error("Bad optional");
    }

    /**
     * @brief Gets the contained value. Will throw an exception if it is empty
     *
     * @return const T& the contained value
     */
    const T &get() const {
        if (has_value()) { return *get_impl(); }
        throw std::runtime_error("Bad optional");
    }

    // Gets the contained value. Or return a given value if this optional
    // is empty
    T get_or_else(const T &v) const {
        if (has_value()) { return *get_impl(); }
        return v;
    }

    // Gets the contained value. Or return the result of the function if this
    // optional is empty
    template <typename Func, typename dummy = decltype(std::declval<Func>()())>
    T get_or_else(Func &&f) const {
        if (has_value()) { return *get_impl(); }
        return f();
    }
};

template <typename T>
optional<typename std::decay<T>::type> some_opt(T &&v) {
    return optional<typename std::decay<T>::type> {std::forward<T>(v)};
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
