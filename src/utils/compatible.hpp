/*******************************************************************************
 * Copyright 2020 Intel Corporation
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

#ifndef LLGA_UTILS_COMPATIBLE_HPP
#define LLGA_UTILS_COMPATIBLE_HPP

#if DNNL_GRAPH_SUPPORT_CXX17
#include <optional>
#endif
#include <memory>
#include <utility>
#include <type_traits>

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

struct nullopt_t {
    enum class _construct { _token };
    explicit constexpr nullopt_t(_construct) {}
};

constexpr const nullopt_t nullopt {nullopt_t::_construct::_token};

template <typename T>
class optional_impl {
public:
    optional_impl() : is_null_(true) {}
    optional_impl(const T &value) {
        is_null_ = false;
        new (&value_) T(value);
    }

    optional_impl(const optional_impl<T> &opt) {
        is_null_ = opt.is_null_;
        if (!is_null_) { new (&value_) T(opt.value()); }
    }

    optional_impl(nullopt_t) : is_null_(true) {}

    ~optional_impl() {
        // explicitly deconstructor
        if (!is_null_) { reinterpret_cast<T *>(&value_)->~T(); }
    }

    void swap(optional_impl<T> &another) {
        std::swap(value_, another.value_);
        std::swap(is_null_, another.is_null_);
    }

    optional_impl<T> &operator=(const optional_impl<T> &another) {
        (optional_impl<T>(another)).swap(*this);
        return *this;
    }

    optional_impl<T> &operator=(const T &value) {
        (optional_impl<T>(value)).swap(*this);
        return *this;
    }

    optional_impl<T> &operator=(nullopt_t) {
        (optional_impl<T>()).swap(*this);
        return *this;
    }

    const T &operator*() const { return *reinterpret_cast<const T *>(&value_); }
    T &operator*() { return *reinterpret_cast<T *>(&value_); }

    const T &value() const {
        if (is_null_) { throw std::logic_error("bad optional access"); }
        return *reinterpret_cast<const T *>(&value_);
    }

    bool operator==(const optional_impl<T> &other) const {
        return this->is_null_ == other.is_null_
                && (this->is_null_ == true || this->value() == other.value());
    }

    bool has_value() const { return !is_null_; }

private:
    typename std::aligned_storage<sizeof(T), alignof(T)>::type value_;
    bool is_null_;
};

class bad_any_cast : public std::bad_cast {
public:
    virtual const char *what() const noexcept { return "bad any_cast"; }
};

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

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

// any structure
// now we only use this any struct for the project.
class any {
public:
    any() = default;
    any(any &&v) {
        clear();
        avtable_ = v.avtable_;
        v.avtable_ = nullptr;
    }
    any(const any &v) { avtable_ = v.avtable_; }

    template <typename T,
            typename = enable_if_t<!std::is_same<T, any &>::value>>
    any(T &&v) {
        clear();
        using value_type = typename std::decay<
                typename std::remove_reference<T>::type>::type;
        avtable_ = std::make_shared<vtable<value_type>>(std::forward<T>(v));
    }

    any &operator=(const any &v) {
        any(v).swap(*this);
        return *this;
    }
    any &operator=(any &&v) {
        v.swap(*this);
        any().swap(v);
        return *this;
    }
    template <typename T>
    any &operator=(T &&v) {
        any(std::forward<T>(v)).swap(*this);
        return *this;
    }

    void clear() {
        if (avtable_) { avtable_ = nullptr; }
    }
    void swap(any &v) { std::swap(avtable_, v.avtable_); }
    bool empty() { return avtable_ == nullptr; }
    const std::type_info &type() const {
        return avtable_ ? avtable_->type() : typeid(void);
    }

    template <typename T, typename T1, typename... Args>
    bool match(T defaults, T1 func1, Args &&... args) const {
        using MatchedT = first_argument<T1>;
        if (type() == typeid(MatchedT)) {
            func1(static_cast<vtable<MatchedT> *>(avtable_.get())->value_);
            return true;
        }
        return match(std::forward<T>(defaults), std::forward<Args>(args)...);
    }

    template <typename T, typename T1>
    bool match(T defaults, T1 func1) const {
        using MatchedT = first_argument<T1>;
        if (type() == typeid(MatchedT)) {
            func1(static_cast<vtable<MatchedT> *>(avtable_.get())->value_);
            return true;
        }
        defaults();
        return false;
    }

private:
    struct any_vtable {
        virtual ~any_vtable() {}
        virtual const std::type_info &type() = 0;
        virtual std::shared_ptr<any_vtable> get_vtable() = 0;
    };
    template <typename T>
    struct vtable : public any_vtable {
        vtable(const T &value) : value_(value) {}
        vtable(T &&value) : value_(std::forward<T>(value)) {}
        vtable &operator=(const vtable &) = delete;
        const std::type_info &type() override { return typeid(T); }
        std::shared_ptr<any_vtable> get_vtable() override {
            return std::make_shared<vtable>(value_);
        }
        T value_;
    };

    std::shared_ptr<any_vtable> avtable_ = nullptr;

    template <typename T>
    friend T *any_cast(any *v);
};

template <typename T>
T *any_cast(any *v) {
    typedef typename std::remove_cv<T>::type value_type;
    return v && v->type() == typeid(T)
            ? &static_cast<any::vtable<value_type> *>(v->avtable_.get())->value_
            : nullptr;
}

template <typename T>
inline const T *any_cast(const any *v) {
    return any_cast<T>(const_cast<any *>(v));
}

template <typename T>
inline T any_cast(any &v) {
    typedef typename std::remove_reference<T>::type nonref;
    auto val = any_cast<nonref>(&v);
    if (val) {
        typedef typename std::conditional<std::is_reference<T>::value, T,
                typename std::add_lvalue_reference<T>::type>::type ref_type;
        return static_cast<ref_type>(*val);
    } else {
        throw bad_any_cast {};
    }
}

template <typename T>
inline T any_cast(const any &v) {
    return any_cast<T>(const_cast<any &>(v));
}

template <typename T>
inline T any_cast(any &&v) {
    static_assert(std::is_rvalue_reference<T &&>::value
                    || std::is_const<
                            typename std::remove_reference<T>::type>::value,
            "should not be used getting non const reference and move object");
    // return any_cast<std::forward<T>>(v);
    return any_cast<T>(v);
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#ifdef DNNL_GRAPH_SUPPORT_CXX17
template <typename T>
using optional = std::optional<T>;
#else
template <typename T>
using optional = optional_impl<T>;
#endif // DNNL_GRAPH_SUPPORT_CXX17

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif
