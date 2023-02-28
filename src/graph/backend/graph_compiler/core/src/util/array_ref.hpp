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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_ARRAY_REF_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_ARRAY_REF_HPP

#include <array>
#include <stddef.h>
#include <vector>
#include "compiler_macros.hpp"
#include <initializer_list>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// ArrayRef implementation taken from LLVM
//===- ArrayRef.h - Array Reference Wrapper -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

template <typename T>
class array_ref {
private:
    const T *ptr_ = nullptr;
    size_t sz_ = 0;

public:
    using value_type = T;
    using const_pointer = const value_type *;
    using const_iterator = const_pointer;
    using const_reference = const value_type &;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = ptrdiff_t;
    using iterator = const_pointer;
    using pointer = value_type *;
    using reference = value_type &;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using size_type = size_t;

    array_ref() = default;
    array_ref(const T *ptr) : ptr_(ptr), sz_(1) {}
    array_ref(const T *ptr, size_t size) : ptr_(ptr), sz_(size) {}

    array_ref(const T *begin, const T *end) : ptr_(begin), sz_(end - begin) {}

#if SC_GNUC_VERSION_GE(9)
// Disable gcc's warning in this constructor as it generates an enormous amount
// of messages. Anyone using ArrayRef should already be aware of the fact that
// it does not do lifetime extension.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-list-lifetime"
#endif
    constexpr array_ref(const std::initializer_list<T> &v)
        : ptr_(v.begin() == v.end() ? nullptr : v.begin()), sz_(v.size()) {}
#if SC_GNUC_VERSION_GE(9)
#pragma GCC diagnostic pop
#endif

    template <typename A>
    array_ref(const std::vector<T, A> &v) : ptr_(v.data()), sz_(v.size()) {}

    template <size_t N>
    constexpr array_ref(const std::array<T, N> &v) : ptr_(v.data()), sz_(N) {}

    template <size_t N>
    constexpr array_ref(const T (&v)[N]) : ptr_(v), sz_(N) {}

    bool empty() const { return sz_ == 0; }

    const T *data() const { return ptr_; }

    size_t size() const { return sz_; }

    const T &front() const { return (*this)[0]; }

    const T &back() const { return (*this)[sz_ - 1]; }

    const T &operator[](size_t i) const {
        assert(i < sz_);
        return ptr_[i];
    }

    /// Disallow accidental assignment from a temporary.
    ///
    /// The declaration here is extra complicated so that "arrayRef = {}"
    /// continues to select the move assignment operator.
    template <typename U>
    typename std::enable_if<std::is_same<U, T>::value, array_ref<T>>::type &
    operator=(U &&)
            = delete;

    /// Disallow accidental assignment from a temporary.
    ///
    /// The declaration here is extra complicated so that "arrayRef = {}"
    /// continues to select the move assignment operator.
    template <typename U>
    typename std::enable_if<std::is_same<U, T>::value, array_ref<T>>::type &
    operator=(std::initializer_list<U>)
            = delete;

    iterator begin() const { return ptr_; }
    iterator end() const { return ptr_ + sz_; }

    reverse_iterator rbegin() const { return reverse_iterator(end()); }
    reverse_iterator rend() const { return reverse_iterator(begin()); }

    std::vector<T> as_vector() const {
        return std::vector<T>(ptr_, ptr_ + sz_);
    }
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
