/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef UTILS_HPP
#define UTILS_HPP

#include <stddef.h>
#include <stdlib.h>

namespace mkldnn { namespace impl {

#define UNUSED(x) ((void)x)

#define CHECK(f) do { \
    status_t status = f; \
    if (status != success) \
    return status; \
} while (0)

namespace utils {

/* SFINAE helper -- anoluge to std::enable_if */
template<bool expr, class T = void> struct enable_if {};
template<class T> struct enable_if<true, T> { typedef T type; };

}

template <typename T, typename P>
inline bool everyone_is(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

template <typename T, typename P>
inline bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) { return one_of(nullptr, ptrs...); }

inline bool implication(bool cause, bool effect) { return !cause || effect; }

template<typename T>
inline void array_copy(T *dst, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i) dst[i] = src[i];
}
template<typename T>
inline bool array_cmp(const T *a1, const T *a2, size_t size) {
    for (size_t i = 0; i < size; ++i) if (a1[i] != a2[i]) return false;
    return true;
}
template<typename T, typename U>
inline void array_set(T *arr, const U& val, size_t size) {
    for (size_t i = 0; i < size; ++i) arr[i] = val;
}

namespace product_impl {
template<size_t> struct int2type{};

template <typename T>
constexpr int product_impl(const T* arr, int2type<0>) { return arr[0]; }

template <typename T, size_t num>
inline T product_impl(const T* arr, int2type<num>) {
    return arr[0]*product_impl(arr+1, int2type<num-1>()); }
}

template <size_t num, typename T>
inline T array_product(const T* arr) {
    return product_impl::product_impl(arr, product_impl::int2type<num-1>());
}

template<typename T>
inline T array_product(const T *arr, size_t size) {
    T prod = 1;
    for (size_t i = 0; i < size; ++i) prod *= arr[i];
    return prod;
}

inline void* malloc(size_t size, int alignment) {
    void *ptr;
    int rc = ::posix_memalign(&ptr, alignment, size);
    return (rc == 0) ? ptr : 0;
}
inline void free(void* p) { ::free(p); }

struct c_compatible {
    enum { default_alignment = 64 };
    static void* operator new(size_t sz) {
        return malloc(sz, default_alignment);
    }
    static void* operator new(size_t sz, void* p) { UNUSED(sz); return p; }
    static void* operator new[](size_t sz) {
        return malloc(sz, default_alignment);
    }
    static void operator delete(void* p) { free(p); }
    static void operator delete[](void* p) { free(p); }
};

inline void yield_thread() { }

}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
