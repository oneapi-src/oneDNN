/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef UTILS_WRAPPER_HPP
#define UTILS_WRAPPER_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_debug.hpp"

#define CHECK_DESTROY(f) \
    do { \
        dnnl_status_t status__ = f; \
        if (status__ != dnnl_success) { \
            BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                    __PRETTY_FUNCTION__, __LINE__, STRINGIFY(f), \
                    status2str(status__), (int)status__); \
            fflush(0); \
            exit(2); \
        } \
    } while (0)

template <typename T>
struct dnnl_api_traits;
//{
//    static void destroy(T t) {}
//};

template <>
struct dnnl_api_traits<dnnl_primitive_t> {
    static void destroy(dnnl_primitive_t t) {
        CHECK_DESTROY(dnnl_primitive_destroy(t));
    }
};

template <>
struct dnnl_api_traits<dnnl_primitive_desc_t> {
    static void destroy(dnnl_primitive_desc_t t) {
        CHECK_DESTROY(dnnl_primitive_desc_destroy(t));
    }
};

template <>
struct dnnl_api_traits<dnnl_primitive_attr_t> {
    static void destroy(dnnl_primitive_attr_t t) {
        CHECK_DESTROY(dnnl_primitive_attr_destroy(t));
    }
};

template <>
struct dnnl_api_traits<dnnl_memory_desc_t> {
    static void destroy(dnnl_memory_desc_t t) {
        CHECK_DESTROY(dnnl_memory_desc_destroy(t));
    }
};

// Generic class providing RAII support for DNNL objects in benchdnn
template <typename T>
struct benchdnn_dnnl_wrapper_t {
    benchdnn_dnnl_wrapper_t(T t = nullptr) : t_(t) {
        static_assert(std::is_pointer<T>::value, "T is not a pointer type.");
    }

    benchdnn_dnnl_wrapper_t &operator=(benchdnn_dnnl_wrapper_t &&rhs) {
        if (this == &rhs) return *this;
        reset(rhs.release());
        return *this;
    }

    benchdnn_dnnl_wrapper_t(benchdnn_dnnl_wrapper_t &&rhs) {
        t_ = nullptr;
        *this = std::move(rhs);
    }

    ~benchdnn_dnnl_wrapper_t() { do_destroy(); }

    T release() {
        T tmp = t_;
        t_ = nullptr;
        return tmp;
    }

    void reset(T t) {
        do_destroy();
        t_ = t;
    }

    operator T() const { return t_; }

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(benchdnn_dnnl_wrapper_t);

private:
    T t_;

    void do_destroy() {
        if (t_) { dnnl_api_traits<T>::destroy(t_); }
    }
};

// Constructs a wrapper object (providing RAII support)
template <typename T>
benchdnn_dnnl_wrapper_t<T> make_benchdnn_dnnl_wrapper(T t) {
    return benchdnn_dnnl_wrapper_t<T>(t);
}

#undef CHECK_DESTROY

#endif
