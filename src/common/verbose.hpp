/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#ifndef COMMON_VERBOSE_HPP
#define COMMON_VERBOSE_HPP

#include <cinttypes>
#include <mutex>
#include <stdio.h>

#include "c_types_map.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "utils.hpp"
#include "z_magic.hpp"

#include "profiler.hpp"
#include "verbose_msg.hpp"

namespace dnnl {
namespace impl {

// Trick to print filename only on all compilers supporting C++11
namespace utility {

template <typename T, size_t S>
inline constexpr size_t get_file_name_offset(
        const T (&str)[S], size_t i = S - 1) {
    return (str[i] == '/' || str[i] == '\\')
            ? i + 1
            : (i > 0 ? get_file_name_offset(str, i - 1) : 0);
}

template <typename T>
inline constexpr size_t get_file_name_offset(T (&str)[1]) {
    return 0;
}
template <typename T, T v>
struct const_expr_value {
    static constexpr const T value = v;
};

} // namespace utility

#define UTILITY_CONST_EXPR_VALUE(exp) \
    utility::const_expr_value<decltype(exp), exp>::value

#define __FILENAME__ (&__FILE__[utility::get_file_name_offset(__FILE__)])

// General formatting macro for verbose.
// msg is typically a constant string pulled from verbose_msg.hpp
// The string can contain format specifiers which are provided in VA_ARGS
// Note: using ##__VAR_ARGS__ is necessary to avoid trailing comma in printf call

#define VFORMAT(stamp, logtype, logsubtype, msg, ...) \
    do { \
        std::string stamp_; \
        if (get_verbose_timestamp()) stamp_ = "," + std::to_string(stamp); \
        printf("onednn_verbose%s," CONCAT2(VERBOSE_, logtype) "%s," msg "\n", \
                stamp_.c_str(), logsubtype, ##__VA_ARGS__); \
    } while (0)

// Macro for boolean checks
#define VCONDCHECK( \
        logtype, logsubtype, component, condition, status, msg, ...) \
    do { \
        if (!(condition)) { \
            if (verbose_has_##logtype()) \
                VFORMAT(get_msec(), logtype, #logsubtype, \
                        #component "," msg ",%s:%d", ##__VA_ARGS__, \
                        __FILENAME__, __LINE__); \
            return status; \
        } \
    } while (0)

// Macro for status checks
#define VCHECK(logtype, logsubtype, component, f, msg, ...) \
    do { \
        status_t _status_ = f; \
        VCONDCHECK(logtype, logsubtype, component, \
                _status_ == status::success, _status_, msg, ##__VA_ARGS__); \
    } while (0)

// Special syntactic sugar for error, plus flush of the output stream
#define VERROR(component, msg, ...) \
    do { \
        if (verbose_has_error()) { \
            VFORMAT(get_msec(), error, "", #component "," msg, ##__VA_ARGS__); \
        } \
        fflush(stdout); \
    } while (0)

// Special syntactic sugar for logging performance
#define VPROF(stamp, logtype, logsubtype, info, duration) \
    { VFORMAT(stamp, logtype, logsubtype, "%s,%g", info, duration); }

bool verbose_has_error();
bool verbose_has_profile_create();
bool verbose_has_profile_exec();
bool verbose_has_dispatch();
int verbose_devinfo();

bool get_verbose_timestamp();

/// A container for primitive desc verbose string.
struct primitive_desc_t;
struct pd_info_t {
    pd_info_t() = default;
    pd_info_t(const pd_info_t &rhs)
        : str_(rhs.str_), is_initialized_(rhs.is_initialized_) {}
    pd_info_t &operator=(const pd_info_t &rhs) {
        is_initialized_ = rhs.is_initialized_;
        str_ = rhs.str_;
        return *this;
    }

    const char *c_str() const { return str_.c_str(); }
    bool is_initialized() const { return is_initialized_; }

    void init(engine_t *engine, const primitive_desc_t *pd);

private:
    std::string str_;

#if defined(DISABLE_VERBOSE)
    bool is_initialized_ = true; // no verbose -> info is always ready
#else
    bool is_initialized_ = false;
#endif

    // Alas, `std::once_flag` cannot be manually set and/or copied (in terms of
    // its state). Hence, when `pd_info_t` is copied the `initialization_flag_`
    // is always reset. To avoid re-initialization we use an extra
    // `is_initialized_` flag, that should be checked before calling `init()`.
    std::once_flag initialization_flag_;
};

std::string md2fmt_str(const memory_desc_t *md);
std::string md2dim_str(const memory_desc_t *md);

} // namespace impl
} // namespace dnnl

#endif
