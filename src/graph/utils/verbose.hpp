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

#ifndef GRAPH_UTILS_VERBOSE_HPP
#define GRAPH_UTILS_VERBOSE_HPP

#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <string>

#include "common/verbose.hpp"

#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

// General formatting macro for verbose.
// msg is typically a constant string pulled from verbose_msg.hpp
// The string can contain format specifiers which are provided in VA_ARGS
// Note: using ##__VAR_ARGS__ is necessary to avoid trailing comma in printf call

#define VFORMATGRAPH(stamp, logtype, logsubtype, msg, ...) \
    do { \
        std::string stamp_; \
        printf("onednn_graph_verbose%s," CONCAT2(VERBOSE_, logtype) "%s," msg \
                                                                    "\n", \
                stamp_.c_str(), logsubtype, ##__VA_ARGS__); \
    } while (0)

// Logging info
#define VINFOGRAPH(logtype, logsubtype, component, msg, ...) \
    do { \
        if (utils::get_graph_verbose(impl::verbose_t::logtype##_##logsubtype)) \
            VFORMATGRAPH(get_msec(), logtype, VERBOSE_##logsubtype, \
                    #component "," msg ",%s:%d", ##__VA_ARGS__, __FILENAME__, \
                    __LINE__); \
    } while (0)

// Macro for boolean checks
#define VCONDCHECKGRAPH( \
        logtype, logsubtype, component, condition, status, msg, ...) \
    do { \
        if (!(condition)) { \
            VINFOGRAPH(logtype, logsubtype, component, msg, ##__VA_ARGS__); \
            return status; \
        } \
    } while (0)

// Macro for status checks
#define VCHECKGRAPH(logtype, logsubtype, component, f, msg, ...) \
    do { \
        status_t _status_ = (f); \
        VCONDCHECKGRAPH(logtype, logsubtype, component, \
                _status_ == status::success, _status_, msg, ##__VA_ARGS__); \
    } while (0)

// Special syntactic sugar for error, plus flush of the output stream
#define VERRORGRAPH(component, msg, ...) \
    do { \
        if (utils::get_graph_verbose(impl::verbose_t::error)) { \
            VFORMATGRAPH( \
                    get_msec(), error, "", #component "," msg, ##__VA_ARGS__); \
        } \
        fflush(stdout); \
    } while (0)

// Special syntactic sugar for logging performance
// NOTE: the VPROF macro does not check for verbose flags, it is the
// responsibility of the caller do check those (it should happen
// anyway to condition collecting stamp/duration)
#define VPROFGRAPH(stamp, logtype, logsubtype, info, duration) \
    { \
        VFORMATGRAPH(stamp, logtype, logsubtype, "%s,%g", info, duration); \
        fflush(stdout); \
    }

uint32_t get_graph_verbose(
        impl::verbose_t::flag_kind kind = impl::verbose_t::none);

struct partition_info_t {
    partition_info_t() = default;
    partition_info_t(const partition_info_t &rhs)
        : str_(rhs.str_), is_initialized_(rhs.is_initialized_) {};
    partition_info_t &operator=(const partition_info_t &rhs) {
        str_ = rhs.str_;
        is_initialized_ = rhs.is_initialized_;
        return *this;
    }

    const char *c_str() const { return str_.c_str(); }
    bool is_initialized() const { return is_initialized_; }

    void init(const engine_t *engine, const compiled_partition_t *partition);

private:
    std::string str_;

#if defined(DISABLE_VERBOSE)
    bool is_initialized_ = true;
#else
    bool is_initialized_ = false;
#endif

    std::once_flag initialization_flag_;
};

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
