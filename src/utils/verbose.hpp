/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef UTILS_VERBOSE_HPP
#define UTILS_VERBOSE_HPP

#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <string>

#include "oneapi/dnnl/dnnl_graph_types.h"

#include "interface/c_types_map.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

// The following code is derived from oneDNN/src/common/utils.hpp.
template <typename T>
struct setting_t {
private:
    T value_;
    bool initialized_;

public:
    setting_t() : initialized_ {false} {}
    setting_t(const T init) : value_ {init}, initialized_ {false} {}
    bool initialized() { return initialized_; }
    T get() { return value_; }
    void set(T new_value) {
        value_ = new_value;
        initialized_ = true;
    }
    setting_t(const setting_t &) = delete;
    setting_t &operator=(const setting_t &) = delete;
};

struct verbose_t {
    int level;
};

#if !defined(DNNL_GRAPH_DISABLE_VERBOSE)
#define DNNL_GRAPH_VERBOSE_BUF_LEN 1024
#else
#define DNNL_GRAPH_VERBOSE_BUF_LEN 1
#endif

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

    void init(const engine_t *engine,
            const ::dnnl_graph_compiled_partition *partition);

private:
    std::string str_;

#if defined(DNNL_GRAPH_DISABLE_VERBOSE)
    bool is_initialized_ = true;
#else
    bool is_initialized_ = false;
#endif

    std::once_flag initialization_flag_;
};

double get_msec();
int get_verbose();

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
