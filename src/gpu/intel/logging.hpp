/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_LOGGING_HPP
#define GPU_INTEL_LOGGING_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

enum class log_level_t {
    off = 0,
    warning = 100,
    suggestion = 120,
    info = 150,
    debug = 160,
    perf = 170,
    trace = 200,
};

// Parses `ONEDNN_VERBOSE` for a `xe=<level>` token independently of the generic
// `debuginfo=<level>` verbose infrastructure. This provides a mechanism to
// isolate gpu specific logs. Accepts a severity name (`xe=info`, `xe=debug`,
// `xe=all`) or the equivalent integer on the `debuginfo` scale (`xe=160`).
inline int get_xe_debug_level() {
    if (!is_dev_mode()) return 0;
    static const int level = [] {
        for (auto &tok : gpu_utils::split(getenv_string_user("VERBOSE"), ",")) {
            if (tok.rfind("xe=", 0) != 0) continue;
            auto v = tok.substr(3);
            if (v == "off") return (int)log_level_t::off;
            if (v == "warning") return (int)log_level_t::warning;
            if (v == "suggestion") return (int)log_level_t::suggestion;
            if (v == "info") return (int)log_level_t::info;
            if (v == "debug") return (int)log_level_t::debug;
            if (v == "perf") return (int)log_level_t::perf;
            if (v == "trace" || v == "all") return (int)log_level_t::trace;
            // Numeric alias (e.g. xe=160); an unrecognized token is a typo.
            if (v.empty()
                    || v.find_first_not_of("0123456789") != std::string::npos) {
                std::cerr << "Error: ONEDNN_VERBOSE xe=" << v
                          << " is not a valid level (use a name off/warning/"
                             "suggestion/info/debug/perf/trace/all or an integer)"
                          << std::endl;
                std::abort();
            }
            return std::atoi(v.c_str());
        }
        return 0;
    }();
    return level;
}

// Left-pad an integer with spaces to at least `width` chars (aligned logs).
inline std::string lpad(long long v, int width) {
    auto s = std::to_string(v);
    return (int)s.size() < width ? std::string(width - s.size(), ' ') + s : s;
}

template <typename T, typename = void>
struct print_helper_t {
    static void call(std::ostream &out, const T &t) { out << t; }
};

template <typename T>
struct print_helper_t<T, decltype(std::declval<T>().str(), void())> {
    static void call(std::ostream &out, const T &t) { out << t.str(); }
};

template <log_level_t level, bool value = true>
class logger_t {
public:
    logger_t(const char *file_name, int line, std::ostream &out = std::cout)
        : file_path_(file_name + std::string(":") + std::to_string(line))
        , out_(out) {}

    logger_t(const logger_t &) = delete;
    logger_t &operator=(const logger_t &) = delete;

    ~logger_t() {
        add_header(true);
        if (lines_.size() == 1) {
            out_ << " " << lines_[0] << std::endl;
        } else {
            out_ << std::endl;
            for (auto &l : lines_) {
                add_header(/*with_file=*/false);
                out_ << "  " << l << std::endl;
            }
        }
    }

    static bool is_enabled() {
        static const int lvl = std::max(
                static_cast<int>(get_verbose_dev_mode(verbose_t::debuginfo)),
                get_xe_debug_level());
        return lvl >= static_cast<int>(level);
    }

    log_level_t get_level() const { return level; }

    operator bool() const { return value; }

    template <typename T>
    logger_t &operator<<(const T &obj) {
        ostringstream_t oss;
        print_helper_t<T>::call(oss, obj);
        auto lines = gpu_utils::split(oss.str(), "\n");
        if (lines_.empty() || lines.empty()) {
            lines_ = std::move(lines);
            return *this;
        }
        lines_.back() += lines[0];
        lines_.insert(lines_.end(), lines.begin() + 1, lines.end());
        return *this;
    }

private:
    void add_header(bool with_file) {
        switch (level) {
            case log_level_t::warning: out_ << "[ WARN]"; break;
            case log_level_t::suggestion: out_ << "[SUGGESTION]"; break;
            case log_level_t::info: out_ << "[ INFO]"; break;
            case log_level_t::debug: out_ << "[DEBUG]"; break;
            case log_level_t::perf: out_ << "[ PERF]"; break;
            case log_level_t::trace: out_ << "[TRACE]"; break;
            default: gpu_error_not_expected();
        }
        if (with_file) out_ << "[" << file_path_ << "]";
    }

    std::string file_path_;
    std::ostream &out_;
    std::vector<std::string> lines_;
};

#define gpu_perf() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level_t::perf>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level_t::perf>( \
                    __FILENAME__, __LINE__)

// Trace can result in overhead making measurement meaningless
#define gpu_perf_no_trace() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level_t::perf>::is_enabled() \
            && !dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level_t::trace>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level_t::perf>( \
                    __FILENAME__, __LINE__)

#define gpu_info() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level_t::info>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level_t::info>( \
                    __FILENAME__, __LINE__)

#define gpu_debug() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level_t::debug>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level_t::debug>( \
                    __FILENAME__, __LINE__)

#define gpu_warning() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level_t::warning>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level_t::warning>( \
                    __FILENAME__, __LINE__)

#define gpu_suggestion() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level_t::suggestion>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level_t::suggestion>( \
                    __FILENAME__, __LINE__)

#define gpu_trace() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level_t::trace>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level_t::trace>( \
                    __FILENAME__, __LINE__)

#define gpu_check(cond) \
    if (!(cond)) \
    return dnnl::impl::gpu::intel::logger_t< \
                   dnnl::impl::gpu::intel::log_level_t::trace>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level_t::trace, false>( \
                    __FILENAME__, __LINE__)

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
