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

#include <iostream>
#include <sstream>
#include <string>

#include "common/utils.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

namespace log_level {
constexpr verbose_t::debug_level off = verbose_t::debug_level::off;
constexpr verbose_t::debug_level warning = verbose_t::debug_level::warn;
constexpr verbose_t::debug_level suggestion = verbose_t::debug_level::warn;
constexpr verbose_t::debug_level info = verbose_t::debug_level::info;
constexpr verbose_t::debug_level perf = verbose_t::debug_level::info;
constexpr verbose_t::debug_level trace = verbose_t::debug_level::trace;
} // namespace log_level

template <typename T, typename = void>
struct print_helper_t {
    static void call(std::ostream &out, const T &t) { out << t; }
};

template <typename T>
struct print_helper_t<T, decltype(std::declval<T>().str(), void())> {
    static void call(std::ostream &out, const T &t) { out << t.str(); }
};

template <verbose_t::debug_level level, bool value = true>
class logger_t {
public:
    logger_t(const char *file_name, int line, std::ostream &out = std::cout)
        : file_path_(file_name + std::string(":") + std::to_string(line))
        , out_(out) {}
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

    static bool is_enabled() { return get_debug_verbose_dev_mode(level); }

    verbose_t::debug_level get_level() const { return level; }

    operator bool() const { return value; }

    template <typename T>
    logger_t &operator<<(const T &obj) {
        std::ostringstream oss;
        print_helper_t<T>::call(oss, obj);
        auto lines = gpu_utils::split(oss.str(), "\n");
        if (lines_.empty() || lines.empty()) {
            lines_ = lines;
            return *this;
        }
        lines_.back() += lines[0];
        lines_.insert(lines_.end(), lines.begin() + 1, lines.end());
        return *this;
    }

private:
    void add_header(bool with_file) {
        switch (level) {
            case log_level::warning: out_ << "[ WARN]"; break;
            case log_level::info: out_ << "[ INFO]"; break;
            case log_level::trace: out_ << "[TRACE]"; break;
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
            dnnl::impl::gpu::intel::log_level::perf>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level::perf>( \
                    __FILENAME__, __LINE__)

// Trace can result in overhead making measurement meaningless
#define gpu_perf_no_trace() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level::perf>::is_enabled() \
            && !dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level::trace>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level::perf>( \
                    __FILENAME__, __LINE__)

#define gpu_info() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level::info>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level::info>( \
                    __FILENAME__, __LINE__)

#define gpu_warning() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level::warning>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level::warning>( \
                    __FILENAME__, __LINE__)

#define gpu_suggestion() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level::suggestion>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level::suggestion>( \
                    __FILENAME__, __LINE__)

#define gpu_trace() \
    dnnl::impl::gpu::intel::logger_t< \
            dnnl::impl::gpu::intel::log_level::trace>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level::trace>( \
                    __FILENAME__, __LINE__)

#define gpu_check(cond) \
    if (!(cond)) \
    return dnnl::impl::gpu::intel::logger_t< \
                   dnnl::impl::gpu::intel::log_level::trace>::is_enabled() \
            && dnnl::impl::gpu::intel::logger_t< \
                    dnnl::impl::gpu::intel::log_level::trace, false>( \
                    __FILENAME__, __LINE__)

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
