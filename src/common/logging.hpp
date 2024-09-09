/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef COMMON_LOGGING_HPP
#define COMMON_LOGGING_HPP

#include <cinttypes>
#include <string>

namespace dnnl {
namespace impl {

#define SEPARATOR_STR "-------------------------------------------------"
#define VERBOSE_LOGGER_NAME "oneDNN"
#define DEFAULT_VERBOSE_LOGFILE_SIZE (1024 * 1024 * 50) // 50 MB
// Sets a default max limit of 500 MB for logging
#define DEFAULT_VERBOSE_NUM_LOGFILES 10

// Handle for logger operations when verbose logging functionality is enabled
struct log_manager_t {
public:
    // Lists the different levels for logging data - these are pre-defined
    // from the spdlog library
    enum log_level_t : uint8_t {
        off,
        trace,
        debug,
        info,
        warn,
        error,
        critical,
    };

    // Static method for singleton pattern instantiation
    static const log_manager_t &get_log_manager() {
        static log_manager_t instance;
        return instance;
    }

    // Prints the message at the specified logging level
    // Implicitly used by the verbose API to save outputs whenever the logger
    // is enabled
    void log(const char *msg, log_level_t log_level) const;

    // Returns the logger status. Logging remains disabled during runtime if:
    // - DNNL_VERBOSE is not enabled or specified
    // - No logfile is provided with DNNL_VERBOSE_LOGFILE
    // - spdlog::rotating_file_mt() fails during instantiation
    bool is_logger_enabled() const { return init_flag_; }

    // When the logfile is specified:
    // - If disabled (the default), the data is printed to logfiles only.
    // - If enabled, the data is printed to both stdout and logfiles.
    // Otherwise, the data is printed to stdout.
    bool is_console_enabled() const { return console_flag_; }

    // Determines the spdlog level from the verbose mode
    // and sets logger to the aligned level.
    // Logging level is set during instantiation but is also updated whenever
    // the verbose levels are functionally updated
    void set_log_level(const std::string &vmode_str) const;

private:
    log_manager_t();

    log_manager_t(const log_manager_t &) = delete;
    log_manager_t &operator=(const log_manager_t &) = delete;

    ~log_manager_t();

    std::string logger_name_ = VERBOSE_LOGGER_NAME;
    std::string logfile_path_;

    // default logger specifications
    unsigned logfile_size_ = DEFAULT_VERBOSE_LOGFILE_SIZE;
    unsigned num_logfiles_ = DEFAULT_VERBOSE_NUM_LOGFILES;
    bool init_flag_ = false;
    bool console_flag_ = false;
};

} // namespace impl
} // namespace dnnl

#endif
