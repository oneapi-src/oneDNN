/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "common/logging.hpp"
#include "common/utils.hpp"

#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/spdlog.h"

namespace dnnl {
namespace impl {

log_manager_t::log_manager_t()
    : logfile_path_(getenv_string_user("VERBOSE_LOGFILE"))
    // enables logging as well as printing to stdout
    , console_flag_(getenv_int_user("VERBOSE_LOG_WITH_CONSOLE", 0)) {

    // logging is automatically disabled when no filepath is provided by
    // DNNL_VERBOSE_LOGFILE
    // in this case, we fall back to printing to stdout
    if (logfile_path_.empty()) {
        console_flag_ = true;
        return;
    }

    // additional run-time controls for logger creation
    if (getenv_int_user("VERBOSE_LOGFILE_SIZE") > 0)
        logfile_size_ = getenv_int_user("VERBOSE_LOGFILE_SIZE");

    if (getenv_int_user("VERBOSE_NUM_LOGFILES") > 0)
        num_logfiles_ = getenv_int_user("VERBOSE_NUM_LOGFILES");

    try {
        // rotating multi-threaded logger instance
        const auto &dnnl_logger = spdlog::rotating_logger_mt(
                logger_name_, logfile_path_, logfile_size_, num_logfiles_);
        init_flag_ = true;

        dnnl_logger->info(SEPARATOR_STR);
        dnnl_logger->info(
                "logger enabled,logfile::{},size::{},num_logfiles::{}",
                logfile_path_, logfile_size_, num_logfiles_);
        dnnl_logger->info(SEPARATOR_STR);
    } catch (spdlog::spdlog_ex &exception) {
        printf("onednn_verbose,info,exception while creating logfile: %s\n",
                exception.what());
    }
}

log_manager_t::~log_manager_t() {
    // spdlog::shutdown() must be called at exit to avoid module segfaults
    spdlog::shutdown();
}

void log_manager_t::log(const char *msg, log_level_t log_level) const {

    const auto &dnnl_logger = spdlog::get(logger_name_);

    // removes trailing newline characters without requiring a custom sink.
    // (by default the fmt library appends a '\n' character to
    // the logged message)
    size_t msg_len = strlen(msg);
    auto nmsg = (msg_len > 0 && msg[msg_len - 1] == '\n')
            ? std::string(msg, msg_len - 1)
            : msg;

    switch (log_level) {
        case off: break;
        case trace: dnnl_logger->trace(nmsg); break;
        case debug: dnnl_logger->debug(nmsg); break;
        case info: dnnl_logger->info(nmsg); break;
        case warn: dnnl_logger->warn(nmsg); break;
        case error: dnnl_logger->error(nmsg); break;
        case critical: dnnl_logger->critical(nmsg); break;
        default: dnnl_logger->error("unknown logging level"); break;
    }
}

void log_manager_t::set_log_level(const std::string &vmode_str) const {
    // The logging level is determined from the verbose mode
    // with the following order of decreasing priority:
    // [trace, debug, info, warn, error, critical, off]
    spdlog::set_level(spdlog::level::off);

    if (vmode_str == "-1" || vmode_str == "all") {
        spdlog::set_level(spdlog::level::trace);
    } else if (vmode_str.rfind("debuginfo=", 0) == 0) {
        spdlog::set_level(spdlog::level::debug);
    } else if (vmode_str == "1" || vmode_str == "2"
            || vmode_str.find("profile") != std::string::npos
            || vmode_str.find("dispatch") != std::string::npos) {
        spdlog::set_level(spdlog::level::info);
    } else if (vmode_str.find("warn") != std::string::npos) {
        spdlog::set_level(spdlog::level::warn);
    } else if (vmode_str.find("check") != std::string::npos) {
        spdlog::set_level(spdlog::level::err);
    } else if (vmode_str.find("error") != std::string::npos) {
        spdlog::set_level(spdlog::level::critical);
    }
}

} // namespace impl
} // namespace dnnl
