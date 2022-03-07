/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "logger.hpp"
#include "utils/utils.hpp"

using namespace dnnl::graph::impl;

dnnl_graph_log_level_t logger_t::default_level_ {dnnl_graph_log_level_disabled};

dnnl_graph_log_level_t logger_t::get_log_level_int() {
    const int len = 12;
    char level_env[len]; // NOLINT
    if (impl::utils::getenv("DNNL_GRAPH_VERBOSE", level_env, len) > 0) {
        int level = atoi(level_env);
        switch (level) {
            default:
            case dnnl_graph_log_level_disabled:
                return dnnl_graph_log_level_disabled;
            case dnnl_graph_log_level_error:
            case dnnl_graph_log_level_info:
            case dnnl_graph_log_level_debug:
                return static_cast<dnnl_graph_log_level_t>(level);
        }
    }
    return default_level_;
}

logger_t::logger_t() : output_stream_(&std::cout) {
    const int len = 12;
    char output_env[len]; // NOLINT
    if (impl::utils::getenv("DNNL_GRAPH_VERBOSE_OUTPUT", output_env, len) > 0) {
        if (strcmp(output_env, "stdout") == 0) {
            output_stream_ = &std::cout;
        } else if (strcmp(output_env, "stderr") == 0) {
            output_stream_ = &std::cerr;
        } else {
            output_file_.open(output_env);
            output_stream_ = &output_file_;
        }
    }
}

void logger_t::log_message_int(int32_t level, const char *message) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const time_t tt = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
#ifdef _WIN32
    auto tm = gmtime(&tt); // NOLINT
#else
    struct tm buf;
    auto tm = gmtime_r(&tt, &buf);
#endif /* _WIN32 */
    if (tm) {
        char buffer[256];
        strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%Sz", tm);
        *output_stream_ << buffer << " ";
    }
    switch (level) {
        case dnnl_graph_log_level_error: *output_stream_ << "[ERROR] "; break;
        case dnnl_graph_log_level_info: *output_stream_ << "[INFO ] "; break;
        case dnnl_graph_log_level_debug: *output_stream_ << "[DEBUG] "; break;
        default: *output_stream_ << "[     ] "; break;
    }
    *output_stream_ << message << std::endl;
}
