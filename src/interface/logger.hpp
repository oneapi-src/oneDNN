/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_INTERFACE_LOGGER_HPP
#define LLGA_INTERFACE_LOGGER_HPP

#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.h"

namespace dnnl {
namespace graph {
namespace impl {

/**
 * Interface for presenting debug messages to user.
 *
 * NOTE: this class is not meant to be used directly. Use DNNL_GRAPH_LOG macros
 * defined later in this file.
 *
 * Debug messages are presented if the verbose mode is enabled during
 * compilation (cmake option: DNNL_GRAPH_VERBOSE=ON) and a proper verbosity
 * level is activated at runtime:
 * export DNNL_GRAPH_VERBOSE=<level>
 *
 * The messages by default end up on stdout, but user can also select stderr or
 * redirect messages to a file:
 * export DNNL_GRAPH_VERBOSE_OUTPUT=<stdout|stderr|path_to_file>
 */
class Logger {
public:
    static void log_message(int32_t level, const char *message) {
        singleton().log_message_int(level, message);
    }

    static dnnl_graph_log_level_t get_log_level() {
        static dnnl_graph_log_level_t active_level {get_log_level_int()};
        return active_level;
    }

#ifdef DNNL_GRAPH_VERBOSE_DISABLE
    static constexpr bool log_level_active(dnnl_graph_log_level_t) {
        return false;
    }
#else
    static bool log_level_active(dnnl_graph_log_level_t level) {
        return (level <= get_log_level());
    }
#endif

#ifdef DNNL_GRAPH_VERBOSE_DISABLE
    static constexpr bool disabled_ = true;
#else
    static constexpr bool disabled_ = false;
#endif

    // This function to have any effect has to be called BEFORE log_message.
    static void set_default_log_level(dnnl_graph_log_level_t level) {
        default_level_ = level;
    }

private:
    Logger();
    void log_message_int(int32_t level, const char *message);
    static dnnl_graph_log_level_t get_log_level_int();

    static Logger &singleton() {
        static Logger logger;
        return logger;
    }

    std::ostream *output_stream_;
    std::ofstream output_file_;
    std::mutex mutex_;
    static dnnl_graph_log_level_t default_level_;
};

/**
 * @brief Class encapsulating a single message with specified verbosity level.
 *
 * The class is not intended to be used directly, as lifespan of LogEntry
 * objects decides when given message will be flushed. Use the DNNL_GRAPH_LOG() and
 * DNNL_GRAPH_LOG_<level>() macros instead.
 */
class LogEntry {
public:
    LogEntry(dnnl_graph_log_level_t level) : level_(level) {}
    ~LogEntry() {
        try {
            dnnl::graph::impl::Logger::log_message(level_, ss_.str().c_str());
        } catch (...) {}
    }
    std::ostream &stream() { return ss_; };

private:
    std::stringstream ss_;
    dnnl_graph_log_level_t level_;
};

/**
 * @brief Macro for composing a debug message.
 *
 * The message is composed using streaming operator syntax:
 * DNNL_GRAPH_LOG_ERROR() << "Variable = " << variable;
 */
#ifndef DNNL_GRAPH_LOG
#define DNNL_GRAPH_LOG(level) \
    if (!dnnl::graph::impl::Logger::log_level_active(level)) { \
    } else \
        dnnl::graph::impl::LogEntry(level).stream()
#define DNNL_GRAPH_LOG_ERROR() DNNL_GRAPH_LOG(dnnl_graph_log_level_error)
#define DNNL_GRAPH_LOG_INFO() DNNL_GRAPH_LOG(dnnl_graph_log_level_info)
#define DNNL_GRAPH_LOG_DEBUG() DNNL_GRAPH_LOG(dnnl_graph_log_level_debug)
#endif

} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
