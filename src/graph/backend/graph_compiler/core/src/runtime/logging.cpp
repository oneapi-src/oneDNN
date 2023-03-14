/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include <iostream>
#include <set>
#include <string>
#include <runtime/config.hpp>
#include <runtime/env_var.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/logging.hpp>
#include <util/string_utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {

#define should_pass_filter(x) true
static std::ostream *stream_target = &std::cerr;

logging_stream_t::logging_stream_t(const char *append)
    : stream_(new std::stringstream {}), append_(append) {}
logging_stream_t::~logging_stream_t() {
    if (stream_) {
        *stream_ << append_;
        (*stream_target) << static_cast<std::stringstream *>(stream_)->str();
        delete stream_;
    }
}

void set_logging_stream(std::ostream *s) {
    stream_target = s;
}

std::ostream *get_logging_stream() {
    return stream_target;
}

static logging_stream_t get_stream(verbose_level level, const char *module_name,
        const char *appender, const char *prefix) {
    if (runtime_config_t::get().verbose_level_ < level) {
        return logging_stream_t();
    }
    if (!module_name || should_pass_filter(module_name)) {
        logging_stream_t ret {appender};
        *(ret.stream_) << prefix;
        if (module_name) { *(ret.stream_) << '[' << module_name << ']' << ' '; }
        return ret;
    }
    return logging_stream_t();
}

logging_stream_t get_info_logging_stream(const char *module_name) {
    return get_stream(INFO, module_name, "\n", "[INFO] ");
}

logging_stream_t get_warning_logging_stream(const char *module_name) {
    return get_stream(WARNING, module_name, "\033[0m\n", "\033[33m[WARN] ");
}

logging_stream_t get_fatal_logging_stream(const char *module_name) {
    return get_stream(FATAL, module_name, "\033[0m\n", "\033[31m[FATAL] ");
}

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
