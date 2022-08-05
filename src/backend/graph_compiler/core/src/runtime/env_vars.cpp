/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#include "env_vars.hpp"
#include <assert.h>
#include <climits>
#include <cstring>
#include <stdlib.h>
#include <string>
#include <vector>
#include "env_var.hpp"
#ifdef _WIN32
#include <windows.h>
#endif
namespace sc {

const char *env_names[] = {"SC_CPU_JIT", "SC_TRACE", "SC_DUMP_GRAPH",
        "SC_GRAPH_DUMP_TENSORS", "SC_VALUE_CHECK", "SC_OPT_LEVEL",
        "SC_BUFFER_SCHEDULE", "SC_KERNEL", "SC_MICRO_KERNEL_OPTIM",
        "SC_DEAD_WRITE_ELIMINATION", "SC_INDEX2VAR", "SC_PRINT_IR",
        "SC_BOUNDARY_CHECK", "SC_PRINT_GENCODE", "SC_KEEP_GENCODE",
        "SC_JIT_CC_OPTIONS_GROUP", "SC_CPU_JIT_FLAGS", "SC_TEMP_DIR",
        "SC_VERBOSE", "SC_RUN_THREADS", "SC_TRACE_INIT_CAP",
        "SC_EXECUTION_VERBOSE", "SC_LOGGING_FILTER", "SC_HOME", "SC_SSA_PASSES",
        "SC_PRINT_PASS_TIME", "SC_PRINT_PASS_RESULT", "SC_JIT_PROFILE",
        "SC_MIXED_FUSION", "SC_COST_MODEL"};

namespace utils {
// TODO(xxx): Copied from onednn, should be removed when merge
int getenv(const char *name, char *buffer, int buffer_size) {
    if (name == nullptr || buffer_size < 0
            || (buffer == nullptr && buffer_size > 0))
        return INT_MIN;

    int result = 0;
    int term_zero_idx = 0;
    size_t value_length = 0;

#ifdef _WIN32
    value_length = GetEnvironmentVariable(name, buffer, buffer_size);
#else
    const char *value = ::getenv(name);
    value_length = value == nullptr ? 0 : strlen(value);
#endif

    if (value_length > INT_MAX)
        result = INT_MIN;
    else {
        int int_value_length = (int)value_length;
        if (int_value_length >= buffer_size) {
#ifdef _WIN32
            if (int_value_length > 0) int_value_length -= 1;
#endif
            result = -int_value_length;
        } else {
            term_zero_idx = int_value_length;
            result = int_value_length;
#ifndef _WIN32
            if (value && buffer) strncpy(buffer, value, buffer_size - 1);
#endif
        }
    }

    if (buffer != nullptr) buffer[term_zero_idx] = '\0';
    return result;
}

int getenv_int(const char *name, int default_value) {
    int value = default_value;
    // # of digits in the longest 32-bit signed int + sign + terminating null
    const int len = 12;
    char value_str[len]; // NOLINT
    if (getenv(name, value_str, len) > 0) value = atoi(value_str);
    return value;
}

std::string getenv_string(const char *name) {
    assert(name);
    assert(strlen(name) != 0);

    const int value_strlen = ::sc::utils::getenv(name, nullptr, 0) * -1;
    assert(value_strlen >= 0);

    if (value_strlen == 0) {
        return std::string();
    } else {
        std::vector<char> buffer(value_strlen + 1);
        const int rc = ::sc::utils::getenv(name, &buffer[0], buffer.size());
        assert(rc == value_strlen);
        return std::string(&buffer[0]);
    }
}
} // namespace utils
} // namespace sc
