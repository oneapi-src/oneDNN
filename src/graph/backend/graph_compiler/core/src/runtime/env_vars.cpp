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
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
#define DEF_ENV(x) "ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_" #x
#define DEF_ENV_TRACE() "ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_KERNEL_TRACE"
const char *env_names[] = {
        DEF_ENV(CPU_JIT),
        DEF_ENV(OPT_LEVEL),
        DEF_ENV_TRACE(),
        DEF_ENV(VERBOSE),
        DEF_ENV(PRINT_PASS_RESULT),
        DEF_ENV(DUMP_GENCODE),
        DEF_ENV(C_INCLUDE),
        DEF_ENV(TRACE_INIT_CAP),
        DEF_ENV(MANAGED_THREAD_POOL),
};

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

    const int value_strlen
            = ::dnnl::impl::graph::gc::utils::getenv(name, nullptr, 0) * -1;
    assert(value_strlen >= 0);

    if (value_strlen == 0) {
        return std::string();
    } else {
        std::vector<char> buffer(value_strlen + 1);
        const int rc = ::dnnl::impl::graph::gc::utils::getenv(
                name, &buffer[0], buffer.size());
        assert(rc == value_strlen);
        return std::string(&buffer[0]);
    }
}
} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
