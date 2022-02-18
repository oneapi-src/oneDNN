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

#include <climits>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif
#include <stdlib.h>
#include <string>
#include "utils.hpp"
#include <compiler/config/env_vars.hpp>
#include <compiler/ir/sc_data_type.hpp>

namespace sc {
namespace utils {

std::string get_dyn_lib_path(void *addr) {
#ifdef _WIN32
    // fix-me: add impl
    throw std::runtime_error("get_dyn_lib_path");
#else
    // On Windows, use GetMappedFileNameW
    Dl_info info;
    if (dladdr(addr, &info)) { return info.dli_fname; }
#endif
    return std::string();
}

static std::string get_sc_home_path_() {
    std::string path;
    char home_path[512];
    if (utils::getenv(
                env_names[env_key::SC_HOME_], home_path, sizeof(home_path))
            != 0) {
        path = home_path;
    } else {
#ifdef SC_HOME
        path = MACRO_2_STR(SC_HOME);
#else
        std::cerr << "environment variable SC_HOME is not set";
#endif
    }
    return path;
}

const std::string &get_sc_home_path() {
    static std::string path = get_sc_home_path_();
    return path;
}

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

uint32_t get_sizeof_etype(sc_data_etype etype) {
    switch (etype) {
        case sc_data_etype::S8:
        case sc_data_etype::U8:
        case sc_data_etype::BOOLEAN: return 1;
        case sc_data_etype::F16:
        case sc_data_etype::BF16: return 2;
        case sc_data_etype::F32:
        case sc_data_etype::S32: return 4;
        case sc_data_etype::GENERIC:
        case sc_data_etype::INDEX:
            return 8; // TODO(xxx): should be target dependent
        default:
            if (etypes::is_pointer(etype)) { return 8; };
            assert(0);
            return 0;
    }
}

uint64_t get_sizeof_type(sc_data_type_t dtype) {
    return get_sizeof_etype(dtype.type_code_) * dtype.lanes_;
}

size_t get_os_page_size() {
#ifdef _WIN32
    // fix-me: (win32) impl
    return 4096;
#else
    static size_t v = getpagesize();
    return v;
#endif
}

std::string get_error_msg(int errnum) {
#if defined(_WIN32) || defined(__APPLE__)
    // fix-me: (win32)
    return "Error message from get_error_msg";
#else
    std::vector<char> buffer(1024); // gnu docs say this is big enough

#if (_POSIX_C_SOURCE >= 200112L) && !_GNU_SOURCE
    // The XSI-compliant version of strerror_r
    if (strerror_r(errnum, &buffer[0], buffer.size()) == 0) {
        return std::string(&buffer[0]);
    } else {
        return "Failed call to strerror_r";
    }
#else
    // The GCC version of strerror_r
    return std::string(strerror_r(errnum, &buffer[0], buffer.size()));
#endif
#endif
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

compiler_configs_t &compiler_configs_t::get() {
    static compiler_configs_t cfg {};
    return cfg;
}

using namespace env_key;
compiler_configs_t::compiler_configs_t() {
    print_gen_code_ = utils::getenv_int(env_names[SC_PRINT_GENCODE], 0);
    const int default_keep_gencode =
#if SC_PROFILING == 1
            1;
#else
            0;
#endif
    keep_gen_code_ = utils::getenv_int(
            env_names[SC_KEEP_GENCODE], default_keep_gencode);
    jit_cc_options_ = utils::getenv_string(env_names[SC_JIT_CC_OPTIONS_GROUP]);
    cpu_jit_flags_ = utils::string_split(
            utils::getenv_string(env_names[SC_CPU_JIT_FLAGS]), " ");
    temp_dir_ = utils::getenv_string(env_names[SC_TEMP_DIR]);
    if (temp_dir_.empty()) { temp_dir_ = "/tmp"; }

    constexpr int default_verbose = 0;
    int tmp_get_verbose_level
            = utils::getenv_int(env_names[SC_VERBOSE], default_verbose);
    if (tmp_get_verbose_level < 0 || tmp_get_verbose_level > 2) {
        tmp_get_verbose_level = 0;
    }
    verbose_level_ = (verbose_level)tmp_get_verbose_level;

    print_pass_time_ = utils::getenv_int("SC_PRINT_PASS_TIME", 0);
    print_pass_result_ = utils::getenv_int("SC_PRINT_PASS_RESULT", 0);
}

} // namespace utils
} // namespace sc
