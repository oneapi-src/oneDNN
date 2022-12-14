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
#include <io.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif
#include <atomic>
#include <stdlib.h>
#include <string>
#include "utils.hpp"
#include <compiler/ir/sc_data_type.hpp>
#include <runtime/config.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/logging.hpp>

#ifdef _WIN32
#define getprocessid GetCurrentProcessId
#else
#define getprocessid getpid
#endif

namespace sc {
namespace utils {

static std::atomic<int32_t> cnt {0};

std::string get_unique_name_for_file() {
    std::stringstream name_maker;
    name_maker << getprocessid() << '_' << ++cnt;
    return name_maker.str();
}

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

uint32_t get_sizeof_etype(sc_data_etype etype) {
    switch (etype) {
        case sc_data_etype::S8:
        case sc_data_etype::U8:
        case sc_data_etype::BOOLEAN: return 1;
        case sc_data_etype::U16:
        case sc_data_etype::F16:
        case sc_data_etype::BF16: return 2;
        case sc_data_etype::U32:
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

compiler_configs_t &compiler_configs_t::get() {
    static compiler_configs_t cfg {};
    return cfg;
}

const std::string &compiler_configs_t::get_temp_dir_path() {
    const std::string &temp_dir = compiler_configs_t::get().temp_dir_;
    if (temp_dir.empty()) {
        COMPILE_ASSERT(0,
                "Current SC_TEMP_DIR can not be writen as temp directory, "
                "please make your writable temp directory and use with "
                "SC_TEMP_DIR=/path/to/temp");
    }
    return temp_dir;
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
    int access_ret = 0;
#ifndef _WIN32
    access_ret = access(temp_dir_.c_str(), W_OK);
#else
    access_ret = _access(temp_dir_.c_str(), 2);
#endif
    if (access_ret != 0) { temp_dir_.clear(); }

    print_pass_time_ = utils::getenv_int(env_names[SC_PRINT_PASS_TIME], 0);
    print_pass_result_ = utils::getenv_int(env_names[SC_PRINT_PASS_RESULT], 0);
    jit_profile_ = utils::getenv_int(env_names[SC_JIT_PROFILE], 0);
}

} // namespace utils

} // namespace sc
