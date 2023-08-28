/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include <fstream>
#include <stdlib.h>
#include <string>
#include "file.hpp"
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

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
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

std::string etype_to_string(sc_data_etype edtype) {
    std::stringstream os;
    os << edtype;
    return os.str();
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

// select nearest even step
int get_nearest_vector_step(const int step) {
    assert(step > 0);
    int nbits = 0, n = step;
    while (n) {
        n = n >> 1;
        nbits++;
    }
    assert(nbits <= 6 || (nbits == 7 && step == 64));
    return (1 << (nbits - 1)) == step ? step : (1 << nbits);
}

compiler_configs_t &compiler_configs_t::get() {
    static compiler_configs_t cfg {};
    return cfg;
}

const std::string &compiler_configs_t::get_temp_dir_path() {
    const std::string &temp_dir = compiler_configs_t::get().temp_dir_;
    return temp_dir;
}

template <typename T>
static void parse_value(const char *name, T &v) {
    auto strv = utils::getenv_string(name);
    if (!strv.empty()) { v = T(std::stoi(strv)); };
}

using namespace env_key;
compiler_configs_t::compiler_configs_t() {
    dump_gen_code_ = utils::getenv_string(env_names[SC_DUMP_GENCODE]);
    print_pass_result_ = utils::getenv_int(env_names[SC_PRINT_PASS_RESULT], 0);

    if (temp_dir_.empty()) {
#ifndef _WIN32
        // use the order defined by POSIX standard
        do {
            temp_dir_ = utils::getenv_string("TMPDIR");
            if (!temp_dir_.empty()) { break; }
            temp_dir_ = utils::getenv_string("TMP");
            if (!temp_dir_.empty()) { break; }
            temp_dir_ = utils::getenv_string("TEMP");
            if (!temp_dir_.empty()) { break; }
            temp_dir_ = utils::getenv_string("TEMPDIR");
            if (!temp_dir_.empty()) { break; }
            temp_dir_ = "/tmp";
        } while (false);
#else
        char temp[MAX_PATH + 2];
        auto ret = GetTempPathA(MAX_PATH + 1, temp);
        if (ret != 0) { temp_dir_ = temp; }
#endif // _WIN32
    }
}

void open_file_for_write(std::ofstream &ret, const std::string &path) {
    ret.open(path);
    COMPILE_ASSERT(ret, "Cannot open file for write:" << path);
}

void open_file_for_read(std::ifstream &ret, const std::string &path) {
    ret.open(path);
    COMPILE_ASSERT(ret, "Cannot open file for read:" << path);
}

static std::string make_temp_path(const std::string &filename) {
    std::string name = compiler_configs_t::get_temp_dir_path();
    name += '/';
    name += filename;
    return name;
}

void open_temp_file_for_write(std::ofstream &ret, const std::string &filename) {
    open_file_for_write(ret, make_temp_path(filename));
}

void open_temp_file_for_read(std::ifstream &ret, const std::string &filename) {
    open_file_for_read(ret, make_temp_path(filename));
}

} // namespace utils

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
