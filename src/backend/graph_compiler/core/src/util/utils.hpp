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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_UTILS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_UTILS_HPP

#include <algorithm>
#include <iostream>
#include <math.h>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "assert.hpp"
#include "def.hpp"
#include "fdstream.hpp"
#include <compiler/ir/sc_data_type.hpp>

namespace sc {
namespace utils {
// get position index from vector
template <typename T>
inline size_t get_index(const std::vector<T> vec, const T &element) {
    auto pos = std::find_if(vec.begin(), vec.end(),
            [&](const T &other) -> bool { return other == element; });
    if (pos == vec.end()) {
        COMPILE_ASSERT(0, "Individual and vector do not match");
        return 0;
    } else {
        return std::distance(vec.begin(), pos);
    }
}

static constexpr size_t divide_and_ceil(size_t x, size_t y) {
    return (x + y - 1) / y;
}

static constexpr size_t rnd_up(const size_t a, const size_t b) {
    return (divide_and_ceil(a, b) * b);
}

static constexpr size_t rnd_dn(const size_t a, const size_t b) {
    return (a / b) * b;
}

template <typename T>
constexpr bool is_one_of(T value, T last) {
    return value == last;
}

template <typename T, typename... Args>
constexpr bool is_one_of(T value, T first, Args... args) {
    return value == first || is_one_of(value, args...);
}

// Copied from onednn utils.hpp
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Reads an environment variable 'name' and stores its string value in the
// 'buffer' of 'buffer_size' bytes (including the terminating zero) on
// success.
//
// - Returns the length of the environment variable string value (excluding
// the terminating 0) if it is set and its contents (including the terminating
// 0) can be stored in the 'buffer' without truncation.
//
// - Returns negated length of environment variable string value and writes
// "\0" to the buffer (if it is not NULL) if the 'buffer_size' is to small to
// store the value (including the terminating 0) without truncation.
//
// - Returns 0 and writes "\0" to the buffer (if not NULL) if the environment
// variable is not set.
//
// - Returns INT_MIN if the 'name' is NULL.
//
// - Returns INT_MIN if the 'buffer_size' is negative.
//
// - Returns INT_MIN if the 'buffer' is NULL and 'buffer_size' is greater than
// zero. Passing NULL 'buffer' with 'buffer_size' set to 0 can be used to
// retrieve the length of the environment variable value string.
//
SC_INTERNAL_API int getenv(const char *name, char *buffer, int buffer_size);

// Reads an integer from the environment
SC_INTERNAL_API int getenv_int(const char *name, int default_value = 0);

// A convenience wrapper for the 'getenv' function defined above.
//
// Note: Due to an apparent limitation in the wrapped 'getenv'
// function, this function makes no distinction between:
// (a) the environment variable not being defined at all, vs.
// (b) the environment variable defined, with a value of empty-string.
//
// This function's behavior is undefined if 'name' is null or the empty-string.
SC_INTERNAL_API std::string getenv_string(const char *name);

template <typename TDst, typename TSrc>
struct bind_assigner_t {
    static void assign(TDst &dst, const TSrc &src) { dst = src; }
};

template <int idx, typename T>
void bind_vector_to_args(const std::vector<T> &v) {}

template <int idx = 0, typename T, typename T2, typename... Args>
void bind_vector_to_args(const std::vector<T> &v, T2 &out1, Args &... args) {
    bind_assigner_t<T2, T>::assign(out1, v[idx]);
    bind_vector_to_args<idx + 1>(v, args...);
}

template <typename T>
void args_to_vector(std::vector<T> &v) {}

template <typename T, typename T2, typename... Args>
void args_to_vector(std::vector<T> &v, T2 &&out1, Args &&... args) {
    v.emplace_back(std::forward<T2>(out1));
    args_to_vector(v, std::move(args)...);
}

template <typename T, typename... Args>
std::vector<T> args_to_vector(Args &&... args) {
    std::vector<T> ret;
    args_to_vector(ret, std::move(args)...);
    return std::move(ret);
}

/**
 * Creates a process and waits until it exits
 * @param program the program path
 * @param args the arguments (including the program name)
 * @param exit_code outputs the exit code of the process if process creating
 *      succeeds. If failed to create the process, this value is not changed
 * @param rstdin the stdin data to dump into the process, nullable
 * @param rstdout receives the stdout data from the process, nullable
 * @param rstderr receives the stderr data from the process, nullable
 * @return true if creating process succeeded
 * */
SC_INTERNAL_API bool create_process_and_await(const std::string &program,
        const std::vector<std::string> &args, int &exit_code,
        const std::string *rstdin = nullptr, std::string *rstdout = nullptr,
        std::string *rstderr = nullptr);

/**
 * Creates a process without waiting for its termination. May redirect stdin,
 * stdout, stderr as stream
 * @param program the program path
 * @param args the arguments (including the program name)
 * @param outhandle outputs handle of the process if process creating
 *      succeeds. If failed to create the process, this value is not changed
 * @param rstdin the stdin stream to dump into the process, nullable
 * @param rstdout receives the stdout data from the process, nullable
 * @param rstderr receives the stderr data from the process, nullable
 * @return true if creating process succeeded
 * */
SC_INTERNAL_API bool create_process(const std::string &program,
        const std::vector<std::string> &args, uintptr_t &outhandle,
        ofdstream_t *rstdin = nullptr, ifdstream_t *rstdout = nullptr,
        ifdstream_t *rstderr = nullptr);

// waits for the termination of the process. Returns true if succeeded
SC_INTERNAL_API bool wait_process(uintptr_t outhandle, int &exit_code);

// Gets the SC_HOME path by env variables. Returns empty string if
// SC_HOME is not set
SC_INTERNAL_API const std::string &get_sc_home_path();

#define MACRO_2_STR_HELPER(x) #x
#define MACRO_2_STR(name) MACRO_2_STR_HELPER(name)

/**
 * Get the factors for a given size, which will be used by the
 * tuner
 * @param X size of a given dimension
 * @return the factors for a given size
 * */
inline std::vector<int> get_factors(const int X) {
    std::vector<int> factors;
    for (auto i = 1; i <= (int)sqrt((double)X); ++i) {
        if (X % i == 0) {
            factors.push_back(i);
            if (X / i != i) factors.push_back(X / i);
        }
    }
    std::sort(factors.begin(), factors.end());
    return factors;
}

/**
 * Get the possible block size list for a given size, which will be used by the
 * tuner
 * @param X size of a given dimension
 * @return a list of block size
 * */
inline std::vector<int> get_blocks(
        const int X, int threshold = 8, int floor = 1024) {
    std::vector<int> blocks;
    for (auto i = 1; i <= (int)sqrt((double)X); ++i) {
        if (X % i == 0) {
            // add a judgement here to prune search space for efficient search
            if (i >= threshold && i <= floor) { blocks.push_back(i); }
            auto div_x = X / i;
            if (div_x >= threshold && div_x <= floor && (div_x != i)) {
                blocks.push_back(div_x);
            }
        }
    }
    if (blocks.empty()) {
        blocks = get_factors(X);
    } else {
        std::sort(blocks.begin(), blocks.end());
    }
    return blocks;
}

/**
 * Gets the size of a etype in bytes
 * */
uint32_t get_sizeof_etype(sc_data_etype etype);

/**
 * Gets the size of a type in bytes
 * */
uint64_t get_sizeof_type(sc_data_type_t dtype);

/**
 * A convenience wrapper around the standard 'strerror_r' function.
 * */
SC_INTERNAL_API std::string get_error_msg(int errnum);

/**
 * Gets the size of OS memory page
 * */
SC_INTERNAL_API size_t get_os_page_size();

/**
 * Gets the file path of a dynamic library
 * @param addr an address of any function in the library
 * @return the library path, or empty if anything goes wrong
 * */
SC_INTERNAL_API std::string get_dyn_lib_path(void *addr);

struct logging_stream_t {
    std::ostream *stream_;
    const char *append_;
    logging_stream_t(std::ostream *stream, const char *append)
        : stream_(stream), append_(append) {}
    ~logging_stream_t() {
        if (stream_) *stream_ << append_;
    }
    operator bool() const { return stream_; };
};

logging_stream_t get_info_logging_stream(const char *module_name = nullptr);
logging_stream_t get_warning_logging_stream(const char *module_name = nullptr);
logging_stream_t get_fatal_logging_stream(const char *module_name = nullptr);

enum verbose_level { FATAL = 0, WARNING, INFO };

void set_logging_stream(std::ostream *s);

struct SC_INTERNAL_API compiler_configs_t {
    bool print_gen_code_;
    bool keep_gen_code_;
    std::string jit_cc_options_;
    std::vector<std::string> cpu_jit_flags_;
    std::string temp_dir_;
    verbose_level verbose_level_;
    bool print_pass_time_;
    bool print_pass_result_;
    bool jit_profile_;

    static compiler_configs_t &get();

private:
    compiler_configs_t();
};

} // namespace utils
} // namespace sc

#define SC_INFO \
    if (auto __sc_stream_temp__ = ::sc::utils::get_info_logging_stream()) \
    (*__sc_stream_temp__.stream_)

#define SC_MODULE_INFO2(NAME) \
    if (auto __sc_stream_temp__ = ::sc::utils::get_info_logging_stream(NAME)) \
    (*__sc_stream_temp__.stream_)

#define SC_MODULE_INFO SC_MODULE_INFO2(__sc_module_name)

#define SC_WARN \
    if (auto __sc_stream_temp__ = ::sc::utils::get_warning_logging_stream()) \
    (*__sc_stream_temp__.stream_)
#define SC_MODULE_WARN \
    if (auto __sc_stream_temp__ \
            = ::sc::utils::get_warning_logging_stream(__sc_module_name)) \
    (*__sc_stream_temp__.stream_)

#define SC_FATAL \
    if (auto __sc_stream_temp__ = ::sc::utils::get_fatal_logging_stream()) \
    (*__sc_stream_temp__.stream_)
#define SC_MODULE_FATAL \
    if (auto __sc_stream_temp__ \
            = ::sc::utils::get_fatal_logging_stream(__sc_module_name)) \
    (*__sc_stream_temp__.stream_)

#define SC_MODULE(NAME) static constexpr const char *__sc_module_name = #NAME;

#endif
