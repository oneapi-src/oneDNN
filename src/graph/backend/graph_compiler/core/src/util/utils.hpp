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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_UTILS_HPP

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
#include <runtime/env_var.hpp>
#include <runtime/logging.hpp>
#include <util/simple_math.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
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
std::unique_ptr<T> make_unique(Args &&...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename TDst, typename TSrc>
struct bind_assigner_t {
    static void assign(TDst &dst, const TSrc &src) { dst = src; }
};

template <int idx, typename T>
void bind_vector_to_args(const std::vector<T> &v) {}

template <int idx = 0, typename T, typename T2, typename... Args>
void bind_vector_to_args(const std::vector<T> &v, T2 &out1, Args &...args) {
    bind_assigner_t<T2, T>::assign(out1, v[idx]);
    bind_vector_to_args<idx + 1>(v, args...);
}

template <typename T, typename T2>
void bind_vector_to_args(const std::vector<T> &v, std::vector<T2> &out) {
    for (size_t idx = 0; idx < v.size(); idx++) {
        bind_assigner_t<T2, T>::assign(out[idx], v[idx]);
    }
}

template <typename T>
void args_to_vector(std::vector<T> &v) {}

template <typename T, typename T2, typename... Args>
void args_to_vector(std::vector<T> &v, T2 &&out1, Args &&...args) {
    v.emplace_back(std::forward<T2>(out1));
    args_to_vector(v, std::move(args)...);
}

template <typename T, typename... Args>
std::vector<T> args_to_vector(Args &&...args) {
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
SC_INTERNAL_API uint32_t get_sizeof_etype(sc_data_etype etype);

/**
 * Gets the size of a type in bytes
 * */
SC_INTERNAL_API uint64_t get_sizeof_type(sc_data_type_t dtype);

/**
 * A convenience wrapper around the standard 'strerror_r' function.
 * */
SC_INTERNAL_API std::string get_error_msg(int errnum);

/**
 * Gets the file path of a dynamic library
 * @param addr an address of any function in the library
 * @return the library path, or empty if anything goes wrong
 * */
SC_INTERNAL_API std::string get_dyn_lib_path(void *addr);

/**
 * Get the nearest even step of the for loop
 */
SC_INTERNAL_API int get_nearest_vector_step(int step);

/**
 * Get the string of etype.
 */
SC_INTERNAL_API std::string etype_to_string(sc_data_etype edtype);

struct SC_INTERNAL_API compiler_configs_t {
    bool print_gen_code_;
    std::string dump_gen_code_;
    std::string jit_cc_options_;
    std::vector<std::string> cpu_jit_flags_;
    bool xbyak_jit_save_obj_ = false;
    bool xbyak_jit_asm_listing_ = false;
    bool xbyak_jit_log_stack_frame_model_ = false;
    bool xbyak_jit_pause_after_codegen_ = false;
    bool diagnose_ = false;
    bool printer_print_address_ = false;
    bool print_pass_time_ = false;
    bool print_pass_result_ = false;
    bool jit_profile_ = false;

    static compiler_configs_t &get();
    static const std::string &get_temp_dir_path();

private:
    compiler_configs_t();
    // set to private to prevent use without permission check
    std::string temp_dir_;
};

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
