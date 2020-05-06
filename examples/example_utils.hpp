/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef EXAMPLE_UTILS_HPP
#define EXAMPLE_UTILS_HPP

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <initializer_list>

#include "dnnl.hpp"
#include "dnnl_debug.h"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP

#ifdef _MSC_VER
#define PRAGMA_MACRo(x) __pragma(x)
#define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
#else
#define PRAGMA_MACRo(x) _Pragma(#x)
#define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
#endif

// MSVC doesn't support collapse clause in omp parallel
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#endif

#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n) PRAGMA_MACRO(omp parallel for collapse(n))
#else // DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n)
#endif

// Exception class to indicate that the example uses a feature that is not
// available on the current systems. It is not treated as an error then, but
// just notifies a user.
struct example_allows_unimplemented : public std::exception {
    example_allows_unimplemented(const char *message) noexcept
        : message(message) {}
    const char *what() const noexcept override { return message; }
    const char *message;
};

inline const char *engine_kind2str_upper(dnnl::engine::kind kind);

// Runs example function with signature void() and catches errors.
// Returns `0` on success, `1` or oneDNN error, and `2` on example error.
inline int handle_example_errors(
        std::initializer_list<dnnl::engine::kind> engine_kinds,
        std::function<void()> example) {
    int exit_code = 0;

    try {
        example();
    } catch (example_allows_unimplemented &e) {
        std::cout << e.message << std::endl;
        exit_code = 0;
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::exception &e) {
        std::cout << "Error in the example: " << e.what() << "." << std::endl;
        exit_code = 2;
    }

    std::string engine_kind_str;
    for (auto it = engine_kinds.begin(); it != engine_kinds.end(); ++it) {
        if (it != engine_kinds.begin()) engine_kind_str += "/";
        engine_kind_str += engine_kind2str_upper(*it);
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << engine_kind_str << "." << std::endl;
    return exit_code;
}

// Same as above, but for functions with signature
// void(dnnl::engine::kind engine_kind, int argc, char **argv).
inline int handle_example_errors(
        std::function<void(dnnl::engine::kind, int, char **)> example,
        dnnl::engine::kind engine_kind, int argc, char **argv) {
    return handle_example_errors(
            {engine_kind}, [&]() { example(engine_kind, argc, argv); });
}

// Same as above, but for functions with signature void(dnnl::engine::kind).
inline int handle_example_errors(
        std::function<void(dnnl::engine::kind)> example,
        dnnl::engine::kind engine_kind) {
    return handle_example_errors(
            {engine_kind}, [&]() { example(engine_kind); });
}

inline dnnl::engine::kind parse_engine_kind(
        int argc, char **argv, int extra_args = 0) {
    // Returns default engine kind, i.e. CPU, if none given
    if (argc == 1) {
        return dnnl::engine::kind::cpu;
    } else if (argc <= extra_args + 2) {
        std::string engine_kind_str = argv[1];
        // Checking the engine type, i.e. CPU or GPU
        if (engine_kind_str == "cpu") {
            return dnnl::engine::kind::cpu;
        } else if (engine_kind_str == "gpu") {
            // Checking if a GPU exists on the machine
            if (dnnl::engine::get_count(dnnl::engine::kind::gpu) == 0) {
                std::cout << "Could not find compatible GPU" << std::endl
                          << "Please run the example with CPU instead"
                          << std::endl;
                exit(1);
            }
            return dnnl::engine::kind::gpu;
        }
    }

    // If all above fails, the example should be ran properly
    std::cout << "Inappropriate engine kind." << std::endl
              << "Please run the example like this: " << argv[0] << " [cpu|gpu]"
              << (extra_args ? " [extra arguments]" : "") << "." << std::endl;
    exit(1);
}

inline const char *engine_kind2str_upper(dnnl::engine::kind kind) {
    if (kind == dnnl::engine::kind::cpu) return "CPU";
    if (kind == dnnl::engine::kind::gpu) return "GPU";
    assert(!"not expected");
    return "<Unknown engine>";
}

inline dnnl::memory::dim product(const dnnl::memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1,
            std::multiplies<dnnl::memory::dim>());
}

// Read from memory, write to handle
inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        for (size_t i = 0; i < bytes; ++i)
            ((uint8_t *)handle)[i] = src[i];
    }
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    else if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();

        cl_int ret = clEnqueueReadBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error("clEnqueueReadBuffer failed.");
    }
#endif
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        for (size_t i = 0; i < bytes; ++i)
            dst[i] = ((uint8_t *)handle)[i];
    }
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    else if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();
        size_t bytes = mem.get_desc().get_size();

        cl_int ret = clEnqueueWriteBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error("clEnqueueWriteBuffer failed.");
    }
#endif
}

#endif
