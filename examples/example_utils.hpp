/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

/// @file
/// Examples C++ Utility Functions

#ifndef EXAMPLE_UTILS_HPP
#define EXAMPLE_UTILS_HPP

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <stdlib.h>
#include <initializer_list>

#include "dnnl.hpp"
#include "dnnl_debug.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "dnnl_sycl.hpp"
#endif

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

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
#include "tbb/version.h"
#if defined(TBB_INTERFACE_VERSION) && (TBB_INTERFACE_VERSION >= 12060)
#include "tbb/global_control.h"
#define DNNL_TBB_NEED_EXPLICIT_FINALIZE
#endif
#endif

// TBB runtime may crash when it is used under CTest. This is a known TBB
// limitation that can be worked around by doing explicit finalization.
// The API to do that was introduced in 2021.6.0. When using an older TBB
// runtime the crash may still happen.
inline void finalize() {
#ifdef DNNL_TBB_NEED_EXPLICIT_FINALIZE
    tbb::task_scheduler_handle handle
            = tbb::task_scheduler_handle {tbb::attach {}};
    tbb::finalize(handle, std::nothrow);
#endif
}

dnnl::engine::kind validate_engine_kind(dnnl::engine::kind akind) {
    // Checking if a GPU exists on the machine
    if (akind == dnnl::engine::kind::gpu) {
        if (dnnl::engine::get_count(dnnl::engine::kind::gpu) == 0) {
            std::cout << "Application couldn't find GPU, please run with CPU "
                         "instead.\n";
            exit(0);
        }
    }
    return akind;
}

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
    finalize();
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
        return validate_engine_kind(dnnl::engine::kind::cpu);
    } else if (argc <= extra_args + 2) {
        std::string engine_kind_str = argv[1];
        // Checking the engine type, i.e. CPU or GPU
        if (engine_kind_str == "cpu") {
            return validate_engine_kind(dnnl::engine::kind::cpu);
        } else if (engine_kind_str == "gpu") {
            return validate_engine_kind(dnnl::engine::kind::gpu);
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
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
            auto src = buffer.get_host_access();
            uint8_t *src_ptr = src.get_pointer();
            if (!src_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                ((uint8_t *)handle)[i] = src_ptr[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *src_ptr = (uint8_t *)mem.get_data_handle();
            if (!src_ptr)
                throw std::runtime_error("get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    ((uint8_t *)handle)[i] = src_ptr[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(handle, src_ptr, size).wait();
            }
        }
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        void *mapped_ptr = mem.map_data();
        if (mapped_ptr) std::memcpy(handle, mapped_ptr, size);
        mem.unmap_data(mapped_ptr);
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        if (!src) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            ((uint8_t *)handle)[i] = src[i];
        return;
    }

    assert(!"not expected");
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
            auto dst = buffer.get_host_access();
            uint8_t *dst_ptr = dst.get_pointer();
            if (!dst_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                dst_ptr[i] = ((uint8_t *)handle)[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
            if (!dst_ptr)
                throw std::runtime_error("get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t *)handle)[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(dst_ptr, handle, size).wait();
            }
        }
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        void *mapped_ptr = mem.map_data();
        if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
        mem.unmap_data(mapped_ptr);
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}

#endif
