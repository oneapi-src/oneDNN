/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#include <atomic>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#ifdef SC_OMP_ENABLED
#include <omp.h>
#endif
#include "config.hpp"
#include "util/bf16.hpp"
#include <compiler/ir/sc_data_type.hpp>
#include <runtime/runtime.hpp>
#include <util/os.hpp>
#include <util/utils.hpp>

SC_MODULE(runtime.support)

using namespace sc;
extern "C" void print_float(float f) {
    printf("%f\n", f);
}

extern "C" void print_index(uint64_t f) {
    printf("%lu\n", f);
}

extern "C" void print_int(int f) {
    printf("%d\n", f);
}

extern "C" void print_str(char *f) {
    fputs(f, stdout);
}

extern "C" uint64_t boundary_check(
        const char *name, uint64_t idx, uint64_t acc_len, uint64_t tsr_len) {
    if (idx >= tsr_len || idx + acc_len > tsr_len) {
        fprintf(stderr,
                "Boundary check for tensor %s failed. idx=%lu acc_len=%lu "
                "tsr_len=%lu\n",
                name, idx, acc_len, tsr_len);
        abort();
    }
    return idx;
}

extern "C" void *sc_global_aligned_alloc(size_t sz, size_t align) {
    return aligned_alloc(align, (sz / align + 1) * align);
}

extern "C" void sc_global_aligned_free(void *ptr, size_t align) {
    aligned_free(ptr);
}

namespace sc {

runtime_config_t &runtime_config_t::get() {
    static runtime_config_t cfg {};
    return cfg;
}

runtime_config_t::runtime_config_t() {
    int ompmaxthreads = 1;
#ifdef SC_OMP_ENABLED
    ompmaxthreads = omp_get_max_threads();
#endif
    threads_per_instance_ = utils::getenv_int("SC_RUN_THREADS", ompmaxthreads);
    amx_exclusive_
            = static_cast<bool>(utils::getenv_int("SC_AMX_EXCLUSIVE", 0));
    if (threads_per_instance_ <= 0) {
        SC_WARN << "thread_pool_num_threads_per_instance <= 0";
        threads_per_instance_ = ompmaxthreads;
    }
    trace_initial_cap_ = 2048 * 1024;
    trace_out_path_ = utils::getenv_string("SC_TRACE");
    execution_verbose_ = (utils::getenv_int("SC_EXECUTION_VERBOSE", 0) == 1);
}

template <typename T>
struct type_converter_t {
    using StreamT = T;
};

template <>
struct type_converter_t<int8_t> {
    using StreamT = int32_t;
};

template <>
struct type_converter_t<uint8_t> {
    using StreamT = uint32_t;
};

template <>
struct type_converter_t<bf16_t> {
    using StreamT = float;
};

template <typename T>
static void set_precision(std::ofstream &ofs) {
    // should be a constexpr-if
    if (std::is_floating_point<T>::value) {
        ofs.precision(std::numeric_limits<T>::max_digits10);
    }
}

template <typename T>
static void dump_str(std::ofstream &ofs, void *tsr, size_t sz) {
    T *buf = reinterpret_cast<T *>(tsr);
    set_precision<T>(ofs);
    for (size_t i = 0; i < sz / sizeof(T); i++) {
        ofs << static_cast<typename type_converter_t<T>::StreamT>(buf[i])
            << '\n';
    }
}
} // namespace sc

extern "C" void sc_dump_tensor(void *tsr, const char *name, const char *shape,
        size_t size, size_t limit, const char *path, bool binary_fmt,
        uint64_t idtype) {
    static std::atomic<int> cnt = {0};
    std::string outpath = std::string(path);
    outpath += '/';
    outpath += name;
    outpath += '.';
    outpath += std::to_string(cnt++);
    if (limit == 0) limit = size;
    size_t real_size = std::min(limit, size);
    if (!binary_fmt) {
        outpath += ".txt";
        std::ofstream ofs(outpath);
        ofs << "shape=" << shape << "\n";
        switch (idtype) {
            case datatypes::s32: dump_str<int32_t>(ofs, tsr, real_size); break;
            case datatypes::f32: dump_str<float>(ofs, tsr, real_size); break;
            case datatypes::bf16: dump_str<bf16_t>(ofs, tsr, real_size); break;
            case datatypes::s8: dump_str<int8_t>(ofs, tsr, real_size); break;
            case datatypes::u8: dump_str<uint8_t>(ofs, tsr, real_size); break;
            default: std::cerr << "Bad type for sc_dump_tensor"; std::abort();
        }
    } else {
        const char *dtype_name;
        switch (idtype) {
            case datatypes::s32: dtype_name = "int32"; break;
            case datatypes::f32: dtype_name = "float32"; break;
            case datatypes::bf16: dtype_name = "bfloat16"; break;
            case datatypes::s8: dtype_name = "int8"; break;
            case datatypes::u8: dtype_name = "uint8"; break;
            default: std::cerr << "Bad type for sc_dump_tensor"; std::abort();
        }
        outpath += ".npy";
        std::string shape_str;
        if (real_size != size) {
            shape_str = std::to_string(real_size);
            shape = shape_str.c_str();
        }
        FILE *of = fopen(outpath.c_str(), "wb");
        // numpy array version 1.0
        const char numpy_magic[] = "\x93NUMPY\x01\x00";
        fwrite(numpy_magic, sizeof(numpy_magic) - 1, 1, of);
        std::stringstream ss;
        ss << "{\"descr\": \"" << dtype_name
           << "\", \"fortran_order\": False, \"shape\": (" << shape << ")}";
        std::string header = ss.str();
        uint16_t headerlen = header.size();
        fwrite(&headerlen, sizeof(headerlen), 1, of);
        fwrite(header.c_str(), header.size(), 1, of);
        char *buf = reinterpret_cast<char *>(tsr);
        fwrite(buf, real_size, 1, of);
        fclose(of);
    }
}

extern "C" void sc_value_check(void *tsr, const char *name, size_t size) {
    // temporarily assume dtype is float32
    float *buf = reinterpret_cast<float *>(tsr);
    for (size_t i = 0; i < size / sizeof(float); i++) {
        float val = static_cast<float>(buf[i]);
        if (std::isnan(val) || std::isinf(val)) {
            SC_MODULE_WARN << "Invalid value (nan or inf) found in tensor "
                           << name << " idx=" << i;
        }
    }
}
