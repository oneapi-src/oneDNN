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

#include <atomic>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <runtime/data_type.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <util/bf16.hpp>
#include <util/utils.hpp>

using namespace sc;

namespace sc {
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
        uint64_t idtype, bool is_dynamic) {
    static std::atomic<int> cnt = {0};
    std::string outpath = std::string(path);
    outpath += '/';
    outpath += name;
    outpath += '.';
    outpath += std::to_string(cnt++);

    using etype_t = sc_data_etype;
    // lower 16 bits are sc_data_etype
    etype_t etype = static_cast<etype_t>(idtype & (0xffff));

    std::string dyn_shape;
    void *real_tsr = tsr;
    if (is_dynamic) {
        auto *dyn_tsr = static_cast<runtime::dynamic_tensor_t *>(tsr);
        int64_t *shape_tsr = dyn_tsr->dims_;
        size = 1;
        for (int i = 0; i < dyn_tsr->ndims_; i++) {
            size *= shape_tsr[i];
            if (i) { dyn_shape += ","; }
            dyn_shape += std::to_string(shape_tsr[i]);
        }
        real_tsr = dyn_tsr->data_;
        shape = dyn_shape.c_str();
        etype = sc_data_etype(dyn_tsr->dtype_);
        size *= utils::get_sizeof_etype(etype);
    }
    if (limit == 0) limit = size;
    size_t real_size = std::min(limit, size);

    if (!binary_fmt) {
        outpath += ".txt";
        std::ofstream ofs(outpath);
        ofs << "shape=" << shape << "\n";
        switch (etype) {
            case etype_t::S32:
                dump_str<int32_t>(ofs, real_tsr, real_size);
                break;
            case etype_t::F32: dump_str<float>(ofs, real_tsr, real_size); break;
            case etype_t::BF16:
                dump_str<bf16_t>(ofs, real_tsr, real_size);
                break;
            case etype_t::S8: dump_str<int8_t>(ofs, real_tsr, real_size); break;
            case etype_t::U8:
                dump_str<uint8_t>(ofs, real_tsr, real_size);
                break;
            default: std::cerr << "Bad type for sc_dump_tensor"; std::abort();
        }
    } else {
        const char *dtype_name;
        switch (etype) {
            case etype_t::S32: dtype_name = "int32"; break;
            case etype_t::F32: dtype_name = "float32"; break;
            case etype_t::BF16: dtype_name = "bfloat16"; break;
            case etype_t::S8: dtype_name = "int8"; break;
            case etype_t::U8: dtype_name = "uint8"; break;
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
        char *buf = static_cast<char *>(real_tsr);
        fwrite(buf, real_size, 1, of);
        fclose(of);
    }
}
