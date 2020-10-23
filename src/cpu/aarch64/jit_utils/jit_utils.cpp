/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
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

#include <mutex>

#include "common/utils.hpp"

#ifndef DNNL_ENABLE_JIT_DUMP
#define DNNL_ENABLE_JIT_DUMP 1
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace jit_utils {

// WARNING: These functions are not thread safe and must be protected by a
// mutex

// TODO (rsdubtso): support prefix for code dumps

void dump_jit_code(const void *code, size_t code_size, const char *code_name) {
#if DNNL_ENABLE_JIT_DUMP
    if (code && get_jit_dump()) {
        static int counter = 0;
#define MAX_FNAME_LEN 256
        char fname[MAX_FNAME_LEN + 1];
        // TODO (Roma): support prefix for code / linux perf dumps
        snprintf(fname, MAX_FNAME_LEN, "dnnl_dump_%s.%d.bin", code_name,
                counter);
        counter++;

        FILE *fp = fopen(fname, "w+");
        // Failure to dump code is not fatal
        if (fp) {
            size_t unused = fwrite(code, code_size, 1, fp);
            UNUSED(unused);
            fclose(fp);
        }
    }
#undef MAX_FNAME_LEN
#else
    UNUSED(code);
    UNUSED(code_size);
    UNUSED(code_name);
#endif
}

void register_jit_code(const void *code, size_t code_size,
        const char *code_name, const char *source_file_name) {
    // The #ifdef guards are required to avoid generating a function that only
    // consists of lock and unlock code
#if DNNL_ENABLE_JIT_PROFILING || DNNL_ENABLE_JIT_DUMP
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    dump_jit_code(
            code, code_size * sizeof(uint32_t) / sizeof(uint8_t), code_name);
#else
    UNUSED(code);
    UNUSED(code_size);
    UNUSED(code_name);
    UNUSED(source_file_name);
#endif
}

} // namespace jit_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
