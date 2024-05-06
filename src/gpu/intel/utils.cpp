/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/intel/utils.hpp"

#include "common/utils.hpp"

#include <cstdio>
#include <mutex>
#include <sstream>

#ifndef DNNL_ENABLE_JIT_DUMP
#define DNNL_ENABLE_JIT_DUMP 1
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gpu_utils {

bool is_jit_dump_enabled() {
#if DNNL_ENABLE_JIT_DUMP
    return get_jit_dump();
#else
    return false;
#endif
}

status_t dump_kernel_binary(
        const std::vector<uint8_t> &binary, const std::string &name) {
    if (!is_jit_dump_enabled()) return status::success;

    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    static int counter = 0;
    std::ostringstream fname;
    fname << "dnnl_dump_gpu_" << name << "." << counter << ".bin";

    FILE *fp = fopen(fname.str().c_str(), "wb+");
    if (!fp) return status::runtime_error;

    fwrite(binary.data(), binary.size(), 1, fp);
    fclose(fp);

    counter++;
    return status::success;
}

} // namespace gpu_utils
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
