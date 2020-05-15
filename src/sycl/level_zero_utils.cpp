/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifdef __linux__
#include <dlfcn.h>
#endif

#ifdef _WIN32
#include "windows.h"
#endif

#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

void *find_ze_symbol(const char *symbol) {
#if defined(__linux__)
    void *handle = dlopen("libze_loader.so", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    HMODULE handle = LoadLibraryA("ze_loader.dll");
#endif
    if (!handle) {
        if (get_verbose())
            printf("dnnl_verbose,gpu,error,cannot find Level Zero loader "
                   "library\n");
        assert(!"not expected");
        return nullptr;
    }

#if defined(__linux__)
    void *f = dlsym(handle, symbol);
#elif defined(_WIN32)
    void *f = GetProcAddress(handle, symbol);
#endif
    if (!f) {
        if (get_verbose())
            printf("dnnl_verbose,gpu,error,cannot find symbol: %s\n", symbol);
        assert(!"not expected");
    }
    return f;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
