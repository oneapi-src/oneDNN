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

#ifndef SYCL_LEVEL_ZERO_UTILS_HPP
#define SYCL_LEVEL_ZERO_UTILS_HPP

#if defined(DNNL_WITH_LEVEL_ZERO)

#if !(defined(__linux__) || defined(_WIN32))
#error "Level Zero is supported with Linux and Windows only"
#endif

#include <level_zero/ze_api.h>

#include "common/verbose.hpp"

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            if (get_verbose()) \
                printf("dnnl_verbose,gpu,ze_error,%d\n", (int)(res_)); \
            return status::runtime_error; \
        } \
    } while (false)

namespace dnnl {
namespace impl {
namespace sycl {

void *find_ze_symbol(const char *symbol);
template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)find_ze_symbol(symbol);
}

status_t func_zeModuleCreate(ze_device_handle_t hDevice,
        const ze_module_desc_t *desc, ze_module_handle_t *phModule,
        ze_module_build_log_handle_t *phBuildLog) {
    using func_type
            = ze_result_t (*)(ze_device_handle_t, const ze_module_desc_t *,
                    ze_module_handle_t *, ze_module_build_log_handle_t *);
    static auto f = find_ze_symbol<func_type>("zeModuleCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, desc, phModule, phBuildLog));
    return status::success;
}

status_t func_zeModuleDestroy(ze_module_handle_t hModule) {
    using func_type = ze_result_t (*)(ze_module_handle_t);
    static auto f = find_ze_symbol<func_type>("zeModuleDestroy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hModule));
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif

#endif // SYCL_LEVEL_ZERO_UTILS_HPP
